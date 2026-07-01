import math
import typing
import warnings

import numpy as np
import torch

from .common import (
    choose_offsets,
    get_block_counts,
    get_dtype_bytes,
    get_instance_size,
    povs_preflight,
    purge_dependent_param,
)
from .constants import MAX_SEED, MIN_CUDA_ARCH, MIN_SEED, MIN_VBLOCK_SIZE
from .numpy import _safe_read_arr, _safe_set_arr
from .torch import _choose_pblock_size, _choose_thr_block_size, _validate_thr_block_size
from .types import Options
from .utils import numpy_to_torch_dtype


def shuffle(
    data: np.memmap,
    iterations: int,
    options: Options | None,
    seed: int,
) -> None:
    """POV Shuffle implementation for `numpy.memmap`.

    :param data: Memmap file wrapper to shuffle in place along the axis 0.
    :param iterations: Number of shuffling iterations to perform.
    :param options: POV Shuffle algorithm options.
    :param seed: Random seed.
    """
    from ._cuda import torch_binding

    options = typing.cast(
        "_MemmapOptions",
        options if options_is_complete(options) else optim_options_for_dataset(data, options),
    )

    preflight(data, iterations, options)

    device = options.memmap_cuda_device_id
    generator = torch.Generator(device=device).manual_seed(seed)

    instances_per_vblock = options.virtual_block_size * options.physical_block_size
    n_pblocks, n_vblocks = get_block_counts(**options._asdict(), deck_size=len(data))
    num_batches = math.ceil(n_vblocks / options.memmap_batch_size)

    pblk_size = options.physical_block_size
    last_pblock_deficit = n_pblocks * pblk_size - len(data)  # zero if deck_size is divisible by pblk_size

    batch_data = torch.empty(
        (options.memmap_batch_size * instances_per_vblock, get_instance_size(data)),
        device=device,
        dtype=numpy_to_torch_dtype(data.dtype),
    )  # Batch instances on device memory
    cpu_buffer = np.empty(batch_data.shape, dtype=data.dtype)  # Buffer of `batch_data` on CPU RAM

    for _ in range(iterations):
        vbid_2_bids = torch.randperm(
            n_vblocks * options.virtual_block_size, device=device, dtype=torch.int64, generator=generator
        ).reshape(n_vblocks, options.virtual_block_size)
        vbid_2_bids[vbid_2_bids >= n_pblocks] = -1

        # Move the vblock that has the smaller pblock to the end, and that pblock within it to the last position
        # This ensures that we only need to handle the last pblock deficit in the tail of the last batch
        if last_pblock_deficit > 0:
            row_idx, col_idx = (vbid_2_bids == n_pblocks - 1).nonzero(as_tuple=False)[0].tolist()
            last_row, last_col = n_vblocks - 1, options.virtual_block_size - 1
            if row_idx != last_row:
                vbid_2_bids[[row_idx, last_row]] = vbid_2_bids[[last_row, row_idx]]
                row_idx = last_row
            if col_idx != last_col:
                vbid_2_bids[row_idx, [col_idx, last_col]] = vbid_2_bids[row_idx, [last_col, col_idx]]

        offset = options.offsets[
            int(torch.randint(len(options.offsets), (1,), device=device, generator=generator).item())
        ]

        for i in range(num_batches):
            batch_assignments = vbid_2_bids[i * options.memmap_batch_size : (i + 1) * options.memmap_batch_size, :]

            num_vblks_in_batch = len(batch_assignments)
            instances_in_batch = instances_per_vblock * num_vblks_in_batch - (
                last_pblock_deficit if i == num_batches - 1 else 0
            )

            batch_data_slice = batch_data[:instances_in_batch]
            cpu_buffer_slice = cpu_buffer[:instances_in_batch]

            seeds = torch.randint(
                MIN_SEED, MAX_SEED, size=(num_vblks_in_batch,), device=device, dtype=torch.int32, generator=generator
            )

            proxy_assignments = torch.arange(0, batch_assignments.numel(), device=device, dtype=torch.int64).reshape(
                batch_assignments.shape
            )
            proxy_assignments[batch_assignments == -1] = -1

            for proxy_pbid, pbid in enumerate(batch_assignments.ravel()):
                if pbid != -1:
                    pblk_data = _safe_read_arr(data, offset, int(pbid) * pblk_size, pblk_size)
                    start = proxy_pbid * pblk_size
                    cpu_buffer_slice[start : start + len(pblk_data)] = pblk_data
            batch_data_slice.copy_(torch.from_numpy(cpu_buffer_slice))

            torch_binding(
                batch_data_slice,
                proxy_assignments,
                seeds,
                0,  # offset has already been accounted for when loading the instances
                options.physical_block_size,
                options.virtual_block_size,
                options.gpu_thread_block_size,
            )

            torch.from_numpy(cpu_buffer_slice).copy_(batch_data_slice)
            for proxy_pbid, pbid in enumerate(batch_assignments.ravel()):
                if pbid != -1:
                    start = proxy_pbid * pblk_size
                    _safe_set_arr(data, offset, int(pbid) * pblk_size, cpu_buffer_slice[start : start + pblk_size])

            data.flush()


class _MemmapOptions(Options):
    virtual_block_size: int
    physical_block_size: int
    offsets: list[int]
    gpu_thread_block_size: int
    memmap_batch_size: int
    memmap_cuda_device_id: int


def options_is_complete(options: Options | None) -> bool:
    return options is not None and None not in (
        options.virtual_block_size,
        options.physical_block_size,
        options.gpu_thread_block_size,
        options.offsets,
        options.memmap_batch_size,
        options.memmap_cuda_device_id,
    )


def optim_options_for_dataset(
    data: np.memmap,
    options: Options | None,
) -> Options:
    """Choose POV Shuffle options for dataset."""
    options = options or Options()

    if options_is_complete(options):
        warnings.warn("All parameters already specified, skipping optimization")
        return options

    instance_size = get_instance_size(data)
    dtype_bytes = get_dtype_bytes(data)

    if options.virtual_block_size is None:
        if options.physical_block_size is not None:
            purge_dependent_param(options, "virtual_block_size", "physical_block_size")
        if options.gpu_thread_block_size is not None:
            purge_dependent_param(options, "virtual_block_size", "gpu_thread_block_size")
        if options.memmap_batch_size is not None:
            purge_dependent_param(options, "virtual_block_size", "memmap_batch_size")

    if options.physical_block_size is None:
        if options.offsets is not None:
            purge_dependent_param(options, "physical_block_size", "offsets")
        if options.memmap_batch_size is not None:
            purge_dependent_param(options, "physical_block_size", "memmap_batch_size")

    return Options(
        memmap_cuda_device_id=(device_id := options.memmap_cuda_device_id or 0),
        virtual_block_size=(vblk := options.virtual_block_size or MIN_VBLOCK_SIZE),
        physical_block_size=(
            pblk := options.physical_block_size or _choose_pblock_size(instance_size, dtype_bytes, vblk, device_id)
        ),
        memmap_batch_size=_choose_batch_size(
            device_id,
            instance_size=instance_size,
            dtype_bytes=dtype_bytes,
            virtual_block_size=vblk,
            physical_block_size=pblk,
        ),
        gpu_thread_block_size=_choose_thr_block_size(
            instance_size=instance_size,
            dtype_bytes=dtype_bytes,
            vblock_size=vblk,
            pblock_size=pblk,
            device_id=device_id,
        ),
        offsets=choose_offsets(
            deck_size=data.shape[0],
            instance_size=instance_size,
            dtype_bytes=dtype_bytes,
            pblock_size=pblk,
        ),
    )


def preflight(
    data: np.memmap,
    iterations: int,
    options: _MemmapOptions,
) -> None:
    """Preflight checks for POV Shuffle on torch tensors."""

    # Dataset preflight  # In sync with: src/povs/__cuda/binds/torch.cpp — dataset preflight
    dtypes = (np.float16, np.float32, np.float64, np.int32, np.int64)
    assert (device_id := options.memmap_cuda_device_id) != -1, "Device must be CUDA"
    assert data.dtype in dtypes, f"Invalid dtype {data.dtype.__str__()}, must be one of {dtypes}"

    # Device preflight  # In sync with: src/povs/__cuda/lib/povs.cu — device preflight
    dev = torch.cuda.get_device_properties(device_id)
    assert (arch := (dev.major, dev.minor)) >= MIN_CUDA_ARCH, f"CUDA arch {arch} must be at least {MIN_CUDA_ARCH}"

    # Standard preflight  # In sync with: src/povs/__cuda/lib/povs.cu — standard preflight
    povs_preflight(data, iterations, options)

    # GPU thread-block preflight  # In sync with: src/povs/__cuda/lib/povs.cu — thread-block preflight
    _validate_thr_block_size(options.gpu_thread_block_size, options.physical_block_size, options.virtual_block_size)


def _choose_batch_size(
    device_id: int,
    instance_size: int,
    dtype_bytes: int,
    virtual_block_size: int,
    physical_block_size: int,
) -> int:
    """Choose batch size as the number of virtual blocks that fit in 80% of currently free GPU memory."""
    free_bytes, _ = torch.cuda.mem_get_info(device_id)
    bytes_per_vblock = virtual_block_size * physical_block_size * instance_size * dtype_bytes
    return max(1, int(free_bytes * 0.8) // bytes_per_vblock)
