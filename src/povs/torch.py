import typing
import warnings

import torch

from .common import (
    choose_offsets,
    get_block_counts,
    get_dtype_bytes,
    get_instance_size,
    povs_preflight,
    purge_dependent_param,
)
from .constants import (
    CUDA_CC_IDEAL_OCCUPANCY,
    CUDA_DEFAULT_IDEAL_OCCUPANCY,
    MAX_BLOCK_SIZE,
    MAX_SEED,
    MIN_CUDA_ARCH,
    MIN_PBLOCK_SIZE,
    MIN_SEED,
    MIN_VBLOCK_SIZE,
    MIN_BLOCk_SIZE,
)
from .types import Options
from .utils import is_power_of_2, round_down_to_power_of_2


def shuffle(
    data: torch.Tensor,
    iterations: int,
    options: Options | None,
    seed: int,
) -> None:
    """POV Shuffle implementation for torch tensors on CUDA.

    :param data: Data tensor to shuffle in place along the axis 0.
    :param iterations: Number of shuffling iterations to perform.
    :param options: POV Shuffle algorithm options.
    :param seed: Random seed or random number generator state.
    """
    from ._cuda import torch_binding

    options = typing.cast(
        "_TorchCudaOptions",
        options if options_is_complete(options) else optim_options_for_dataset(data, options),
    )

    preflight(data, iterations, options)

    device = data.get_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    n_pblocks, n_vblocks = get_block_counts(**options._asdict(), deck_size=len(data))

    for _ in range(iterations):
        vbid_2_bids = torch.randperm(
            n_vblocks * options.virtual_block_size, device=device, dtype=torch.int64, generator=generator
        ).reshape(n_vblocks, options.virtual_block_size)
        vbid_2_bids[vbid_2_bids >= n_pblocks] = -1

        seeds = torch.randint(
            MIN_SEED, MAX_SEED, size=(n_vblocks,), device=device, dtype=torch.int32, generator=generator
        )

        offset = options.offsets[
            int(torch.randint(len(options.offsets), (1,), device=device, generator=generator).item())
        ]

        torch_binding(
            data,
            vbid_2_bids,
            seeds,
            offset,
            options.physical_block_size,
            options.virtual_block_size,
            options.gpu_thread_block_size,
        )


class _TorchCudaOptions(Options):
    virtual_block_size: int
    physical_block_size: int
    offsets: list[int]
    gpu_thread_block_size: int


def options_is_complete(options: Options | None) -> bool:
    return options is not None and None not in (
        options.virtual_block_size,
        options.physical_block_size,
        options.gpu_thread_block_size,
        options.offsets,
    )


def optim_options_for_dataset(
    data: torch.Tensor,
    options: Options | None,
) -> Options:
    """Choose POV Shuffle options for dataset."""
    options = options or Options()

    if options_is_complete(options):
        warnings.warn("All parameters already specified, skipping optimization")
        return options

    assert (device_id := data.get_device()) != -1, "Tensor device must be CUDA"
    instance_size = get_instance_size(data)
    dtype_bytes = get_dtype_bytes(data)

    if options.virtual_block_size is None:
        if options.physical_block_size is not None:
            purge_dependent_param(options, "virtual_block_size", "physical_block_size")
        if options.gpu_thread_block_size is not None:
            purge_dependent_param(options, "virtual_block_size", "gpu_thread_block_size")

    if options.physical_block_size is None and options.offsets is not None:
        purge_dependent_param(options, "physical_block_size", "offsets")

    return Options(
        virtual_block_size=(vblk := options.virtual_block_size or MIN_VBLOCK_SIZE),
        physical_block_size=(
            pblk := options.physical_block_size or _choose_pblock_size(instance_size, dtype_bytes, vblk, device_id)
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
    data: torch.Tensor,
    iterations: int,
    options: _TorchCudaOptions,
) -> None:
    """Preflight checks for POV Shuffle on torch tensors."""

    # Dataset preflight  # In sync with: src/povs/__cuda/binds/torch.cpp — dataset preflight
    dtypes = (torch.float16, torch.float32, torch.float64, torch.int32, torch.int64)
    assert (device_id := data.get_device()) != -1, "Tensor device must be CUDA"
    assert data.dtype in dtypes, f"Invalid dtype {data.dtype.__str__()}, must be one of {dtypes}"
    assert data.is_contiguous(), "Tensor must be contiguous"

    # Device preflight  # In sync with: src/povs/__cuda/lib/povs.cu — device preflight
    dev = torch.cuda.get_device_properties(device_id)
    assert (arch := (dev.major, dev.minor)) >= MIN_CUDA_ARCH, f"CUDA arch {arch} must be at least {MIN_CUDA_ARCH}"

    # Standard preflight  # In sync with: src/povs/__cuda/lib/povs.cu — standard preflight
    povs_preflight(data, iterations, options)

    # GPU thread-block preflight  # In sync with: src/povs/__cuda/lib/povs.cu — thread-block preflight
    _validate_thr_block_size(options.gpu_thread_block_size, options.physical_block_size, options.virtual_block_size)


def _validate_thr_block_size(
    thr_block_size: int,
    vblock_size: int,
    pblock_size: int,
) -> None:
    prefix = f"thread-block size ({thr_block_size})"
    assert is_power_of_2(thr_block_size), f"{prefix} must be a power of 2"
    assert thr_block_size <= MAX_BLOCK_SIZE, f"{prefix} must not exceed {MAX_BLOCK_SIZE}"
    assert thr_block_size >= MIN_BLOCk_SIZE, f"{prefix} must be at least {MIN_BLOCk_SIZE}"
    assert thr_block_size <= (total := vblock_size * pblock_size), f"{prefix} must not exceed total instances ({total})"


def _choose_pblock_size(
    instance_size: int,
    dtype_bytes: int,
    vblock_size: int,
    device_id: int,
) -> int:
    """Choose optimized physical block size for CUDA POV Shuffle.

    Take the number of thread blocks per SM under theoretical optimal occupancy in the absence of memory constraints.
    Choose the largest valid physical block size whose imposed memory constraint would still allow that occupancy to
    happen, or the closest to that occupancy.
    """

    # Set lower bounds
    min_pblock_size = max(vblock_size, MIN_PBLOCK_SIZE)
    min_thr_blocks_per_sm = 1  # at least one thread block must fit in each SM
    min_instances_per_thr_block = min_pblock_size * vblock_size
    min_instances_per_sm = min_thr_blocks_per_sm * min_instances_per_thr_block

    # Inspect the device
    dev = torch.cuda.get_device_properties(device_id)
    sm_smem_bytes: int = dev.shared_memory_per_multiprocessor
    _, target_thr_blocks_per_sm = _get_cc_ideal_occ((dev.major, dev.minor))

    # Determine how many instances can fit in the SM
    instance_bytes = instance_size * dtype_bytes
    needed_smem_bytes_per_instance = instance_bytes + 32  # one int32 for the instance permutation index
    instances_per_sm = sm_smem_bytes // needed_smem_bytes_per_instance

    # Validate against lower bound
    assert instances_per_sm >= min_instances_per_sm, (
        "Cannot fit enough instances in SM shared memory "
        f"({needed_smem_bytes_per_instance=}, {sm_smem_bytes=}, {min_instances_per_sm=})"
    )

    # Approximate a pblock size for closest to ideal occupancy
    thr_blocks_per_sm = min(target_thr_blocks_per_sm, instances_per_sm // min_instances_per_thr_block)
    instances_per_thr_block = instances_per_sm // thr_blocks_per_sm
    pblock_size = instances_per_thr_block // vblock_size

    # Coerce to power of 2 while preserving proximity to ideal occupancy
    if is_power_of_2(pblock_size):
        return pblock_size

    lower_pblk_size = round_down_to_power_of_2(pblock_size)
    upper_pblk_size = 2 * lower_pblk_size

    occupancy_at_lower_pblk_size = instances_per_sm // (lower_pblk_size * vblock_size)
    occupancy_at_upper_pblk_size = instances_per_sm // (upper_pblk_size * vblock_size)

    occ_diff_at_lower_pblk_size = abs(occupancy_at_lower_pblk_size - target_thr_blocks_per_sm)
    occ_diff_at_upper_pblk_size = abs(occupancy_at_upper_pblk_size - target_thr_blocks_per_sm)

    return lower_pblk_size if occ_diff_at_lower_pblk_size < occ_diff_at_upper_pblk_size else upper_pblk_size


def _choose_thr_block_size(
    instance_size: int,
    dtype_bytes: int,
    vblock_size: int,
    pblock_size: int,
    device_id: int,
) -> int:
    """Choose optimized thread block size for CUDA POV Shuffle."""

    dev = torch.cuda.get_device_properties(device_id)
    max_threads_per_sm = dev.max_threads_per_multi_processor
    target_thr_block_size, _ = _get_cc_ideal_occ((dev.major, dev.minor))

    # Given how much shared memory each thread block needs to accommodate `vblock_size * pblock_size` instances
    # of the given size and dtype, determine how many thread blocks we can fit in the SM.
    max_thr_blocks_per_sm = _get_occupancy_for_smem_constraint(
        instance_size, dtype_bytes, vblock_size, pblock_size, device_id
    )

    # Distribute the max threads of the SM to the amount of thread blocks calculated in the previous step.
    # Giving more than this amount of threads to each block would mean fitting fewer blocks in the SM
    # than the shared memory constraint allows, thus making the block size the bottleneck, instead of memory.
    max_threads_per_block = max_threads_per_sm // max_thr_blocks_per_sm

    thr_block_size = min(
        MAX_BLOCK_SIZE,
        target_thr_block_size,
        max_threads_per_block,
        pblock_size * vblock_size,  # Don't use more threads than there are instances.
    )

    return max(MIN_BLOCk_SIZE, round_down_to_power_of_2(thr_block_size))


def _get_occupancy_for_smem_constraint(
    instance_size: int,
    dtype_bytes: int,
    vblock_size: int,
    pblock_size: int,
    device_id: int,
) -> int:
    """Get the number of GPU thread blocks per SM under the shared memory constraint
    created by the specified problem parameters."""

    dev = torch.cuda.get_device_properties(device_id)
    sm_smem_bytes: int = dev.shared_memory_per_multiprocessor

    instance_bytes = instance_size * dtype_bytes
    needed_smem_bytes_per_instance = instance_bytes + 32  # one int32 for the instance permutation index
    instances_per_sm = sm_smem_bytes // needed_smem_bytes_per_instance

    instances_per_thr_block = vblock_size * pblock_size
    return instances_per_sm // instances_per_thr_block


def _get_cc_ideal_occ(cc: tuple[int, int]) -> tuple[int, int]:
    if details := CUDA_CC_IDEAL_OCCUPANCY.get(cc):
        return details

    warnings.warn(f"Unknown compute capability: {cc}")
    return CUDA_DEFAULT_IDEAL_OCCUPANCY
