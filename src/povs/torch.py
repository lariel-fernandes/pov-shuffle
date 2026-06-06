import warnings

import torch

from ._cuda import torch_binding
from .common import choose_offsets, get_dtype_bytes, get_instance_size, povs_preflight
from .constants import (
    CUDA_CC_IDEAL_OCCUPANCY,
    CUDA_DEFAULT_IDEAL_OCCUPANCY,
    MAX_BLOCK_SIZE,
    MAX_SEED,
    MIN_CUDA_ARCH,
    MIN_PBLOCK_SIZE,
    MIN_SEED,
    MIN_VBLOCK_SIZE,
)
from .types import FullOptions, Options
from .utils import (
    is_power_of_2,
    round_down_to_multiple,
    round_down_to_power_of_2,
)


def shuffle(
    data: torch.Tensor,
    iterations: int = 1,
    options: FullOptions | None = None,
    seed: int | torch.Generator | None = None,
) -> None:
    """POV Shuffle implementation for torch tensors on CUDA.

    :param data: Data tensor to shuffle in place along the axis 0.
    :param iterations: Number of shuffling iterations to perform.
    :param options: POV Shuffle algorithm options.
    :param seed: Random seed or random number generator state.
    """
    # Dataset preflight
    assert (device_id := data.get_device()) != -1, "Tensor device must be CUDA"
    assert data.dtype in (torch.float16, torch.float32, torch.float64, torch.int32, torch.int64)
    assert data.is_contiguous()

    # Device preflight
    device = torch.cuda.get_device_properties(device_id)
    assert (device.major, device.minor) >= MIN_CUDA_ARCH

    # Resolve options
    options = options or optim_options_for_dataset(data)

    # Coerce to numerical seed
    seed = seed if isinstance(seed, int) else int(torch.randint(MIN_SEED, MAX_SEED, (1,), generator=seed).item())

    # Validate parameters
    povs_preflight(iterations, options)

    # Delegate to bound CUDA library
    torch_binding(
        data,
        torch.tensor(options.offsets, dtype=torch.int64),
        iterations,
        options.physical_block_size,
        options.virtual_block_size,
        seed,
        choose_thr_block_size(
            instance_size=get_instance_size(data),
            dtype_bytes=get_dtype_bytes(data),
            vblock_size=options.virtual_block_size,
            pblock_size=options.physical_block_size,
            device_id=device_id,
        ),
    )


def optim_options_for_dataset(
    data: torch.Tensor,
    partial_options: Options | None = None,
) -> FullOptions:
    """Choose POV Shuffle options for dataset."""
    assert (device_id := data.get_device()) != -1, "Tensor device must be CUDA"
    instance_size = get_instance_size(data)
    dtype_bytes = get_dtype_bytes(data)

    partial_options = partial_options or Options()

    return FullOptions(
        virtual_block_size=(vblk := partial_options.virtual_block_size or MIN_VBLOCK_SIZE),
        physical_block_size=(
            pblk := partial_options.physical_block_size
            or choose_pblock_size(instance_size, dtype_bytes, vblk, device_id)
        ),
        offsets=choose_offsets(
            instance_size=instance_size,
            dtype_bytes=dtype_bytes,
            pblock_size=pblk,
        ),
    )


def choose_pblock_size(
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


def choose_thr_block_size(
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

    if thr_block_size > pblock_size and thr_block_size % pblock_size != 0:
        thr_block_size = round_down_to_multiple(thr_block_size, pblock_size)  # Round down to closest multiple of pblk

    if thr_block_size < pblock_size and pblock_size % thr_block_size != 0:
        thr_block_size = round_down_to_power_of_2(thr_block_size)  # Round down to closest divisor of pblk

    return thr_block_size


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
