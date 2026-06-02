import warnings

import torch

from .utils import is_power_of_2, round_down_to_power_of_2
from .constants import _CUDA_CC_IDEAL_OCCUPANCY, _DEFAULT_IDEAL_OCC_BLOCKS_PER_SM, _DEFAULT_IDEAL_OCC_BLOCK_SIZE


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
    max_thr_block_size = 0  # TODO: get max threads per block from device properties
    target_thr_block_size, _ = _get_cc_ideal_occ((dev.major, dev.minor))

    max_thr_blocks_per_sm = _get_occupancy_for_smem_constraint(
        instance_size, dtype_bytes, vblock_size, pblock_size, device_id
    )

    return round_down_to_power_of_2(
        min(
            target_thr_block_size,
            max_thr_block_size,  # Don't use more threads than the SM allows
            pblock_size * vblock_size,  # Don't use more threads than there are instances
            max_threads_per_sm // max_thr_blocks_per_sm,  # Don't use so many threads that
            # we can't fit as many thread blocks in the SM as the shared memory constraint allows
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
    min_pblock_size = vblock_size
    min_thr_blocks_per_sm = 1  # at least one thread block must fit in each SM
    min_instances_per_thr_block = min_pblock_size * vblock_size
    min_instances_per_sm = min_thr_blocks_per_sm * min_instances_per_thr_block

    # Inspect the device
    dev = torch.cuda.get_device_properties(device_id)
    sm_smem_bytes: int = dev.shared_memory_per_multiprocessor  # TODO: confirm that this property is returned in bytes
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


# TODO: functions to choose offset parameters


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
    if details := _CUDA_CC_IDEAL_OCCUPANCY.get(cc):
        return details

    warnings.warn(f"Unknown compute capability: {cc}")

    return (
        _DEFAULT_IDEAL_OCC_BLOCK_SIZE,
        _DEFAULT_IDEAL_OCC_BLOCKS_PER_SM,
    )
