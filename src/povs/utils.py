import math

from .constants import ALLOWED_VIRTUAL_BLOCK_SIZES
from .types import POVSOptions


def get_valid_offsets(
    physical_block_size: int,
    offset_step_size: int,
    max_offset_steps: int,
    **__,
) -> list[int]:
    """Get offsets that are not multiples of the physical block size."""

    all_offsets = [(n_steps * offset_step_size) for n_steps in range(1, max_offset_steps + 1)]
    return [offset for offset in all_offsets if offset % physical_block_size != 0]


def get_block_counts(
    deck_size: int,
    physical_block_size: int,
    virtual_block_size: int,
    **__,
) -> tuple[int, int]:
    """Get the number of physical and virtual blocks, with rounding up."""

    n_blocks = math.ceil(deck_size / physical_block_size)
    n_vblocks = math.ceil(n_blocks / virtual_block_size)
    return n_blocks, n_vblocks


def povs_preflight(
    iterations: int,
    options: POVSOptions,
) -> list[int]:
    """Common validations and preparations before running any POV Shuffle implementation."""

    assert iterations >= 1

    # Validate block sizes
    assert is_power_of_2(options.physical_block_size)
    assert options.virtual_block_size in ALLOWED_VIRTUAL_BLOCK_SIZES
    assert options.virtual_block_size <= options.physical_block_size

    # Validate offset parameters
    assert options.max_offset_steps >= 2
    assert options.offset_step_size > 0
    assert options.offset_step_size % options.physical_block_size != 0

    # Require at least 2 offsets that are not multiples of the physical block size
    valid_offsets = get_valid_offsets(**options._asdict())
    assert len(valid_offsets) >= 2
    return valid_offsets


def is_power_of_2(x: int) -> bool:
    return x > 0 and (x & (x - 1) == 0)


def round_down_to_power_of_2(x: int) -> int:
    return 1 << (x.bit_length() - 1)
