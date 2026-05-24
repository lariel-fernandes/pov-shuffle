import math

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
    # Validate parameters
    assert iterations >= 1
    assert options.virtual_block_size >= 2
    assert options.max_offset_steps >= 2
    assert options.offset_step_size % options.physical_block_size != 0

    # Collect offsets that are not multiples of the physical block size
    valid_offsets = get_valid_offsets(**options._asdict())
    assert len(valid_offsets) >= 2
    return valid_offsets
