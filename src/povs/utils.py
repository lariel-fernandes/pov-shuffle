import math


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
