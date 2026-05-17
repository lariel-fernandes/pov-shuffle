from typing import NamedTuple


class POVSOptions(NamedTuple):
    """POV Shuffle algorithm options.

    :param physical_block_size: Physical block size, in number of array elements.
    :param virtual_block_size: Virtual block size, in number of physical blocks.
    :param offset_step_size: Offset step size, in number of array elements.
    :param max_offset_steps: Maximum number of offset steps (inclusive).
    """

    physical_block_size: int = 32
    virtual_block_size: int = 4
    offset_step_size: int = 4
    max_offset_steps: int = 8
