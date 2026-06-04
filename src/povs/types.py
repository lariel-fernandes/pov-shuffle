from typing import NamedTuple


class POVSOptions(NamedTuple):
    """POV Shuffle algorithm options.

    :param physical_block_size: Physical block size, in number of array elements.
    :param virtual_block_size: Virtual block size, in number of physical blocks.
    :param offsets: List of valid offsets to sample at random in each shuffling iteration, in number of array elements.
    """

    physical_block_size: int
    virtual_block_size: int
    offsets: list[int]
