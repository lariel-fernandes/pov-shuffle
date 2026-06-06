from typing import NamedTuple


class Options(NamedTuple):
    """POV Shuffle algorithm options.

    Ordered from upstream to downstream in the param optimization chain.

    :param physical_block_size: Physical block size, in number of array elements.
    :param virtual_block_size: Virtual block size, in number of physical blocks.
    :param offsets: List of valid offsets to sample at random in each shuffling iteration, in number of array elements.
    """

    physical_block_size: int | None = None
    virtual_block_size: int | None = None
    offsets: list[int] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.virtual_block_size is None:
            assert self.physical_block_size is None, "Cannot specify pblock size without specifying vblock size"


class FullOptions(Options):
    """Fully specified POV Shuffle algorithm options."""

    physical_block_size: int
    virtual_block_size: int
    offsets: list[int]
