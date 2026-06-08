from typing import NamedTuple


class Options(NamedTuple):
    """POV Shuffle algorithm options.

    Ordered from upstream to downstream in the param optimization chain.

    :param physical_block_size: Physical block size, in number of array elements.
    :param virtual_block_size: Virtual block size, in number of physical blocks.
    :param offsets: List of valid offsets to sample at random in each shuffling iteration, in number of array elements.
    :param gpu_thread_block_size: GPU thread-block size (only used in CUDA implementation).
    """

    physical_block_size: int | None = None
    virtual_block_size: int | None = None
    offsets: list[int] | None = None
    gpu_thread_block_size: int | None = None

    def is_fully_specified(self, cuda_required: bool) -> bool:
        return None not in self[: None if cuda_required else -1]


class FullOptions(Options):
    """Fully specified POV Shuffle algorithm options."""

    physical_block_size: int
    virtual_block_size: int
    offsets: list[int]
    gpu_thread_block_size: int
