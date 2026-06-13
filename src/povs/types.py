from typing import NamedTuple


class BuildParams(NamedTuple):
    """Build-time parameters used to compile the CUDA extension."""

    debug_mode: bool
    cuda_arch: list[str]
    vblock_sizes: list[int]
    pblock_sizes: list[int]
    instance_sizes: list[int]
    cartesian_instancing: bool
    instantiations: list | None = None
    instantiations_all_types: list | None = None


class Options(NamedTuple):
    """POV Shuffle algorithm options.

    Ordered from upstream to downstream in the param optimization chain.

    :param virtual_block_size: Virtual block size, in number of physical blocks.
    :param physical_block_size: Physical block size, in number of instances.
    :param offsets: List of valid offsets to sample at random in each shuffling iteration, in number of instances.
    :param gpu_thread_block_size: GPU thread-block size (only used in CUDA implementation).
    """

    virtual_block_size: int | None = None
    physical_block_size: int | None = None
    offsets: list[int] | None = None
    gpu_thread_block_size: int | None = None

    def is_fully_specified(self, cuda_required: bool) -> bool:
        return None not in self[: None if cuda_required else -1]


class FullOptions(Options):
    """Fully specified POV Shuffle algorithm options."""

    virtual_block_size: int
    physical_block_size: int
    offsets: list[int]
    gpu_thread_block_size: int
