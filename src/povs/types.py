from typing import NamedTuple


class BuildParams(NamedTuple):
    """Build-time parameters used to compile the CUDA extension."""

    debug_mode: bool
    cuda_arch: list[str]
    vblock_sizes: list[int]
    pblock_sizes: list[int]
    instance_sizes: list[int]
    cartesian_instancing: bool
    instantiations: list
    instantiations_all_types: list


class Options(NamedTuple):
    """POV Shuffle algorithm options.

    :param virtual_block_size: Virtual block size, in number of physical blocks.
    :param physical_block_size: Physical block size, in number of instances.
    :param offsets: List of valid offsets to sample at random in each shuffling iteration, in number of instances.

    Params for the torch+cuda implementation:
    :param gpu_thread_block_size: GPU thread-block size (only used in CUDA implementation).
    """

    virtual_block_size: int | None = None
    physical_block_size: int | None = None
    offsets: list[int] | None = None

    gpu_thread_block_size: int | None = None
