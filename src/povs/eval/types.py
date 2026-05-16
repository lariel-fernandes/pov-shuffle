from typing import NamedTuple


class PovsOptions(NamedTuple):
    """POV Shuffle algorithm options"""

    base_block_size: int = 32
    max_block_clumping: int = 4
    base_offset: int = 4
    max_offset_factor: int = 8
