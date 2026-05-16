from typing import TypeVar

import numpy as np
import sys


def pov_shuffle(
    data: np.ndarray,
    base_block_size: int = 32,
    max_block_clumping: int = 4,
    base_offset: int = 4,
    max_offset_factor: int = 8,
    seed_or_state: int | np.random.RandomState = None,
) -> None:
    """Pseudo-parallel POV Shuffle implementation based on NumPy and CPU processing.
    
    :param data: Data array to shuffle in place.
    :param base_block_size: Base physical block size.
    :param max_block_clumping: Max physical blocks to combine into each virtual block.
    :param base_offset: Base block start offset.
    :param max_offset_factor: Max multiplier of the block start offset.
    :param seed_or_state: Random seed or random number generator state.
    """

    assert max_block_clumping >= 3
    assert max_offset_factor >= 2
    assert base_offset % base_block_size != 0

    rng = (
        seed_or_state
        if isinstance(seed_or_state, np.random.RandomState)
        else np.random.RandomState(seed=seed_or_state if isinstance(seed_or_state, int) else np.random.randint(0, 1000))
    )
