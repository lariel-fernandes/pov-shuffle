import math
import warnings

import numpy as np


def pov_shuffle(
    data: np.ndarray,
    iterations: int = 3,
    base_block_size: int = 32,
    max_block_clumping: int = 4,
    base_offset: int = 4,
    max_offset_factor: int = 8,
    seed: int | np.random.RandomState = None,
) -> None:
    """Pseudo-parallel POV Shuffle implementation based on NumPy and CPU processing.

    :param data: Data array to shuffle in place.
    :param iterations: Number of shuffling iterations to perform.
    :param base_block_size: Base physical block size.
    :param max_block_clumping: Max physical blocks to combine into each virtual block.
    :param base_offset: Base block start offset.
    :param max_offset_factor: Max multiplier of the block start offset.
    :param seed: Random seed or random number generator state.
    """
    rng = (
        seed
        if isinstance(seed, np.random.RandomState)
        else np.random.RandomState(seed=seed if isinstance(seed, int) else np.random.randint(0, 1000))
    )

    if len(data) <= 2 * base_block_size:
        warnings.warn("Data size is no larger than twice the block size. Falling back to a true shuffle.")
        rng.shuffle(data)
        return

    # Validate parameters
    assert iterations >= 1
    assert max_block_clumping >= 3
    assert max_offset_factor >= 2
    assert base_offset % base_block_size != 0

    # Collect offsets that are not multiples of the base block size
    valid_offsets = [x for x in range(1, max_offset_factor + 1) if (x * base_offset) % base_block_size != 0]
    assert len(valid_offsets) >= 2

    n_blocks = math.ceil(len(data) / base_block_size)  # number of physical blocks

    for _ in range(iterations):
        offset = valid_offsets[rng.randint(0, len(valid_offsets))]
        blocks_to_clump = rng.randint(2, max_block_clumping + 1)

        n_vblocks = math.ceil(n_blocks / blocks_to_clump)  # number of virtual blocks
        seeds = rng.randint(0, 1000, size=n_vblocks)  # random seed for each virtual block

        # Build a mapping of each virtual block ID to its physical block IDs
        vbid_2_bids = np.arange(n_vblocks * blocks_to_clump)
        rng.shuffle(vbid_2_bids)
        vbid_2_bids = vbid_2_bids.reshape((n_vblocks, blocks_to_clump))

        def _worker(vbid: int):
            """Processes a single virtual block ID."""
            bids = vbid_2_bids[vbid]  # Read physical block IDs

            # Read the physical blocks into local array and shuffle with seed
            local = np.concat([
                _safe_read_arr(data, offset, bid * base_block_size, base_block_size) for bid in bids if bid < n_blocks
            ])
            np.random.RandomState(seed=seeds[vbid]).shuffle(local)

            # Write back shuffled data to physical blocks
            for i, bid in enumerate(bids):
                if bid < n_blocks:
                    shuffled = local[i * base_block_size : (i + 1) * base_block_size]
                    _safe_set_arr(data, offset, bid * base_block_size, shuffled)

        np.vectorize(_worker)(np.arange(n_vblocks))  # Process all virtual blocks


def _safe_read_arr(arr: np.ndarray, offset: int, start: int, length: int) -> np.ndarray:
    """Safely read a slice of an offset array while wrapping around and avoiding over-indexing."""

    # Require a valid start index
    assert start < len(arr)

    # Cap the slice length to avoid over-indexing
    length = min(length, len(arr) - start)

    # Apply the start offset
    start += offset

    # Read the slice and wrap around
    return arr.take(range(start, start + length), mode="wrap")


def _safe_set_arr(arr: np.ndarray, offset: int, start: int, val: np.ndarray) -> None:
    """Safely set a slice of an offset array while wrapping around and avoiding over-indexing."""

    # Require a valid start index
    assert start < len(arr)

    # Cap the slice length to avoid over-indexing
    length = min(len(val), len(arr) - start)

    # Apply the start offset
    start += offset

    # Set the slice and wrap around
    arr.put(range(start, start + length), val[:length], mode="wrap")
