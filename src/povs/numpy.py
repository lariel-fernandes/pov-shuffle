import numpy as np

from povs import POVSOptions
from povs.utils import get_block_counts, get_valid_offsets


def pov_shuffle(
    data: np.ndarray,
    iterations: int = 1,
    options: POVSOptions = POVSOptions(),
    seed: int | np.random.Generator = None,
) -> None:
    """Pseudo-parallel POV Shuffle implementation based on NumPy and CPU processing.

    :param data: Data array to shuffle in place.
    :param iterations: Number of shuffling iterations to perform.
    :param options: POV Shuffle algorithm options.
    :param seed: Random seed or random number generator state.
    """
    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed=seed if isinstance(seed, int) else np.random.randint(0, 1000))
    )

    # Validate parameters
    assert iterations >= 1
    assert options.virtual_block_size >= 2
    assert options.max_offset_steps >= 2
    assert options.offset_step_size % options.physical_block_size != 0

    # Collect offsets that are not multiples of the physical block size
    valid_offsets = get_valid_offsets(**options._asdict())
    assert len(valid_offsets) >= 2

    # Calculate number of blocks with rounding up
    n_blocks, n_vblocks = get_block_counts(**options._asdict(), deck_size=len(data))

    for _ in range(iterations):
        offset = valid_offsets[rng.integers(0, len(valid_offsets))]
        seeds = rng.integers(0, 1000, size=n_vblocks)  # random seed for each virtual block

        # Build a mapping of each virtual block ID to its physical block IDs
        vbid_2_bids = np.arange(n_vblocks * options.virtual_block_size)
        rng.shuffle(vbid_2_bids)
        vbid_2_bids = vbid_2_bids.reshape((n_vblocks, options.virtual_block_size))

        def _worker(vbid: int):
            """Processes a single virtual block ID."""
            bids = vbid_2_bids[vbid]  # Read physical block IDs

            # Read the physical blocks into local array and shuffle with seed
            local = np.concat([
                _safe_read_arr(data, offset, bid * options.physical_block_size, options.physical_block_size)
                for bid in bids
                if bid < n_blocks
            ])
            np.random.RandomState(seed=seeds[vbid]).shuffle(local)

            # Write back shuffled data to physical blocks
            for i, bid in enumerate(bids):
                if bid < n_blocks:
                    shuffled = local[i * options.physical_block_size : (i + 1) * options.physical_block_size]
                    _safe_set_arr(data, offset, bid * options.physical_block_size, shuffled)

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
