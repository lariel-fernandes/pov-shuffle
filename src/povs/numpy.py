import numpy as np

from .common import choose_offsets, get_block_counts, get_dtype_bytes, get_instance_size, povs_preflight
from .constants import MAX_SEED, MIN_PBLOCK_SIZE, MIN_SEED, MIN_VBLOCK_SIZE
from .types import FullOptions, Options


def shuffle(
    data: np.ndarray,
    iterations: int = 1,
    options: FullOptions | None = None,
    seed: int | np.random.Generator | None = None,
) -> None:
    """Pseudo-parallel POV Shuffle implementation based on NumPy and CPU processing.

    This is just meant for testing and validations, not for production use,
    as a standard O(n) shuffle on the CPU should always perform better than
    the iterative parallel version in a pseudo-parallel execution.

    :param data: Data array to shuffle in place along the axis 0.
    :param iterations: Number of shuffling iterations to perform.
    :param options: POV Shuffle algorithm options.
    :param seed: Random seed or random number generator state.
    """
    # Resolve options
    options = options or optim_options_for_dataset(data)

    # Coerce seed to generator
    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(seed=seed if isinstance(seed, int) else np.random.randint(MIN_SEED, MAX_SEED))
    )

    # Validate parameters
    povs_preflight(iterations, options)

    # Block count arithmetic
    n_pblocks, n_vblocks = get_block_counts(**options._asdict(), deck_size=len(data))

    for _ in range(iterations):
        # WARNING: The sequence of rng usages in the next 3 code blocks must match the one in the CUDA
        #          implementation for reproducibility (shuffling, then seed sampling, then offset sampling)

        # Build a mapping of each virtual block ID to its physical block IDs
        vbid_2_bids = np.arange(n_vblocks * options.virtual_block_size)
        rng.shuffle(vbid_2_bids)
        vbid_2_bids = vbid_2_bids.reshape((n_vblocks, options.virtual_block_size))

        seeds = rng.integers(0, 1000, size=n_vblocks)  # random seed for each virtual block
        offset = options.offsets[rng.integers(0, len(options.offsets))]  # Sample a pblocks start offset

        def _worker(vbid: int):
            """Processes a single virtual block ID."""
            bids = vbid_2_bids[vbid]  # Read physical block IDs

            # Read the physical blocks into local array and shuffle with seed
            local = np.concat([
                _safe_read_arr(data, offset, bid * options.physical_block_size, options.physical_block_size)
                for bid in bids
                if bid < n_pblocks
            ])
            np.random.RandomState(seed=seeds[vbid]).shuffle(local)

            # Write back shuffled data to physical blocks
            for i, bid in enumerate(bids):
                if bid < n_pblocks:
                    shuffled = local[i * options.physical_block_size : (i + 1) * options.physical_block_size]
                    _safe_set_arr(data, offset, bid * options.physical_block_size, shuffled)

        np.vectorize(_worker)(np.arange(n_vblocks))  # Process all virtual blocks


def optim_options_for_dataset(
    data: np.ndarray,
    partial_options: Options | None = None,
) -> FullOptions:
    """Choose POV Shuffle options for dataset."""
    partial_options = partial_options or Options()

    return FullOptions(
        virtual_block_size=partial_options.virtual_block_size or MIN_VBLOCK_SIZE,
        physical_block_size=(pblk := partial_options.physical_block_size or MIN_PBLOCK_SIZE),
        offsets=choose_offsets(
            instance_size=get_instance_size(data),
            dtype_bytes=get_dtype_bytes(data),
            pblock_size=pblk,
        ),
    )


def _safe_read_arr(arr: np.ndarray, offset: int, start: int, length: int) -> np.ndarray:
    """Safely read a slice of an offset array while wrapping around and avoiding over-indexing."""

    # Require a valid start index
    assert start < len(arr)

    # Cap the slice length to avoid over-indexing
    length = min(length, len(arr) - start)

    # Apply the start offset
    start += offset

    # Read the slice and wrap around along axis 0 to preserve instance shape
    indices = np.arange(start, start + length) % len(arr)
    return arr[indices]


def _safe_set_arr(arr: np.ndarray, offset: int, start: int, val: np.ndarray) -> None:
    """Safely set a slice of an offset array while wrapping around and avoiding over-indexing."""

    # Require a valid start index
    assert start < len(arr)

    # Cap the slice length to avoid over-indexing
    length = min(len(val), len(arr) - start)

    # Apply the start offset
    start += offset

    # Set the slice and wrap around along axis 0 to preserve instance shape
    indices = np.arange(start, start + length) % len(arr)
    arr[indices] = val[:length]
