import numpy as np


def get_tvd(samples: np.ndarray) -> float:
    """Get the mean per-position Total Variation Distance against a uniform distribution.

    :param samples: Array of shape (num_samples, deck_size) containing independent permutations of a deck.
                    The deck is assumed to contain all and only the monotonically increasing indices 0 to `deck_size-1`.
    """
    num_samples, deck_size = samples.shape

    # Build (deck_size, deck_size) array where [i, j] is how often value i appeared at position j
    counts = np.apply_along_axis(
        lambda col: np.bincount(col, minlength=deck_size),
        axis=0,
        arr=samples,
    )

    probs = counts / num_samples
    baseline_prob = 1 / deck_size
    return float(0.5 * np.abs(probs - baseline_prob).sum(axis=0).mean())
