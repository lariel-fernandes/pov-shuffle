from itertools import permutations

import numpy as np


def get_tvd(samples: np.ndarray) -> float:
    """Get the mean per-position Total Variation Distance against a uniform distribution.

    This is a measure of position bias, checking how likely it is that item `i` goes to position `j` when shuffling,
    and calculating the difference to a truly uniform shuffle. It is calculated for `j` and then averaged.
    The result is from 0 (most uniform) to 1 (least uniform).

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


def get_ngram_tvd(samples: np.ndarray, n: int) -> float:
    """Get TVD of the n-gram increments distribution against uniform.

    This is a measure of sequence bias, checking e.g. if sequences like (1,2,4) and (5,6,8) (increments of 1 and 2) are
    more likely than (1,3,7) and (5,7,11) (increments of 2 and 4), which should not be the case in a uniform shuffle.
    The result is from 0 (most uniform) to 1 (least uniform).

    For each n-gram (v_0,...,v_{n-1}), the event outcome is defined as the tuple of modular distances from v_0:
    ((v_1-v_0) % deck_size, ..., (v_{n-1}-v_0) % deck_size).
    Valid outcomes are all ordered selections of n-1 distinct values from {1,...,deck_size-1}.

    :param samples: Array of shape (num_samples, deck_size) containing independent permutations of a deck.
                    The deck is assumed to contain all and only the monotonically increasing indices 0 to `deck_size-1`.
    :param n: Degree of the n-gram.
    """
    num_samples, deck_size = samples.shape

    # Extract n-grams for every starting position across all samples (with position wrap-around)
    col_indices = np.arange(deck_size).reshape(deck_size, 1) + np.arange(n).reshape(1, n)  # (deck_size, n)
    ngrams = samples.take(col_indices, axis=1, mode="wrap")  # (num_samples, deck_size, n)

    # Modular distances from first element: values in {1,...,deck_size-1} for distinct elements
    diffs = (ngrams[:, :, 1:] - ngrams[:, :, :1]) % deck_size  # (num_samples, deck_size, n-1)
    flat_diffs = diffs.reshape(-1, n - 1)  # (num_samples * deck_size, n-1)

    unique_diffs, counts = np.unique(flat_diffs, axis=0, return_counts=True)
    total = num_samples * deck_size
    obs_probs = {tuple(row): count / total for row, count in zip(unique_diffs, counts)}

    # Valid tuples are all permutations of n-1 distinct values from {1,...,deck_size-1}
    valid_diffs = list(permutations(range(1, deck_size), n - 1))
    baseline_prob = 1 / len(valid_diffs)
    tvd = sum(abs(obs_probs.get(t, 0.0) - baseline_prob) for t in valid_diffs)
    return float(0.5 * tvd)
