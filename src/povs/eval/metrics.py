import collections

import numpy as np


def get_tvd(permutations: np.ndarray) -> float:
    """Get the Total Variation Difference of the permutations against a uniform distribution.

    :param permutations: Array of shape (num_permutations, deck_size) containing different permutations of a deck.
                         The deck is assumed to contain all and only the monotonically increasing indices 0 to `deck_size-1`.

    For each position `j`, the outcomes are defined as each index `i` that can show up
    in that position, with a baseline uniform probability of `1 / deck_size`.
    The observed TVD is then averaged over the deck positions.
    """
    num_permutations, deck_size = permutations.shape
    baseline = 1 / deck_size

    measures = np.zeros(deck_size, dtype=float)

    for j in range(deck_size):
        abs_differences = 0
        counts = collections.Counter(permutations[:, j])

        for i in range(deck_size):
            count = counts.get(i, 0)
            abs_differences += abs(baseline - (count / num_permutations))

        measures[j] = abs_differences / 2

    return float(measures.mean())
