from math import perm as _perm

import numpy as np


def get_tvd_num_valid(deck_size: int) -> int:
    """Number of valid events for positional TVD: one travel-distance value per deck position."""
    return deck_size


def get_ngram_tvd_num_valid(deck_size: int, n: int) -> int:
    """Number of valid events for n-gram TVD: ordered selections of n-1 distinct values from {1,...,deck_size-1}."""
    return _perm(deck_size - 1, n - 1)


def get_sample_deficit(num_samples: int, deck_size: int, num_valid: int) -> int:
    """Signed gap between the event space and the observation budget.

    :returns: ``num_valid - num_samples * deck_size``. Positive means undersampled (more observations
              would be needed to cover all valid events at least once). Zero means exactly covered.
              Negative means oversampled (surplus observations).
    """
    return num_valid - num_samples * deck_size


def get_tvd(samples: np.ndarray) -> float:
    """Mean per-position Total Variation Distance via the travel-distance distribution.

    For each element in each sample, the travel distance is ``t = (dest_position - src_value) % deck_size``.
    In a truly uniform shuffle each travel distance is equally likely, so the reference distribution
    is uniform over ``{0, ..., deck_size - 1}``.

    **Adaptive TVD formula**::

        total     = num_samples * deck_size   # total observations
        num_valid = deck_size                 # number of valid events
        p_ref     = 1 / min(total, num_valid)

        TVD = 0.5 * [ sum_observed |count/total - p_ref|
                    + (min(total, num_valid) - num_observed) * p_ref ]

    Since ``total = num_samples * deck_size >= deck_size = num_valid`` for any ``num_samples >= 1``,
    this metric is always in the oversampled regime and reduces to the standard TVD of the
    travel-distance distribution against uniform.

    :param samples: Array of shape ``(num_samples, deck_size)`` containing independent permutations.
                    Values must be the monotonically increasing indices ``0`` to ``deck_size - 1``.
    """
    num_samples, deck_size = samples.shape
    total = num_samples * deck_size
    num_valid = deck_size
    p_ref = 1.0 / min(total, num_valid)

    positions = np.tile(np.arange(deck_size), num_samples)
    travel_distances = (positions - samples.ravel()) % deck_size
    counts = np.bincount(travel_distances, minlength=deck_size)
    # bincount fills zeros for unobserved distances, so the sum already covers the full event space
    return float(0.5 * np.abs(counts / total - p_ref).sum())


def get_ngram_tvd(samples: np.ndarray, n: int) -> float:
    """TVD of the n-gram relative-difference distribution against uniform.

    For each consecutive n-gram ``(v_0, ..., v_{n-1})`` in a sample (with position wrap-around),
    the event is the tuple of modular distances from ``v_0``:
    ``((v_1 - v_0) % N, ..., (v_{n-1} - v_0) % N)``.
    Valid events are all ordered selections of ``n-1`` distinct values from ``{1, ..., N-1}``,
    giving ``num_valid = P(N-1, n-1)`` equally-likely outcomes under a uniform shuffle.

    **Adaptive TVD formula**::

        total     = num_samples * deck_size
        num_valid = P(deck_size - 1, n - 1)
        p_ref     = 1 / min(total, num_valid)

        TVD = 0.5 * [ sum_observed |count/total - p_ref|
                    + (min(total, num_valid) - num_observed) * p_ref ]

    Observed events are counted sparsely (no enumeration of all valid tuples), so the unobserved
    contribution is computed analytically. This makes the formula O(total) in time and memory
    regardless of ``n`` or ``deck_size``.

    **Sampling regimes**:

    - ``n = 2``: ``num_valid = deck_size - 1``, ``total / num_valid ≈ num_samples``. Always
      oversampled for reasonable ``num_samples``; reduces to standard TVD.
    - ``n = 3``: ``num_valid ≈ deck_size²``. Undersampled when ``num_samples < deck_size``.
      In that regime ``p_ref = 1 / total``, so a perfectly uniform shuffler scores TVD ≈ 0
      (unobserved events are not penalised beyond the sample budget), while a biased shuffler
      that revisits patterns scores higher because it observes fewer distinct events.

    :param samples: Array of shape ``(num_samples, deck_size)`` containing independent permutations.
    :param n: Degree of the n-gram.
    """
    num_samples, deck_size = samples.shape
    total = num_samples * deck_size
    num_valid = _perm(deck_size - 1, n - 1)
    p_ref = 1.0 / min(total, num_valid)

    col_indices = np.arange(deck_size).reshape(deck_size, 1) + np.arange(n).reshape(1, n)
    ngrams = samples.take(col_indices, axis=1, mode="wrap")  # (num_samples, deck_size, n)
    diffs = (ngrams[:, :, 1:] - ngrams[:, :, :1]) % deck_size  # (num_samples, deck_size, n-1)
    del ngrams  # free before encoding keys to keep peak memory bounded
    flat_diffs = diffs.reshape(-1, n - 1)  # (total, n-1)

    if n == 2:
        # Diffs are scalars in {1,...,deck_size-1}; bincount fills zeros for unobserved values
        counts = np.bincount(flat_diffs.ravel(), minlength=deck_size)[1:]
        # bincount includes zeros, so the full event space is already covered
        return float(0.5 * np.abs(counts / total - p_ref).sum())
    else:
        # Encode each diff tuple as a single int64 key, then count sparsely with np.unique
        powers = np.array([deck_size**i for i in range(n - 2, -1, -1)], dtype=np.int64)
        keys = flat_diffs.astype(np.int64) @ powers
        _, counts = np.unique(keys, return_counts=True)
        num_observed = len(counts)
        # np.unique only returns non-zero counts; add unobserved contribution analytically
        observed = float(0.5 * np.sum(np.abs(counts / total - p_ref)))
        unobserved = 0.5 * (min(total, num_valid) - num_observed) * p_ref
        return observed + unobserved
