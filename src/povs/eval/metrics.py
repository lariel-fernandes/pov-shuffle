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
    total = int(num_samples) * deck_size
    num_valid = deck_size
    p_ref = 1.0 / min(total, num_valid)

    # Process in batches: avoid allocating (num_samples * deck_size,) in one shot.
    _MAX_BYTES = 256 * 1024 * 1024  # 256 MB cap for the (batch, deck_size) int64 array
    batch_size = max(1, _MAX_BYTES // (deck_size * 8))
    arange = np.arange(deck_size, dtype=np.int64)
    counts = np.zeros(deck_size, dtype=np.int64)
    for start in range(0, num_samples, batch_size):
        batch = samples[start : start + batch_size].astype(np.int64)  # (b, deck_size)
        tds = (arange - batch) % deck_size  # (b, deck_size), broadcast
        counts += np.bincount(tds.ravel(), minlength=deck_size)
    # bincount fills zeros for unobserved distances, so the sum already covers the full event space
    return float(0.5 * np.abs(counts / total - p_ref).sum())


def get_ngram_tvd(samples: np.ndarray, n: int, skip: int = 0) -> float:
    """TVD of the n-gram relative-travel-distance distribution against uniform.

    For each n-gram of observed values ``(v_0, ..., v_{n-1})`` at positions spaced ``skip+1`` apart
    (with wrap-around), the event is the tuple of relative travel distances
    ``(rtd_1, ..., rtd_{n-1})`` from :func:`_relative_travel_distances`, reduced modulo
    ``deck_size`` for counting. The modular reduction collapses the signed RTD range into
    ``deck_size - 1`` equally-likely equivalence classes under a uniform shuffle.

    Valid events are all tuples where the underlying value differences
    ``(v_k - v_0) mod N`` form an ordered selection of ``n-1`` distinct non-zero values from
    ``{1, ..., N-1}``, giving ``num_valid = P(N-1, n-1)`` equally-likely outcomes under a uniform
    shuffle. This count is the same for all values of ``skip``.

    **Adaptive TVD formula**::

        total     = num_samples * deck_size
        num_valid = P(deck_size - 1, n - 1)
        p_ref     = 1 / min(total, num_valid)

        TVD = 0.5 * [ sum_observed |count/total - p_ref|
                    + (min(total, num_valid) - num_observed) * p_ref ]

    Observed events are counted sparsely (no enumeration of all valid tuples), so the unobserved
    contribution is computed analytically. Episodes are processed in batches to keep peak memory
    proportional to ``batch_size * deck_size * n`` rather than ``num_samples * deck_size * n``.

    **Sampling regimes**:

    - ``n = 2``: ``num_valid = deck_size - 1``, ``total / num_valid ≈ num_samples``. Always
      oversampled for reasonable ``num_samples``; reduces to standard TVD.
    - ``n = 3``: ``num_valid ≈ deck_size²``. Undersampled when ``num_samples < deck_size``.
      In that regime ``p_ref = 1 / total``, so a perfectly uniform shuffler scores TVD ≈ 0
      (unobserved events are not penalised beyond the sample budget), while a biased shuffler
      that revisits patterns scores higher because it observes fewer distinct events.

    :param samples: Array of shape ``(num_samples, deck_size)`` containing independent permutations.
    :param n: Degree of the n-gram.
    :param skip: Number of positions skipped between consecutive n-gram elements.
                 ``0`` (default) means adjacent elements; ``1`` means every other element, etc.
    """
    num_samples, deck_size = samples.shape
    total = num_samples * deck_size
    num_valid = _perm(deck_size - 1, n - 1)
    p_ref = 1.0 / min(total, num_valid)

    step = skip + 1
    col_indices = np.arange(deck_size).reshape(deck_size, 1) + np.arange(0, n * step, step).reshape(1, n)

    # Batch over episodes to cap the (batch_size, deck_size, n) intermediate arrays
    _MAX_BYTES = 256 * 1024 * 1024  # 256 MB
    bytes_per_episode = deck_size * max(n * samples.itemsize, (n - 1) * 8)
    batch_size = max(1, _MAX_BYTES // bytes_per_episode)

    if n == 2:
        # n=2: RTDs are scalars; accumulate bincount across batches
        excluded = (deck_size - skip - 1) % deck_size
        counts = np.zeros(deck_size, dtype=np.int64)
        for start in range(0, num_samples, batch_size):
            batch = samples[start : start + batch_size]
            ngrams = batch.take(col_indices, axis=1, mode="wrap")  # (b, deck_size, 2)
            rtds = _relative_travel_distances(ngrams, skip)  # (b, deck_size, 1)
            del ngrams
            counts += np.bincount((rtds % deck_size).ravel().astype(np.int64), minlength=deck_size)
        counts = np.delete(counts, excluded)
        return float(0.5 * np.abs(counts / total - p_ref).sum())
    else:
        # n>2: encode RTD tuples as int64 keys.
        # Two memory concerns:
        #   1. Intermediate (batch, pos, n) arrays: already bounded by _MAX_BYTES via batch_size.
        #   2. The final keys array: total = num_samples * deck_size int64 values, which for
        #      large decks can reach several GB and then np.unique needs a second sorted copy.
        #      Cap it by subsampling positions — the metric is already undersampled at large
        #      deck sizes so the estimate is statistically equivalent.
        _MAX_KEY_BYTES = 512 * 1024 * 1024  # 512 MB for the pre-allocated keys array
        max_keys = _MAX_KEY_BYTES // 8
        if total > max_keys:
            pos_per_sample = max(1, max_keys // num_samples)
            sampled_pos = np.sort(np.random.choice(deck_size, pos_per_sample, replace=False))
            step_offsets = np.arange(0, n * step, step)
            eff_col_indices = (sampled_pos.reshape(-1, 1) + step_offsets.reshape(1, n)) % deck_size
            effective_total = num_samples * pos_per_sample
        else:
            eff_col_indices = col_indices
            pos_per_sample = deck_size
            effective_total = total
        # Recompute p_ref against the actual observation count after subsampling
        p_ref = 1.0 / min(effective_total, num_valid)

        # Rebatch based on (possibly smaller) pos_per_sample, then pre-allocate the key array.
        # Pre-allocation (vs accumulating a list + concatenating) avoids the 2x memory spike
        # from holding both the list and the concatenated result simultaneously.
        bytes_per_ep = pos_per_sample * max(n * samples.itemsize, (n - 1) * 8)
        ep_batch = max(1, _MAX_BYTES // bytes_per_ep)
        powers = np.array([deck_size**i for i in range(n - 2, -1, -1)], dtype=np.int64)
        all_keys = np.empty(effective_total, dtype=np.int64)
        offset = 0
        for start in range(0, num_samples, ep_batch):
            batch = samples[start : start + ep_batch]
            b = len(batch)
            chunk = b * pos_per_sample
            ngrams = batch.take(eff_col_indices, axis=1, mode="wrap")  # (b, pos_per_sample, n)
            rtds = _relative_travel_distances(ngrams, skip)
            del ngrams
            flat_rtds = (rtds % deck_size).reshape(-1, n - 1).astype(np.int64)
            del rtds
            all_keys[offset : offset + chunk] = flat_rtds @ powers
            del flat_rtds
            offset += chunk

        _, counts = np.unique(all_keys, return_counts=True)
        del all_keys
        num_observed = len(counts)
        # np.unique only returns non-zero counts; add unobserved contribution analytically
        observed = float(0.5 * np.sum(np.abs(counts / effective_total - p_ref)))
        unobserved = 0.5 * (min(effective_total, num_valid) - num_observed) * p_ref
        return observed + unobserved


def _relative_travel_distances(ngrams: np.ndarray, skip: int = 0) -> np.ndarray:
    """Signed relative travel distance of each non-anchor element in a batch of n-grams.

    For an n-gram ``(v_0, v_1, ..., v_{n-1})`` observed at positions spaced ``skip+1`` apart,
    the relative travel distance of ``v_k`` is how much closer it became to the anchor ``v_0``
    as a result of shuffling:

        rtd_k = (v_k - v_0) - k * (skip + 1)

    ``v_k - v_0`` is the original signed positional gap; ``k * (skip + 1)`` is the current
    separation. Positive values mean the pair got closer; negative means they drifted further apart.

    Under a uniform shuffle, ``rtd_k`` takes ``deck_size - 1`` equally-likely values in the
    approximate range ``[-deck_size, +deck_size]``.

    :param ngrams: Array of shape ``(..., n)`` containing observed values.
    :param skip: Number of positions skipped between consecutive n-gram observations.
                 ``0`` (default) means adjacent; ``1`` means every other element, etc.
    :returns: Array of shape ``(..., n-1)``.
    """
    n = ngrams.shape[-1]
    step = skip + 1
    expected = step * np.arange(1, n, dtype=np.int64)
    diffs = ngrams[..., 1:].astype(np.int64) - ngrams[..., :1].astype(np.int64)
    return diffs - expected
