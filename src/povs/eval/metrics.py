import math

import numpy as np


def get_pos_tvd_event_space(deck_size: int) -> int:
    """Number of valid events for positional TVD: one travel-distance value per deck position."""
    return deck_size


def get_ngram_tvd_event_space(deck_size: int, n: int) -> int:
    """Number of valid events for n-gram TVD: tuples of n-1 relative travel distances to the n-gram anchor.

    Distances are modulo deck_size, so ranging from 1 to deck_size-1"""
    return math.perm(deck_size - 1, n - 1)


def get_sample_deficit(num_episodes: int, deck_size: int, event_space: int) -> int:
    """Signed gap between the event space and the observation budget.

    :returns: ``num_valid - num_samples * deck_size``. Positive means undersampled (more observations
              would be needed to cover all valid events at least once). Zero means exactly covered.
              Negative means oversampled (surplus observations).
    """
    return event_space - num_episodes * deck_size


def get_tvd(episodes: np.ndarray) -> float:
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

    :param episodes: Array of shape ``(num_samples, deck_size)`` containing independent permutations.
                    Values must be the monotonically increasing indices ``0`` to ``deck_size - 1``.
    """
    # TODO: adjust nomenclature of variables in docstring to match the one used in code
    num_episodes, deck_size = episodes.shape
    num_observations = int(num_episodes) * deck_size
    event_space = get_pos_tvd_event_space(deck_size)
    p_ref = 1.0 / min(num_observations, event_space)  # Reference event probability

    num_tds_per_episode = deck_size  # Observed travel distances per shuffling episode
    td_bytes = 8  # Travel distance is int64
    peak_bytes_per_episode = num_tds_per_episode * td_bytes  # Bytes for storing travel distances of each episode

    # Process in batches: avoid allocating (num_episodes * deck_size,) in one shot.
    _MAX_BYTES = 256 * 1024 * 1024  # 256 MB cap for the (batch, deck_size) int64 array
    max_episodes_per_batch = max(1, _MAX_BYTES // peak_bytes_per_episode)

    indices = np.arange(deck_size, dtype=np.int64)  # Identity array for determining travel distances
    td_counts = np.zeros(deck_size, dtype=np.int64)  # Travel distance event counts

    for start in range(0, num_episodes, max_episodes_per_batch):
        batch_episodes = episodes[start : start + max_episodes_per_batch].astype(np.int64)  # (batch_size, deck_size)

        # Because the deck itself is an identity array, the travel distance of each element after shuffling is simply
        # the difference between its value (original index) and the index where it landed (aligned value in `indices`).
        distances = (indices - batch_episodes) % deck_size  # (batch_size, deck_size), broadcast subtraction
        td_counts += np.bincount(distances.ravel(), minlength=deck_size)

    # bincount fills zeros for unobserved distances, so the sum already covers the full event space
    p_obs = td_counts / num_observations  # Observed travel distance probabilities
    return float(np.abs(p_obs - p_ref).sum() / 2)


def get_ngram_tvd(episodes: np.ndarray, n: int, skip: int = 0) -> float:
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
      (unobserved events are not penalized beyond the sample budget), while a biased shuffler
      that revisits patterns scores higher because it observes fewer distinct events.

    :param episodes: Array of shape ``(num_samples, deck_size)`` containing independent permutations.
    :param n: Degree of the n-gram.
    :param skip: Number of positions skipped between consecutive n-gram elements.
                 ``0`` (default) means adjacent elements; ``1`` means every other element, etc.
    """
    rtd_tup_size = n - 1  # Each tuple of relative travel distances from n-gram anchor to n-gram items has size n - 1
    rtd_bytes = 8  # Relative travel distances are represented as int64

    num_episodes, deck_size = episodes.shape
    num_observations = num_episodes * deck_size
    event_space = get_ngram_tvd_event_space(deck_size, n)
    p_ref = 1.0 / min(num_observations, event_space)  # Reference event probability

    # Sampling step between n-gram items
    # For skip=0 (not a skip-gram), the step is 1 (adjacent items). For skip=1, the step is 2 (every other item), etc.
    step = skip + 1

    # After-shuffle distances between each ngram item and the anchor: 0 for the anchor itself and k*step for other items
    # E.g.: for n=3 and skip=1 (i.e. step=2): [0, 2, 4]
    ngram_steps = np.arange(0, n * step, step)

    # Indexer with shape (deck_size, n) for selecting `deck_size` n-grams of size n
    # E.g. for n=2 and skip=0: [[0, 1], [1, 2], [2, 3], ...]
    ngrams_indexer = np.arange(deck_size).reshape(deck_size, 1) + ngram_steps.reshape(1, n)

    # Batch over episodes to cap the (batch_size, deck_size, n) intermediate arrays
    _MAX_BYTES = 256 * 1024 * 1024  # 256 MB

    if n == 2:
        # n=2: RTDs are scalars; accumulate bincount across batches
        key_counts = np.zeros(deck_size, dtype=np.int64)

        # v_0 and v_1 can't be the same instance.
        # Plugging v_1=v_0 and k=1 (one n-gram step) in the RTD formula gives:
        # (dist_before - dist_after) % deck = (v_1 - v_0 - k * step) % deck = (0 - 1 * step) % deck
        excluded_rtd = (-step) % deck_size

        # Determine batch_size to respect _MAX_BYTES
        ngrams_mem_peak = deck_size * n * episodes.itemsize
        rtds_mem_peak = deck_size * rtd_tup_size * rtd_bytes
        peak_bytes_per_episode = max(ngrams_mem_peak, rtds_mem_peak)  # use the worst of the two mem peaks
        max_episodes_per_batch = max(1, _MAX_BYTES // peak_bytes_per_episode)

        for start in range(0, num_episodes, max_episodes_per_batch):
            episodes_batch = episodes[start : start + max_episodes_per_batch]
            ngrams = episodes_batch.take(ngrams_indexer, axis=1, mode="wrap")  # (batch_size, deck_size, 2)
            rtds = _relative_travel_distances(ngrams, skip)  # (batch_size, deck_size, 1)
            del ngrams
            key_counts += np.bincount((rtds % deck_size).ravel().astype(np.int64), minlength=deck_size)

        key_counts = np.delete(key_counts, excluded_rtd)
        p_obs = key_counts / num_observations  # Observed RTD probabilities
        return float(1 / 2 * np.abs(p_obs - p_ref).sum())
    else:
        # n>2: encode RTD tuples as int64 keys using a bijection that interprets each tuple as an integer in the base `deck_size`
        # E.g.: deck_size=10, n=4: bijection(rtd_1, rtd_2, rtd_3) = rtd_1 * 10^2 + rtd_2 * 10^1 + rtd_3 * 10^0
        powers = np.array([deck_size**i for i in range(n - 2, -1, -1)], dtype=np.int64)

        # Determine how many distinct keys can be stored in memory at once. In the worst case, every observed n-gram
        # has a distinct RTD tuple, therefore a distinct key, so effectively this is a cap on the observation budget.
        key_bytes = 8  # Each key is represented as int64
        _MAX_KEYS_BYTES = 512 * 1024 * 1024  # Limit keys storage to 512 MB (RAM)
        max_keys = _MAX_KEYS_BYTES // key_bytes

        if num_observations > max_keys:
            num_ngrams_per_episode = max(1, max_keys // num_episodes)
            ngram_positions = np.sort(np.random.choice(deck_size, num_ngrams_per_episode, replace=False))
            effective_ngrams_indexer = (ngram_positions.reshape(-1, 1) + ngram_steps.reshape(1, n)) % deck_size
            effective_num_observations = num_episodes * num_ngrams_per_episode
        else:
            effective_ngrams_indexer = ngrams_indexer
            num_ngrams_per_episode = deck_size
            effective_num_observations = num_observations

        # Recompute p_ref against the actual observation count after subsampling
        p_ref = 1.0 / min(effective_num_observations, event_space)

        # Determine batch_size in number of episodes to respect _MAX_BYTES
        ngrams_mem_peak = num_ngrams_per_episode * n * episodes.itemsize
        rtds_mem_peak = num_ngrams_per_episode * rtd_tup_size * rtd_bytes
        peak_bytes_per_episode = max(ngrams_mem_peak, rtds_mem_peak)  # use the worst of the two mem peaks
        max_episodes_per_batch = max(1, _MAX_BYTES // peak_bytes_per_episode)

        # Rebatch based on (possibly smaller) ngrams_per_episode, then pre-allocate the key array.
        all_keys = np.empty(effective_num_observations, dtype=np.int64)
        keys_offset = 0
        for start in range(0, num_episodes, max_episodes_per_batch):
            # Sample episodes batch
            episodes_batch = episodes[start : start + max_episodes_per_batch]
            num_episodes_in_batch = len(episodes_batch)
            num_ngrams_in_batch = num_episodes_in_batch * num_ngrams_per_episode

            # (num_episodes_in_batch, num_ngrams_per_episode, n)
            ngrams = episodes_batch.take(effective_ngrams_indexer, axis=1, mode="wrap")

            # (num_episodes_in_batch, num_ngrams_per_episode, n - 1)
            rtds = _relative_travel_distances(ngrams, skip)
            del ngrams

            # (num_ngrams_in_batch, n - 1)
            flat_rtds = (rtds % deck_size).reshape(-1, n - 1).astype(np.int64)
            del rtds

            # Scalar product broadcast: (num_ngrams_in_batch, n - 1) @ (n - 1,) = (num_ngrams_in_batch,)
            # Every RTD tuple of size n - 1 is combined with the powers for the `deck_size` base, resulting in the bijection:
            # int64_key = rtd_0 * deck^(n - 2) + ... + rtd_i * deck^(n - 2 - i) + ... rtd_{n - 2} * deck^0
            all_keys[keys_offset : keys_offset + num_ngrams_in_batch] = flat_rtds @ powers
            del flat_rtds

            keys_offset += num_ngrams_in_batch

        _, key_counts = np.unique(all_keys, return_counts=True)
        del all_keys

        num_distinct_observed_events = len(key_counts)
        p_obs = key_counts / effective_num_observations  # Observed probabilities for event keys
        del key_counts

        # Number of distinct unobserved events, capped by the observation budget
        num_missing_events = min(event_space, effective_num_observations) - num_distinct_observed_events

        # Adaptive TVD with contribution of unobserved events
        return float(1 / 2 * (num_missing_events * p_ref + np.sum(np.abs(p_obs - p_ref))))


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

    # n - 1 distances to n-gram anchor after shuffling
    dist_after = step * np.arange(1, n, dtype=np.int64)

    # n - 1 distances to n-gram anchor before shuffling
    # Assuming deck is an identity array, distances before can be obtained by subtracting elements.
    dist_before = ngrams[..., 1:].astype(np.int64) - ngrams[..., :1].astype(np.int64)

    return dist_before - dist_after
