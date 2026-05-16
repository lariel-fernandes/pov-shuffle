import numpy as np
from tqdm import tqdm

from povs.numpy import pov_shuffle

from .metrics import get_tvd
from .types import PovsOptions


def tvd_per_iteration(
    deck_size: int,
    num_samples: int,
    max_iterations: int,
    options: PovsOptions,
    state: np.random.RandomState,
) -> tuple[np.ndarray, float]:
    """Measure the Total Variation Distance of the deck shuffling as a function of the number of shuffle iterations.

    :param deck_size: The generated deck is a monotonically increasing sequence from 0 to `deck_size-1`
    :param num_samples: The number of deck permutations to perform, for estimating the resulting distribution.
    :param max_iterations: Test between 1 and max iterations (inclusive).
    :param options: POV Shuffle algorithm options.
    :param state: Random number generator state for reproducibility.
    """
    deck = np.arange(deck_size)

    tvds = np.zeros(max_iterations, dtype=float)

    for num_iterations in tqdm(range(1, max_iterations + 1), desc="Num Iterations"):
        samples = deck.repeat(num_samples).reshape(num_samples, deck_size)

        for sample_id in tqdm(range(num_samples), desc="Samples", leave=False):
            sample = deck.copy()
            pov_shuffle(sample, iterations=num_iterations, seed=state, **options._asdict())
            samples[sample_id] = sample

        tvds[num_iterations - 1] = get_tvd(samples)

    samples = deck.repeat(num_samples).reshape(num_samples, deck_size)
    for sample_id in tqdm(range(num_samples), desc="Samples (baseline)", leave=False):
        sample = deck.copy()
        state.shuffle(sample)
        samples[sample_id] = sample
    baseline_tvd = get_tvd(samples)

    return tvds, baseline_tvd
