import numpy as np
from tqdm import tqdm

from povs import POVSOptions
from povs.numpy import pov_shuffle

from .metrics import get_tvd


def tvd_per_iteration(
    deck_size: int,
    num_samples: int,
    max_iterations: int,
    options: POVSOptions,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Measure the Total Variation Distance of the deck shuffling as a function of the number of shuffle iterations.

    :param deck_size: The generated deck is a monotonically increasing sequence from 0 to `deck_size-1`
    :param num_samples: The number of deck permutations to perform, for estimating the resulting distribution.
    :param max_iterations: Test between 1 and max iterations (inclusive).
    :param options: POV Shuffle algorithm options.
    :param rng: Random number generator for reproducibility.
    """
    deck = np.arange(deck_size)
    tvds = np.zeros(max_iterations, dtype=float)

    # Initialize samples and determine the TVD of the POV Shuffle after each iteration
    samples = np.tile(deck, num_samples).reshape(num_samples, deck_size)
    for num_iterations in tqdm(range(1, max_iterations + 1), desc="Iterations"):
        for sample_id in tqdm(range(num_samples), desc="Samples", leave=False):
            pov_shuffle(samples[sample_id], iterations=1, seed=rng, options=options)
        tvds[num_iterations - 1] = get_tvd(samples)

    # Re-initialize samples and determine the TVD of a perfect shuffle (baseline)
    samples = np.tile(deck, num_samples).reshape(num_samples, deck_size)
    for sample_id in tqdm(range(num_samples), desc="Samples (baseline)"):
        rng.shuffle(samples[sample_id])
    baseline_tvd = get_tvd(samples)

    return tvds, baseline_tvd
