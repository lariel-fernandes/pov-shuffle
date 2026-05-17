from typing import NamedTuple

import numpy as np
from tqdm import tqdm

from povs import POVSOptions
from povs.numpy import pov_shuffle

from .metrics import get_ngram_tvd, get_tvd


class TVDPerIterResult(NamedTuple):
    tvds: np.ndarray  # shape (max_iterations,)
    baseline_tvd: float
    ngram_tvds: np.ndarray  # shape (max_iterations, len(ngram_degrees))
    baseline_ngram_tvds: np.ndarray  # shape (len(ngram_degrees),)


def tvd_per_iteration(
    deck_size: int,
    num_samples: int,
    max_iterations: int,
    options: POVSOptions,
    rng: np.random.Generator,
    ngram_degrees: list[int],
) -> TVDPerIterResult:
    """Measure the Total Variation Distance of the deck shuffling as a function of the number of shuffle iterations.

    :param deck_size: The generated deck is a monotonically increasing sequence from 0 to `deck_size-1`
    :param num_samples: The number of deck permutations to perform, for estimating the resulting distribution.
    :param max_iterations: Test between 1 and max iterations (inclusive).
    :param options: POV Shuffle algorithm options.
    :param rng: Random number generator for reproducibility.
    :param ngram_degrees: N-gram degrees for which to compute TVD at each iteration.
    """
    deck = np.arange(deck_size)
    tvds = np.zeros(max_iterations, dtype=float)
    ngram_tvds = np.zeros((max_iterations, len(ngram_degrees)), dtype=float)

    # Initialize samples and determine the TVD of the POV Shuffle after each iteration
    samples = np.tile(deck, num_samples).reshape(num_samples, deck_size)
    for i in tqdm(range(max_iterations), desc="Iterations"):
        for sample_id in tqdm(range(num_samples), desc="Samples", leave=False):
            pov_shuffle(samples[sample_id], iterations=1, seed=rng, options=options)
        tvds[i] = get_tvd(samples)
        ngram_tvds[i, :] = np.array([get_ngram_tvd(samples, n) for n in ngram_degrees])

    # Re-initialize samples and determine the TVD of a perfect shuffle (baseline)
    samples = np.tile(deck, num_samples).reshape(num_samples, deck_size)
    for sample_id in tqdm(range(num_samples), desc="Samples (baseline)"):
        rng.shuffle(samples[sample_id])
    baseline_tvd = get_tvd(samples)
    baseline_ngram_tvds = np.array([get_ngram_tvd(samples, n) for n in ngram_degrees])

    return TVDPerIterResult(
        tvds=tvds,
        baseline_tvd=baseline_tvd,
        ngram_tvds=ngram_tvds,
        baseline_ngram_tvds=baseline_ngram_tvds,
    )
