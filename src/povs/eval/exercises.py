from typing import NamedTuple

import numpy as np
import torch
from tqdm import tqdm

from povs import POVSOptions
from povs import pov_shuffle as _pov_shuffle
from povs.numpy import pov_shuffle

from .metrics import get_ngram_tvd, get_tvd
from .params import OptionsSetEntry
from .utils import time_cuda_op


class ShuffleTimePerDeckSizeResult(NamedTuple):
    pov_times_ms: list[list[float]]  # outer: per deck_size, inner: per run
    baseline_times_ms: list[list[float]]  # outer: per deck_size, inner: per run


class ShuffleTimePerOptionsResult(NamedTuple):
    labels: list[str]
    times_ms: list[list[float]]  # outer: per options set, inner: per run


def shuffle_time_per_deck_size(
    deck_sizes: list[int],
    iterations: int,
    instance_size: int,
    num_runs: int,
    num_warmup_runs: int,
    povs_options_per_deck_size: dict[int, POVSOptions],
    seed: int,
) -> ShuffleTimePerDeckSizeResult:
    """Measure POV Shuffle time and Fisher-Yates CUDA baseline time across deck sizes.

    :param deck_sizes: Deck sizes to benchmark.
    :param iterations: Number of POV Shuffle iterations per timed call.
    :param instance_size: Feature dimension; each CUDA tensor has shape (deck_size, instance_size).
    :param num_runs: Number of timed runs per deck size.
    :param num_warmup_runs: Number of warm-up calls before measurement.
    :param povs_options_per_deck_size: POV Shuffle options to use for each deck size.
    :param seed: Base seed; each deck size gets an independent derived seed.
    """
    pov_times = []
    baseline_times = []

    for i, deck_size in enumerate(tqdm(deck_sizes, desc="Deck sizes")):
        options = povs_options_per_deck_size[deck_size]
        data = torch.zeros(deck_size, instance_size, dtype=torch.float32, device="cuda")

        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed + i)
        pov_times.append(
            time_cuda_op(
                lambda: _pov_shuffle(data, iterations=iterations, options=options, seed=gen),
                num_warmup=num_warmup_runs,
                num_runs=num_runs,
            )
        )

        baseline_gen = torch.Generator(device="cuda")
        baseline_gen.manual_seed(seed + i + len(deck_sizes))
        baseline_times.append(
            time_cuda_op(
                lambda: data.copy_(data[torch.randperm(data.shape[0], device=data.device, generator=baseline_gen)]),
                num_warmup=num_warmup_runs,
                num_runs=num_runs,
            )
        )

    return ShuffleTimePerDeckSizeResult(pov_times_ms=pov_times, baseline_times_ms=baseline_times)


def shuffle_time_per_options(
    options_sets: list[OptionsSetEntry],
    deck_size: int,
    instance_size: int,
    iterations: int,
    num_runs: int,
    num_warmup_runs: int,
    seed: int,
) -> ShuffleTimePerOptionsResult:
    """Measure POV Shuffle time for each options configuration on a fixed deck size.

    :param options_sets: Labeled options configurations to benchmark.
    :param deck_size: Fixed number of elements in the deck.
    :param instance_size: Feature dimension; each CUDA tensor has shape (deck_size, instance_size).
    :param iterations: Number of POV Shuffle iterations per timed call.
    :param num_runs: Number of timed runs per options set.
    :param num_warmup_runs: Number of warm-up calls before measurement.
    :param seed: Base seed; each options set gets an independent derived seed.
    """
    labels = []
    times = []

    for i, entry in enumerate(tqdm(options_sets, desc="Options sets")):
        data = torch.zeros(deck_size, instance_size, dtype=torch.float32, device="cuda")
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed + i)

        labels.append(entry.label)
        times.append(
            time_cuda_op(
                lambda: _pov_shuffle(data, iterations=iterations, options=entry.options, seed=gen),
                num_warmup=num_warmup_runs,
                num_runs=num_runs,
            )
        )

    return ShuffleTimePerOptionsResult(labels=labels, times_ms=times)


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
