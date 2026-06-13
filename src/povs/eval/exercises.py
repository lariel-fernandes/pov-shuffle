from typing import NamedTuple

import numpy as np
import torch
from tqdm import tqdm

from povs import optim_options_for_dataset, shuffle
from povs.types import FullOptions, Options

from .metrics import get_ngram_tvd, get_tvd
from .utils import time_cuda_op


class ShuffleTimePerDeckSizeResult(NamedTuple):
    pov_times_ms: list[list[float] | None]  # outer: per deck_size, inner: per run, None if error
    baseline_times_ms: list[list[float] | None]  # outer: per deck_size, inner: per run, None if error
    option_sets: list[FullOptions | None]  # one per deck_size
    pov_errors: list[Exception | None]  # one per deck_size
    baseline_errors: list[Exception | None]  # one per deck_size


def shuffle_time_per_deck_size(
    deck_sizes: list[int],
    iterations: int,
    instance_size: int,
    num_runs: int,
    num_warmup_runs: int,
    povs_options_per_deck_size: dict[int, Options | None],
    default_options: Options | None,
    dtype: torch.dtype,
    seed: int,
    tolerate_errors: bool,
    cuda_device_id: int = 0,
) -> ShuffleTimePerDeckSizeResult:
    """Measure POV Shuffle time and Fisher-Yates CUDA baseline time across deck sizes.

    :param deck_sizes: Deck sizes to benchmark.
    :param iterations: Number of POV Shuffle iterations per timed call.
    :param instance_size: Feature dimension; each CUDA tensor has shape (deck_size, instance_size).
    :param num_runs: Number of timed runs per deck size.
    :param num_warmup_runs: Number of warm-up calls before measurement.
    :param povs_options_per_deck_size: POV Shuffle options to use for each deck size.
    :param default_options: Default POV Shuffle options for deck sizes without specific options.
    :param dtype: Numeric data type.
    :param seed: Base seed; each deck size gets an independent derived seed.
    :param tolerate_errors: If False, re-raise the first error immediately; if True, record it and continue.
    :param cuda_device_id: Integer ID of the CUDA device on which to allocate and time operations.
    """
    pov_times = []
    baseline_times = []
    option_sets = []
    pov_errors: list[Exception | None] = []
    baseline_errors: list[Exception | None] = []

    for i, deck_size in enumerate(tqdm(deck_sizes, desc="Deck sizes")):
        data = torch.zeros(deck_size, instance_size, dtype=dtype, device=f"cuda:{cuda_device_id}")
        pov_error: Exception | None = None
        baseline_error: Exception | None = None
        pov_run_times: list[float] | None = None
        baseline_run_times: list[float] | None = None
        cur_options: FullOptions | None = None

        try:
            cur_options = optim_options_for_dataset(data, povs_options_per_deck_size.get(deck_size) or default_options)
            pov_run_times = time_cuda_op(
                lambda: shuffle(data, iterations=iterations, options=cur_options, seed=seed),
                num_warmup=num_warmup_runs,
                num_runs=num_runs,
            )
        except Exception as e:
            if not tolerate_errors:
                raise
            pov_error = e

        try:
            baseline_gen = torch.Generator(device=f"cuda:{cuda_device_id}")
            baseline_gen.manual_seed(seed + i + len(deck_sizes))
            baseline_run_times = time_cuda_op(
                lambda: _baseline_torch_shuffle(data, baseline_gen),
                num_warmup=num_warmup_runs,
                num_runs=num_runs,
            )
        except Exception as e:
            if not tolerate_errors:
                raise
            baseline_error = e

        option_sets.append(cur_options)
        pov_times.append(pov_run_times)
        baseline_times.append(baseline_run_times)
        pov_errors.append(pov_error)
        baseline_errors.append(baseline_error)

    return ShuffleTimePerDeckSizeResult(
        pov_times_ms=pov_times,
        baseline_times_ms=baseline_times,
        option_sets=option_sets,
        pov_errors=pov_errors,
        baseline_errors=baseline_errors,
    )


class TVDPerIterResult(NamedTuple):
    options: FullOptions  # options after inference for dataset
    tvds: np.ndarray  # shape (max_iterations,)
    baseline_tvd: float
    ngram_tvds: np.ndarray  # shape (max_iterations, len(ngram_degrees))
    baseline_ngram_tvds: np.ndarray  # shape (len(ngram_degrees),)


def tvd_per_iteration(
    deck_size: int,
    num_samples: int,
    max_iterations: int,
    options: Options | None,
    rng: np.random.Generator,
    ngram_degrees: list[int],
    device: str,
) -> TVDPerIterResult:
    """Measure the Total Variation Distance of the deck shuffling as a function of the number of shuffle iterations.

    :param deck_size: The generated deck is a monotonically increasing sequence from 0 to `deck_size-1`
    :param num_samples: The number of deck permutations to perform, for estimating the resulting distribution.
    :param max_iterations: Test between 1 and max iterations (inclusive).
    :param options: POV Shuffle algorithm options.
    :param rng: Random number generator for reproducibility.
    :param ngram_degrees: N-gram degrees for which to compute TVD at each iteration.
    :param device: Torch device on which to allocate and shuffle the deck tensor.
    """
    deck = torch.arange(deck_size, device=device)
    options = optim_options_for_dataset(deck, options)

    tvds = np.zeros(max_iterations, dtype=float)
    ngram_tvds = np.zeros((max_iterations, len(ngram_degrees)), dtype=float)

    # Initialize samples and determine the TVD of the POV Shuffle after each iteration
    samples = deck.unsqueeze(0).expand(num_samples, -1).clone()
    for i in tqdm(range(max_iterations), desc="Iterations"):
        for sample_id in tqdm(range(num_samples), desc="Samples", leave=False):
            shuffle(samples[sample_id], iterations=1, seed=rng, options=options)
        samples_np = samples.cpu().numpy()
        tvds[i] = get_tvd(samples_np)
        ngram_tvds[i, :] = np.array([get_ngram_tvd(samples_np, n) for n in ngram_degrees])

    # Re-initialize samples and determine the TVD of a perfect shuffle (baseline)
    samples_np = deck.unsqueeze(0).expand(num_samples, -1).clone().cpu().numpy()
    for sample_id in tqdm(range(num_samples), desc="Samples (baseline)"):
        rng.shuffle(samples_np[sample_id])
    baseline_tvd = get_tvd(samples_np)
    baseline_ngram_tvds = np.array([get_ngram_tvd(samples_np, n) for n in ngram_degrees])

    return TVDPerIterResult(
        tvds=tvds,
        baseline_tvd=baseline_tvd,
        ngram_tvds=ngram_tvds,
        baseline_ngram_tvds=baseline_ngram_tvds,
        options=options,
    )


def _baseline_torch_shuffle(data: torch.Tensor, gen: torch.Generator) -> None:
    """For a fair comparison with the algorithm we perform a truly uniform, zero-copy, in-place shuffle."""
    n = data.shape[0]
    perm = torch.randperm(n, device=data.device, generator=gen).cpu().numpy()
    visited = np.zeros(n, dtype=bool)
    buf = torch.empty_like(data[0])
    for start in range(n):
        if visited[start]:
            continue
        j = int(perm[start])
        if j == start:
            visited[start] = True
            continue
        buf.copy_(data[start])
        i = start
        while j != start:
            data[i].copy_(data[j])
            visited[i] = True
            i = j
            j = int(perm[i])
        data[i].copy_(buf)
        visited[i] = True
