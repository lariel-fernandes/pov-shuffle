from typing import NamedTuple

import numpy as np
import torch
from tqdm import tqdm

from povs import optim_options_for_dataset, shuffle
from povs.types import FullOptions, Options

from .metrics import get_ngram_tvd, get_ngram_tvd_num_valid, get_sample_deficit, get_tvd, get_tvd_num_valid
from .utils import time_cpu_op, time_cuda_op


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
            baseline_rng = np.random.default_rng(seed + i + len(deck_sizes))
            baseline_data = torch.zeros(deck_size, instance_size, dtype=dtype).numpy()
            baseline_run_times = time_cpu_op(
                lambda: baseline_rng.shuffle(baseline_data),
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


class BreakingPointPerDeckSizeResult(NamedTuple):
    options: dict  # deck_size -> FullOptions (after inference)
    positional_breaking_points: dict  # deck_size -> iteration (1-indexed) | None if not converged
    ngram_breaking_points: dict  # deck_size -> {degree -> iteration | None}
    sample_deficits: dict  # deck_size -> {metric_name -> int}


def breaking_point_per_deck_size(
    deck_sizes: list,
    num_samples: int,
    ngram_degrees: list,
    positional_tolerance: float,
    ngram_tolerances: dict,
    default_ngram_tolerance: float,
    max_iterations_per_deck_size: dict,
    default_max_iterations: int,
    povs_options_per_deck_size: dict,
    default_options,
    rng: np.random.Generator,
    dtype,
    device: str,
) -> BreakingPointPerDeckSizeResult:
    """Find the breaking point iteration per deck size for all bias metrics.

    For each deck size, runs POV shuffle iteratively until every bias metric's TVD is within tolerance
    of the corresponding baseline TVD, or the iteration cap is reached.

    :param deck_sizes: Deck sizes to test.
    :param num_samples: Number of independent shuffles sampled to estimate the output distribution.
    :param ngram_degrees: N-gram degrees for which bias convergence is measured.
    :param positional_tolerance: Convergence threshold for positional TVD.
    :param ngram_tolerances: Per-degree convergence thresholds, keyed by n-gram degree.
    :param default_ngram_tolerance: Fallback threshold for degrees absent from ``ngram_tolerances``.
    :param max_iterations_per_deck_size: Hard iteration cap per deck size.
    :param default_max_iterations: Fallback cap for deck sizes without a specific entry.
    :param povs_options_per_deck_size: POV Shuffle options per deck size; ``None`` uses ``default_options``.
    :param default_options: Default POV Shuffle options.
    :param rng: Random number generator for reproducibility.
    :param dtype: Torch dtype for the deck tensor (e.g. ``torch.int32``).
    :param device: Torch device on which the deck tensor lives and is shuffled.
    """
    options_out = {}
    positional_bps = {}
    ngram_bps = {}
    deficits_out = {}

    for deck_size in tqdm(deck_sizes, desc="Deck sizes"):
        deck = torch.arange(deck_size, dtype=dtype, device=device)
        cur_options = optim_options_for_dataset(deck, povs_options_per_deck_size.get(deck_size) or default_options)
        options_out[deck_size] = cur_options
        max_iter = max_iterations_per_deck_size.get(deck_size, default_max_iterations)

        # Baseline: measure TVD of np.shuffle
        baseline_samples = deck.unsqueeze(0).expand(num_samples, -1).clone().cpu().numpy()
        for s in tqdm(range(num_samples), desc="Baseline samples", leave=False):
            rng.shuffle(baseline_samples[s])
        baseline_pos = get_tvd(baseline_samples)
        baseline_ngram = {n: get_ngram_tvd(baseline_samples, n) for n in ngram_degrees}

        # Track per-metric convergence
        pos_bp: int | None = None
        ngram_bp: dict = {n: None for n in ngram_degrees}

        samples = deck.unsqueeze(0).expand(num_samples, -1).clone()
        for iteration in tqdm(range(1, max_iter + 1), desc="Iterations", leave=False):
            for s in range(num_samples):
                shuffle(samples[s], iterations=1, seed=rng, options=cur_options)
            samples_np = samples.cpu().numpy()

            if pos_bp is None:
                pov_pos = get_tvd(samples_np)
                if pov_pos - baseline_pos < positional_tolerance:
                    pos_bp = iteration

            for n in ngram_degrees:
                if ngram_bp[n] is None:
                    tol = ngram_tolerances.get(n, default_ngram_tolerance)
                    pov_ng = get_ngram_tvd(samples_np, n)
                    if pov_ng - baseline_ngram[n] < tol:
                        ngram_bp[n] = iteration

            if pos_bp is not None and all(v is not None for v in ngram_bp.values()):
                break

        positional_bps[deck_size] = pos_bp
        ngram_bps[deck_size] = ngram_bp
        deficits_out[deck_size] = {
            "positional": get_sample_deficit(num_samples, deck_size, get_tvd_num_valid(deck_size)),
            **{
                f"{n}-gram": get_sample_deficit(num_samples, deck_size, get_ngram_tvd_num_valid(deck_size, n))
                for n in ngram_degrees
            },
        }

    return BreakingPointPerDeckSizeResult(
        options=options_out,
        positional_breaking_points=positional_bps,
        ngram_breaking_points=ngram_bps,
        sample_deficits=deficits_out,
    )


class TVDPerIterResult(NamedTuple):
    options: FullOptions  # options after inference for dataset
    tvds: np.ndarray  # shape (max_iterations,)
    baseline_tvd: float
    ngram_tvds: np.ndarray  # shape (max_iterations, len(ngram_degrees))
    baseline_ngram_tvds: np.ndarray  # shape (len(ngram_degrees),)
    sample_deficits: dict  # metric_name -> int


def tvd_per_iteration(
    deck_size: int,
    num_samples: int,
    max_iterations: int,
    options: Options | None,
    rng: np.random.Generator,
    ngram_degrees: list[int],
    dtype: torch.dtype,
    device: str,
) -> TVDPerIterResult:
    """Measure the Total Variation Distance of the deck shuffling as a function of the number of shuffle iterations.

    :param deck_size: The generated deck is a monotonically increasing sequence from 0 to `deck_size-1`
    :param num_samples: The number of deck permutations to perform, for estimating the resulting distribution.
    :param max_iterations: Test between 1 and max iterations (inclusive).
    :param options: POV Shuffle algorithm options.
    :param rng: Random number generator for reproducibility.
    :param ngram_degrees: N-gram degrees for which to compute TVD at each iteration.
    :param dtype: Torch dtype for the deck tensor.
    :param device: Torch device on which to allocate and shuffle the deck tensor.
    """
    deck = torch.arange(deck_size, dtype=dtype, device=device)
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

    sample_deficits = {
        "positional": get_sample_deficit(num_samples, deck_size, get_tvd_num_valid(deck_size)),
        **{
            f"{n}-gram": get_sample_deficit(num_samples, deck_size, get_ngram_tvd_num_valid(deck_size, n))
            for n in ngram_degrees
        },
    }

    return TVDPerIterResult(
        tvds=tvds,
        baseline_tvd=baseline_tvd,
        ngram_tvds=ngram_tvds,
        baseline_ngram_tvds=baseline_ngram_tvds,
        options=options,
        sample_deficits=sample_deficits,
    )
