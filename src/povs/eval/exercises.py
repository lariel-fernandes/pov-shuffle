import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from tqdm import tqdm

from povs import optim_options_for_dataset, shuffle
from povs.types import FullOptions, Options

from .lstm import LSTMSettings, lstm_predictability
from .metrics import get_ngram_tvd, get_ngram_tvd_event_space, get_pos_tvd_event_space, get_sample_deficit, get_tvd
from .types import NgramSpec
from .utils import ngram_metric_name, time_cpu_op, time_cuda_op

_logger = logging.getLogger(__name__)


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


class _BreakingPointPerDeckSizeResult(NamedTuple):
    options: FullOptions
    sample_deficit: int
    positional_breaking_point: int | None
    ngram_breaking_points: dict[tuple[int, int], int | None]  # (n, skip) -> iteration | None
    lstm_breaking_point: int | None


class BreakingPointPerDeckSizeResult(NamedTuple):
    deck_sizes: dict[int, _BreakingPointPerDeckSizeResult]  # deck_size -> _BreakingPointPerDeckSizeResult


def breaking_point_per_deck_size(
    deck_sizes: list,
    num_samples: int,
    ngram_degrees: list,
    ngram_skips: list,
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
    lstm_settings: LSTMSettings | None = None,
    lstm_tolerance: float = 0.01,
    checkpoint_dir: Path | None = None,
) -> BreakingPointPerDeckSizeResult:
    """Find the breaking point iteration per deck size for all bias metrics.

    For each deck size, runs POV shuffle iteratively until every bias metric's TVD is within tolerance
    of the corresponding baseline TVD, or the iteration cap is reached.

    :param deck_sizes: Deck sizes to test.
    :param num_samples: Number of independent shuffles sampled to estimate the output distribution.
    :param ngram_degrees: N-gram degrees for which bias convergence is measured.
    :param ngram_skips: Skip values paired with ``ngram_degrees``; ``0`` means adjacent elements.
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
    :param lstm_settings: If provided, trains an LSTM at each iteration and tracks its convergence to
        the baseline predictability using ``lstm_tolerance`` as the threshold.
    :param lstm_tolerance: Convergence threshold for LSTM predictability.
    :param checkpoint_dir: If provided, episodes and metrics are persisted under
        ``<checkpoint_dir>/<deck_size>/baseline/`` and ``<checkpoint_dir>/<deck_size>/pov_<k>/``.
        Existing checkpoints are loaded instead of recomputed, enabling crash recovery.
    """
    ngram_pairs = list(zip(ngram_degrees, ngram_skips))
    options_out = {}
    positional_bps = {}
    ngram_bps = {}
    deficits_out = {}
    lstm_bps: dict | None = {} if lstm_settings is not None else None

    for deck_size in tqdm(deck_sizes, desc="Deck sizes"):
        deck = torch.arange(deck_size, dtype=dtype, device=device)
        cur_options = optim_options_for_dataset(deck, povs_options_per_deck_size.get(deck_size) or default_options)
        options_out[deck_size] = cur_options
        max_iter = max_iterations_per_deck_size.get(deck_size, default_max_iterations)
        ds_ckpt = checkpoint_dir / str(deck_size) if checkpoint_dir is not None else None
        _logger.info("deck=%d: starting (max_iterations=%d)", deck_size, max_iter)

        # --- Baseline ---
        baseline_samples = _ckpt_load_episodes(ds_ckpt, "baseline")
        if baseline_samples is None:
            _logger.info("deck=%d baseline: generating %d episodes", deck_size, num_samples)
            baseline_samples = deck.unsqueeze(0).expand(num_samples, -1).clone().cpu().numpy()
            for s in tqdm(range(num_samples), desc="Baseline samples", leave=False):
                rng.shuffle(baseline_samples[s])
            _ckpt_save_episodes(ds_ckpt, "baseline", baseline_samples)
        else:
            _logger.info("deck=%d baseline: loaded episodes from checkpoint", deck_size)

        baseline_metrics = _ckpt_load_metrics(ds_ckpt, "baseline") or {}

        if "positional" not in baseline_metrics:
            _logger.info("deck=%d baseline: computing positional TVD", deck_size)
            baseline_metrics["positional"] = get_tvd(baseline_samples)
            _ckpt_save_metrics(ds_ckpt, "baseline", baseline_metrics)
        else:
            _logger.info("deck=%d baseline: positional TVD loaded from checkpoint", deck_size)

        for n, skip in ngram_pairs:
            key = ngram_metric_name(n, skip)
            if key not in baseline_metrics:
                _logger.info("deck=%d baseline: computing %s TVD", deck_size, key)
                baseline_metrics[key] = get_ngram_tvd(baseline_samples, n, skip=skip)
                _ckpt_save_metrics(ds_ckpt, "baseline", baseline_metrics)
            else:
                _logger.info("deck=%d baseline: %s TVD loaded from checkpoint", deck_size, key)

        if lstm_settings is not None and "lstm_predictability" not in baseline_metrics:
            _logger.info("deck=%d baseline: computing LSTM predictability", deck_size)
            baseline_metrics["lstm_predictability"] = lstm_predictability(
                baseline_samples, deck_size, lstm_settings, device
            )
            _ckpt_save_metrics(ds_ckpt, "baseline", baseline_metrics)
        elif lstm_settings is not None:
            _logger.info("deck=%d baseline: LSTM predictability loaded from checkpoint", deck_size)

        baseline_pos = baseline_metrics["positional"]
        baseline_ngram = {(n, skip): baseline_metrics[ngram_metric_name(n, skip)] for n, skip in ngram_pairs}
        baseline_lstm_val = baseline_metrics.get("lstm_predictability")

        # --- Iterations ---
        pos_bp: int | None = None
        ngram_bp: dict = {(n, skip): None for n, skip in ngram_pairs}
        lstm_bp: int | None = None

        samples = deck.unsqueeze(0).expand(num_samples, -1).clone()
        for iteration in tqdm(range(1, max_iter + 1), desc="Iterations", leave=False):
            ckpt_name = f"pov_{iteration}"
            episodes = _ckpt_load_episodes(ds_ckpt, ckpt_name)
            if episodes is not None:
                _logger.info("deck=%d iter=%d: loaded episodes from checkpoint", deck_size, iteration)
                samples = torch.tensor(episodes, dtype=dtype, device=device)
                samples_np = episodes
            else:
                _logger.info("deck=%d iter=%d: shuffling %d episodes", deck_size, iteration, num_samples)
                for s in range(num_samples):
                    shuffle(samples[s], iterations=1, seed=rng, options=cur_options)
                samples_np = samples.cpu().numpy()
                _ckpt_save_episodes(ds_ckpt, ckpt_name, samples_np)

            iter_metrics = _ckpt_load_metrics(ds_ckpt, ckpt_name) or {}

            if pos_bp is None and "positional" not in iter_metrics:
                _logger.info("deck=%d iter=%d: computing positional TVD", deck_size, iteration)
                iter_metrics["positional"] = get_tvd(samples_np)
                _ckpt_save_metrics(ds_ckpt, ckpt_name, iter_metrics)
            elif pos_bp is None:
                _logger.info("deck=%d iter=%d: positional TVD loaded from checkpoint", deck_size, iteration)
            if (
                pos_bp is None
                and "positional" in iter_metrics
                and iter_metrics["positional"] - baseline_pos < positional_tolerance
            ):
                pos_bp = iteration
                _logger.info("deck=%d iter=%d: positional breaking point", deck_size, iteration)

            for n, skip in ngram_pairs:
                key = ngram_metric_name(n, skip)
                if ngram_bp[(n, skip)] is None and key not in iter_metrics:
                    _logger.info("deck=%d iter=%d: computing %s TVD", deck_size, iteration, key)
                    iter_metrics[key] = get_ngram_tvd(samples_np, n, skip=skip)
                    _ckpt_save_metrics(ds_ckpt, ckpt_name, iter_metrics)
                elif ngram_bp[(n, skip)] is None:
                    _logger.info("deck=%d iter=%d: %s TVD loaded from checkpoint", deck_size, iteration, key)
                if (
                    ngram_bp[(n, skip)] is None
                    and key in iter_metrics
                    and iter_metrics[key] - baseline_ngram[(n, skip)] < ngram_tolerances.get(n, default_ngram_tolerance)
                ):
                    ngram_bp[(n, skip)] = iteration
                    _logger.info("deck=%d iter=%d: %s breaking point", deck_size, iteration, key)

            if lstm_settings is not None and lstm_bp is None and "lstm_predictability" not in iter_metrics:
                _logger.info("deck=%d iter=%d: computing LSTM predictability", deck_size, iteration)
                iter_metrics["lstm_predictability"] = lstm_predictability(samples_np, deck_size, lstm_settings, device)
                _ckpt_save_metrics(ds_ckpt, ckpt_name, iter_metrics)
            elif lstm_settings is not None and lstm_bp is None:
                _logger.info("deck=%d iter=%d: LSTM predictability loaded from checkpoint", deck_size, iteration)
            if (
                lstm_settings is not None
                and lstm_bp is None
                and "lstm_predictability" in iter_metrics
                and iter_metrics["lstm_predictability"] - baseline_lstm_val < lstm_tolerance
            ):
                lstm_bp = iteration
                _logger.info("deck=%d iter=%d: LSTM breaking point", deck_size, iteration)

            tvd_converged = pos_bp is not None and all(v is not None for v in ngram_bp.values())
            lstm_converged = lstm_settings is None or lstm_bp is not None
            if tvd_converged and lstm_converged:
                break

        positional_bps[deck_size] = pos_bp
        ngram_bps[deck_size] = ngram_bp
        if lstm_bps is not None:
            lstm_bps[deck_size] = lstm_bp
        deficits_out[deck_size] = {
            "positional": get_sample_deficit(num_samples, deck_size, get_pos_tvd_event_space(deck_size)),
            **{
                ngram_metric_name(n, skip): get_sample_deficit(
                    num_samples, deck_size, get_ngram_tvd_event_space(deck_size, n)
                )
                for n, skip in ngram_pairs
            },
        }

    return BreakingPointPerDeckSizeResult(
        options=options_out,
        positional_breaking_points=positional_bps,
        ngram_breaking_points=ngram_bps,
        sample_deficits=deficits_out,
        lstm_breaking_points=lstm_bps,
    )


def _ckpt_load_episodes(ds_ckpt: Path | None, name: str) -> np.ndarray | None:
    if ds_ckpt is None:
        return None
    d = ds_ckpt / name
    npy = d / "episodes.npy"
    if npy.exists():
        return np.load(npy)
    npz = d / "episodes.npz"
    return np.load(npz)["arr_0"] if npz.exists() else None


def _ckpt_save_episodes(ds_ckpt: Path | None, name: str, episodes: np.ndarray) -> None:
    if ds_ckpt is None:
        return
    d = ds_ckpt / name
    d.mkdir(parents=True, exist_ok=True)
    # np.save is much faster than np.savez_compressed for near-random data (permutations
    # compress poorly, so zlib burns CPU cycles for minimal size reduction)
    np.save(d / "episodes.npy", episodes)


def _ckpt_load_metrics(ds_ckpt: Path | None, name: str) -> dict | None:
    if ds_ckpt is None:
        return None
    path = ds_ckpt / name / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def _ckpt_save_metrics(ds_ckpt: Path | None, name: str, metrics: dict) -> None:
    if ds_ckpt is None:
        return
    d = ds_ckpt / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(metrics))


class BiasPerIterResult(NamedTuple):
    options: FullOptions  # options after inference for dataset
    sample_deficits: dict[str, int]  # metric_name -> int

    baseline_pos_tvd: float
    pos_tvds: np.ndarray  # shape (max_iterations,)

    baseline_ngram_tvds: np.ndarray  # shape (len(ngrams_specs),)
    ngram_tvds: np.ndarray  # shape (max_iterations, len(ngrams_specs))

    baseline_lstm_predictability: float | None = None
    lstm_predictabilities: np.ndarray | None = None  # shape (max_iterations,)


def bias_per_iteration(
    deck_size: int,
    num_samples: int,
    max_iterations: int,
    options: Options | None,
    rng: np.random.Generator,
    ngrams_specs: list[NgramSpec],
    dtype: torch.dtype,
    device: str,
    lstm_settings: LSTMSettings | None = None,
) -> BiasPerIterResult:
    """Measure the Total Variation Distance of the deck shuffling as a function of the number of shuffle iterations.

    :param deck_size: The generated deck is a monotonically increasing sequence from 0 to `deck_size-1`
    :param num_samples: The number of deck permutations to perform, for estimating the resulting distribution.
    :param max_iterations: Test between 1 and max iterations (inclusive).
    :param options: POV Shuffle algorithm options.
    :param rng: Random number generator for reproducibility.
    :param ngrams_specs: N-gram degrees to measure TVD, optionally with skip values to define skip-grams.
    :param dtype: Torch dtype for the deck tensor.
    :param device: Torch device on which to allocate and shuffle the deck tensor.
    :param lstm_settings: If provided, trains an LSTM to measure RTD predictability using the
        context window size from ``settings.context_length``.
    """
    deck = torch.arange(deck_size, dtype=dtype, device=device)
    options = optim_options_for_dataset(deck, options)

    tvds = np.zeros(max_iterations, dtype=float)
    ngram_tvds = np.zeros((max_iterations, len(ngrams_specs)), dtype=float)
    lstm_pred = np.zeros(max_iterations) if lstm_settings is not None else None

    # Initialize samples and determine the TVD of the POV Shuffle after each iteration
    samples = deck.unsqueeze(0).expand(num_samples, -1).clone()
    for i in tqdm(range(max_iterations), desc="Iterations"):
        for sample_id in tqdm(range(num_samples), desc="Samples", leave=False):
            shuffle(samples[sample_id], iterations=1, seed=rng, options=options)
        samples_np = samples.cpu().numpy()
        tvds[i] = get_tvd(samples_np)
        ngram_tvds[i, :] = np.array([get_ngram_tvd(samples_np, n, skip=skip) for n, skip in ngrams_specs])
        if lstm_settings is not None and lstm_pred is not None:
            lstm_pred[i] = lstm_predictability(samples_np, deck_size, lstm_settings, device)

    # Re-initialize samples and determine the TVD of a perfect shuffle (baseline)
    samples_np = deck.unsqueeze(0).expand(num_samples, -1).clone().cpu().numpy()
    for sample_id in tqdm(range(num_samples), desc="Samples (baseline)"):
        rng.shuffle(samples_np[sample_id])
    baseline_tvd = get_tvd(samples_np)
    baseline_ngram_tvds = np.array([get_ngram_tvd(samples_np, n, skip=skip) for n, skip in ngrams_specs])
    baseline_lstm = (
        lstm_predictability(samples_np, deck_size, lstm_settings, device) if lstm_settings is not None else None
    )

    sample_deficits = {
        "positional": get_sample_deficit(num_samples, deck_size, get_pos_tvd_event_space(deck_size)),
        **{
            ngram_metric_name(n, skip): get_sample_deficit(
                num_samples, deck_size, get_ngram_tvd_event_space(deck_size, n)
            )
            for n, skip in ngrams_specs
        },
    }

    return BiasPerIterResult(
        pos_tvds=tvds,
        baseline_pos_tvd=baseline_tvd,
        ngram_tvds=ngram_tvds,
        baseline_ngram_tvds=baseline_ngram_tvds,
        options=options,
        sample_deficits=sample_deficits,
        lstm_predictabilities=lstm_pred,
        baseline_lstm_predictability=baseline_lstm,
    )
