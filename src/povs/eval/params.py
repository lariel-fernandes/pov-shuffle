from typing import NamedTuple

from povs import Options

from .lstm import LSTMSettings
from .types import NgramSpec


class BreakingPointParams(NamedTuple):
    """Breaking point per deck size experiment parameters.

    - `seed`: RNG seed for reproducibility.
    - `num_episodes`: Number of independent shuffles sampled to estimate the output distribution.
    - `deck_sizes`: Deck sizes to test, in ascending order.
    - `ngram_specs`: N-gram degrees for which TVD was measured, optionally with skip values to define skip-grams.
    - `positional_tolerance`: Convergence threshold for positional TVD: ``tvd_pov - tvd_baseline < tolerance``.
    - `ngram_tolerances`: Per-degree convergence thresholds, keyed by n-gram degree.
    - `default_ngram_tolerance`: Fallback threshold for n-gram degrees absent from ``ngram_tolerances``.
    - `max_iterations_per_deck_size`: Hard iteration cap per deck size, keyed by deck size.
    - `default_max_iterations`: Fallback cap for deck sizes absent from ``max_iterations_per_deck_size``.
    - `povs_options_per_deck_size`: POV Shuffle options per deck size; replaced with optimized ``FullOptions``
      in the report.
    - `default_options`: Default POV Shuffle options for deck sizes without specific options.
    - `dtype`: Torch dtype name for the deck tensor (e.g. ``"int32"``, ``"int64"``).
    - `device`: Torch device on which the deck tensor lives and is shuffled (e.g. ``"cpu"``, ``"cuda"``).
    - `lstm_settings`: If provided, an LSTM is trained at each iteration and its breaking point is tracked
      alongside the TVD metrics. A single shared architecture is used across all deck sizes.
    - `lstm_tolerance`: Convergence threshold for LSTM predictability: ``pred_pov - pred_baseline < tolerance``.
    """

    seed: int
    num_episodes: int
    deck_sizes: list[int]
    ngram_specs: list[NgramSpec]
    positional_tolerance: float
    ngram_tolerances: dict[int, float]
    default_ngram_tolerance: float
    max_iterations_per_deck_size: dict[int, int]
    default_max_iterations: int
    povs_options_per_deck_size: dict[int, Options | None]
    default_options: Options | None
    dtype: str
    device: str
    lstm_settings: LSTMSettings | None = None
    lstm_tolerance: float = 0.01


class TimePerDeckSizeParams(NamedTuple):
    """Time per deck size experiment parameters.

    - `seed`: RNG seed for reproducibility.
    - `deck_sizes`: Deck sizes to benchmark, in ascending order.
    - `iterations`: Number of POV Shuffle iterations per timed call.
    - `instance_size`: Feature dimension of each deck element; tensor shape is `(deck_size, instance_size)`.
    - `num_runs`: Number of timed runs per deck size (for averaging).
    - `num_warmup_runs`: Number of warm-up calls before timing begins (not measured).
    - `povs_options_per_deck_size`: POV Shuffle options to use for each deck size, keyed by deck size.
    - `tolerate_errors`: If ``True``, errors during timing are recorded and the experiment continues.
      If ``False``, the first error is re-raised immediately.
    - `cuda_device_id`: Integer ID of the CUDA device on which tensors are allocated and benchmarked.
    """

    seed: int
    deck_sizes: list[int]
    iterations: int
    instance_size: int
    num_runs: int
    num_warmup_runs: int
    povs_options_per_deck_size: dict[int, Options | None]
    default_options: Options | None
    dtype: str
    tolerate_errors: bool
    cuda_device_id: int = 0


class BiasPerIterParams(NamedTuple):
    """TVD per iteration experiment parameters.

    - `seed`: RNG seed for reproducibility.
    - `deck_size`: Number of elements in the deck (dataset size proxy).
    - `num_episodes`: Number of independent shuffles sampled to estimate the output distribution.
    - `max_iterations`: Number of shuffle iterations tested (from 1 to this value, inclusive).
    - `povs_options`: POV Shuffle algorithm options used in this run.
    - `ngram_specs`: N-gram degrees for which TVD was measured, optionally with skip values to define skip-grams.
    - `dtype`: Torch dtype name for the deck tensor (e.g. ``"int32"``, ``"int64"``).
    - `device`: Torch device on which the deck tensor lives and is shuffled (e.g. ``"cpu"``, ``"cuda"``).
    """

    seed: int
    num_episodes: int
    deck_size: int
    max_iterations: int
    ngram_specs: list[NgramSpec]
    povs_options: Options | None
    dtype: str
    device: str
    lstm_settings: LSTMSettings | None = None
