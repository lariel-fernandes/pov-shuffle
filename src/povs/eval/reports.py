from typing import NamedTuple

import matplotlib.figure
import pandas as pd

from .params import BiasPerIterParams, BreakingPointParams, TimePerDeckSizeParams


class TimePerDeckSizeReport(NamedTuple):
    """POV Shuffle time per deck size experiment report.

    **Results & Metrics**:

    - `plot`: Matplotlib figure showing shuffle time vs deck size for POV Shuffle and Fisher-Yates CUDA baseline,
              with a secondary y-axis showing the speedup ratio.

    - `timings`: DataFrame with one row per deck size. Columns:
      - `deck_size`: Number of elements in the deck.
      - `pov_mean_ms`: Mean POV Shuffle time in milliseconds across timing runs.
      - `pov_std_ms`: Standard deviation of POV Shuffle time.
      - `baseline_mean_ms`: Mean Fisher-Yates (CUDA randperm + copy) time in milliseconds.
      - `baseline_std_ms`: Standard deviation of baseline time.
      - `speedup`: Ratio `baseline_mean_ms / pov_mean_ms`. Values > 1 mean POV Shuffle is faster.

    - `pov_errors`: Exceptions raised during POV Shuffle timing, keyed by deck size.
    - `baseline_errors`: Exceptions raised during baseline timing, keyed by deck size.
    - `cuda_device_name`: Name of the GPU used (e.g. ``"NVIDIA A100 80GB PCIe"``).
    - `cuda_compute_capability`: Compute capability of the GPU as a float (e.g. ``8.9``).
    """

    params: TimePerDeckSizeParams
    timings: pd.DataFrame
    plot: matplotlib.figure.Figure
    pov_errors: dict[int, Exception]
    baseline_errors: dict[int, Exception]
    cuda_device_name: str
    cuda_compute_capability: float


class BiasPerIterReport(NamedTuple):
    """Bias per iteration experiment report.

    **Results & Metrics**:

    - `plot`: Matplotlib figure visualising bias convergence across iterations.
    - `host_shuffle_load`: Amount of shuffle happening on the host side within each iteration (non-parallel),
                           as a percentage of the `deck_size`.

    - `ideal_worker_count`: Total tasks, i.e. how many parallel workers that would be required
                            for full parallelization, given the `povs_options` and `deck_size`.

    - `worker_data_scan_per_iter`: Fraction of the dataset scanned by each worker per iteration.
    - `num_valid_offsets`: Number of valid offsets that may have been used in the shuffle iterations,
                           depending on the offset parameters in `povs_options`.

    - `baseline_tvd`: Observed positional TVD for a true uniform shuffle on the same dataset.
    - `baseline_ngram_tvds`: Observed N-gram TVD of the baseline shuffle, one value per (degree, skip) pair.
    - `baseline_lstm_predictability`: LSTM predictability of the baseline shuffle (if LSTM enabled).

    - `biases`: DataFrame with one row per iteration. Columns:
      - `iteration`: Iteration number (1-indexed).
      - `cumulative_exposure`: Fraction of the dataset scanned by each worker up to that iteration.
      - `positional`: Positional TVD at that iteration.
      - ``"{n}-gram"`` / ``"{n}-gram (skip {s})"``: N-gram TVD at that iteration.
      - `lstm_predictability`: LSTM predictability at that iteration (only present if LSTM is enabled).

    - `sample_deficits`: How many more samples would be needed to observe all valid events at least once,
      per metric. Keys: ``"positional"``, ``"{n}-gram"`` / ``"{n}-gram (skip {s})"``.
      Zero when exactly covered; negative when oversampled.
    """

    params: BiasPerIterParams
    worker_data_scan_per_iter: float
    num_valid_offsets: int
    ideal_worker_count: int
    host_shuffle_load: float
    baseline_tvd: float
    baseline_ngram_tvds: list[float]
    biases: pd.DataFrame
    plot: matplotlib.figure.Figure
    sample_deficits: dict[str, int]
    baseline_lstm_predictability: float | None = None


class BreakingPointPerDeckSizeReport(NamedTuple):
    """Breaking point per deck size experiment report.

    **Results & Metrics**:

    - `plot`: Matplotlib figure showing the breaking point (iterations to convergence) vs deck size,
              one line per bias metric. Missing data points indicate non-convergence within the iteration limit.

    - `breaking_points`: DataFrame with one row per deck size. Columns:
      - `deck_size`: Number of elements in the deck.
      - `positional`: Iteration at which positional bias converged; ``NaN`` if not converged.
      - ``"{n}-gram"`` / ``"{n}-gram (skip {s})"``: Iteration at which n-gram bias converged; ``NaN`` if not.
      - `lstm_predictability`: Iteration at which LSTM predictability converged; ``NaN`` if not (only present
        when ``lstm_settings`` is configured).
      - `overall`: Latest convergence iteration across all metrics (only set when all metrics converged).

    - `non_convergences`: Metrics that did not converge within the iteration limit, keyed by deck size.
      Each value is a list of metric names (e.g. ``["positional", "3-gram (skip 2)"]``).

    - `sample_deficits`: Sample deficit per deck size per metric. Outer key: deck size. Inner key:
      metric name (``"positional"``, ``"{n}-gram"``). Value: ``num_valid - num_samples * deck_size``
      (positive = undersampled; negative = oversampled).
    """

    params: BreakingPointParams
    breaking_points: pd.DataFrame
    non_convergences: dict[int, list[str]]
    plot: matplotlib.figure.Figure
    sample_deficits: dict[int, dict[str, int]]
