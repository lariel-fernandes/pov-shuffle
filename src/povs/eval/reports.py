from typing import NamedTuple

import matplotlib.figure
import pandas as pd

from .params import TimePerDeckSizeParams, TVDPerIterParams


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


class TVDPerIterReport(NamedTuple):
    """TVD per iteration experiment report.

    **Results & Metrics**:

    - `plot`: Matplotlib figure visualising TVD convergence across iterations.
    - `host_shuffle_load`: Amount of shuffle happening on the host side within each iteration (non-parallel),
                           as a percentage of the `deck_size`.

    - `ideal_worker_count`: Total tasks, i.e. how many parallel workers that would be required
                            for full parallelization, given the `povs_options` and `deck_size`.

    - `worker_data_scan_per_iter`: Fraction of the dataset scanned by each worker per iteration.
    - `num_valid_offsets`: Number of valid offsets that may have been used in the shuffle iterations,
                           depending on the offset parameters in `povs_options`.

    - `baseline_tvd`: Observed TVD for a true uniform shuffle on the same dataset; lower bound for a perfect shuffler.
                      In theory this should be zero, but if the sample size is too small and the deck size too large,
                      the statistic may not converge to zero.

    - `baseline_ngram_tvds`: Observed N-gram TVD of the baseline shuffle, one value per degree in `ngram_degrees`.
                             Same considerations as `baseline_tvd` apply, with the degree of the ngram distribution
                             increasing the amount of samples required for convergence.

    - `tvds`: DataFrame with one row per iteration. Columns:
      - `iteration`: Iteration number (1-indexed).
      - `tvd`: Total Variation Distance of the POV Shuffle at that iteration.
      - `cumulative_exposure`: Fraction of the dataset scanned by each worker up to that iteration.

    - `ngram_tvds`: DataFrame with one row per iteration and one column per degree in `ngram_degrees`
    """

    params: TVDPerIterParams
    worker_data_scan_per_iter: float
    num_valid_offsets: int
    ideal_worker_count: int
    host_shuffle_load: float
    baseline_tvd: float
    baseline_ngram_tvds: list[float]
    tvds: pd.DataFrame
    ngram_tvds: pd.DataFrame
    plot: matplotlib.figure.Figure
