from typing import NamedTuple

import matplotlib.figure
import pandas as pd

from povs import POVSOptions


class TVDPerIterReport(NamedTuple):
    """TVD per iteration experiment report.

    **Parameters**:

    - `seed`: RNG seed for reproducibility.
    - `deck_size`: Number of elements in the deck (dataset size proxy).
    - `num_samples`: Number of independent shuffles sampled to estimate the output distribution.
    - `max_iterations`: Number of shuffle iterations tested (from 1 to this value, inclusive).
    - `povs_options`: POV Shuffle algorithm options used in this run.
    - `ngram_degrees`: N-gram degrees for which TVD was measured.

    **Results & Metrics**:

    - `plot`: Matplotlib figure visualising TVD convergence across iterations.
    - `worker_data_scan_per_iter`: Fraction of the dataset scanned by each worker per iteration.

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

    seed: int
    deck_size: int
    num_samples: int
    max_iterations: int
    worker_data_scan_per_iter: float
    povs_options: POVSOptions
    ngram_degrees: list[int]
    baseline_tvd: float
    baseline_ngram_tvds: list[float]
    tvds: pd.DataFrame
    ngram_tvds: pd.DataFrame
    plot: matplotlib.figure.Figure
