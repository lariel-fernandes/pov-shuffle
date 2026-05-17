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

    **Results & Metrics**:

    - `plot`: Matplotlib figure visualising TVD convergence across iterations.
    - `worker_data_scan_per_iter`: Fraction of the dataset scanned by each worker per iteration.
    - `baseline_tvd`: TVD measured for a true uniform shuffle on the same dataset; lower bound for a perfect shuffler.
    - `tvds`: DataFrame with one row per iteration. Columns:
      - `iteration`: Iteration number (1-indexed).
      - `tvd`: Total Variation Distance of the POV Shuffle at that iteration.
      - `cumulative_exposure`: Fraction of the dataset scanned by each worker up to that iteration.
    """

    seed: int
    deck_size: int
    num_samples: int
    max_iterations: int
    worker_data_scan_per_iter: float
    povs_options: POVSOptions
    baseline_tvd: float
    tvds: pd.DataFrame
    plot: matplotlib.figure.Figure
