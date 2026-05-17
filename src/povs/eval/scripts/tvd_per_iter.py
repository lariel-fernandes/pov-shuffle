from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from povs import POVSOptions

from ..exercises import tvd_per_iteration
from ..io import save_tvd_per_iter_report
from ..plots import plot_tvd_per_iteration
from ..reports import TVDPerIterReport

# Parameters
results_path = Path("./data")
seed = 42
num_samples = 1000
deck_size = 512
max_iterations = 6
povs_options = POVSOptions(
    physical_block_size=8,
    virtual_block_size=3,
    offset_step_size=4,
    max_offset_steps=16,
)

# Infer details
start_time = datetime.now()
worker_data_scan_per_iter = (povs_options.physical_block_size * povs_options.virtual_block_size) / num_samples

# Run experiment
tvds, baseline = tvd_per_iteration(
    deck_size=deck_size,
    num_samples=num_samples,
    max_iterations=max_iterations,
    options=povs_options,
    rng=np.random.default_rng(seed),
)

# Put together the report
report = TVDPerIterReport(
    seed=seed,
    deck_size=deck_size,
    num_samples=num_samples,
    max_iterations=max_iterations,
    worker_data_scan_per_iter=worker_data_scan_per_iter,
    povs_options=povs_options,
    baseline_tvd=baseline,
    tvds=pd.DataFrame({
        "iteration": range(1, max_iterations + 1),
        "tvd": tvds,
        "cumulative_exposure": [i * worker_data_scan_per_iter for i in range(1, max_iterations + 1)],
    }),
    plot=plot_tvd_per_iteration(
        tvds=tvds,
        baseline=baseline,
        worker_data_scan_per_iter=worker_data_scan_per_iter,
    ),
)

# Save the report
save_tvd_per_iter_report(
    report=report,
    path=results_path / Path(__file__).stem / start_time.strftime("%Y-%m-%dT%H:%M:%S"),
)
