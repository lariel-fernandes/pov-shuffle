from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from povs import POVSOptions
from povs.utils import get_block_counts, get_valid_offsets

from ..exercises import tvd_per_iteration
from ..io import save_tvd_per_iter_report
from ..plots import plot_tvd_per_iteration
from ..reports import TVDPerIterReport

# Parameters
results_path = Path("./data")
seed = 42
num_samples = 3000
deck_size = 1024
max_iterations = 6
ngram_degrees = [2, 3]
povs_options = POVSOptions(
    physical_block_size=32,
    virtual_block_size=3,
    offset_step_size=4,
    max_offset_steps=16,
)

# Infer details
start_time = datetime.now()
_, num_vblocks = get_block_counts(deck_size=deck_size, **povs_options._asdict())
worker_data_scan_per_iter = (povs_options.physical_block_size * povs_options.virtual_block_size) / num_samples

# Run experiment
result = tvd_per_iteration(
    deck_size=deck_size,
    num_samples=num_samples,
    max_iterations=max_iterations,
    options=povs_options,
    rng=np.random.default_rng(seed),
    ngram_degrees=ngram_degrees,
)

# Put together the report
report = TVDPerIterReport(
    seed=seed,
    deck_size=deck_size,
    num_samples=num_samples,
    max_iterations=max_iterations,
    worker_data_scan_per_iter=worker_data_scan_per_iter,
    povs_options=povs_options,
    ngram_degrees=ngram_degrees,
    baseline_tvd=result.baseline_tvd,
    baseline_ngram_tvds=result.baseline_ngram_tvds.tolist(),
    tvds=pd.DataFrame({
        "iteration": range(1, max_iterations + 1),
        "tvd": result.tvds,
        "cumulative_exposure": [i * worker_data_scan_per_iter for i in range(1, max_iterations + 1)],
    }),
    ngram_tvds=pd.DataFrame(
        result.ngram_tvds,
        columns=[f"{n}-gram" for n in ngram_degrees],
    ).assign(iteration=range(1, max_iterations + 1)),
    plot=plot_tvd_per_iteration(
        tvds=result.tvds,
        baseline=result.baseline_tvd,
        worker_data_scan_per_iter=worker_data_scan_per_iter,
        ngram_tvds=result.ngram_tvds,
        ngram_degrees=ngram_degrees,
        baseline_ngram_tvds=result.baseline_ngram_tvds,
    ),
    num_valid_offsets=len(get_valid_offsets(**povs_options._asdict())),
    ideal_worker_count=num_vblocks,
    host_shuffle_load=(num_vblocks * povs_options.virtual_block_size) / deck_size,
)

# Save the report
save_tvd_per_iter_report(
    report=report,
    path=results_path / Path(__file__).stem / start_time.strftime("%Y-%m-%dT%H:%M:%S"),
)
