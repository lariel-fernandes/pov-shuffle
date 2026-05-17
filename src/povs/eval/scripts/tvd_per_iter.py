from datetime import datetime
from pathlib import Path

import numpy as np

from povs import POVSOptions
from povs.eval.exercises import tvd_per_iteration
from povs.eval.plots import plot_tvd_per_iteration

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

# Run experiment
tvds, baseline = tvd_per_iteration(deck_size, num_samples, max_iterations, povs_options, np.random.default_rng(seed))

# Generate plots
plot = plot_tvd_per_iteration(
    tvds,
    baseline=baseline,
    worker_data_scan_per_iter=(povs_options.physical_block_size * povs_options.virtual_block_size) / num_samples,
)

# Save results
experiment_dir = results_path / Path(__file__).stem / datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
experiment_dir.mkdir(parents=True, exist_ok=True)
plot.savefig(experiment_dir / "tvd_per_iteration.png")
