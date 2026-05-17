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

# Save results
results_path.mkdir(parents=True, exist_ok=True)
plot_tvd_per_iteration(tvds, baseline).savefig(results_path / "tvd_per_iteration.png")
