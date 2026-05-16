from pathlib import Path

import numpy as np

from povs.eval.exercises import tvd_per_iteration
from povs.eval.plots import plot_tvd_per_iteration
from povs.eval.types import PovsOptions

# Parameters
results_path = Path("./data")
seed = 42
num_samples = 1000
deck_size = 1024
max_iterations = 6
povs_options = PovsOptions(
    base_block_size=16,
    max_block_clumping=5,
    base_offset=8,
    max_offset_factor=32,
)

# Run experiment
tvds, baseline = tvd_per_iteration(deck_size, num_samples, max_iterations, povs_options, np.random.RandomState(seed))

# Save results
results_path.mkdir(parents=True, exist_ok=True)
plot_tvd_per_iteration(tvds, baseline).savefig(results_path / "tvd_per_iteration.png")
