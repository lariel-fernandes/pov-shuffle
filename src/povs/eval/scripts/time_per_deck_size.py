from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from povs import Options

from ..exercises import shuffle_time_per_deck_size
from ..io import save_time_per_deck_size_report
from ..params import TimePerDeckSizeParams
from ..plots import plot_time_per_deck_size
from ..reports import TimePerDeckSizeReport

# Parameters
params = TimePerDeckSizeParams(
    seed=42,
    deck_sizes=[256, 512, 1024, 2048, 4096, 8192, 16384],
    iterations=1,
    instance_size=128,
    num_runs=50,
    num_warmup_runs=10,
    povs_options_per_deck_size={
        256: Options(physical_block_size=16, virtual_block_size=3),
        512: Options(physical_block_size=32, virtual_block_size=3),
        1024: Options(physical_block_size=32, virtual_block_size=3),
        2048: Options(physical_block_size=32, virtual_block_size=4),
        4096: Options(physical_block_size=64, virtual_block_size=4),
        8192: Options(physical_block_size=64, virtual_block_size=4),
        16384: Options(physical_block_size=128, virtual_block_size=4),
    },
    dtype=torch.float32.__str__(),
)

# Run experiment
start_time = datetime.now()
result = shuffle_time_per_deck_size(
    deck_sizes=params.deck_sizes,
    iterations=params.iterations,
    instance_size=params.instance_size,
    num_runs=params.num_runs,
    num_warmup_runs=params.num_warmup_runs,
    povs_options_per_deck_size=params.povs_options_per_deck_size,
    dtype=getattr(torch, params.dtype),
    seed=params.seed,
)

# Compute stats
pov_means = [float(np.mean(t)) for t in result.pov_times_ms]
pov_stds = [float(np.std(t)) for t in result.pov_times_ms]
baseline_means = [float(np.mean(t)) for t in result.baseline_times_ms]
baseline_stds = [float(np.std(t)) for t in result.baseline_times_ms]
speedups = [b / p for b, p in zip(baseline_means, pov_means)]

# Put together the report
report = TimePerDeckSizeReport(
    params=params._replace(
        povs_options_per_deck_size={
            size: options
            for size, options in zip(
                params.povs_options_per_deck_size.keys(),
                result.option_sets,
            )
        }
    ),
    timings=pd.DataFrame({
        "deck_size": params.deck_sizes,
        "pov_mean_ms": pov_means,
        "pov_std_ms": pov_stds,
        "baseline_mean_ms": baseline_means,
        "baseline_std_ms": baseline_stds,
        "speedup": speedups,
    }),
    plot=plot_time_per_deck_size(
        deck_sizes=params.deck_sizes,
        pov_means_ms=pov_means,
        pov_stds_ms=pov_stds,
        baseline_means_ms=baseline_means,
        baseline_stds_ms=baseline_stds,
    ),
)

# Save the report
save_time_per_deck_size_report(
    report=report,
    path=Path("./data") / Path(__file__).stem / start_time.strftime("%Y-%m-%dT%H:%M:%S"),
)
