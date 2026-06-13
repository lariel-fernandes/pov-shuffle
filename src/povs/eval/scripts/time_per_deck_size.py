from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from povs.types import Options

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
    povs_options_per_deck_size={},
    default_options=Options(virtual_block_size=2, physical_block_size=16),
    dtype=torch.float32.__str__().split(".")[-1],
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
    default_options=params.default_options,
    dtype=getattr(torch, params.dtype),
    seed=params.seed,
)

# Compute stats (None where the run failed)
pov_means = [float(np.mean(t)) if t is not None else None for t in result.pov_times_ms]
pov_stds = [float(np.std(t)) if t is not None else None for t in result.pov_times_ms]
baseline_means = [float(np.mean(t)) if t is not None else None for t in result.baseline_times_ms]
baseline_stds = [float(np.std(t)) if t is not None else None for t in result.baseline_times_ms]
speedups = [b / p if b is not None and p is not None else None for b, p in zip(baseline_means, pov_means)]

# Put together the report
report = TimePerDeckSizeReport(
    params=params._replace(
        povs_options_per_deck_size={
            size: options for size, options in zip(params.deck_sizes, result.option_sets) if options is not None
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
