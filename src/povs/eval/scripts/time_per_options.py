from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from povs import FullOptions

from ..exercises import shuffle_time_per_options
from ..io import save_time_per_options_report
from ..params import OptionsSetEntry, TimePerOptionsParams
from ..plots import plot_time_per_options
from ..reports import TimePerOptionsReport

# Parameters
params = TimePerOptionsParams(
    seed=42,
    deck_size=4096,
    instance_size=128,
    iterations=1,
    num_runs=50,
    num_warmup_runs=10,
    options_sets=[
        OptionsSetEntry(
            "pbs=16 vbs=2",
            FullOptions(physical_block_size=16, virtual_block_size=2, offset_step_size=4, max_offset_steps=16),
        ),
        OptionsSetEntry(
            "pbs=16 vbs=4",
            FullOptions(physical_block_size=16, virtual_block_size=4, offset_step_size=4, max_offset_steps=16),
        ),
        OptionsSetEntry(
            "pbs=32 vbs=2",
            FullOptions(physical_block_size=32, virtual_block_size=2, offset_step_size=4, max_offset_steps=16),
        ),
        OptionsSetEntry(
            "pbs=32 vbs=4",
            FullOptions(physical_block_size=32, virtual_block_size=4, offset_step_size=4, max_offset_steps=16),
        ),
        OptionsSetEntry(
            "pbs=64 vbs=2",
            FullOptions(physical_block_size=64, virtual_block_size=2, offset_step_size=8, max_offset_steps=16),
        ),
        OptionsSetEntry(
            "pbs=64 vbs=4",
            FullOptions(physical_block_size=64, virtual_block_size=4, offset_step_size=8, max_offset_steps=16),
        ),
    ],
)

# Run experiment
start_time = datetime.now()
result = shuffle_time_per_options(
    options_sets=params.options_sets,
    deck_size=params.deck_size,
    instance_size=params.instance_size,
    iterations=params.iterations,
    num_runs=params.num_runs,
    num_warmup_runs=params.num_warmup_runs,
    seed=params.seed,
)

# Compute stats
means = [np.mean(t) for t in result.times_ms]
stds = [np.std(t) for t in result.times_ms]
mins = [np.min(t) for t in result.times_ms]
maxs = [np.max(t) for t in result.times_ms]

# Put together the report
report = TimePerOptionsReport(
    params=params,
    timings=pd.DataFrame({
        "label": result.labels,
        "mean_ms": means,
        "std_ms": stds,
        "min_ms": mins,
        "max_ms": maxs,
    }),
    plot=plot_time_per_options(
        labels=result.labels,
        means_ms=means,
        stds_ms=stds,
    ),
)

# Save the report
save_time_per_options_report(
    report=report,
    path=Path("./data") / Path(__file__).stem / start_time.strftime("%Y-%m-%dT%H:%M:%S"),
)
