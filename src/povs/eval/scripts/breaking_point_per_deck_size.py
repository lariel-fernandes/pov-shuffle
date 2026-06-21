import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from povs import Options

from ..exercises import breaking_point_per_deck_size
from ..io import save_breaking_point_report
from ..params import BreakingPointParams
from ..plots import plot_breaking_point_per_deck_size
from ..reports import BreakingPointPerDeckSizeReport

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
args = parser.parse_args()

# Parameters
params = BreakingPointParams(
    seed=42,
    num_samples=500,
    deck_sizes=[10_000, 50_000, 100_000, 500_000, 1_000_000],
    ngram_degrees=[2, 3],
    positional_tolerance=0.01,
    ngram_tolerances={},
    default_ngram_tolerance=0.01,
    max_iterations_per_deck_size={},
    default_max_iterations=20,
    povs_options_per_deck_size={},
    default_options=Options(
        virtual_block_size=2,
        physical_block_size=32,
    ),
    dtype=torch.int32.__str__().split(".")[-1],
    device=args.device,
)

# Run experiment
start_time = datetime.now()
result = breaking_point_per_deck_size(
    deck_sizes=params.deck_sizes,
    num_samples=params.num_samples,
    ngram_degrees=params.ngram_degrees,
    positional_tolerance=params.positional_tolerance,
    ngram_tolerances=params.ngram_tolerances,
    default_ngram_tolerance=params.default_ngram_tolerance,
    max_iterations_per_deck_size=params.max_iterations_per_deck_size,
    default_max_iterations=params.default_max_iterations,
    povs_options_per_deck_size=params.povs_options_per_deck_size,
    default_options=params.default_options,
    rng=np.random.default_rng(params.seed),
    dtype=getattr(torch, params.dtype),
    device=params.device,
)

# Non-convergence tracking
non_convergences: dict[int, list[str]] = {}
for deck_size in params.deck_sizes:
    nc = []
    if result.positional_breaking_points[deck_size] is None:
        nc.append("positional")
    for n in params.ngram_degrees:
        if result.ngram_breaking_points[deck_size][n] is None:
            nc.append(f"{n}-gram")
    if nc:
        non_convergences[deck_size] = nc

# Build breaking_points DataFrame
metric_cols = ["positional"] + [f"{n}-gram" for n in params.ngram_degrees]
rows = []
for deck_size in params.deck_sizes:
    row = {"deck_size": deck_size, "positional": result.positional_breaking_points[deck_size]}
    for n in params.ngram_degrees:
        row[f"{n}-gram"] = result.ngram_breaking_points[deck_size][n]
    converged = [row[k] for k in metric_cols if row[k] is not None]
    row["overall"] = max(converged) if len(converged) == len(metric_cols) else None
    rows.append(row)
breaking_points = pd.DataFrame(rows)

# Put together the report
report = BreakingPointPerDeckSizeReport(
    params=params._replace(povs_options_per_deck_size=result.options),
    breaking_points=breaking_points,
    non_convergences=non_convergences,
    sample_deficits=result.sample_deficits,
    plot=plot_breaking_point_per_deck_size(
        deck_sizes=params.deck_sizes,
        positional_breaking_points=[result.positional_breaking_points[s] for s in params.deck_sizes],
        ngram_breaking_points={
            n: [result.ngram_breaking_points[s][n] for s in params.deck_sizes] for n in params.ngram_degrees
        },
        max_iterations=[
            params.max_iterations_per_deck_size.get(s, params.default_max_iterations) for s in params.deck_sizes
        ],
    ),
)

# Save the report
save_breaking_point_report(
    report=report,
    path=Path("./data") / Path(__file__).stem / start_time.strftime("%Y-%m-%dT%H:%M:%S"),
)
