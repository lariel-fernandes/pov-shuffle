import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from povs import Options

from ..exercises import breaking_point_per_deck_size
from ..io import save_breaking_point_report
from ..lstm import LSTMSettings
from ..params import BreakingPointParams
from ..plots import plot_breaking_point_per_deck_size
from ..reports import BreakingPointPerDeckSizeReport
from ..utils import ngram_metric_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint")
parser.add_argument("--cleanup", action="store_true", help="Delete checkpoint dirs of all other runs after saving")
args = parser.parse_args()

exp_dir = Path("./data") / Path(__file__).stem

if args.resume:
    latest = max((d for d in exp_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
    start_time = datetime.strptime(latest.name, "%Y-%m-%dT%H:%M:%S")
else:
    start_time = datetime.now()

run_dir = exp_dir / start_time.strftime("%Y-%m-%dT%H:%M:%S")
checkpoint_dir = run_dir / "checkpoint"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Parameters
params = BreakingPointParams(
    seed=42,
    num_samples=500,
    deck_sizes=[10_000, 50_000, 100_000, 500_000, 1_000_000],
    ngram_degrees=[2, 3],
    ngram_skips=[0, 0],
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
    device="cuda",
    lstm_settings=LSTMSettings(
        layers=[32],
        context_length=16,
        num_epochs=15,
        batch_size=512,
        max_sequences=2_000_000,
    ),
)

ngram_pairs = list(zip(params.ngram_degrees, params.ngram_skips))

# Run experiment
result = breaking_point_per_deck_size(
    deck_sizes=params.deck_sizes,
    num_samples=params.num_samples,
    ngram_degrees=params.ngram_degrees,
    ngram_skips=params.ngram_skips,
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
    lstm_settings=params.lstm_settings,
    lstm_tolerance=params.lstm_tolerance,
    checkpoint_dir=checkpoint_dir,
)

# Non-convergence tracking
non_convergences: dict[int, list[str]] = {}
for deck_size in params.deck_sizes:
    nc = []
    if result.positional_breaking_points[deck_size] is None:
        nc.append("positional")
    for n, skip in ngram_pairs:
        if result.ngram_breaking_points[deck_size][(n, skip)] is None:
            nc.append(ngram_metric_name(n, skip))
    if result.lstm_breaking_points is not None and result.lstm_breaking_points[deck_size] is None:
        nc.append("lstm_predictability")
    if nc:
        non_convergences[deck_size] = nc

# Build breaking_points DataFrame
metric_cols = ["positional"] + [ngram_metric_name(n, skip) for n, skip in ngram_pairs]
if result.lstm_breaking_points is not None:
    metric_cols.append("lstm_predictability")
rows = []
for deck_size in params.deck_sizes:
    row = {"deck_size": deck_size, "positional": result.positional_breaking_points[deck_size]}
    for n, skip in ngram_pairs:
        row[ngram_metric_name(n, skip)] = result.ngram_breaking_points[deck_size][(n, skip)]
    if result.lstm_breaking_points is not None:
        row["lstm_predictability"] = result.lstm_breaking_points[deck_size]
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
            ngram_metric_name(n, skip): [result.ngram_breaking_points[s][(n, skip)] for s in params.deck_sizes]
            for n, skip in ngram_pairs
        },
        max_iterations=[
            params.max_iterations_per_deck_size.get(s, params.default_max_iterations) for s in params.deck_sizes
        ],
        lstm_breaking_points=(
            [result.lstm_breaking_points[s] for s in params.deck_sizes]
            if result.lstm_breaking_points is not None
            else None
        ),
    ),
)

# Save the report
save_breaking_point_report(report=report, path=run_dir / "report")

# Clean up checkpoint dirs of all other runs
if args.cleanup:
    for d in exp_dir.iterdir():
        if d.is_dir() and d != run_dir:
            shutil.rmtree(d / "checkpoint", ignore_errors=True)
