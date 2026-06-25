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
from ..types import NgramSpec

# Parameters
params = BreakingPointParams(
    seed=42,
    num_episodes=500,
    deck_sizes=[10_000, 50_000, 100_000, 500_000, 1_000_000],
    ngram_specs=[NgramSpec.parse(x) for x in [2, 3]],
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

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint")
parser.add_argument("--cleanup", action="store_true", help="Delete checkpoints of all other runs after saving")
args = parser.parse_args()

# Resolve experiment directories
exp_dir = Path("./data") / Path(__file__).stem
if args.resume:
    latest = max((d for d in exp_dir.iterdir() if d.is_dir()), key=lambda d: d.name)
    start_time = datetime.strptime(latest.name, "%Y-%m-%dT%H:%M:%S")
else:
    start_time = datetime.now()
run_dir = exp_dir / start_time.strftime("%Y-%m-%dT%H:%M:%S")
(checkpoint_dir := run_dir / "checkpoint").mkdir(parents=True, exist_ok=True)

# Run experiment
result = breaking_point_per_deck_size(
    deck_sizes=params.deck_sizes,
    num_episodes=params.num_episodes,
    ngram_specs=params.ngram_specs,
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

# Build breaking_points DataFrame
rows = []
for deck_size in params.deck_sizes:
    deck_result = result.deck_sizes[deck_size]
    breaking_points = {
        "positional": deck_result.positional_breaking_point,
        **{spec.title: deck_result.ngram_breaking_points[spec] for spec in params.ngram_specs},
        **({"lstm_predictability": deck_result.lstm_breaking_point} if params.lstm_settings is not None else {}),
    }
    converged = [bp or 0 for bp in breaking_points.values()]
    rows.append({"deck_size": deck_size} | breaking_points | {"overall": max(converged) if all(converged) else None})
df_breaking_points = pd.DataFrame(rows)

# Put together the report
report = BreakingPointPerDeckSizeReport(
    params=params._replace(povs_options_per_deck_size={d: result.deck_sizes[d].options for d in params.deck_sizes}),
    breaking_points=df_breaking_points,
    sample_deficits={d: result.deck_sizes[d].sample_deficit for d in params.deck_sizes},
    non_convergences={d: result.deck_sizes[d].non_convergences or [] for d in params.deck_sizes},
    plot=plot_breaking_point_per_deck_size(
        deck_sizes=params.deck_sizes,
        positional_breaking_points=[result.deck_sizes[d].positional_breaking_point for d in params.deck_sizes],
        ngram_breaking_points={
            spec.title: [result.deck_sizes[d].ngram_breaking_points[spec] for d in params.deck_sizes]
            for spec in params.ngram_specs
        },
        max_iterations=[
            params.max_iterations_per_deck_size.get(s, params.default_max_iterations) for s in params.deck_sizes
        ],
        lstm_breaking_points=(
            [result.deck_sizes[d].lstm_breaking_point for d in params.deck_sizes]
            if params.lstm_settings is not None
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
