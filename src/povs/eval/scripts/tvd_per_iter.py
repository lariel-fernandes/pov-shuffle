from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from povs import Options
from povs.common import get_block_counts

from ..exercises import tvd_per_iteration
from ..io import save_tvd_per_iter_report
from ..params import TVDPerIterParams
from ..plots import plot_tvd_per_iteration
from ..reports import TVDPerIterReport
from ..utils import ngram_metric_name

# Parameters
params = TVDPerIterParams(
    seed=42,
    num_samples=3000,
    deck_size=1024,
    max_iterations=6,
    ngram_degrees=[2, 3],
    ngram_skips=[0, 0],
    povs_options=Options(
        virtual_block_size=2,
        physical_block_size=32,
    ),
    dtype=torch.int32.__str__().split(".")[-1],
    device="cuda",
)

# Run experiment
start_time = datetime.now()
result = tvd_per_iteration(
    deck_size=params.deck_size,
    num_samples=params.num_samples,
    max_iterations=params.max_iterations,
    options=params.povs_options,
    rng=np.random.default_rng(params.seed),
    ngram_degrees=params.ngram_degrees,
    ngram_skips=params.ngram_skips,
    dtype=getattr(torch, params.dtype),
    device=params.device,
    lstm_settings=params.lstm_settings,
)

ngram_names = [ngram_metric_name(n, skip) for n, skip in zip(params.ngram_degrees, params.ngram_skips)]

# Put together the report
report = TVDPerIterReport(
    params=params._replace(povs_options=result.options),
    worker_data_scan_per_iter=(
        worker_data_scan_per_iter := (result.options.physical_block_size * result.options.virtual_block_size)
        / params.num_samples
    ),
    baseline_tvd=result.baseline_tvd,
    baseline_ngram_tvds=result.baseline_ngram_tvds.tolist(),
    tvds=pd.DataFrame({
        "iteration": range(1, params.max_iterations + 1),
        "tvd": result.tvds,
        "cumulative_exposure": [i * worker_data_scan_per_iter for i in range(1, params.max_iterations + 1)],
    }),
    ngram_tvds=pd.DataFrame(
        result.ngram_tvds,
        columns=ngram_names,
    ).assign(iteration=range(1, params.max_iterations + 1)),
    plot=plot_tvd_per_iteration(
        tvds=result.tvds,
        baseline=result.baseline_tvd,
        worker_data_scan_per_iter=worker_data_scan_per_iter,
        ngram_tvds=result.ngram_tvds,
        ngram_names=ngram_names,
        baseline_ngram_tvds=result.baseline_ngram_tvds,
    ),
    num_valid_offsets=len(result.options.offsets),
    ideal_worker_count=(num_vblocks := get_block_counts(deck_size=params.deck_size, **result.options._asdict())[1]),
    host_shuffle_load=(num_vblocks * result.options.virtual_block_size) / params.deck_size,
    sample_deficits=result.sample_deficits,
    lstm_predictabilities=(
        pd.DataFrame(result.lstm_predictabilities, columns=ngram_names).assign(
            iteration=range(1, params.max_iterations + 1)
        )
        if result.lstm_predictabilities is not None
        else None
    ),
    baseline_lstm_predictabilities=(
        result.baseline_lstm_predictabilities.tolist() if result.baseline_lstm_predictabilities is not None else None
    ),
)

# Save the report
save_tvd_per_iter_report(
    report=report,
    path=Path("./data") / Path(__file__).stem / start_time.strftime("%Y-%m-%dT%H:%M:%S"),
)
