from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from povs import Options
from povs.common import get_block_counts

from ..exercises import bias_per_iteration
from ..io import save_bias_per_iter_report
from ..lstm import LSTMSettings
from ..params import BiasPerIterParams
from ..plots import plot_bias_per_iteration
from ..reports import BiasPerIterReport
from ..utils import ngram_metric_name

# Parameters
params = BiasPerIterParams(
    seed=42,
    num_samples=500,
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
    lstm_settings=LSTMSettings(layers=[32], context_length=16, num_epochs=15, batch_size=512),
)

# Run experiment
start_time = datetime.now()
result = bias_per_iteration(
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
worker_data_scan_per_iter = (
    result.options.physical_block_size * result.options.virtual_block_size
) / params.num_samples

# Put together the report
report = BiasPerIterReport(
    params=params._replace(povs_options=result.options),
    worker_data_scan_per_iter=worker_data_scan_per_iter,
    baseline_tvd=result.baseline_tvd,
    baseline_ngram_tvds=result.baseline_ngram_tvds.tolist(),
    biases=pd.DataFrame({
        "iteration": range(1, params.max_iterations + 1),
        "cumulative_exposure": [i * worker_data_scan_per_iter for i in range(1, params.max_iterations + 1)],
        "positional": result.tvds,
        **{name: result.ngram_tvds[:, i] for i, name in enumerate(ngram_names)},
        **({"lstm_predictability": result.lstm_predictabilities} if result.lstm_predictabilities is not None else {}),
    }),
    plot=plot_bias_per_iteration(
        tvds=result.tvds,
        baseline=result.baseline_tvd,
        worker_data_scan_per_iter=worker_data_scan_per_iter,
        ngram_tvds=result.ngram_tvds,
        ngram_names=ngram_names,
        baseline_ngram_tvds=result.baseline_ngram_tvds,
        lstm_predictabilities=result.lstm_predictabilities,
        baseline_lstm_predictability=result.baseline_lstm_predictabilities,
    ),
    num_valid_offsets=len(result.options.offsets),
    ideal_worker_count=(num_vblocks := get_block_counts(deck_size=params.deck_size, **result.options._asdict())[1]),
    host_shuffle_load=(num_vblocks * result.options.virtual_block_size) / params.deck_size,
    sample_deficits=result.sample_deficits,
    baseline_lstm_predictability=result.baseline_lstm_predictabilities,
)

# Save the report
save_bias_per_iter_report(
    report=report,
    path=Path("./data") / Path(__file__).stem / start_time.strftime("%Y-%m-%dT%H:%M:%S"),
)
