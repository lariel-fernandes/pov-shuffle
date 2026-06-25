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
from ..metrics import get_cumulative_worker_exposure, get_worker_data_scan_per_iter
from ..params import BiasPerIterParams
from ..plots import plot_bias_per_iteration
from ..reports import BiasPerIterReport
from ..types import NgramSpec

# Parameters
params = BiasPerIterParams(
    seed=42,
    num_episodes=500,
    deck_size=1024,
    max_iterations=6,
    ngram_specs=[NgramSpec.parse(x) for x in [2, 3]],
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
    num_samples=params.num_episodes,
    max_iterations=params.max_iterations,
    options=params.povs_options,
    rng=np.random.default_rng(params.seed),
    ngram_specs=params.ngram_specs,
    dtype=getattr(torch, params.dtype),
    device=params.device,
    lstm_settings=params.lstm_settings,
)

# Put together the report
report = BiasPerIterReport(
    params=params._replace(povs_options=result.options),
    worker_data_scan_per_iter=(
        worker_data_scan_per_iter := get_worker_data_scan_per_iter(
            result.options.virtual_block_size,
            result.options.physical_block_size,
            params.deck_size,
        )
    ),
    baseline_tvd=result.baseline_pos_tvd,
    baseline_ngram_tvds=result.baseline_ngram_tvds.tolist(),
    biases=pd.DataFrame({
        "iteration": range(1, params.max_iterations + 1),
        "cumulative_exposure": get_cumulative_worker_exposure(worker_data_scan_per_iter, params.max_iterations),
        "positional": result.pos_tvds,
        **{spec.title: result.ngram_tvds[:, i] for i, spec in enumerate(params.ngram_specs)},
        **({"lstm_predictability": result.lstm_predictabilities} if result.lstm_predictabilities is not None else {}),
    }),
    plot=plot_bias_per_iteration(
        tvds=result.pos_tvds,
        baseline=result.baseline_pos_tvd,
        worker_data_scan_per_iter=worker_data_scan_per_iter,
        ngram_tvds=result.ngram_tvds,
        ngram_names=[x.title for x in params.ngram_specs],
        baseline_ngram_tvds=result.baseline_ngram_tvds,
        lstm_predictabilities=result.lstm_predictabilities,
        baseline_lstm_predictability=result.baseline_lstm_predictability,
    ),
    num_valid_offsets=len(result.options.offsets),
    ideal_worker_count=(num_vblocks := get_block_counts(deck_size=params.deck_size, **result.options._asdict())[1]),
    host_shuffle_load=(num_vblocks * result.options.virtual_block_size) / params.deck_size,
    sample_deficits=result.sample_deficits,
    baseline_lstm_predictability=result.baseline_lstm_predictability,
)

# Save the report
save_bias_per_iter_report(
    report=report,
    path=Path("./data") / Path(__file__).stem / start_time.strftime("%Y-%m-%dT%H:%M:%S"),
)
