Bias per iteration experiment report.

    **Results & Metrics**:

    - `plot`: Matplotlib figure visualising bias convergence across iterations.
    - `host_shuffle_load`: Amount of shuffle happening on the host side within each iteration (non-parallel),
                           as a percentage of the `deck_size`.

    - `ideal_worker_count`: Total tasks, i.e. how many parallel workers that would be required
                            for full parallelization, given the `povs_options` and `deck_size`.

    - `worker_data_scan_per_iter`: Fraction of the dataset scanned by each worker per iteration.
    - `num_valid_offsets`: Number of valid offsets that may have been used in the shuffle iterations,
                           depending on the offset parameters in `povs_options`.

    - `baseline_tvd`: Observed positional TVD for a true uniform shuffle on the same dataset.
    - `baseline_ngram_tvds`: Observed N-gram TVD of the baseline shuffle, one value per (degree, skip) pair.
    - `baseline_lstm_predictability`: LSTM predictability of the baseline shuffle (if LSTM enabled).

    - `biases`: DataFrame with one row per iteration. Columns:
      - `iteration`: Iteration number (1-indexed).
      - `cumulative_exposure`: Fraction of the dataset scanned by each worker up to that iteration.
      - `positional`: Positional TVD at that iteration.
      - ``"{n}-gram"`` / ``"{n}-gram (skip {s})"``: N-gram TVD at that iteration.
      - `lstm_predictability`: LSTM predictability at that iteration (only present if LSTM is enabled).

    - `sample_deficits`: How many more samples would be needed to observe all valid events at least once,
      per metric. Keys: ``"positional"``, ``"{n}-gram"`` / ``"{n}-gram (skip {s})"``.
      Zero when exactly covered; negative when oversampled.
    