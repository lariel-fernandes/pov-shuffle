TVD per iteration experiment report.

    **Results & Metrics**:

    - `plot`: Matplotlib figure visualising TVD convergence across iterations.
    - `host_shuffle_load`: Amount of shuffle happening on the host side within each iteration (non-parallel),
                           as a percentage of the `deck_size`.

    - `ideal_worker_count`: Total tasks, i.e. how many parallel workers that would be required
                            for full parallelization, given the `povs_options` and `deck_size`.

    - `worker_data_scan_per_iter`: Fraction of the dataset scanned by each worker per iteration.
    - `num_valid_offsets`: Number of valid offsets that may have been used in the shuffle iterations,
                           depending on the offset parameters in `povs_options`.

    - `baseline_tvd`: Observed TVD for a true uniform shuffle on the same dataset; lower bound for a perfect shuffler.
                      In theory this should be zero, but if the sample size is too small and the deck size too large,
                      the statistic may not converge to zero.

    - `baseline_ngram_tvds`: Observed N-gram TVD of the baseline shuffle, one value per degree in `ngram_degrees`.
                             Same considerations as `baseline_tvd` apply, with the degree of the ngram distribution
                             increasing the amount of samples required for convergence.

    - `tvds`: DataFrame with one row per iteration. Columns:
      - `iteration`: Iteration number (1-indexed).
      - `tvd`: Total Variation Distance of the POV Shuffle at that iteration.
      - `cumulative_exposure`: Fraction of the dataset scanned by each worker up to that iteration.

    - `ngram_tvds`: DataFrame with one row per iteration and one column per degree in `ngram_degrees`
    