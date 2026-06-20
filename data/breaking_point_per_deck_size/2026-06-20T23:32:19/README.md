Breaking point per deck size experiment report.

    **Results & Metrics**:

    - `plot`: Matplotlib figure showing the breaking point (iterations to convergence) vs deck size,
              one line per bias metric. Missing data points indicate non-convergence within the iteration limit.

    - `breaking_points`: DataFrame with one row per deck size. Columns:
      - `deck_size`: Number of elements in the deck.
      - `positional`: Iteration at which positional bias converged; ``NaN`` if not converged.
      - `{n}-gram`: Iteration at which n-gram bias of degree ``n`` converged; ``NaN`` if not converged.
      - `overall`: Latest convergence iteration across all metrics (only set when all metrics converged).

    - `non_convergences`: Metrics that did not converge within the iteration limit, keyed by deck size.
      Each value is a list of metric names (e.g. ``["positional", "3-gram"]``).

    - `sample_deficits`: Sample deficit per deck size per metric. Outer key: deck size. Inner key:
      metric name (``"positional"``, ``"{n}-gram"``). Value: ``num_valid - num_samples * deck_size``
      (positive = undersampled; negative = oversampled).
    