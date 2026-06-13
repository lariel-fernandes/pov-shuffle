POV Shuffle time per deck size experiment report.

    **Results & Metrics**:

    - `plot`: Matplotlib figure showing shuffle time vs deck size for POV Shuffle and Fisher-Yates CUDA baseline,
              with a secondary y-axis showing the speedup ratio.

    - `timings`: DataFrame with one row per deck size. Columns:
      - `deck_size`: Number of elements in the deck.
      - `pov_mean_ms`: Mean POV Shuffle time in milliseconds across timing runs.
      - `pov_std_ms`: Standard deviation of POV Shuffle time.
      - `baseline_mean_ms`: Mean Fisher-Yates (CUDA randperm + copy) time in milliseconds.
      - `baseline_std_ms`: Standard deviation of baseline time.
      - `speedup`: Ratio `baseline_mean_ms / pov_mean_ms`. Values > 1 mean POV Shuffle is faster.

    - `pov_errors`: Exceptions raised during POV Shuffle timing, keyed by deck size.
    - `baseline_errors`: Exceptions raised during baseline timing, keyed by deck size.
    - `cuda_device_name`: Name of the GPU used (e.g. ``"NVIDIA A100 80GB PCIe"``).
    - `cuda_compute_capability`: Compute capability of the GPU as a float (e.g. ``8.9``).
    