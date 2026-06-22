import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def plot_time_per_deck_size(
    deck_sizes: list[int],
    pov_means_ms: list[float | None],
    pov_stds_ms: list[float | None],
    baseline_means_ms: list[float | None],
    baseline_stds_ms: list[float | None],
) -> matplotlib.figure.Figure:
    """Plot shuffle time vs deck size for POV Shuffle and Fisher-Yates CUDA baseline.

    :param deck_sizes: Deck sizes along the x-axis.
    :param pov_means_ms: Mean POV Shuffle time per deck size in ms; None entries are omitted.
    :param pov_stds_ms: Standard deviation of POV Shuffle time per deck size.
    :param baseline_means_ms: Mean Fisher-Yates (CUDA) time per deck size in ms; None entries are omitted.
    :param baseline_stds_ms: Standard deviation of baseline time per deck size.
    :returns: Matplotlib figure.
    """
    pov_valid = [(s, m, st) for s, m, st in zip(deck_sizes, pov_means_ms, pov_stds_ms) if m is not None]
    baseline_valid = [(s, m, st) for s, m, st in zip(deck_sizes, baseline_means_ms, baseline_stds_ms) if m is not None]

    pov_sizes = [s for s, _, _ in pov_valid]
    pov_means = np.array([m for _, m, _ in pov_valid])
    pov_stds = np.array([st for _, _, st in pov_valid])

    baseline_sizes = [s for s, _, _ in baseline_valid]
    baseline_means = np.array([m for _, m, _ in baseline_valid])
    baseline_stds = np.array([st for _, _, st in baseline_valid])

    pov_mean_by_size = {s: m for s, m, _ in pov_valid}
    baseline_mean_by_size = {s: m for s, m, _ in baseline_valid}
    speedup_sizes = sorted(set(pov_mean_by_size) & set(baseline_mean_by_size))
    speedups = [baseline_mean_by_size[s] / pov_mean_by_size[s] for s in speedup_sizes]

    fig, ax = plt.subplots()

    if len(pov_sizes) > 0:
        ax.plot(pov_sizes, pov_means, marker="o", color="C0", label="POV")
        ax.fill_between(pov_sizes, pov_means - pov_stds, pov_means + pov_stds, color="C0", alpha=0.2)
    if len(baseline_sizes) > 0:
        ax.plot(baseline_sizes, baseline_means, marker="o", color="C1", label="Baseline")
        ax.fill_between(
            baseline_sizes, baseline_means - baseline_stds, baseline_means + baseline_stds, color="C1", alpha=0.2
        )

    ax.set_xscale("log", base=10)
    ax.set_xlabel("Deck Size")
    ax.set_ylabel("Time (ms)")
    fig.suptitle("Time per Deck Size")
    ax.set_title("Close-to-uniform | Zero-copy", fontsize=9)
    ax.legend(loc="upper left")
    ax.grid(True)

    ax2 = ax.twinx()
    if len(speedup_sizes) > 0:
        ax2.plot(speedup_sizes, speedups, marker="s", linestyle="--", color="C2", label="Speedup (baseline / POV)")
    ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("Speedup (×)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_breaking_point_per_deck_size(
    deck_sizes: list,
    positional_breaking_points: list,
    ngram_breaking_points: dict,
    max_iterations: list,
    lstm_breaking_points: list | None = None,
) -> matplotlib.figure.Figure:
    """Plot breaking point (iterations to convergence) vs deck size for each bias metric.

    :param deck_sizes: Deck sizes along the x-axis.
    :param positional_breaking_points: Convergence iteration per deck size; ``None`` where not converged.
    :param ngram_breaking_points: Convergence iterations keyed by formatted metric name (e.g. ``"2-gram"``,
        ``"3-gram (skip 2)"``), values are lists indexed per deck size.
    :param max_iterations: Resolved iteration cap per deck size; used as the y-position for non-convergence markers.
    :param lstm_breaking_points: Convergence iteration per deck size for LSTM predictability; omitted if ``None``.
    :returns: Matplotlib figure.
    """
    fig, ax = plt.subplots()

    did_not_converge_added = False

    def _plot_metric(values, label, color):
        nonlocal did_not_converge_added
        valid_x, valid_y = [], []
        nc_x, nc_y = [], []
        for x, v, m in zip(deck_sizes, values, max_iterations):
            if v is not None:
                valid_x.append(x)
                valid_y.append(v)
            else:
                nc_x.append(x)
                nc_y.append(m)

        if valid_x:
            ax.plot(valid_x, valid_y, marker="o", color=color, label=label)

        if nc_x:
            nc_label = "did not converge" if not did_not_converge_added else None
            ax.scatter(nc_x, nc_y, marker="^", color=color, zorder=5, label=nc_label)
            did_not_converge_added = True

    n_metrics = 1 + len(ngram_breaking_points) + (1 if lstm_breaking_points is not None else 0)
    colors = [f"C{i}" for i in range(n_metrics)]
    _plot_metric(positional_breaking_points, "Positional bias", colors[0])
    for i, (name, values) in enumerate(ngram_breaking_points.items()):
        _plot_metric(values, f"{name} bias", colors[i + 1])
    if lstm_breaking_points is not None:
        _plot_metric(lstm_breaking_points, "LSTM pred. bias", colors[len(ngram_breaking_points) + 1])

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xscale("log", base=10)
    ax.set_xlabel("Deck Size")
    ax.set_ylabel("Breaking Point (iterations)")
    fig.suptitle("Breaking Point per Deck Size")
    ax.set_title("Iterations until bias converges to baseline", fontsize=9)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bias_per_iteration(
    tvds: np.ndarray,
    baseline: float,
    worker_data_scan_per_iter: float,
    ngram_tvds: np.ndarray,
    ngram_names: list[str],
    baseline_ngram_tvds: np.ndarray,
    lstm_predictabilities: np.ndarray | None = None,
    baseline_lstm_predictability: float | None = None,
) -> matplotlib.figure.Figure:
    """Plot bias metrics as a function of shuffle iterations.

    :param tvds: 1-D array of positional TVD values, one per iteration (index 0 = iteration 1).
    :param baseline: Observed positional TVD for the baseline shuffle on the same dataset size.
    :param worker_data_scan_per_iter: Percentage of data seen by each worker per iteration.
    :param ngram_tvds: Array of shape (max_iterations, len(ngram_names)) with n-gram TVD per iteration.
    :param ngram_names: Formatted metric names (e.g. ``"2-gram"``, ``"3-gram (skip 2)"``) corresponding
        to columns in ``ngram_tvds``.
    :param baseline_ngram_tvds: Observed n-gram TVD for the baseline shuffle, one value per entry in ngram_names.
    :param lstm_predictabilities: 1-D array of LSTM predictability values per iteration; omitted if ``None``.
    :param baseline_lstm_predictability: LSTM predictability of the baseline shuffle; omitted if ``None``.
    :returns: Matplotlib figure.
    """
    fig, ax = plt.subplots()
    iterations = range(1, len(tvds) + 1)

    ax.plot(iterations, tvds, marker="o", color="C0", label="Positional bias")
    ax.axhline(y=baseline, color="C0", linestyle="--")
    for i, name in enumerate(ngram_names):
        color = f"C{i + 1}"
        ax.plot(iterations, ngram_tvds[:, i], marker="o", color=color, label=f"{name} bias")
        ax.axhline(y=baseline_ngram_tvds[i], color=color, linestyle="--")

    if lstm_predictabilities is not None:
        lstm_color = f"C{len(ngram_names) + 1}"
        ax_pred = ax.twinx()
        all_pred = list(lstm_predictabilities) + (
            [baseline_lstm_predictability] if baseline_lstm_predictability is not None else []
        )
        y_lo = min(0.4, np.floor(min(all_pred) * 10) / 10)
        y_hi = max(0.6, np.ceil(max(all_pred) * 10) / 10)
        ax_pred.set_ylim(y_lo, y_hi)
        ax_pred.set_ylabel("LSTM Predictability (R²)")
        ax_pred.plot(iterations, lstm_predictabilities, marker="o", color=lstm_color, label="LSTM pred. bias")
        if baseline_lstm_predictability is not None:
            ax_pred.axhline(y=baseline_lstm_predictability, color=lstm_color, linestyle="--")
        lstm_handles, lstm_labels = ax_pred.get_legend_handles_labels()
    else:
        lstm_handles, lstm_labels = [], []

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Bias")
    fig.suptitle("Bias per Iteration")
    ax.set_title("POV and baseline deviations from uniform", fontsize=9)
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    baseline_handle = Line2D([0], [0], color="black", linestyle="--")
    ax.legend(handles=handles + lstm_handles + [baseline_handle], labels=labels + lstm_labels + ["Baseline"])

    ax_top = ax.secondary_xaxis(
        "top",
        functions=(
            lambda x: x * worker_data_scan_per_iter,
            lambda x: x / worker_data_scan_per_iter,
        ),
    )
    ax_top.set_xlabel("Data Scan per Worker (%)")
    fig.tight_layout()

    return fig
