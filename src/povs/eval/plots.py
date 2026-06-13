import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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
        ax.plot(pov_sizes, pov_means, marker="o", color="C0", label="POV Shuffle")
        ax.fill_between(pov_sizes, pov_means - pov_stds, pov_means + pov_stds, color="C0", alpha=0.2)
    if len(baseline_sizes) > 0:
        ax.plot(baseline_sizes, baseline_means, marker="o", color="C1", label="Fisher-Yates (CUDA)")
        ax.fill_between(
            baseline_sizes, baseline_means - baseline_stds, baseline_means + baseline_stds, color="C1", alpha=0.2
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Deck Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("POV Shuffle — Time per Deck Size")
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


def plot_tvd_per_iteration(
    tvds: np.ndarray,
    baseline: float,
    worker_data_scan_per_iter: float,
    ngram_tvds: np.ndarray,
    ngram_degrees: list[int],
    baseline_ngram_tvds: np.ndarray,
) -> matplotlib.figure.Figure:
    """Plot Total Variation Distance as a function of shuffle iterations.

    :param tvds: 1-D array of positional TVD values, one per iteration (index 0 = iteration 1).
    :param baseline: Observed positional TVD for the baseline shuffle on the same dataset size.
    :param worker_data_scan_per_iter: Percentage of data seen by each worker per iteration.
    :param ngram_tvds: Array of shape (max_iterations, len(ngram_degrees)) with n-gram TVD per iteration.
    :param ngram_degrees: N-gram degrees corresponding to columns in ngram_tvds.
    :param baseline_ngram_tvds: Observed n-gram TVD for the baseline shuffle, one value per degree in ngram_degrees.
    :returns: Matplotlib figure.
    """
    fig, ax = plt.subplots()
    iterations = range(1, len(tvds) + 1)

    ax.plot(iterations, tvds, marker="o", color="C0", label="Positional bias")
    ax.axhline(y=baseline, color="C0", linestyle="--")
    for i, n in enumerate(ngram_degrees):
        color = f"C{i + 1}"
        ax.plot(iterations, ngram_tvds[:, i], marker="o", color=color, label=f"{n}-gram bias")
        ax.axhline(y=baseline_ngram_tvds[i], color=color, linestyle="--")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Total Variation Distance")
    ax.set_title("POV Shuffle — TVD per Iteration")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linestyle="--"))
    labels.append("Baseline")
    ax.legend(handles=handles, labels=labels)

    ax2 = ax.secondary_xaxis(
        "top",
        functions=(
            lambda x: x * worker_data_scan_per_iter,
            lambda x: x / worker_data_scan_per_iter,
        ),
    )
    ax2.set_xlabel("Data Scan per Worker (%)")
    fig.tight_layout()

    return fig
