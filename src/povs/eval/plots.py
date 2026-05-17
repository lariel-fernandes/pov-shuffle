import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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
    ax.set_ylabel("TVD")
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
