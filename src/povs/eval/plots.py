import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def plot_tvd_per_iteration(
    tvds: np.ndarray, baseline: float, worker_data_scan_per_iter: float
) -> matplotlib.figure.Figure:
    """Plot Total Variation Distance as a function of shuffle iterations.

    :param tvds: 1-D array of TVD values, one per iteration (index 0 = iteration 1).
    :param baseline: Observed TVD for the baseline shuffle on the same dataset size.
    :param worker_data_scan_per_iter: Percentage of data seen by each worker per iteration.
    :returns: Matplotlib figure.
    """
    fig, ax = plt.subplots()
    ax.plot(range(1, len(tvds) + 1), tvds, marker="o")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("TVD")
    ax.set_title("POV Shuffle — TVD per Iteration")
    ax.grid(True)
    ax.axhline(y=baseline, color="r", linestyle="--", label="Baseline")
    ax.legend()

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
