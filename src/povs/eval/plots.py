import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def plot_tvd_per_iteration(tvds: np.ndarray, baseline: float | None = None) -> matplotlib.figure.Figure:
    """Plot Total Variation Distance as a function of shuffle iterations.

    :param tvds: 1-D array of TVD values, one per iteration (index 0 = iteration 1).
    :param baseline: Observed TVD for the baseline shuffle on the same dataset size.
    :returns: Matplotlib figure.
    """
    fig, ax = plt.subplots()
    ax.plot(range(1, len(tvds) + 1), tvds, marker="o")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("TVD")
    ax.set_title("POV Shuffle — TVD per Iteration")
    ax.grid(True)

    # If a baseline is specified, set a horizontal line
    if baseline is not None:
        ax.axhline(y=baseline, color="r", linestyle="--", label="Baseline")

    return fig
