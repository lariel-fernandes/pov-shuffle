import numpy as np
import torch
import torch as t

from . import numpy as povs_numpy
from . import torch as povs_torch
from .constants import MAX_SEED, MIN_SEED
from .types import POVSOptions

__all__ = [
    "POVSOptions",
    "pov_shuffle",
    "get_povs_options_for_dataset",
]


def pov_shuffle(
    data: t.Tensor | np.ndarray,
    iterations: int = 1,
    options: POVSOptions | None = None,
    seed: int | t.Generator | np.random.Generator | None = None,
) -> None:
    """POV Shuffle - Implementation routing for numpy (cpu) vs torch (cuda).

    :param data: Data tensor or array to shuffle in place along the axis 0.
    :param iterations: Number of shuffling iterations to perform.
    :param options: POV Shuffle algorithm options.
    :param seed: Random seed or random number generator state.
    """

    # Use numpy implementation for CPU tensors
    if isinstance(data, t.Tensor) and data.get_device() == -1:
        data = data.numpy()

    options = options or get_povs_options_for_dataset(data)

    if isinstance(data, np.ndarray):
        if isinstance(seed, t.Generator):
            # For numpy impl, coerce torch generator to numerical seed
            seed = int(torch.randint(MIN_SEED, MAX_SEED, (1,), generator=seed).item())
        povs_numpy.pov_shuffle(data, iterations, options, seed)

    elif isinstance(data, t.Tensor):
        if isinstance(seed, np.random.Generator):
            # For torch impl, coerce numpy generator to numerical seed
            seed = int(seed.integers(MIN_SEED, MAX_SEED))
        povs_torch.pov_shuffle(data, iterations, options, seed)

    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray.")


def get_povs_options_for_dataset(
    data: np.ndarray | t.Tensor,
) -> POVSOptions:
    """Choose POV Shuffle options for dataset."""

    # Use numpy implementation for CPU tensors
    if isinstance(data, t.Tensor) and data.get_device() == -1:
        data = data.numpy()

    if isinstance(data, np.ndarray):
        return povs_numpy.choose_options_for_dataset(data)

    return povs_torch.choose_options_for_dataset(data)
