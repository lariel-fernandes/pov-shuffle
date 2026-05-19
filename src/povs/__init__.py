import numpy as np
import torch
import torch as t

from .numpy import pov_shuffle as pov_shuffle_numpy
from .options import POVSOptions
from .torch import pov_shuffle as pov_shuffle_torch

__all__ = [
    "POVSOptions",
]

# TODO:
#   - do some plot that compares against a standard local block shuffle!
#   - add the exercise of breaking point by dataset size


def pov_shuffle(
    data: t.Tensor | np.ndarray,
    iterations: int = 1,
    options: POVSOptions = POVSOptions(),
    seed: int | t.Generator | np.random.Generator = None,
) -> None:
    """POV Shuffle - Implementation routing for numpy (cpu) vs torch (cuda).

    :param data: Data tensor or array to shuffle in place along the axis 0.
    :param iterations: Number of shuffling iterations to perform.
    :param options: POV Shuffle algorithm options.
    :param seed: Random seed or random number generator state.
    """

    # Use numpy CPU implementation for CPU tensors
    if isinstance(data, t.Tensor) and data.get_device() == -1:
        data = data.numpy()

    if isinstance(data, np.ndarray):
        if isinstance(seed, t.Generator):
            seed = int(torch.randint(0, 1000, (1,), generator=seed).item())
        pov_shuffle_numpy(data, iterations, options, seed)

    elif isinstance(data, t.Tensor):
        if isinstance(seed, np.random.Generator):
            seed = int(seed.integers(0, 1000))
        pov_shuffle_torch(data, iterations, options, seed)

    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray.")
