import numpy as np
import torch
import torch as t

from .constants import MAX_SEED, MIN_SEED
from .numpy import pov_shuffle as pov_shuffle_numpy
from .torch import pov_shuffle as pov_shuffle_torch
from .types import POVSOptions

__all__ = [
    "POVSOptions",
]


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

    # Use numpy implementation for CPU tensors
    if isinstance(data, t.Tensor) and data.get_device() == -1:
        data = data.numpy()

    if isinstance(data, np.ndarray):
        if isinstance(seed, t.Generator):
            # For numpy impl, coerce torch generator to numerical seed
            seed = int(torch.randint(MIN_SEED, MAX_SEED, (1,), generator=seed).item())
        pov_shuffle_numpy(data, iterations, options, seed)

    elif isinstance(data, t.Tensor):
        if isinstance(seed, np.random.Generator):
            # For torch impl, coerce numpy generator to numerical seed
            seed = int(seed.integers(MIN_SEED, MAX_SEED))
        pov_shuffle_torch(data, iterations, options, seed)

    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray.")
