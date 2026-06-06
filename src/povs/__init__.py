import warnings

import numpy as np
import torch as t

from . import numpy as povs_numpy
from . import torch as povs_torch
from .common import get_int_seed
from .types import FullOptions, Options

__all__ = [
    "shuffle",
    "Options",
    "optim_options_for_dataset",
]


def shuffle(
    data: t.Tensor | np.ndarray,
    iterations: int = 1,
    options: Options | None = None,
    seed: int | t.Generator | np.random.Generator | None = None,
) -> None:
    """POV Shuffle - Implementation routing for numpy (cpu) vs torch (cuda).

    :param data: Data tensor or array to shuffle in place along the axis 0.
    :param iterations: Number of shuffling iterations to perform.
    :param options: POV Shuffle algorithm options. If options are partially specified,
                    the missing downstream parameters are chosen using `optim_options_for_dataset`.
                    Specified downstream parameters are ignored if at least one upstream parameter is unspecified.
    :param seed: Random seed or random number generator state.
    """

    # Use numpy implementation for CPU tensors
    if isinstance(data, t.Tensor) and data.get_device() == -1:
        data = data.numpy()

    # Resolve options
    options = options or Options()
    options = optim_options_for_dataset(data, options) if None in options else FullOptions(*options)

    # Dispatch shuffle
    if isinstance(data, np.ndarray):
        povs_numpy.shuffle(data, iterations, options, get_int_seed(seed))
    elif isinstance(data, t.Tensor):
        povs_torch.shuffle(data, iterations, options, get_int_seed(seed))
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray.")


def optim_options_for_dataset(
    data: np.ndarray | t.Tensor,
    partial_options: Options | None = None,
) -> FullOptions:
    """Choose POV Shuffle options for dataset."""
    partial_options = partial_options or Options()

    if None not in partial_options:
        warnings.warn("All parameters already specified, skipping optimization")
        return FullOptions(*partial_options)

    for i, param in Options._fields:
        if partial_options[i] is None and any(x is not None for x in partial_options[i + 1 :]):
            warnings.warn(f"Upstream param {param} is not specified, ignoring specification of downstream params")
            partial_options = Options(*partial_options[: i + 1])
            break

    # Use numpy implementation for CPU tensors
    if isinstance(data, t.Tensor) and data.get_device() == -1:
        data = data.numpy()

    if isinstance(data, np.ndarray):
        return povs_numpy.optim_options_for_dataset(data, partial_options)

    return povs_torch.optim_options_for_dataset(data, partial_options)
