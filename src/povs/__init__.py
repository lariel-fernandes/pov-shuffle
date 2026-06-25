from typing import Any

import numpy as np
import torch as t

from . import numpy as povs_numpy
from . import torch as povs_torch
from .common import get_build_params, get_int_seed
from .types import BuildParams, Options

__all__ = [
    "shuffle",
    "Options",
    "BuildParams",
    "get_build_params",
    "optim_options_for_dataset",
]


def shuffle(
    data: np.ndarray | t.Tensor,
    iterations: int = 1,
    options: Options | None = None,
    seed: int | np.random.Generator | Any | None = None,
) -> None:
    """POV Shuffle - Main interface with implementation routing.

    :param data: `torch.Tensor` or `numpy.ndarray` to be shuffled in place along the axis 0.
    :param iterations: Number of shuffling iterations to perform.
    :param options: POV Shuffle algorithm options.
                    Partial or missing options are resolved as in `povs.optim_options_for_dataset`.
    :param seed: Integer seed, `np.random.Generator` or `torch.Generator`.
    """
    seed = get_int_seed(seed)

    if isinstance(data, np.ndarray):
        povs_numpy.shuffle(data, iterations, options, seed)
        return

    if isinstance(data, t.Tensor):
        if data.get_device() == -1:
            shuffle(data.numpy(), iterations, options, seed)
            return

        povs_torch.shuffle(data, iterations, options, seed)
        return

    raise TypeError(f"Unsupported data type: {type(data)}")


def optim_options_for_dataset(
    data: np.ndarray | t.Tensor,
    options: Options | None = None,
) -> Options:
    """Resolve partial or missing POV Shuffle algorithm options, optimizing for the specified dataset.

    :param data: `torch.Tensor` or `numpy.ndarray`.
    :param options: Starting point for the POV Shuffle algorithm options.
    :returns: Recommended POV Shuffle algorithm options for the specified dataset.
    """
    options = options or Options()

    if isinstance(data, np.ndarray):
        return (
            options if povs_numpy.options_is_complete(options) else povs_numpy.optim_options_for_dataset(data, options)
        )

    if isinstance(data, t.Tensor):
        if data.get_device() == -1:
            return optim_options_for_dataset(data.numpy(), options)

        return (
            options if povs_torch.options_is_complete(options) else povs_torch.optim_options_for_dataset(data, options)
        )

    raise TypeError(f"Unsupported data type: {type(data)}")
