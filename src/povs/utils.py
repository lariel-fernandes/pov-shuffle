import math

import numpy as np
import torch


def numpy_to_torch_dtype(dtype: np.dtype) -> torch.dtype:
    return getattr(torch, np.dtype(dtype).name)


def is_power_of_2(x: int) -> bool:
    return x > 0 and (x & (x - 1) == 0)


def round_down_to_power_of_2(x: int) -> int:
    return 1 << (x.bit_length() - 1)


def least_factor_to_make_multiple(x, y):
    """Smallest k such that k * x is a multiple of y."""
    return y // math.gcd(x, y)
