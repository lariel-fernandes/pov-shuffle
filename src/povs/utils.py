import math


def is_power_of_2(x: int) -> bool:
    return x > 0 and (x & (x - 1) == 0)


def round_down_to_power_of_2(x: int) -> int:
    return 1 << (x.bit_length() - 1)


def round_down_to_multiple(x: int, y: int) -> int:
    """Closest k <= x that is a multiple of y."""
    return (x // y) * y


def least_factor_to_make_multiple(x, y):
    """Smallest k such that k * x is a multiple of y."""
    return y // math.gcd(x, y)
