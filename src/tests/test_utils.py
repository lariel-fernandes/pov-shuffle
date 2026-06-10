import pytest

from povs.utils import (
    is_power_of_2,
    least_factor_to_make_multiple,
    round_down_to_power_of_2,
)


@pytest.mark.parametrize(
    "x, expected",
    [
        (1, True),  # 2^0
        (2, True),
        (4, True),
        (1024, True),
        (0, False),  # not positive
        (-1, False),  # not positive
        (3, False),  # odd non-power
        (6, False),  # even non-power
    ],
)
def test_is_power_of_2(x, expected):
    assert is_power_of_2(x) == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 4),
        (7, 4),
        (8, 8),
        (9, 8),
    ],
)
def test_round_down_to_power_of_2(x, expected):
    assert round_down_to_power_of_2(x) == expected


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (1, 6, 6),  # x=1: need all 6 copies
        (6, 6, 1),  # x == y: already a multiple
        (3, 6, 2),  # x divides y evenly
        (6, 3, 1),  # x is a multiple of y
        (4, 6, 3),  # gcd=2, result=3
        (3, 4, 4),  # coprime, result=y
        (1, 1, 1),  # trivial
    ],
)
def test_least_factor_to_make_multiple(x, y, expected):
    result = least_factor_to_make_multiple(x, y)
    assert result == expected
    assert result >= 1, "factor must be at least 1 — zero would make k*x=0, never a meaningful multiple"
    assert (result * x) % y == 0, "k*x must be a multiple of y"
