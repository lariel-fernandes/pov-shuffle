from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pytest

from povs.eval.metrics import get_tvd


@dataclass(frozen=True)
class _TestGetTVD:
    samples: np.ndarray
    expected_tvd: float = 0
    tolerance: float = 0
    relative: bool = False
    desc: str | None = None


_test_get_tvd_cases: list[_TestGetTVD] = [
    _TestGetTVD(
        desc="all permutations of range(3) -> 0",
        samples=np.array(list(permutations(range(3)))),
    ),
    _TestGetTVD(
        desc="identity only -> 2/3",
        samples=np.tile(np.arange(3), 3).reshape(3, 3),
        expected_tvd=2 / 3,
    ),
    _TestGetTVD(
        desc="all cyclic rotations -> 0 (mean of marginals is uniform)",
        samples=np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]),
    ),
    _TestGetTVD(
        desc="half identity half reversal -> 4/9",
        samples=np.array([[0, 1, 2]] * 3 + [[2, 1, 0]] * 3),
        expected_tvd=4 / 9,
    ),
]


@pytest.mark.parametrize(
    "case_idx",
    list(range(len(_test_get_tvd_cases))),
    ids=lambda case_idx: _test_get_tvd_cases[case_idx].desc or case_idx,
)
def test_get_tvd(case_idx: int):
    case = _test_get_tvd_cases[case_idx]
    result = get_tvd(case.samples)
    kwargs = {"rtol" if case.relative else "atol": case.tolerance}
    assert np.isclose(result, case.expected_tvd, **kwargs), result
