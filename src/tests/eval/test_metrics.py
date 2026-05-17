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
        desc="all permutations -> 0",
        samples=np.array(list(permutations(range(3)))),
    ),
    _TestGetTVD(
        desc="no permutation -> high TVD",
        samples=np.tile(np.arange(3), 3).reshape(3, 3),
        expected_tvd=1,
        tolerance=0.5,
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
