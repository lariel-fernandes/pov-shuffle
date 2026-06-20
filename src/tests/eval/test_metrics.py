from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pytest

from povs.eval.metrics import get_ngram_tvd, get_tvd


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
        desc="half identity half reversal -> 1/3 (reversal contributes distance 0,1,2 equally)",
        samples=np.array([[0, 1, 2]] * 3 + [[2, 1, 0]] * 3),
        expected_tvd=1 / 3,
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


@dataclass(frozen=True)
class _TestGetNgramTVD:
    samples: np.ndarray
    n: int
    expected_tvd: float = 0
    tolerance: float = 0
    relative: bool = False
    desc: str | None = None


_test_get_ngram_tvd_cases: list[_TestGetNgramTVD] = [
    _TestGetNgramTVD(
        desc="all permutations of range(3), n=2 -> 0",
        samples=np.array(list(permutations(range(3)))),
        n=2,
    ),
    _TestGetNgramTVD(
        desc="all permutations of range(3), n=3 -> 0",
        samples=np.array(list(permutations(range(3)))),
        n=3,
    ),
    _TestGetNgramTVD(
        desc="identity only, n=2 -> 1/2 (distance 1 always, distance 2 never)",
        samples=np.tile(np.arange(3), 3).reshape(3, 3),
        n=2,
        expected_tvd=1 / 2,
    ),
    _TestGetNgramTVD(
        desc="identity only, n=3 -> 1/2 (tuple (1,2) always, (2,1) never)",
        samples=np.tile(np.arange(3), 3).reshape(3, 3),
        n=3,
        expected_tvd=1 / 2,
    ),
]


@pytest.mark.parametrize(
    "case_idx",
    list(range(len(_test_get_ngram_tvd_cases))),
    ids=lambda case_idx: _test_get_ngram_tvd_cases[case_idx].desc or case_idx,
)
def test_get_ngram_tvd(case_idx: int):
    case = _test_get_ngram_tvd_cases[case_idx]
    result = get_ngram_tvd(case.samples, case.n)
    kwargs = {"rtol" if case.relative else "atol": case.tolerance}
    assert np.isclose(result, case.expected_tvd, **kwargs), result
