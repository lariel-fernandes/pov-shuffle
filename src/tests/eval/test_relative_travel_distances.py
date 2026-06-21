from dataclasses import dataclass

import numpy as np
import pytest

from povs.eval.metrics import _relative_travel_distances


@dataclass(frozen=True)
class _TestRTD:
    ngrams: np.ndarray
    skip: int
    expected: np.ndarray
    desc: str | None = None


_test_rtd_cases: list[_TestRTD] = [
    _TestRTD(
        desc="bigram unchanged relative distance: rtd = 0",
        ngrams=np.array([[[1, 2]]]),  # v0=1, v1=2, step=1: (2-1)-1 = 0
        skip=0,
        expected=np.array([[[0]]]),
    ),
    _TestRTD(
        desc="bigram got 7 positions closer (example 2 from glossary, step=3)",
        ngrams=np.array([[[0, 10]]]),  # v0=0, v1=10, step=3: (10-0)-3 = 7
        skip=2,
        expected=np.array([[[7]]]),
    ),
    _TestRTD(
        desc="bigram drifted further apart: negative rtd",
        ngrams=np.array([[[5, 3]]]),  # v0=5, v1=3, step=1: (3-5)-1 = -3
        skip=0,
        expected=np.array([[[-3]]]),
    ),
    _TestRTD(
        desc="3-gram adjacent (glossary example 3): rtd_1=4, rtd_2=-9",
        ngrams=np.array([[[10, 15, 3]]]),  # v0=10, v1=15, v2=3, step=1
        skip=0,
        expected=np.array([[[4, -9]]]),  # (15-10)-1=4, (3-10)-2=-9
    ),
    _TestRTD(
        desc="3-gram skip=1 (step=2): rtd uses k*step as current separation",
        ngrams=np.array([[[0, 10, 4]]]),  # v0=0, v1=10, v2=4, step=2
        skip=1,
        expected=np.array([[[8, 0]]]),  # (10-0)-2=8, (4-0)-4=0
    ),
    _TestRTD(
        desc="identity n-gram (v_k = v_0 + k): all rtds are zero",
        ngrams=np.array([[[3, 4, 5]]]),  # v0=3, v1=4, v2=5, step=1
        skip=0,
        expected=np.array([[[0, 0]]]),  # (4-3)-1=0, (5-3)-2=0
    ),
    _TestRTD(
        desc="batch of bigrams: shape (2, 3, 2) -> output (2, 3, 1)",
        ngrams=np.array([
            [[0, 1], [2, 5], [7, 3]],
            [[1, 1], [0, 4], [6, 2]],
        ]),
        skip=0,
        expected=np.array([
            [[(1 - 0) - 1], [(5 - 2) - 1], [(3 - 7) - 1]],  # 0, 2, -5
            [[(1 - 1) - 1], [(4 - 0) - 1], [(2 - 6) - 1]],  # -1, 3, -5
        ]),
    ),
]


@pytest.mark.parametrize(
    "case_idx",
    list(range(len(_test_rtd_cases))),
    ids=lambda case_idx: _test_rtd_cases[case_idx].desc or case_idx,
)
def test_relative_travel_distances(case_idx: int):
    case = _test_rtd_cases[case_idx]
    result = _relative_travel_distances(case.ngrams, case.skip)
    assert result.shape == case.expected.shape, f"shape mismatch: {result.shape} != {case.expected.shape}"
    np.testing.assert_array_equal(result, case.expected)
