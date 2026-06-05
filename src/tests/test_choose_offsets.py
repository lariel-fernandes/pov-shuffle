from dataclasses import dataclass

import pytest

from povs.common import choose_offsets


@dataclass(frozen=True)
class _Case:
    instance_size: int
    dtype_bytes: int
    pblock_size: int
    max_offsets: int | None
    expected: list[int]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=i)
        for i, case in enumerate([
            _Case(instance_size=1, dtype_bytes=4, pblock_size=16, max_offsets=None, expected=[0, 4, 8, 12]),
            _Case(instance_size=1, dtype_bytes=2, pblock_size=16, max_offsets=None, expected=[0, 8]),
            _Case(instance_size=1, dtype_bytes=1, pblock_size=32, max_offsets=None, expected=[0, 16]),
            _Case(instance_size=4, dtype_bytes=4, pblock_size=32, max_offsets=None, expected=list(range(32))),
            _Case(instance_size=1, dtype_bytes=4, pblock_size=32, max_offsets=3, expected=[0, 4, 8]),
            _Case(instance_size=1, dtype_bytes=4, pblock_size=16, max_offsets=0, expected=[0, 4, 8, 12]),
            _Case(instance_size=1, dtype_bytes=4, pblock_size=16, max_offsets=-1, expected=[0, 4, 8, 12]),
        ])
    ],
)
def test_choose_offsets(case: _Case) -> None:
    assert choose_offsets(case.instance_size, case.dtype_bytes, case.pblock_size, case.max_offsets) == case.expected
