from dataclasses import dataclass
from itertools import product

import pytest

from povs.common import _validate_offsets, _validate_pblock_size, choose_offsets


@dataclass
class _Case:
    instance_size: int
    dtype_bytes: int
    pblock_size: int
    max_offsets: int | None
    id: str | None = None

    def __post_init__(self):
        if self.id is None:
            self.id = f"{self.instance_size}x{self.dtype_bytes}B_pblk{self.pblock_size}"
            if self.max_offsets is not None:
                self.id += f"_max{self.max_offsets}"


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case.id)
        for case in [
            _Case(*vals)
            for vals in product(
                [1, 2, 4],  # instance_size
                [16, 32, 64],  # dtype_bytes
                [16, 32, 64],  # pblock_size
                [None, 2, 4],  # max_offsets
            )
        ]
    ],
)
def test_choose_offsets(case: _Case) -> None:
    _validate_pblock_size(case.pblock_size)
    offsets = choose_offsets(case.instance_size, case.dtype_bytes, case.pblock_size, case.max_offsets)
    _validate_offsets(offsets, case.pblock_size)
