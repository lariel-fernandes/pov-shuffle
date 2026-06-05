from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from povs.torch import choose_pblock_size


@dataclass(frozen=True)
class _Case:
    instance_size: int
    dtype_bytes: int
    vblock_size: int
    sm_smem_bytes: int
    device_major: int
    device_minor: int
    expected: int


# Arithmetic key (for cases using instance_size=1, dtype_bytes=4, vblock_size=2):
#   needed_smem_per_instance = 1*4 + 32 = 36 bytes
#   instances_per_sm         = sm_smem_bytes // 36
#   min_instances_per_thr_block = max(vblock_size, 16) * vblock_size = 32
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=i)
        for i, case in enumerate([
            # instances_per_sm=2048, thr_blocks=32, instances_per_thr_block=64, pblock=32 → exact power of 2 (early return)
            _Case(
                instance_size=1,
                dtype_bytes=4,
                vblock_size=2,
                sm_smem_bytes=73728,
                device_major=9,
                device_minor=0,
                expected=32,
            ),
            # instances_per_sm=1536, pblock=24 → lower=16(diff=16) vs upper=32(diff=8) → upper wins
            _Case(
                instance_size=1,
                dtype_bytes=4,
                vblock_size=2,
                sm_smem_bytes=55296,
                device_major=9,
                device_minor=0,
                expected=32,
            ),
            # instances_per_sm=1152, pblock=18 → lower=16(diff=4) vs upper=32(diff=14) → lower wins
            _Case(
                instance_size=1,
                dtype_bytes=4,
                vblock_size=2,
                sm_smem_bytes=41472,
                device_major=9,
                device_minor=0,
                expected=16,
            ),
            # instances_per_sm=512, memory constrains thr_blocks to 16 (target=32); pblock=16 → exact power of 2
            _Case(
                instance_size=1,
                dtype_bytes=4,
                vblock_size=2,
                sm_smem_bytes=18432,
                device_major=9,
                device_minor=0,
                expected=16,
            ),
            # CC (8,0): target_thr_blocks_per_sm=16; instances_per_sm=1024, thr_blocks=16, pblock=32 → exact power of 2
            _Case(
                instance_size=1,
                dtype_bytes=4,
                vblock_size=2,
                sm_smem_bytes=36864,
                device_major=8,
                device_minor=0,
                expected=32,
            ),
            # Unknown CC → default (64, 24): needed_smem=40, instances_per_sm=1536, thr_blocks=24, pblock=32 → exact power of 2
            _Case(
                instance_size=2,
                dtype_bytes=4,
                vblock_size=2,
                sm_smem_bytes=61440,
                device_major=99,
                device_minor=0,
                expected=32,
            ),
        ])
    ],
)
def test_choose_pblock_size(case: _Case) -> None:
    dev = SimpleNamespace(
        major=case.device_major,
        minor=case.device_minor,
        shared_memory_per_multiprocessor=case.sm_smem_bytes,
    )
    with patch("povs.torch.torch.cuda.get_device_properties", return_value=dev):
        result = choose_pblock_size(case.instance_size, case.dtype_bytes, case.vblock_size, device_id=0)
    assert result == case.expected
