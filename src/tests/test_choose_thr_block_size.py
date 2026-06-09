from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from povs.common import _validate_pblock_size, _validate_vblock_size  # noqa
from povs.torch import _choose_thr_block_size, _validate_thr_block_size  # noqa


@dataclass(frozen=True)
class _Case:
    instance_size: int
    dtype_bytes: int
    vblock_size: int
    pblock_size: int
    sm_smem_bytes: int
    max_threads_per_sm: int
    device_major: int
    device_minor: int


# Arithmetic key (for cases using instance_size=1, dtype_bytes=4):
#   needed_smem_per_instance = 1*4 + 32 = 36 bytes
#   instances_per_sm         = sm_smem_bytes // 36
#   instances_per_thr_block  = vblock_size * pblock_size
#   max_thr_blocks_per_sm    = instances_per_sm // instances_per_thr_block
#   max_threads_per_block    = max_threads_per_sm // max_thr_blocks_per_sm
#   thr_block_size (raw)     = min(target_thr_block_size, max_threads_per_block, pblock_size * vblock_size)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=str(i))
        for i, case in enumerate([
            # raw=min(64,64,128)=64; 64>32, 64%32=0 → no rounding; result equals target
            _Case(
                instance_size=1,
                dtype_bytes=4,
                vblock_size=4,
                pblock_size=32,
                sm_smem_bytes=147456,
                max_threads_per_sm=2048,
                device_major=9,
                device_minor=0,
            ),
            # max_thr_blk=64, max_thr_per_blk=32; raw=min(64,32,32)=32; 32>16, 32%16=0 → no rounding; capped by pblock×vblock
            _Case(
                instance_size=1,
                dtype_bytes=4,
                vblock_size=2,
                pblock_size=16,
                sm_smem_bytes=73728,
                max_threads_per_sm=2048,
                device_major=9,
                device_minor=0,
            ),
            # max_thr_blk=32, max_thr_per_blk=48; raw=min(64,48,64)=48; 48>32, 48%32≠0 → round_down_to_multiple(48,32)=32
            _Case(
                instance_size=1,
                dtype_bytes=4,
                vblock_size=2,
                pblock_size=32,
                sm_smem_bytes=73728,
                max_threads_per_sm=1536,
                device_major=9,
                device_minor=0,
            ),
            # max_thr_blk=32, max_thr_per_blk=24; raw=min(64,24,64)=24; 24<32, 32%24≠0 → round_down_to_power_of_2(24)=16
            _Case(
                instance_size=1,
                dtype_bytes=4,
                vblock_size=2,
                pblock_size=32,
                sm_smem_bytes=73728,
                max_threads_per_sm=768,
                device_major=9,
                device_minor=0,
            ),
            # max_thr_blk=32, max_thr_per_blk=16; raw=min(64,16,64)=16; 16<32, 32%16=0 → no rounding; exact divisor
            _Case(
                instance_size=1,
                dtype_bytes=4,
                vblock_size=2,
                pblock_size=32,
                sm_smem_bytes=73728,
                max_threads_per_sm=512,
                device_major=9,
                device_minor=0,
            ),
            # CC (8,0): target_thr_block_size=64; max_thr_blk=16, max_thr_per_blk=128; raw=min(64,128,64)=64; 64>32, 64%32=0 → no rounding
            _Case(
                instance_size=1,
                dtype_bytes=4,
                vblock_size=2,
                pblock_size=32,
                sm_smem_bytes=36864,
                max_threads_per_sm=2048,
                device_major=8,
                device_minor=0,
            ),
        ])
    ],
)
def test_choose_thr_block_size(case: _Case) -> None:
    dev = SimpleNamespace(
        shared_memory_per_multiprocessor=case.sm_smem_bytes,
        max_threads_per_multi_processor=case.max_threads_per_sm,
        major=case.device_major,
        minor=case.device_minor,
    )

    _validate_vblock_size(case.vblock_size)
    _validate_pblock_size(case.pblock_size)

    with patch("povs.torch.torch.cuda.get_device_properties", return_value=dev):
        thr_block_size = _choose_thr_block_size(
            case.instance_size, case.dtype_bytes, case.vblock_size, case.pblock_size, device_id=0
        )
        _validate_thr_block_size(thr_block_size, case.vblock_size, case.pblock_size)
