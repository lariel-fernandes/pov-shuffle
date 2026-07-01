import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

from povs import Options
from povs.memmap import shuffle

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available")

# Partial options: gpu_thread_block_size and memmap_batch_size are filled in by optim_options_for_dataset
_OPTS_SMALL = Options(physical_block_size=16, virtual_block_size=2, offsets=[4, 8], memmap_cuda_device_id=0)


@dataclass
class _Case:
    deck_size: int
    instance_shape: tuple[int, ...]
    iterations: int
    options: Options | None
    id: str | None = None

    def __post_init__(self):
        if self.id is None:
            shape = "x".join(map(str, self.instance_shape))
            opts = list(self.options).__repr__().replace(" ", "") if self.options else ""
            self.id = f"deck{self.deck_size}_shape{shape}_iter{self.iterations}{'_opts' if opts else ''}{opts}"


def _make_memmap(deck_size: int, instance_shape: tuple[int, ...], path: Path) -> np.memmap:
    flat = np.arange(deck_size * math.prod(instance_shape), dtype=np.float32).reshape(deck_size, *instance_shape)
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(deck_size, *instance_shape))
    mm[:] = flat
    mm.flush()
    return mm


def _check_integrity(data: np.memmap, deck_size: int, instance_shape: tuple[int, ...]) -> None:
    instance_size = math.prod(instance_shape)
    flat = np.asarray(data).reshape(deck_size, instance_size)
    instance_ids = flat[:, 0].astype(np.int64) // instance_size
    assert sorted(instance_ids.tolist()) == list(range(deck_size)), "deck integrity failed"
    assert instance_ids.tolist() != list(range(deck_size)), "deck was not shuffled"
    expected = (instance_ids[:, None] * instance_size + np.arange(instance_size)).astype(np.float32)
    assert np.array_equal(flat, expected), "instance integrity failed"


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case.id)
        for case in [
            _Case(*vals)
            for vals in product(
                [
                    32,  # 2 full pblocks, 1 vblock — clean baseline
                    48,  # n_pblocks=3 not divisible by vblock_size=2 — padding slot only
                    40,  # deck_size % pblk != 0 — partial pblock + padding slot
                    128,  # larger deck
                ],
                [(8,), (2, 4)],
                [1, 3],
                [_OPTS_SMALL, None],
            )
        ]
    ],
)
def test_pov_shuffle_memmap(case: _Case, tmp_path: Path) -> None:
    mm = _make_memmap(case.deck_size, case.instance_shape, tmp_path / "deck.memmap")
    shuffle(mm, iterations=case.iterations, options=case.options, seed=42)
    _check_integrity(mm, case.deck_size, case.instance_shape)


def test_pov_shuffle_memmap_padding_block(tmp_path: Path) -> None:
    """Regression: every valid physical block must be shuffled when n_pblocks % vblock_size != 0.

    deck_size=48, pblock_size=16 → n_pblocks=3, not divisible by vblock_size=2.
    The last virtual block has one padding slot; the real block paired with it must still be shuffled.
    """
    n_pblocks = 3  # ceil(48 / 16)
    pblock_size = 16

    for seed in range(20):
        mm = _make_memmap(48, (8,), tmp_path / f"deck_{seed}.memmap")
        original = np.array(mm)
        shuffle(mm, iterations=1, options=_OPTS_SMALL, seed=seed)

        for pbid in range(n_pblocks):
            block = slice(pbid * pblock_size, pbid * pblock_size + pblock_size)
            assert not np.array_equal(mm[block], original[block]), (
                f"Physical block {pbid} was not shuffled with seed={seed}"
            )


def test_pov_shuffle_memmap_partial_pblock(tmp_path: Path) -> None:
    """Regression: partial last physical block must be shuffled without data corruption.

    deck_size=40, pblock_size=16 → n_pblocks=3, last pblock has only 8 valid instances.
    n_pblocks=3 is also not divisible by vblock_size=2, exercising both edge cases together.
    The partial block's garbage tail must be excluded from the kernel so it is not shuffled
    into valid data — verified by _check_integrity.
    """
    deck_size = 40

    for seed in range(20):
        mm = _make_memmap(deck_size, (8,), tmp_path / f"deck_{seed}.memmap")
        shuffle(mm, iterations=1, options=_OPTS_SMALL, seed=seed)
        _check_integrity(mm, deck_size, (8,))
