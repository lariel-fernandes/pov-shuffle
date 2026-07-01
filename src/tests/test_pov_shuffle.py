import math
import re
from dataclasses import dataclass
from itertools import product

import numpy as np
import pytest
import torch

from povs import BuildParams, Options, get_build_params, optim_options_for_dataset, shuffle

_OPTS_SMALL = Options(physical_block_size=16, virtual_block_size=2, offsets=[4, 8])


@pytest.fixture(scope="session")
def build_params() -> BuildParams:
    return get_build_params()


_MARK_SKIP_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device available")


@dataclass
class _Case:
    deck_size: int
    instance_shape: tuple[int, ...]
    iterations: int
    options: Options | None
    device: str
    id: str | None = None

    def __post_init__(self):
        if self.id is None:
            shape = "x".join(map(str, self.instance_shape))
            dev = re.sub(r":.*", "", self.device)
            opts = list(self.options).__repr__().replace(" ", "") if self.options else ""
            self.id = f"deck{self.deck_size}_shape{shape}_iter{self.iterations}{'_opts' if opts else ''}{opts}_{dev}"


@pytest.mark.parametrize(
    "case",
    [
        *[
            pytest.param(case, id=case.id)
            for case in [
                _Case(*vals)
                for vals in product(
                    [32, 128],
                    [(8,), (2, 4), (2, 2, 2)],
                    [1, 3],
                    [_OPTS_SMALL, None],
                    ["numpy", "cpu"],
                )
            ]
        ],
        *[
            pytest.param(case, id=case.id, marks=_MARK_SKIP_CUDA)
            for case in [
                _Case(*vals, device="cuda:0")
                for vals in product(
                    [128],
                    [(8,), (2, 4)],
                    [1, 3],
                    [_OPTS_SMALL, None],
                )
            ]
        ],
    ],
)
def test_pov_shuffle(case: _Case, build_params: BuildParams) -> None:
    data = _make_deck(case.deck_size, case.instance_shape, case.device)
    options = optim_options_for_dataset(data) if case.options is None else case.options
    shuffle(data, iterations=case.iterations, options=options, seed=42)
    _check_integrity(data, case.deck_size, case.instance_shape)


def _make_deck(deck_size: int, instance_shape: tuple[int, ...], device: str) -> np.ndarray | torch.Tensor:
    flat = np.arange(deck_size * math.prod(instance_shape), dtype=np.float32).reshape(deck_size, *instance_shape)
    if device == "numpy":
        return flat
    tensor = torch.from_numpy(flat.copy())
    return tensor if device == "cpu" else tensor.to(device)


@pytest.mark.parametrize(
    "device",
    [
        "numpy",
        "cpu",
        pytest.param("cuda:0", marks=_MARK_SKIP_CUDA),
    ],
)
def test_pov_shuffle_partial_pblock(device: str) -> None:
    """Regression: every valid physical block must be shuffled when deck_size % pblock_size != 0.

    deck_size=40, pblock_size=16 → n_pblocks=3, last pblock has only 8 valid instances.
    n_pblocks=3 is also not divisible by vblock_size=2, so one padding slot is present too,
    exercising both edge cases simultaneously.
    """
    opts = Options(physical_block_size=16, virtual_block_size=2, offsets=[4, 8])
    deck_size = 40

    for seed in range(20):
        data = _make_deck(deck_size, (8,), device)
        shuffle(data, iterations=1, options=opts, seed=seed)
        _check_integrity(data, deck_size, (8,))


@pytest.mark.parametrize(
    "device",
    [
        "numpy",
        "cpu",
        pytest.param("cuda:0", marks=_MARK_SKIP_CUDA),
    ],
)
def test_pov_shuffle_padding_block(device: str) -> None:
    """Regression: every valid physical block must be shuffled when n_pblocks % vblock_size != 0.

    deck_size=48, pblock_size=16 → n_pblocks=3, not divisible by vblock_size=2.
    The last virtual block has one padding slot (phantom ID 3). The valid block paired
    with the phantom must still be shuffled; its data at original positions must change.
    With ~50% probability per seed the phantom lands at position 0, triggering the bug.
    Over 20 seeds the probability of no seed triggering it is ≈ 0.5^20 < 10^-6.
    """
    opts = Options(physical_block_size=16, virtual_block_size=2, offsets=[4, 8])
    n_pblocks = 3  # ceil(48 / 16)

    for seed in range(20):
        data = _make_deck(48, (8,), device)
        original = data.clone() if isinstance(data, torch.Tensor) else data.copy()
        shuffle(data, iterations=1, options=opts, seed=seed)
        data_arr = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        orig_arr = original.cpu().numpy() if isinstance(original, torch.Tensor) else original

        for pbid in range(n_pblocks):
            block = slice(pbid * 16, pbid * 16 + 16)
            assert not np.array_equal(data_arr[block], orig_arr[block]), (
                f"Physical block {pbid} was not shuffled with seed={seed} on {device}"
            )


def _check_integrity(data: np.ndarray | torch.Tensor, deck_size: int, instance_shape: tuple[int, ...]) -> None:
    instance_size = math.prod(instance_shape)
    arr = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
    flat = arr.reshape(deck_size, instance_size)

    instance_ids = flat[:, 0].astype(np.int64) // instance_size

    assert sorted(instance_ids.tolist()) == list(range(deck_size)), "deck integrity failed"
    assert instance_ids.tolist() != list(range(deck_size)), "deck was not shuffled"

    expected = (instance_ids[:, None] * instance_size + np.arange(instance_size)).astype(np.float32)
    assert np.array_equal(flat, expected), "instance integrity failed"
