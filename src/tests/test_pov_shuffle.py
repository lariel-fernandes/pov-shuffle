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
            self.id = f"deck{self.deck_size}_shape{shape}_iter{self.iterations}_opts{opts}_{dev}"


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
    if options.physical_block_size not in build_params.pblock_sizes:
        pytest.skip(f"pblock_size={options.physical_block_size} not compiled in {build_params.pblock_sizes}")
    shuffle(data, iterations=case.iterations, options=options, seed=42)
    _check_integrity(data, case.deck_size, case.instance_shape)


def _make_deck(deck_size: int, instance_shape: tuple[int, ...], device: str) -> np.ndarray | torch.Tensor:
    flat = np.arange(deck_size * math.prod(instance_shape), dtype=np.float32).reshape(deck_size, *instance_shape)
    if device == "numpy":
        return flat
    tensor = torch.from_numpy(flat.copy())
    return tensor if device == "cpu" else tensor.to(device)


def _check_integrity(data: np.ndarray | torch.Tensor, deck_size: int, instance_shape: tuple[int, ...]) -> None:
    instance_size = math.prod(instance_shape)
    arr = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
    flat = arr.reshape(deck_size, instance_size)

    instance_ids = flat[:, 0].astype(np.int64) // instance_size

    assert sorted(instance_ids.tolist()) == list(range(deck_size)), "deck integrity failed"

    expected = (instance_ids[:, None] * instance_size + np.arange(instance_size)).astype(np.float32)
    assert np.array_equal(flat, expected), "instance integrity failed"
