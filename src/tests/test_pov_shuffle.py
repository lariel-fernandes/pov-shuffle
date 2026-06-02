import math
from dataclasses import dataclass
from itertools import product

import numpy as np
import pytest
import torch

from povs import POVSOptions, pov_shuffle

_CUDA_AVAILABLE = torch.cuda.is_available()
_OPTS_SMALL = POVSOptions(physical_block_size=4, virtual_block_size=2, offset_step_size=1, max_offset_steps=4)


@dataclass(frozen=True)
class _Case:
    deck_size: int
    instance_shape: tuple[int, ...]
    iterations: int
    options: POVSOptions
    device: str

    @property
    def id(self) -> str:
        shape = "x".join(map(str, self.instance_shape))
        return f"deck{self.deck_size}_shape{shape}_iter{self.iterations}_pbs{self.options.physical_block_size}_{self.device.replace(':', '')}"


def _generate_cases() -> list:
    cpu_cases = [
        _Case(ds, shape, iters, opts, dev)
        for ds, shape, iters, opts, dev in product(
            [32, 128],
            [(8,), (2, 4), (2, 2, 2)],
            [1, 3],
            [_OPTS_SMALL, POVSOptions()],
            ["numpy", "cpu"],
        )
    ]
    cuda_cases = [
        _Case(ds, shape, iters, opts, "cuda:0")
        for ds, shape, iters, opts in product(
            [128],
            [(8,), (2, 4)],
            [1, 3],
            [_OPTS_SMALL, POVSOptions()],
        )
    ]
    result = [pytest.param(c, id=c.id) for c in cpu_cases]
    result += [
        pytest.param(
            c,
            id=c.id,
            marks=pytest.mark.skipif(not _CUDA_AVAILABLE, reason="No CUDA device available"),
        )
        for c in cuda_cases
    ]
    return result


def _make_deck(deck_size: int, instance_shape: tuple[int, ...], device: str) -> np.ndarray | torch.Tensor:
    S = math.prod(instance_shape)
    flat = np.arange(deck_size * S, dtype=np.float32).reshape(deck_size, *instance_shape)
    if device == "numpy":
        return flat
    tensor = torch.from_numpy(flat.copy())
    return tensor if device == "cpu" else tensor.to(device)


def _check_integrity(data: np.ndarray | torch.Tensor, deck_size: int, instance_shape: tuple[int, ...]) -> None:
    S = math.prod(instance_shape)
    arr = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
    flat = arr.reshape(deck_size, S)

    instance_ids = flat[:, 0].astype(np.int64) // S

    assert sorted(instance_ids.tolist()) == list(range(deck_size)), "deck integrity failed"

    expected = (instance_ids[:, None] * S + np.arange(S)).astype(np.float32)
    assert np.array_equal(flat, expected), "instance integrity failed"


@pytest.mark.parametrize("case", _generate_cases())
def test_pov_shuffle(case: _Case) -> None:
    data = _make_deck(case.deck_size, case.instance_shape, case.device)
    pov_shuffle(data, iterations=case.iterations, options=case.options, seed=42)
    _check_integrity(data, case.deck_size, case.instance_shape)
