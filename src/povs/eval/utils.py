import time
import traceback
from collections.abc import Callable
from typing import Any, Type

import yaml as _yaml


def time_cuda_op(op: Callable, num_warmup: int, num_runs: int) -> list[float]:
    """Time a CUDA operation using CUDA events. Returns per-run elapsed times in milliseconds.

    :param op: Callable to time; must submit work to the default CUDA stream.
    :param num_warmup: Number of warm-up calls before measurement (not timed).
    :param num_runs: Number of timed calls.
    """
    import torch

    for _ in range(num_warmup):
        op()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        op()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return times_ms


def time_cpu_op(op: Callable, num_warmup: int, num_runs: int) -> list[float]:
    """Time a CPU operation using perf_counter. Returns per-run elapsed times in milliseconds.

    :param op: Callable to time.
    :param num_warmup: Number of warm-up calls before measurement (not timed).
    :param num_runs: Number of timed calls.
    """
    for _ in range(num_warmup):
        op()

    times_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        op()
        times_ms.append((time.perf_counter() - t0) * 1000)

    return times_ms


class _Dumper(_yaml.Dumper):
    """Patch of `yaml.Dumper`.

    Additional features:
       - Serializes NamedTuple as dict
    """

    def represent_data(self, data):
        if isinstance(data, tuple) and callable(f := getattr(data, "_asdict", None)):
            data = f()

        if isinstance(data, Exception):
            data = "".join(traceback.format_exception(type(data), data, data.__traceback__))

        return super().represent_data(data)


class yaml:
    """Namespace for extensions of the `yaml` package."""

    @staticmethod
    def dump(
        data: Any,
        *,
        sort_keys: bool = False,
        width: float = float("inf"),
        Dumper: Type[_yaml.Dumper] = _Dumper,
        **kwargs,
    ) -> str:
        """Drop-in replacement of yaml.dump with more convenient defaults."""
        return str(
            _yaml.dump(
                data,
                width=width,
                sort_keys=sort_keys,
                Dumper=Dumper,
                **kwargs,
            ),
        )
