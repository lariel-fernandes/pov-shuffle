from __future__ import annotations

import typing

import torch

__all__: list[str] = ["torch_binding"]

def torch_binding(
    arg0: torch.Tensor,
    arg1: torch.Tensor,
    arg2: torch.Tensor,
    arg3: typing.SupportsInt,
    arg4: typing.SupportsInt,
    arg5: typing.SupportsInt,
    arg6: typing.SupportsInt,
) -> None:
    """
    PyTorch binding for POV Shuffle in CUDA
    """
