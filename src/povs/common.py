# SPDX-FileCopyrightText: Copyright 2025 Dash0 Inc.
import math

import numpy as np
import torch

from povs import POVSOptions
from povs.constants import ALLOWED_VIRTUAL_BLOCK_SIZES, MIN_OFFSETS, MIN_PHYSICAL_BLOCK_SIZE
from povs.utils import is_power_of_2, least_factor_to_make_multiple


def get_dtype_bytes(dataset: np.ndarray | torch.Tensor) -> int:
    return dataset.element_size() if isinstance(dataset, torch.Tensor) else dataset.itemsize


def get_instance_size(dataset: np.ndarray | torch.Tensor) -> int:
    total_elements = dataset.numel() if isinstance(dataset, torch.Tensor) else dataset.size
    return total_elements // dataset.shape[0] if dataset.ndim > 1 else 1


def povs_preflight(
    iterations: int,
    options: POVSOptions,
) -> None:
    """Common validations and preparations before running any POV Shuffle implementation."""

    assert iterations >= 1
    assert is_power_of_2(options.physical_block_size)
    assert options.physical_block_size >= MIN_PHYSICAL_BLOCK_SIZE
    assert options.virtual_block_size in ALLOWED_VIRTUAL_BLOCK_SIZES
    assert len(options.offsets) >= MIN_OFFSETS


def get_block_counts(
    deck_size: int,
    physical_block_size: int,
    virtual_block_size: int,
    **__,
) -> tuple[int, int]:
    """Get the number of physical and virtual blocks, with rounding up."""

    n_pblocks = math.ceil(deck_size / physical_block_size)
    n_vblocks = math.ceil(n_pblocks / virtual_block_size)
    return n_pblocks, n_vblocks


def choose_offsets(
    instance_size: int,
    dtype_bytes: int,
    pblock_size: int,
    max_offsets: int | None = None,
) -> list[int]:
    """Choose offsets for POVSOptions.

    Find a base offset that keeps 128bit memory alignment and use multiples of it that don't exceed `pblock_size`.
    """
    instance_bits = instance_size * dtype_bytes * 8
    base_offset = least_factor_to_make_multiple(instance_bits, 128)
    num_offsets = pblock_size // base_offset

    if max_offsets is not None and max_offsets > 0:
        num_offsets = min(num_offsets, max_offsets)

    return [base_offset * i for i in range(num_offsets)]
