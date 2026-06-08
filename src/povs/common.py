# SPDX-FileCopyrightText: Copyright 2025 Dash0 Inc.
import math

import numpy as np
import torch

from .constants import MAX_SEED, MIN_OFFSETS, MIN_PBLOCK_SIZE, MIN_SEED, MIN_VBLOCK_SIZE
from .types import FullOptions
from .utils import is_power_of_2, least_factor_to_make_multiple


def get_int_seed(seed: int | torch.Generator | np.random.Generator | None) -> int:
    if isinstance(seed, int):
        return seed

    if isinstance(seed, torch.Generator):
        return int(torch.randint(MIN_SEED, MAX_SEED, (1,), generator=seed).item())

    if seed is None:
        seed = np.random.default_rng()

    return int(seed.integers(MIN_SEED, MAX_SEED))


def get_dtype_bytes(dataset: np.ndarray | torch.Tensor) -> int:
    return dataset.element_size() if isinstance(dataset, torch.Tensor) else dataset.itemsize


def get_instance_size(dataset: np.ndarray | torch.Tensor) -> int:
    total_elements = dataset.numel() if isinstance(dataset, torch.Tensor) else dataset.size
    return total_elements // dataset.shape[0] if dataset.ndim > 1 else 1


def povs_preflight(
    data: np.ndarray | torch.Tensor,
    iterations: int,
    options: FullOptions,
) -> None:
    """Common validations and preparations before running any POV Shuffle implementation."""

    assert iterations >= 1
    _validate_vblock_size(options.virtual_block_size)
    _validate_pblock_size(options.physical_block_size)
    _validate_offsets(
        options.offsets,
        deck_size=data.shape[0],
        instance_size=get_instance_size(data),
        dtype_bytes=get_dtype_bytes(data),
        pblock_size=options.physical_block_size,
    )


def _validate_vblock_size(vblock_size: int) -> None:
    assert is_power_of_2(vblock_size)
    assert vblock_size >= MIN_VBLOCK_SIZE


def _validate_pblock_size(pblock_size: int) -> None:
    assert is_power_of_2(pblock_size)
    assert pblock_size >= MIN_PBLOCK_SIZE


def _validate_offsets(
    offsets: list[int],
    deck_size: int,
    instance_size: int,
    dtype_bytes: int,
    pblock_size: int,
) -> None:
    assert len(offsets) >= MIN_OFFSETS
    for offset in offsets:
        less_than_deck_size, zero_or_not_divisible_by_pblk, preserves_alignment = _offset_is_valid(
            offset,
            deck_size=deck_size,
            instance_size=instance_size,
            dtype_bytes=dtype_bytes,
            pblock_size=pblock_size,
        )
        assert less_than_deck_size
        assert zero_or_not_divisible_by_pblk
        assert preserves_alignment


def _offset_is_valid(
    offset: int,
    deck_size: int,
    instance_size: int,
    dtype_bytes: int,
    pblock_size: int,
) -> tuple[bool, bool, bool]:
    instance_bits = instance_size * dtype_bytes * 8

    less_than_deck_size = offset < deck_size
    zero_or_not_divisible_by_pblk = offset == 0 or offset % pblock_size != 0
    preserves_alignment = is_power_of_2(offset_bits := offset * instance_bits) and offset_bits >= 16
    return less_than_deck_size, zero_or_not_divisible_by_pblk, preserves_alignment


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
    max_offsets: int | None = 16,
) -> list[int]:
    """Choose offsets for POVSOptions.

    Find a base offset that keeps a decent memory alignment and use
    multiples of it that have distinct rests when divided by `pblock_size`.
    """
    assert max_offsets is None or max_offsets >= MIN_OFFSETS
    max_offsets = max_offsets or MIN_OFFSETS

    instance_bits = instance_size * dtype_bytes * 8
    base_offset = _choose_base_offset(instance_bits, pblock_size)

    offsets = []
    rests = set()
    for i in range(max_offsets):
        offset = i * base_offset
        rest = 0 if offset == 0 else offset % pblock_size

        if rest not in rests:
            offsets.append(offset)
            rests.add(rest)

    return offsets


def _choose_base_offset(
    instance_bits: int,
    pblock_size: int,
    alignment: int = 128,
) -> int:
    base_offset = least_factor_to_make_multiple(instance_bits, alignment)

    if base_offset % pblock_size == 0:
        if alignment > 16:
            return _choose_base_offset(instance_bits, pblock_size, alignment // 2)
        raise ValueError("Cannot find a memory-aligned base offset that is not a multiple of the physical block size")

    return base_offset
