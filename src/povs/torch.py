import torch

from ._cuda import torch_binding
from .options import POVSOptions
from .utils import get_valid_offsets


def pov_shuffle(
    data: torch.Tensor,
    iterations: int = 1,
    options: POVSOptions = POVSOptions(),
    seed: int | torch.Generator = None,
) -> None:
    """POV Shuffle implementation for torch tensors on CUDA.

    :param data: Data tensor to shuffle in place along the axis 0.
    :param iterations: Number of shuffling iterations to perform.
    :param options: POV Shuffle algorithm options.
    :param seed: Random seed or random number generator state.
    """
    assert data.get_device() != -1, "Tensor device must be CUDA"

    # Validate parameters
    assert iterations >= 1
    assert options.virtual_block_size >= 2
    assert options.max_offset_steps >= 2
    assert options.offset_step_size % options.physical_block_size != 0

    # Collect offsets that are not multiples of the physical block size
    valid_offsets = get_valid_offsets(**options._asdict())
    assert len(valid_offsets) >= 2

    torch_binding(
        data,
        torch.tensor(valid_offsets, dtype=torch.int64),
        iterations,
        options.physical_block_size,
        options.virtual_block_size,
        seed if isinstance(seed, int) else int(torch.randint(0, 1000, (1,), generator=seed).item()),
    )
