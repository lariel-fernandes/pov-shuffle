import torch

from ._cuda import torch_binding
from .constants import MAX_SEED, MIN_SEED
from .types import POVSOptions
from .utils import povs_preflight


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

    # Coerce to numerical seed
    seed = seed if isinstance(seed, int) else int(torch.randint(MIN_SEED, MAX_SEED, (1,), generator=seed).item())

    # Validate parameters and get valid offsets
    offsets = povs_preflight(iterations, options)

    # Delegate to bound CUDA library
    torch_binding(
        data,
        torch.tensor(offsets, dtype=torch.int64),
        iterations,
        options.physical_block_size,
        options.virtual_block_size,
        seed,
    )
