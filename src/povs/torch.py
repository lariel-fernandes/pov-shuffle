import torch

from ._cuda import torch_binding
from .options import POVSOptions


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

    torch_binding(
        data,
        iterations,
        *options,
        seed if isinstance(seed, int) else int(torch.randint(0, 1000, (1,), generator=seed).item()),
    )
