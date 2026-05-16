def pov_shuffle():
    """POV Shuffle implementation for torch tensors.

    For CPU tensors, it falls back to using the pseudo-parallel implementation from the `povs.numpy` module.
    For CUDA tensors, it uses the truly parallel CUDA implementation from the module `povs.ext.torch`
    """
