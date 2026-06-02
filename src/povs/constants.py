# Closed-open interval for sampling a numerical seed from a distribution.
# Aligned with the respective constants in `povs.cu` for reproducibility.
MIN_SEED = 1
MAX_SEED = 1000

ALLOWED_VIRTUAL_BLOCK_SIZES = (2, 4, 8, 16)

# Lookup of Cuda Compute capability (major, minor) tuple to a tuple of:
# - Thread-Block size for ideal occupancy (disregarding memory constraints, which are handled separately)
# - Number of thread-blocks per SM when using that thread-block size (again, if not constrained by memory)
_CUDA_CC_IDEAL_OCCUPANCY: dict[tuple[int, int], tuple[int, int]] = {
    (12, 0): (24, 64),  # Hopper (H100)
    (11, 0): (24, 64),  # Blackwell (H100)
    (10, 3): (32, 64),  # Hopper (H100)
    (10, 0): (32, 64),  # Hopper (H100)
    (9, 0): (32, 64),  # Hopper (H100)
    (8, 9): (24, 64),  # Ada Lovelace (RTX 4xxx, L4)
    (8, 7): (16, 64),  # Ampere consumer (RTX 3xxx, A10, A30, A40)
    (8, 6): (16, 64),  # Ampere consumer (RTX 3xxx, A10, A30, A40)
    (8, 0): (32, 64),  # Ampere datacenter (A100)
    (7, 5): (16, 64),  # Turing (RTX 20xx, T4)
    (7, 0): (32, 64),  # Volta (V100)
}
# TODO: adjust ideal occupancy block sizes

# Conservative defaults for unknown compute capabilities
_DEFAULT_IDEAL_OCC_BLOCK_SIZE = 64
_DEFAULT_IDEAL_OCC_BLOCKS_PER_SM = 24
