# Closed-open interval for sampling a numerical seed from a distribution.
# Aligned with the respective constants in `povs.cu` for reproducibility.
MIN_SEED = 1
MAX_SEED = 1000

# Minimum number of valid offsets so that the random offset sampling
# at each shuffle iteration doesn't yield always the same fixed offset.
MIN_OFFSETS = 2

# In the case of smallest instance size (1) and smallest dtype bytes (16) this pblock size allows
# `povs.optim._choose_offsets` to produce at least `MIN_OFFSETS` valid offsets (namely the offset list [0, 8])
MIN_PHYSICAL_BLOCK_SIZE = 16

# In the case of smallest pblock size (16), these vblock sizes preserve the requirements:
# - Being larger than 1
# - Being a power of 2
# - Not exceeding pblock size
# - Being a divisor of pblock size
ALLOWED_VIRTUAL_BLOCK_SIZES = (2, 4, 8, 16)
MIN_VIRTUAL_BLOCK_SIZE = min(ALLOWED_VIRTUAL_BLOCK_SIZES)

# Lookup of Cuda Compute capability (major, minor) tuple to a tuple of:
# - Thread-Block size for ideal occupancy (disregarding memory constraints, which are handled separately)
# - Number of thread-blocks per SM when using that thread-block size (again, if not constrained by memory)
CUDA_CC_IDEAL_OCCUPANCY: dict[tuple[int, int], tuple[int, int]] = {
    (12, 0): (64, 24),
    (11, 0): (64, 24),
    (10, 3): (64, 32),
    (10, 0): (64, 32),
    (9, 0): (64, 32),
    (8, 9): (64, 24),
    (8, 7): (96, 16),
    (8, 6): (96, 16),
    (8, 0): (64, 16),
    (7, 5): (64, 16),
}
CUDA_DEFAULT_IDEAL_OCCUPANCY = (64, 24)
