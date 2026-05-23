#include "torch.h"

#include <torch/torch.h>

#include "../lib/povs.h"

// clang-format off
#ifndef PBLOCK_SIZE_CASES
#define PBLOCK_SIZE_CASES(lambda) \
    case 8: { constexpr int PBLOCK_SIZE = 8; return lambda(); }
#endif

#ifndef VBLOCK_SIZE_CASES
#define VBLOCK_SIZE_CASES(lambda) \
    case 2: { constexpr int VBLOCK_SIZE = 2; return lambda(); }
#endif
// clang-format on

#define PBLOCK_SIZE_DISPATCH(x, lambda)                                  \
    [&]() {                                                              \
        switch (x) {                                                     \
            PBLOCK_SIZE_CASES(lambda)                                    \
            default: TORCH_CHECK(false, "Unsupported PBLOCK_SIZE: ", x); \
        }                                                                \
    }()

#define VBLOCK_SIZE_DISPATCH(x, lambda)                                  \
    [&]() {                                                              \
        switch (x) {                                                     \
            VBLOCK_SIZE_CASES(lambda)                                    \
            default: TORCH_CHECK(false, "Unsupported VBLOCK_SIZE: ", x); \
        }                                                                \
    }()

void torch_povs(
    torch::Tensor X, // Data to shuffle in place along the axis 0
    torch::Tensor O, // Valid block start offsets
    int iterations,
    int pblock_size,
    int vblock_size,
    int seed
)
{
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(!O.is_cuda(), "O must be a CPU tensor");
}
