#include "torch.h"

#include <torch/torch.h>

#include "../lib/povs.h"

/** Macros */
#pragma region Macros

// clang-format off
#ifndef PBLOCK_SIZE_CASES
#define PBLOCK_SIZE_CASES(lambda) \
    case 8: { constexpr int PBLOCK_SIZE = 8; return lambda(); }
#endif

#ifndef VBLOCK_SIZE_CASES
#define VBLOCK_SIZE_CASES(lambda) \
    case 2: { constexpr int VBLOCK_SIZE = 2; return lambda(); }
#endif

#ifndef INSTANCE_SIZE_CASES
#define INSTANCE_SIZE_CASES(lambda) \
    case 2: { constexpr int INSTANCE_SIZE = 1; return lambda(); }
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

#define INSTANCE_SIZE_DISPATCH(x, lambda)                                  \
    [&]() {                                                                \
        switch (x) {                                                       \
            INSTANCE_SIZE_CASES(lambda)                                    \
            default: TORCH_CHECK(false, "Unsupported INSTANCE_SIZE: ", x); \
        }                                                                  \
    }()

#pragma endregion

void torch_povs(
    const torch::Tensor& X, // Data to shuffle in place along the axis 0
    const torch::Tensor& O, // Valid block start offsets
    const int iterations,
    const int pblock_size,
    const int vblock_size,
    const int block_size,
    const int seed
)
{
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(O.is_cpu(), "O must be a CPU tensor");

    const long* Oh_ptr = O.data_ptr<long>(); // O pointer in host memory
    const long num_offsets = O.size(0);
    const long num_instances = X.size(0);

    // Calculate instance size as the flattened row size.
    // For a tensor of rank N, this will be the product of dimensions 1 through N-1
    // For a tensor of rank 1, this will be 1
    long instance_size = 1;
    for (int i = 1; i < X.dim(); ++i)
        instance_size *= X.size(i);

    AT_DISPATCH_FLOATING_TYPES_AND3(at::kHalf, at::kInt, at::kLong, X.scalar_type(), "povs", [&] {
        auto* Xg_ptr = X.data_ptr<scalar_t>(); // X pointer in device global memory

        PBLOCK_SIZE_DISPATCH(pblock_size, [&]() {
            VBLOCK_SIZE_DISPATCH(vblock_size, [&]() {
                INSTANCE_SIZE_DISPATCH(vblock_size, [&]() {
                    // clang-format off
                    (povs_cuda<scalar_t, PBLOCK_SIZE, VBLOCK_SIZE, INSTANCE_SIZE>)(
                        Xg_ptr, num_instances,
                        Oh_ptr, num_offsets,
                        iterations, seed, X.get_device(), block_size
                    );
                    // clang-format on
                });
            });
        });
    });
}
