#include "torch.h"

#include <torch/torch.h>

#include "../lib/povs.h"

#if __has_include("povs_cuda_dispatch.gen.h")
#include "povs_cuda_dispatch.gen.h"
#else
// clang-format off
#define POVS_CUDA_DISPATCH(vblock_val, pblock_val, instance_val, scalar_type_val, fn) \
    do { \
        if (vblock_val == 2 && pblock_val == 8 && instance_val == 1 && scalar_type_val == at::kFloat) { constexpr int VBLOCK_SIZE = 2, PBLOCK_SIZE = 8, INSTANCE_SIZE = 1; using scalar_t = float; fn; } \
        else { TORCH_CHECK(false, "Unsupported povs_cuda combination:", " vblock=", vblock_val, " pblock=", pblock_val, " instance=", instance_val, " scalar_type=", at::toString(scalar_type_val)); } \
    } while (0)
// clang-format on
#endif

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
    // Dataset preflight — keep in sync with povs.torch.preflight, dataset section
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(
        X.scalar_type() == at::kHalf || X.scalar_type() == at::kFloat || X.scalar_type() == at::kDouble || X.scalar_type() == at::kInt ||
            X.scalar_type() == at::kLong,
        "Unsupported data type"
    );
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

    // clang-format off
    POVS_CUDA_DISPATCH(vblock_size, pblock_size, instance_size, X.scalar_type(), {
        auto* Xg_ptr = X.data_ptr<scalar_t>();
        (povs_cuda<VBLOCK_SIZE, PBLOCK_SIZE, INSTANCE_SIZE, scalar_t>)(
            Xg_ptr, num_instances,
            Oh_ptr, num_offsets,
            iterations, seed, X.get_device(), block_size
        );
    });
    // clang-format on
}
