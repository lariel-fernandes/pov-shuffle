#include "povs.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include "./utils.h"

template <typename DType, int PBlockSize, int VBlockSize>
void povs_cuda(
    DType* Xg_ptr,     // Instances to shuffle in place.
                       // Col-major (instance_size, num_instances) -- equivalent to Row-major (num_instances, instance_size)
    const int* Oh_ptr, // Valid block start offsets with dim (num_valid_offsets,)
    const int num_instances,
    const int instance_size,
    const int iterations,
    const int seed
)
{
    cudaError_t cudaStatus = cudaSuccess;

    printf(
        "povs_cuda: num_instances=%d, instance_size=%d, iterations=%d, pblock_size=%d, vblock_size=%d, seed=%d\n",
        num_instances,
        instance_size,
        iterations,
        PBlockSize,
        VBlockSize,
        seed
    );
    printf("povs_cuda: sizeof(DType)=%zu\n", sizeof(DType));
    printf("povs_cuda: Xg_ptr=%p\n", Xg_ptr);
    printf("povs_cuda: Oh_ptr=%p\n", Oh_ptr);

cleanup:
    if (cudaStatus != cudaSuccess) {
        exit(cudaStatus);
    }
}

int main()
{
    return 0;
}

#if __has_include("povs_cuda_template_instances.gen.inc")
#include "povs_cuda_template_instances.gen.inc"
#else
INSTANTIATE_POVS_CUDA(float, 8, 2)
#endif
