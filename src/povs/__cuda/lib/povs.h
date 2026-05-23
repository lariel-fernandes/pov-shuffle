#ifndef POVS_H
#define POVS_H

#include <ATen/ATen.h>

template <typename DType, int PBlockSize, int VBlockSize>
void povs_cuda(
    DType* Xg_ptr,
    long num_instances,
    long instance_size,
    const long* Oh_ptr,
    long num_offsets,
    int iterations,
    int seed,
    int8_t device_id
);

#define INSTANTIATE_POVS_CUDA(DType, PBlockSize, VBlockSize) \
    template void povs_cuda<DType, PBlockSize, VBlockSize>(  \
        DType * Xg_ptr,                                      \
        long num_instances,                                  \
        long instance_size,                                  \
        const long* Oh_ptr,                                  \
        long num_offsets,                                    \
        int iterations,                                      \
        int seed,                                            \
        int8_t device_id                                     \
    );

#define INSTANTIATE_POVS_CUDA_ALL_TYPES(PBlockSize, VBlockSize) \
    INSTANTIATE_POVS_CUDA(int, PBlockSize, VBlockSize)          \
    INSTANTIATE_POVS_CUDA(long, PBlockSize, VBlockSize)         \
    INSTANTIATE_POVS_CUDA(float, PBlockSize, VBlockSize)        \
    INSTANTIATE_POVS_CUDA(double, PBlockSize, VBlockSize)

#endif
