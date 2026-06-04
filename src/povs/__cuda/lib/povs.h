#ifndef POVS_H
#define POVS_H

#include <ATen/ATen.h>

template <typename DType, int PBlockSize, int VBlockSize, int InstanceSize>
void povs_cuda(
    DType* Xg_ptr,
    long num_instances,
    const long* Oh_ptr,
    long num_offsets,
    int iterations,
    int seed,
    int8_t device_id,
    int block_size
);

#define INSTANTIATE_POVS_CUDA(DType, PBlockSize, VBlockSize, InstanceSize) \
    template void povs_cuda<DType, PBlockSize, VBlockSize, InstanceSize>(  \
        DType * Xg_ptr,                                                    \
        long num_instances,                                                \
        const long* Oh_ptr,                                                \
        long num_offsets,                                                  \
        int iterations,                                                    \
        int seed,                                                          \
        int8_t device_id,                                                  \
        int block_size                                                     \
    );

#define INSTANTIATE_POVS_CUDA_ALL_TYPES(PBlockSize, VBlockSize, InstanceSize) \
    INSTANTIATE_POVS_CUDA(c10::Half, PBlockSize, VBlockSize, InstanceSize)    \
    INSTANTIATE_POVS_CUDA(int, PBlockSize, VBlockSize, InstanceSize)          \
    INSTANTIATE_POVS_CUDA(long, PBlockSize, VBlockSize, InstanceSize)         \
    INSTANTIATE_POVS_CUDA(float, PBlockSize, VBlockSize, InstanceSize)        \
    INSTANTIATE_POVS_CUDA(double, PBlockSize, VBlockSize, InstanceSize)

#endif
