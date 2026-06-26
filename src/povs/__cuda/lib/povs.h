#ifndef POVS_H
#define POVS_H

#include <ATen/ATen.h>

template <int VBlockSize, int PBlockSize, int InstanceSize, typename DType>
void povs_cuda(DType* Xg_ptr, long num_instances, const long* Ag_ptr, const int* Sg_ptr, long offset, int block_size);

#define INSTANTIATE_POVS_CUDA(VBlockSize, PBlockSize, InstanceSize, DType)                                     \
    template void povs_cuda<VBlockSize, PBlockSize, InstanceSize, DType>(                                      \
        DType * Xg_ptr, long num_instances, const long* Ag_ptr, const int* Sg_ptr, long offset, int block_size \
    );

#define INSTANTIATE_POVS_CUDA_ALL_TYPES(VBlockSize, PBlockSize, InstanceSize) \
    INSTANTIATE_POVS_CUDA(VBlockSize, PBlockSize, InstanceSize, c10::Half)    \
    INSTANTIATE_POVS_CUDA(VBlockSize, PBlockSize, InstanceSize, int)          \
    INSTANTIATE_POVS_CUDA(VBlockSize, PBlockSize, InstanceSize, long)         \
    INSTANTIATE_POVS_CUDA(VBlockSize, PBlockSize, InstanceSize, float)        \
    INSTANTIATE_POVS_CUDA(VBlockSize, PBlockSize, InstanceSize, double)

#endif
