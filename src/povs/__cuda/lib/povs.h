#ifndef POVS_H
#define POVS_H

template <typename DType, int PBlockSize, int VBlockSize>
void povs_cuda(DType* Xg_ptr, const int* Oh_ptr, int num_instances, int instance_size, int iterations, int seed);

#define INSTANTIATE_POVS_CUDA(DType, PBlockSize, VBlockSize)                                              \
    template void povs_cuda<DType, PBlockSize, VBlockSize>(                                               \
        DType * Xg_ptr, const int* Oh_ptr, int num_instances, int instance_size, int iterations, int seed \
    );

#define INSTANTIATE_POVS_CUDA_ALL_TYPES(PBlockSize, VBlockSize) \
    INSTANTIATE_POVS_CUDA(int, PBlockSize, VBlockSize)          \
    INSTANTIATE_POVS_CUDA(long, PBlockSize, VBlockSize)         \
    INSTANTIATE_POVS_CUDA(float, PBlockSize, VBlockSize)        \
    INSTANTIATE_POVS_CUDA(double, PBlockSize, VBlockSize)

#endif
