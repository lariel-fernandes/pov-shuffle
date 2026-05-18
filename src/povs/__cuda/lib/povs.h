#ifndef POVS_H
#define POVS_H

template <typename DType, int PBlockSize, int VBlockSize>
DType* povs_cuda();

#define INSTANTIATE_POVS_CUDA(DType, PBlockSize, VBlockSize) \
    template DType* povs_cuda<DType, PBlockSize, VBlockSize>();

#define INSTANTIATE_POVS_CUDA_ALL_TYPES(PBlockSize, VBlockSize) \
    INSTANTIATE_POVS_CUDA(int, PBlockSize, VBlockSize)          \
    INSTANTIATE_POVS_CUDA(long, PBlockSize, VBlockSize)         \
    INSTANTIATE_POVS_CUDA(float, PBlockSize, VBlockSize)        \
    INSTANTIATE_POVS_CUDA(double, PBlockSize, VBlockSize)   

#endif
