# include "povs.h"

template <typename DType, int PBlockSize, int VBlockSize>
DType* povs_cuda(

)
{
    cudaError_t cudaStatus = cudaSuccess;
    DType* x = (DType*)malloc(sizeof(DType) * PBlockSize * VBlockSize);
    return x;
}

int main() {
    return 0;
}

#if __has_include("povs_cuda_template_instances.gen.inc")
#include "povs_cuda_template_instances.gen.inc"
#else
INSTANTIATE_POVS_CUDA_ALL_TYPES(8, 2)
#endif
