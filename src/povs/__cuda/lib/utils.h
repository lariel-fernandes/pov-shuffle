#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#include <cuda_runtime.h>

// If not a success: print error, store it and return true
inline bool cuda_check_status(const cudaError_t status, const char* file, const int line, cudaError_t* status_ptr)
{
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(status));
        *status_ptr = status;
        return true;
    }
    return false;
}

// If expression errors: print error, store it and go to the indicated label
#define CUDA_CHECK_STATUS(status_ptr, label, expr)                                 \
    {                                                                              \
        if (cuda_check_status((expr), __FILE__, __LINE__, status_ptr)) goto label; \
    }

// Same as CUDA_CHECK_STATUS for async operations
#define CUDA_CHECK_LAST_STATUS(status_ptr, label)                                              \
    {                                                                                          \
        if (cuda_check_status(cudaGetLastError(), __FILE__, __LINE__, status_ptr)) goto label; \
    }

// Round-up division
template <typename DType>
DType div_round_up(DType a, DType b)
{
    return (a + b - 1) / b;
};
int div_round_up(int a, int b);
long div_round_up(long a, long b);

#endif
