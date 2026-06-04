#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#include <cuda_runtime.h>

/** CUDA error handling */
#pragma region CUDA error handling

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

#pragma endregion

/** Generic utilities */
#pragma region Generic utilities

// Round-up division
template <typename DTypeA, typename DTypeB>
auto div_round_up(DTypeA a, DTypeB b)
{
    using DTypeC = std::common_type_t<DTypeA, DTypeB>;
    const auto a_cast = static_cast<DTypeC>(a);
    const auto b_cast = static_cast<DTypeC>(b);
    const auto one = static_cast<DTypeC>(1);
    return (a_cast + b_cast - one) / b_cast;
};
template auto div_round_up(long a, int b);

// Shuffle flat array in place
template <typename DType>
void shuffle_array(DType* arr, const long size, std::mt19937& rng)
{
    for (long i = 0; i < size; i++) {
        std::uniform_int_distribution dist(i, size - 1);
        long j = dist(rng);
        std::swap(arr[i], arr[j]);
    }
}
template void shuffle_array(long arr[], long size, std::mt19937& rng);

#pragma endregion

#endif
