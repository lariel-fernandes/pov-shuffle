#include "povs.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include "./utils.h"

// clang-format off
#define DISPATCH_CUDA_ARCH(cuda_arch, lambda) \
    [&]() {                                   \
        if      (cuda_arch >= 900) { constexpr int kCudaArch = 900; lambda(); } \
        else if (cuda_arch >= 800) { constexpr int kCudaArch = 800; lambda(); } \
        else                       { constexpr int kCudaArch = 700; lambda(); } \
    }()
// clang-format on

// Host function to get cuda architecture of specified device ID
int get_device_cuda_arch(int8_t device_id)
{
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device_id);
    return prop.major * 100 + prop.minor * 10;
}

// Device function to get the cuda architecture of the current device code
constexpr int __device__ get_cuda_arch()
{
#if defined(__CUDA_ARCH__)
    return __CUDA_ARCH__;
#else
    return 700; // Fall back to oldest supported arch
#endif
}

// Get GPU program block size for CUDA arch
// (not to mistake with pblock or vblock sizes, which refer to shuffle algorithm parameters)
template <int cuda_arch>
constexpr int get_block_size()
{
    return 64;
}

template <typename DType, int CudaArch, int BitWidth>
constexpr auto __device__ get_copy_atom()
{
    using namespace cute;
    if constexpr (CudaArch >= 800) return Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint_bit_t<BitWidth>>, DType>{};
    return Copy_Atom<UniversalCopy<uint_bit_t<BitWidth>>, DType>{};
}

template <typename DType, int CudaArch, int BlockSize>
auto __device__ get_tiled_copy()
{
    using namespace cute;
    constexpr int BitWidth = 128;
    constexpr int CopyWidth = BitWidth / 8 / sizeof(DType);
    constexpr auto CopyAtom = get_copy_atom<DType, CudaArch, BitWidth>();
    auto value_layout = make_layout(make_shape(Int<CopyWidth>{}));
    auto thread_layout = make_layout(make_shape(Int<BlockSize / CopyWidth>{}, Int<CopyWidth>{})
    ); // TODO: consider using a flat shape when the instance size is 1
    return make_tiled_copy(CopyAtom, thread_layout, value_layout);
}

/** Kernel for a single iteration of the POV Shuffle
 *
 *  Each GPU program block maps to a single virtual Block through `blockIdx.x`
 *  Each virtual block maps to `VBlockSize` physical blocks through the `V` mapping
 */
template <typename DType, int PBlockSize, int VBlockSize>
__global__ void povs_kernel(
    const DType* Xg_ptr, // Instances to shuffle in place - Global device memory pointer.
                         // Col-major (instance_size, num_instances)
    const long* Vg_ptr,  // Virtual block assignments - Global device memory pointer.
                         // Col-major (VBlockSize, num_vblocks)
    const long offset,   // Physical block start offset
    const long num_instances,
    const long instance_size,
    const int seed
)
{
}

template <typename DType, int PBlockSize, int VBlockSize>
void povs_cuda(
    DType* Xg_ptr, // Instances to shuffle in place - Global device memory pointer.
                   // Col-major (instance_size, num_instances) -- equivalent to Row-major (num_instances, instance_size)
    const long num_instances,
    const long instance_size,

    const long* Oh_ptr, // Valid block start offsets with dim (num_offsets,) - Host memory pointer
    const long num_offsets,

    const int iterations,
    const int seed,
    const int8_t device_id
)
{
    cudaError_t cudaStatus = cudaSuccess;
    const int cuda_arch = get_device_cuda_arch(device_id);
    const long num_pblocks = div_round_up(num_instances, static_cast<long>(PBlockSize));
    const long num_vblocks = div_round_up(num_pblocks, static_cast<long>(VBlockSize));

    // Allocate device pointers
    long* Vg_ptr = nullptr; // Virtual block assignments - Global device mem pointer. Col-major (VBlockSize, num_vblocks)
    CUDA_CHECK_STATUS(&cudaStatus, cleanup, cudaSetDevice(device_id));
    CUDA_CHECK_STATUS(&cudaStatus, cleanup, cudaMalloc(&Vg_ptr, sizeof(long) * num_vblocks * VBlockSize));

    // TODO: iterate, choose random offset, shuffle assignments

    DISPATCH_CUDA_ARCH(cuda_arch, [&] {
        const int num_blocks = num_pblocks;
        constexpr int block_size = get_block_size<kCudaArch>();

        (povs_kernel<DType, PBlockSize, VBlockSize>
         <<<num_blocks, block_size>>>(Xg_ptr, Vg_ptr, Oh_ptr[0], num_instances, instance_size, seed));
    });
    CUDA_CHECK_LAST_STATUS(&cudaStatus, cleanup);
#ifdef STANDALONE_BUILD
    // Only synchronize in standalone build (otherwise, the caller is responsible for synchronization)
    CUDA_CHECK_STATUS(&cudaStatus, cleanup, cudaDeviceSynchronize());
#endif

cleanup:
    cudaFree(Vg_ptr);
    if (cudaStatus != cudaSuccess) exit(cudaStatus);
}

int main()
{
    return 0;
}

#if __has_include("povs_cuda_template_instances.gen.inc")
#include "povs_cuda_template_instances.gen.inc"
#else
INSTANTIATE_POVS_CUDA_ALL_TYPES(8, 2)
#endif
