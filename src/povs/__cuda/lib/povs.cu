#include "povs.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <random>

#include "./utils.h"

// clang-format off
#define DISPATCH_CUDA_ARCH(cuda_arch, lambda) \
    [&]() {                                   \
        if      (cuda_arch >= 900) { constexpr int kCudaArch = 900; return lambda(); } \
        else if (cuda_arch >= 800) { constexpr int kCudaArch = 800; return lambda(); } \
        else                       { constexpr int kCudaArch = 700; return lambda(); } \
    }()
// clang-format on

/** Device-side helper functions */

constexpr int __device__ get_cuda_arch()
{
#if defined(__CUDA_ARCH__)
    return __CUDA_ARCH__;
#else
    return 700; // Fall back to oldest supported architecture
#endif
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
    // TODO: consider using a flat shape when the instance size is 1
    auto thread_layout = make_layout(make_shape(Int<BlockSize / CopyWidth>{}, Int<CopyWidth>{}));
    return make_tiled_copy(CopyAtom, thread_layout, value_layout);
}

/** POV Shuffle - Single iteration Kernel
 *
 *  Each GPU program block maps to a single virtual Block through `blockIdx.x`
 *  Each virtual block maps to `VBlockSize` physical blocks through the `V` mapping
 */
template <typename DType, int PBlockSize, int VBlockSize, int InstanceSize>
__global__ void povs_kernel(
    const DType* Xg_ptr, // Instances to shuffle in place - Global device memory pointer.
                         // Col-major (InstanceSize, num_instances)
    const long* Ag_ptr,  // Physical to Virtual block assignments - Global device memory pointer.
                         // Col-major (VBlockSize, num_vblocks)
    const long offset,   // Physical block start offset
    const long num_instances,
    const int seed
)
{
}

/** Host-side helper functions */

// Get GPU program block size for CUDA arch, optimized for SM occupancy
// (not to mistake with pblock or vblock sizes, which refer to shuffle algorithm parameters)
template <int cuda_arch>
constexpr int get_block_size()
{
    return 64;
}

/** POV Shuffle -- CUDA host program */
template <typename DType, int PBlockSize, int VBlockSize, int InstanceSize>
void povs_cuda(
    DType* Xg_ptr, // Instances to shuffle in place - Global device memory pointer.
                   // Col-major (InstanceSize, num_instances) -- equivalent to Row-major (num_instances, InstanceSize)
    const long num_instances,

    const long* Oh_ptr, // Valid block start offsets with dim (num_offsets,) - Host memory pointer
    const long num_offsets,

    const int iterations,
    const int seed,
    const int8_t device_id
)
{
    // CUDA boilerplate
    cudaError_t cudaStatus = cudaSuccess;
    const int cuda_arch = get_device_cuda_arch(device_id);

    // Shuffle block arithmetic
    const long num_pblocks = div_round_up(num_instances, PBlockSize);
    const long num_vblocks = div_round_up(num_pblocks, VBlockSize);
    const long num_assignments = num_vblocks * VBlockSize; // Number of physical to virtual block assignments. This can be larger than
                                                           // num_pblocks, in which case some vblocks will get -1 (padding) assignments

    // Initialize random distributions
    std::mt19937 rng(seed);
    std::uniform_int_distribution offset_dist(0l, num_offsets - 1);

    // Allocate host pointers
    long* Ah_ptr = new long[num_assignments]; // Physical to Virtual block assignments - Host memory pointer with shape (num_assignments,)
    for (int i = 0; i < num_assignments; ++i)
        Ah_ptr[i] = i < num_pblocks ? i : -1; // Initialize with identity mapping for every valid pblock ID, padding with -1

    // Allocate device pointers
    int* Sg_ptr = nullptr;  // Virtual block random seeds - Global device memory pointer with shape (num_vblocks,)
    long* Ag_ptr = nullptr; // Physical to Virtual block assignments - Global device mem pointer. Col-major (VBlockSize, num_vblocks)
                            // Each column [:,j] holds the IDs of the ith pblocks assigned to the jth vblock
    CUDA_CHECK_STATUS(&cudaStatus, cleanup, cudaSetDevice(device_id));
    CUDA_CHECK_STATUS(&cudaStatus, cleanup, cudaMalloc(&Sg_ptr, sizeof(int) * num_vblocks));
    CUDA_CHECK_STATUS(&cudaStatus, cleanup, cudaMalloc(&Ag_ptr, sizeof(long) * num_assignments));

    // Iterate Kernel submissions
    for (int iter = 0; iter < iterations; ++iter) {
        const bool interrupted = DISPATCH_CUDA_ARCH(cuda_arch, [&] {
            bool lambda_completed = false; // Flag for detecting if the lambda execution was cut short by an error

            // GPU program block arithmetic
            const int num_blocks = num_pblocks;
            constexpr int block_size = get_block_size<kCudaArch>();

            // Sample an offset and shuffle the assignments
            const long offset = Oh_ptr[offset_dist(rng)];
            shuffle_array(Ah_ptr, num_pblocks, rng); // Shuffle the physical to virtual block mapping on the host
            CUDA_CHECK_STATUS(&cudaStatus, end_lambda, cudaMemcpy(Ag_ptr, Ah_ptr, sizeof(long) * num_assignments, cudaMemcpyHostToDevice));

            // Submit kernel
            (povs_kernel<DType, PBlockSize, VBlockSize, InstanceSize>
             <<<num_blocks, block_size>>>(Xg_ptr, Ag_ptr, offset, num_instances, seed));

            lambda_completed = true; // If we reached this point, the lambda executed successfully
        end_lambda:
            return !lambda_completed;
        });

        if (interrupted) goto cleanup;
        CUDA_CHECK_LAST_STATUS(&cudaStatus, cleanup);
    }

#ifdef STANDALONE_BUILD
    // Because the kernel submissions for all iterations are queued into the same stream, there is no need to synchronize inbetween.
    // For standalone builds, we synchronize once after all kernel submissions.
    // In a full build, the caller is responsible for the CUDA stream synchronization (usually managed by Torch)
    CUDA_CHECK_STATUS(&cudaStatus, cleanup, cudaDeviceSynchronize());
#endif

cleanup:
    free(Ah_ptr);
    cudaFree(Ag_ptr);
    cudaFree(Sg_ptr);
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
