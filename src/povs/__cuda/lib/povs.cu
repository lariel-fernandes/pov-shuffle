#include "povs.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <random>

#include "./utils.h"

// Closed-open interval for sampling a numerical seed from a distribution.
// Aligned with the respective constants in the Python module `povs.constants` for reproducibility.
static constexpr int MIN_SEED = 0;
static constexpr int MAX_SEED = 1000;

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

// template <typename DType, int CudaArch, int BitWidth>
// constexpr auto __device__ get_copy_atom()
// {
//     using namespace cute;
//     if constexpr (CudaArch >= 800) return Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint_bit_t<BitWidth>>, DType>{};
//     return Copy_Atom<UniversalCopy<uint_bit_t<BitWidth>>, DType>{};
// }
//
// template <typename DType, int CudaArch, int BlockSize>
// auto __device__ get_tiled_copy()
// {
//     using namespace cute;
//     constexpr int BitWidth = 128;
//     constexpr int CopyWidth = BitWidth / 8 / sizeof(DType);
//     constexpr auto CopyAtom = get_copy_atom<DType, CudaArch, BitWidth>();
//     auto value_layout = make_layout(make_shape(Int<CopyWidth>{}));
//     // TODO: consider using a flat shape when the instance size is 1
//     auto thread_layout = make_layout(make_shape(Int<BlockSize / CopyWidth>{}, Int<CopyWidth>{}));
//     return make_tiled_copy(CopyAtom, thread_layout, value_layout);
// }

/** POV Shuffle - Single iteration Kernel
 *
 *  Each GPU program block maps to a single virtual Block through `blockIdx.x`
 *  Each virtual block maps to `VBlockSize` physical blocks through the `V` mapping
 */
template <typename DType, int PBlockSize, int VBlockSize, int InstanceSize, int BlockSize>
__global__ void povs_kernel(
    const DType* Xg_ptr, // Instances to shuffle in place - Global device memory pointer.
                         // Col-major (InstanceSize, num_instances)
    const long* Ag_ptr,  // Physical to Virtual block assignments - Global device memory pointer.
                         // Col-major (VBlockSize, num_vblocks)
    const int* Sg_ptr,   // Random generator seeds - Global device memory pointer with shape (num_vblocks,)
    const long offset,   // Physical block start offset
    const long num_instances
)
{
    using namespace cute;
    const uint vblock_id = blockIdx.x;
    const int seed = Sg_ptr[vblock_id];
    // auto tiled_copy = get_tiled_copy<DType, get_cuda_arch(), BlockSize>();

    // Block-owned assignment tensors in global device mem and register mem, respectively, both with shape (VBlockSize,).
    // Each position `i` contains the index of the assigned physical block, or -1 if padding.
    auto bAg = make_tensor(Ag_ptr + (VBlockSize * vblock_id), make_layout(make_shape(Int<VBlockSize>{})));
    auto bAr = make_fragment_like(bAg);
    for (int i = 0; i < VBlockSize; ++i)
        bAr[i] = bAg[i];

    // Block-owned data instances tensor in shared SM memory, Col-major with shape (InstanceSize, PBlockSize, VBlockSize).
    __shared__ DType bXs_ptr[size(InstanceSize * PBlockSize * VBlockSize)];
    auto bXs = make_tensor(make_smem_ptr(bXs_ptr), make_layout(make_shape(Int<InstanceSize>{}, Int<PBlockSize>{}, Int<VBlockSize>{})));

    // Block-owned permutation index tensor in shared SM memory with shape (PBlockSize * VBlockSize,).
    // Finding the value `j` at position `i` means that bXs[:,j] should travel to bXg[:,i] as the result of shuffling.
    __shared__ int bIs_ptr[size(PBlockSize * VBlockSize)];
    auto bIs = make_tensor(make_smem_ptr(bIs_ptr), make_layout(make_shape(Int<PBlockSize * VBlockSize>{})));

    // Define predicate of block-owned data instances
    auto bXp = lazy::transform(make_identity_tensor(bXs.shape()), [&](auto coord) {
        const auto iid_in_pblk = get<1>(coord);           // instance id within pblock
        const auto pbid_in_vblk = get<2>(coord);          // pblock id within vblock
        const auto pbid = bAr[pbid_in_vblk];              // Global pblock id that was assigned to this vblock
        const auto iid = pbid * PBlockSize + iid_in_pblk; // Global instance ID that is part of that pblock
        if (pbid == -1) return false;                     // Check for assignment padding
        if (iid >= num_instances) return false;           // Check for instance over-indexing in the last pblock
        return true;
    });

    // For each pblock id within this vblock
    for (int pbid_in_vblk = 0; pbid_in_vblk < VBlockSize; ++pbid_in_vblk) {
        const auto pbid = bAr[pbid_in_vblk]; // Global pblock id that was assigned to this vblock
        if (pbid == -1) continue;            // Skip for assignment padding

        // Pblock start and wrap around arithmetic
        const auto iid_start = pbid * PBlockSize;                     // Global instance ID that is the start of the assigned pblock
        auto oiid_start = iid_start + offset;                         // Offset global instance ID that is the start of the assigned pblock
        if (oiid_start >= num_instances) oiid_start -= num_instances; // Wrap around to compensate the offset
        const auto tail_length = max(0l, static_cast<long>(PBlockSize) - (num_instances - oiid_start)); // Wrapped around pblock instances
        const auto head_length = PBlockSize - tail_length; // Number of instances in the pblock that are before the wrap around

        // Block-owned data instances tensor in global device mem, local partitioned for that pblock, with shape (InstanceSize, PBlockSize).
        auto bXg_pblk = make_tensor(Xg_ptr + (oiid_start * InstanceSize), make_layout(make_shape(Int<InstanceSize>{}, Int<PBlockSize>{})));
        auto bXg_pblk_pred = lazy::transform(make_identity_tensor(bXg_pblk.shape()), [&](auto coord) {
            const auto iid_in_pblk = get<1>(coord);       // instance id within pblock
            const auto iid = iid_start + iid_in_pblk;     // Global instance ID that is part of that pblock
            if (iid >= num_instances) return false;       // Check for instance over-indexing in the last pblock
            if (iid_in_pblk >= head_length) return false; // Check if instance is part of the pblock tail that goes over the wrap around
            return true;
        });

        // Tail of bXg_pblk that goes over the wrap around
        auto bXg_pblk_tail =
            make_tensor(Xg_ptr - (head_length * InstanceSize), make_layout(make_shape(Int<InstanceSize>{}, Int<PBlockSize>{})));
        auto bXg_pblk_tail_pred = lazy::transform(make_identity_tensor(bXg_pblk_tail.shape()), [&](auto coord) {
            const auto iid_in_pblk = get<1>(coord);      // instance id within pblock
            const auto iid = iid_start + iid_in_pblk;    // Global instance ID that is part of that pblock
            if (iid >= num_instances) return false;      // Check for instance over-indexing in the last pblock
            if (iid_in_pblk < head_length) return false; // Check if instance is part of the pblock head that is before the wrap around
            return true;
        });

        // Copy bXg_pblk to bXs[:,:head_length,pbid_in_vblk] where predicate bXg_pblk_pred allows, vectorizing by a thread layout
        // Copy bXg_pblk_tail to bXs[:,head_length:,pbid_in_vblk] where predicate bXg_pblk_tail_pred allows, vectorizing by a thread layout
    }

    // Data instances tensor in global device memory, Col-major with shape (InstanceSize, num_instances)
    auto Xg = make_tensor(Xg_ptr, make_layout(make_shape(Int<InstanceSize>{}, num_instances)));

    // Partition bIs into tIs, initialize with identity using address diff from pointer start, let threadId.x == 0 shuffle bIs in place with
    // seed (possibly pass the tensor's pointer to __host__ __device__ shuffle) Use the same ptr diff arith to map writing bXs[:,[i]] to
    // bXg_pblock[:,offset_wrap_around(i)]
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
    std::uniform_int_distribution seed_dist(MIN_SEED, MAX_SEED - 1);

    // Allocate host pointers
    long* Sh_ptr = new long[num_vblocks];     // Random generator seeds - Host memory pointer with shape (num_vblocks,)
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

        // Dispatch lambda and capture interruption flag
        const bool interrupted = DISPATCH_CUDA_ARCH(cuda_arch, [&] {
            bool lambda_completed = false; // Flag for detecting if the lambda execution was cut short by an error

            // GPU program block arithmetic
            const int num_blocks = num_pblocks;
            constexpr int BlockSize = get_block_size<kCudaArch>();

            long offset = 0l; // Declare variable before the CUDA status check macros
                              // to avoid `error: transfer of control bypasses initialization of ...`

            // WARNING: The sequence of rng usages in the next 3 code blocks must match the one in the numpy
            //          implementation for reproducibility (shuffling, then seed sampling, then offset sampling)

            // Shuffle the pblock to vblock assignments
            shuffle_array(Ah_ptr, num_pblocks, rng);
            CUDA_CHECK_STATUS(&cudaStatus, end_lambda, cudaMemcpy(Ag_ptr, Ah_ptr, sizeof(long) * num_assignments, cudaMemcpyHostToDevice));

            // Sample random seeds for every vblock
            for (int i = 0; i < num_vblocks; ++i)
                Sh_ptr[i] = seed_dist(rng);
            CUDA_CHECK_STATUS(&cudaStatus, end_lambda, cudaMemcpy(Sg_ptr, Sh_ptr, sizeof(int) * num_vblocks, cudaMemcpyHostToDevice));

            // Sample a pblock start offset
            offset = Oh_ptr[offset_dist(rng)];

            // Submit kernel
            (povs_kernel<DType, PBlockSize, VBlockSize, InstanceSize, BlockSize>
             <<<num_blocks, BlockSize>>>(Xg_ptr, Ag_ptr, Sg_ptr, offset, num_instances));

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
    free(Sh_ptr);
    cudaFree(Ag_ptr);
    cudaFree(Sg_ptr);
    if (cudaStatus != cudaSuccess) exit(cudaStatus);
}

#ifdef STANDALONE_BUILD
// Main function for trying out the standalone CUDA program (not included in package build)
int main()
{
    using DType = float;
    constexpr int PBlockSize = 8;
    constexpr int VBlockSize = 2;
    constexpr int InstanceSize = 1;

    constexpr long num_instances = 64;
    constexpr int iterations = 1;
    constexpr int seed = 42;
    constexpr int8_t device_id = 0;

    auto* Xh_ptr = new DType[num_instances];
    for (long i = 0; i < num_instances; ++i)
        Xh_ptr[i] = static_cast<DType>(i);

    constexpr int num_offsets = 1;
    constexpr long Oh_ptr[num_offsets] = {0};

    DType* Xg_ptr = nullptr;
    cudaError_t status = cudaMalloc(&Xg_ptr, sizeof(DType) * num_instances);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        delete[] Xh_ptr;
        return 1;
    }
    cudaMemcpy(Xg_ptr, Xh_ptr, sizeof(DType) * num_instances, cudaMemcpyHostToDevice);

    povs_cuda<DType, PBlockSize, VBlockSize, InstanceSize>(Xg_ptr, num_instances, Oh_ptr, num_offsets, iterations, seed, device_id);
    cudaMemcpy(Xh_ptr, Xg_ptr, sizeof(DType) * num_instances, cudaMemcpyDeviceToHost);

    constexpr long print_limit = 32;
    printf("Output (first %ld): ", print_limit);
    for (long i = 0; i < print_limit; ++i)
        printf("%.0f ", Xh_ptr[i]);
    printf("\n");

    cudaFree(Xg_ptr);
    delete[] Xh_ptr;
    return 0;
}
#endif

#if __has_include("povs_cuda_template_instances.gen.inc")
#include "povs_cuda_template_instances.gen.inc"
#else
INSTANTIATE_POVS_CUDA_ALL_TYPES(8, 2, 1)
#endif
