#include "povs.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <random>

#include "./utils.h"

/** Constants */
#pragma region Constants

// Closed-open interval for sampling a numerical seed from a distribution.
// Aligned with the respective constants in the Python module `povs.constants` for reproducibility.
static constexpr int MIN_SEED = 1;
static constexpr int MAX_SEED = 1000;

#pragma endregion

/** Macros */
#pragma region Macros

// clang-format off
#define DISPATCH_CUDA_ARCH(cuda_arch, lambda) \
    [&]() {                                   \
        if      (cuda_arch >= 900) { constexpr int kCudaArch = 900; return lambda(); } \
        else if (cuda_arch >= 800) { constexpr int kCudaArch = 800; return lambda(); } \
        else                       { constexpr int kCudaArch = 700; return lambda(); } \
    }()
// clang-format on

#pragma endregion

/** Device-side helper functions */
#pragma region Device helper functions

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
    if constexpr (CudaArch >= 800 && BitWidth >= 32)
        return Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint_bit_t<BitWidth>>, DType>{};
    else
        return Copy_Atom<UniversalCopy<uint_bit_t<BitWidth>>, DType>{};
}

template <typename DType, int CudaArch, int BlockSize, int InstanceSize>
auto __device__ get_tiled_copy()
{
    using namespace cute;
    constexpr int InstanceBits = InstanceSize * static_cast<int>(sizeof(DType)) * 8;
    constexpr int BitWidth = InstanceBits >= 128 ? 128 : InstanceBits >= 64 ? 64 : InstanceBits >= 32 ? 32 : 16;
    constexpr int CopyWidth = BitWidth / 8 / sizeof(DType);
    constexpr auto CopyAtom = get_copy_atom<DType, CudaArch, BitWidth>();
    auto value_layout = make_layout(make_shape(Int<CopyWidth>{}));
    auto thread_layout = make_layout(make_shape(Int<BlockSize / CopyWidth>{}, Int<CopyWidth>{}));
    return make_tiled_copy(CopyAtom, thread_layout, value_layout);
}

#pragma endregion

/** CUDA Kernel */
#pragma region CUDA Kernel

/** POV Shuffle - Single iteration Kernel
 *
 *  Each GPU program block maps to a single virtual Block through `blockIdx.x`
 *  Each virtual block maps to `VBlockSize` physical blocks through the `A` mapping
 */
template <typename DType, int PBlockSize, int VBlockSize, int InstanceSize, int BlockSize>
__global__ void povs_kernel(
    DType* Xg_ptr,      // Instances to shuffle in place - Global device memory pointer.
                        // Col-major (InstanceSize, num_instances)
    const long* Ag_ptr, // Physical to Virtual block assignments - Global device memory pointer.
                        // Col-major (VBlockSize, num_vblocks)
    const int* Sg_ptr,  // Random generator seeds - Global device memory pointer with shape (num_vblocks,)
    const long offset,  // Physical block start offset
    const long num_instances
)
{
    using namespace cute;
    const uint vblock_id = blockIdx.x;
    const int seed = Sg_ptr[vblock_id];

    // Define tilers for distributing copy throughput and computation across threads
#pragma region Tilers
    auto copy_tiler = get_tiled_copy<DType, get_cuda_arch(), BlockSize, InstanceSize>();
    auto thr_copy_tiler = copy_tiler.get_slice(threadIdx.x);
    auto thr_comp_tiler = make_layout(make_shape(Int<VBlockSize>{}, Int<BlockSize / VBlockSize>{}));
#pragma endregion

    // Define reusable layouts
#pragma region Layouts
    // Hierarchy: Virtual block contains physical blocks, which contain instances, which are treated as contiguous 1-D vectors, regardless
    // of their original shape. The shuffle operates on the instances, keeping each instance intact but changing their order.
    // Data layouts index every element within every instance, while instance layouts index each instance as a whole, ignoring InstanceSize.
    auto instance_layout = make_layout(make_shape(Int<InstanceSize>{}));                                        // Single instance layout
    auto pblk_data_layout = make_layout(make_shape(Int<InstanceSize>{}, Int<PBlockSize>{}));                    // PBlock data layout
    auto vblk_data_layout = make_layout(make_shape(Int<InstanceSize>{}, Int<PBlockSize>{}, Int<VBlockSize>{})); // VBlock data layout
    auto vblk_inst_layout = make_layout(make_shape(Int<PBlockSize>{}, Int<VBlockSize>{}));                      // VBlock instances layout
#pragma endregion

#pragma region Tensors
    // Define tensors for virtual block data
    __shared__ DType bXs_ptr[size(vblk_data_layout)];                 // Block-owned shared memory pointer
    auto bXs = make_tensor(make_smem_ptr(bXs_ptr), vblk_data_layout); // Block-owned tensor in SM shared memory

    // Define tensors for the assignments of global physical block IDs to the current virtual block
    auto bAg = make_tensor(Ag_ptr + vblock_id * VBlockSize, make_shape(Int<VBlockSize>{})); // Block-owned in global device mem
    auto bAr = make_fragment_like(bAg);                                                     // Block-owned in registers

    // Define tensors for the instances permutation index
    __shared__ int bPs_ptr[size(vblk_inst_layout)];                   // Block-owned shared memory pointer
    auto bPs = make_tensor(make_smem_ptr(bPs_ptr), vblk_inst_layout); // Block-owned tensor in SM shared memory
    auto tPs = local_partition(bPs, thr_comp_tiler, threadIdx.x);     // Thread-owned tensor in SM shared memory
#pragma endregion

#pragma region Lazy transforms
    // Define layout identities for coordinate mapping
    auto pblk_data_ident = make_identity_tensor(shape(pblk_data_layout));
    auto vblk_inst_ident = make_identity_tensor(shape(vblk_inst_layout));
    auto thr_vblk_inst_ident = local_partition(vblk_inst_ident, thr_comp_tiler, threadIdx.x);

    // Define predicates for boundary checking
    auto vblk_inst_pred = lazy::transform(vblk_inst_ident, [&](auto coord) {
        const auto iid_in_pblk = get<0>(coord);           // Instance id within physical block
        const auto pbid_in_vblk = get<1>(coord);          // Physical block id within virtual block
        const auto pbid = bAr[pbid_in_vblk];              // Global id of assigned physical block
        const auto iid = PBlockSize * pbid + iid_in_pblk; // Global instance ID that is part of that physical block
        if (pbid == -1) return false;                     // Guard against assignment padding
        if (iid >= num_instances) return false;           // Guard against instance over-indexing in the last physical block
        return true;
    });
    auto thr_vblk_inst_pred = local_partition(vblk_inst_pred, thr_comp_tiler, threadIdx.x);
#pragma endregion

    // ### Stage 0 ###
    // Copy this vblock's assignment map to registers for fast access and sort the assignments to ensure gap-free predicates
    {
        copy(bAg, bAr); // Copy from global device mem to registers

        // Sort bAr ascending
        {
            // Treating -1 (padding) as +infinity so it sinks to the end.
            // Guarantees the truncated last pblock (highest ID, if assigned here) lands at the last valid slot,
            // keeping the permutation index gap-free for the Fisher-Yates shuffle in Stage 2.
            for (int i = 1; i < VBlockSize; ++i) {
                long key = bAr[i];
                int j = i - 1;
                while (j >= 0 && key != -1 && (bAr[j] == -1 || bAr[j] > key)) {
                    bAr[j + 1] = bAr[j];
                    --j;
                }
                bAr[j + 1] = key;
            }
        }
    }

    // ### Stage 1 ###
    // Vectorized predicated copy of assigned physical blocks from global device mem to SM shared memory
    {
        // For each pblock id within this vblock
        for (int pbid_in_vblk = 0; pbid_in_vblk < VBlockSize; ++pbid_in_vblk) {
            const auto pbid = bAr[pbid_in_vblk]; // Global pblock id that was assigned to this vblock
            if (pbid == -1) continue;            // Skip for assignment padding

            // Pblock start and wrap around arithmetic
            const auto iid_start = pbid * PBlockSize; // Global instance ID that is the start of the assigned pblock
            auto oiid_start = iid_start + offset;     // Offset global instance ID that is the start of the assigned pblock
            if (oiid_start >= num_instances) oiid_start -= num_instances; // Wrap around to compensate the offset
            const auto tail_length = max(0l, static_cast<long>(PBlockSize) - (num_instances - oiid_start)); // Wrapped around pblk instances
            const auto head_length = PBlockSize - tail_length; // Number of instances in the pblock that are before the wrap around

            // Base predicate to prevent instance over-indexing in the last physical block
            auto pblock_pred = lazy::transform(pblk_data_ident, [&](auto coord) {
                const auto iid_in_pblk = get<1>(coord);   // instance id within pblock
                const auto iid = iid_start + iid_in_pblk; // Global instance ID that is part of that pblock
                return iid < num_instances;
            });

            // Block-owned data instances tensors in global device mem, local partitioned for that pblock, with shape (InstanceSize,
            // PBlockSize) For the pblock that goes over the wrap around, the head and tail point to the instances before and after the
            // wrap, respectively.
            auto bXg_pblk_head = make_tensor(Xg_ptr + (oiid_start * InstanceSize), pblk_data_layout);
            auto bXg_pblk_tail = make_tensor(Xg_ptr - (head_length * InstanceSize), pblk_data_layout);

            // Define predicates for the head and tail, guarding against over-indexing in each case
            auto bXg_pblk_head_pred = lazy::transform(pblk_data_ident, [&](auto coord) {
                const auto iid_in_pblk = get<1>(coord);                 // instance id within pblock
                return pblock_pred[coord] && iid_in_pblk < head_length; // Guard against over-indexing and ignore tail instances
            });
            auto bXg_pblk_tail_pred = lazy::transform(pblk_data_ident, [&](auto coord) {
                const auto iid_in_pblk = get<1>(coord);                  // instance id within pblock
                return pblock_pred[coord] && iid_in_pblk >= head_length; // Guard against over-indexing and ignore head instances
            });

            // Vectorized, predicated copy head and tail from global device memory to shared SM memory
            // clang-format off
            copy_if(
                copy_tiler,
                bXg_pblk_head_pred,
                thr_copy_tiler.partition_S(bXg_pblk_head),
                thr_copy_tiler.partition_D(bXs(_, _, pbid_in_vblk))
            );
            copy_if(
                copy_tiler,
                bXg_pblk_tail_pred,
                thr_copy_tiler.partition_S(bXg_pblk_tail),
                thr_copy_tiler.partition_D(bXs(_, _, pbid_in_vblk))
            );
        }
        cp_async_fence(); cp_async_wait<0>(); __syncthreads(); // Sync on shared mem writes (possibly with streaming depending on copy atom)
        // clang-format on
    }

    // ### Stage 2 ###
    // Generate permutation index in shared SM memory (populate with identity in parallel, then run a single-threaded shuffle)
    {
        // Parallel initialize permutation index
        for (int i = 0; i < size(tPs); ++i) {
            if (!thr_vblk_inst_pred[i]) continue;                 // Guard against vblock instances predicate
            auto vblk_inst_coord = thr_vblk_inst_ident[i];        // vblock instances identity -> coordinate in vblock instances layout
            auto iid_in_vblk = vblk_inst_layout(vblk_inst_coord); // layout(coordinate) -> flat instance id within vblock
            tPs[i] = iid_in_vblk;
        }

        __syncthreads(); // Sync on shared mem writes

        // Find the last valid index (inclusive) in the permutation index tensor, which determines the boundary for the Fisher-Yates shuffle
        int shuffle_boundary = 0;
        {
            for (int i = size(vblk_inst_pred) - 1; i >= 0; --i) {
                if (vblk_inst_pred[i]) {
                    shuffle_boundary = i;
                    break;
                }
            }
        }

        // Fisher-Yates shuffle of the permutation index in shared SM memory
        {
            // clang-format off
            if (threadIdx.x == 0) {
                auto rng = static_cast<uint32_t>(seed);
                for (int i = 0; i <= shuffle_boundary; ++i) {
                    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;                                                 // xorshift32
                    const int j = i + static_cast<int>((static_cast<uint64_t>(rng) * (shuffle_boundary - i + 1)) >> 32); // Lemire fast range
                    int temp = bPs[i];
                    bPs[i] = bPs[j];
                    bPs[j] = temp;
                }
            }
            // clang-format on
        }

        __syncthreads(); // Have other threads wait for the thread 0 to finish writing the permutated indices in shared memory
    }

    // ### Stage 3 ###
    // Parallel write permutated instances from shared SM memory back to mapped positions in global device memory
    {
        for (int i = 0; i < size(tPs); ++i) {
            if (!thr_vblk_inst_pred[i]) continue;                // Guard against vblock instances predicate
            const auto iid_in_vblk = tPs[i];                     // Instance ID within the flattened virtual block
            const auto src_coord = vblk_inst_ident[iid_in_vblk]; // vblk inst ident at flat inst ID -> src coord in vblk inst layout
            const auto dst_coord = thr_vblk_inst_ident[i];       // tiled vblk inst ident at iter i -> dst coord in vblk inst layout

            // Destination pointer arithmetic
            const int iid_in_pblk = get<0>(dst_coord);        // Instance ID within physical block
            const int pbid_in_vblk = get<1>(dst_coord);       // Physical block ID within virtual block
            const int pbid = bAr[pbid_in_vblk];               // Global physical block ID
            const auto iid = pbid * PBlockSize + iid_in_pblk; // Global target instance ID
            auto oiid = iid + offset;                         // Offset global target instance ID
            if (oiid >= num_instances) oiid -= num_instances; // Wrap around to compensate the offset

            // Thread-owned tensor pointing to the target instance in global device memory
            auto tXg_instance = make_tensor(Xg_ptr + (oiid * InstanceSize), instance_layout);

            // Copy whole instance from shared memory to global device memory
            copy(bXs(prepend(src_coord, _)), tXg_instance);
            // TODO: consider using a single threaded vectorized copy atom for 128-bit buffer copying of large instances
        }
    }
}

#pragma endregion

/** Host-side helper functions */
#pragma region Host helper functions

// Get GPU program block size for CUDA arch, optimized for SM occupancy
// (not to mistake with pblock or vblock sizes, which refer to shuffle algorithm parameters)
template <int cuda_arch>
constexpr int get_block_size()
{
    return 64;
}

#pragma endregion

/** CUDA host program */
#pragma region CUDA host program

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
    int* Sh_ptr = new int[num_vblocks];       // Random generator seeds - Host memory pointer with shape (num_vblocks,)
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
            const int num_blocks = num_vblocks;
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
    delete[] Ah_ptr;
    delete[] Sh_ptr;
    cudaFree(Ag_ptr);
    cudaFree(Sg_ptr);
    if (cudaStatus != cudaSuccess) exit(cudaStatus);
}

#pragma endregion

/** Standalone runner (no lib interface) */
#pragma region Standalone runner

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
    auto* Yh_ptr = new bool[num_instances];
    for (long i = 0; i < num_instances; ++i) {
        Xh_ptr[i] = static_cast<DType>(i);
        Yh_ptr[i] = false;
    }

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

    constexpr long print_limit = 64;
    printf("Output (first %ld): ", print_limit);
    for (long i = 0; i < num_instances; ++i) {
        if (i < print_limit) printf("%.0f ", Xh_ptr[i]);
        Yh_ptr[static_cast<long>(Xh_ptr[i])] = true; // Mark the instance ID as seen in the output
    }
    printf("\n");

    int seen = 0;
    for (long i = 0; i < num_instances; ++i)
        if (Yh_ptr[i]) seen++;
    printf("Seen %d / %ld\n", seen, num_instances);

    cudaFree(Xg_ptr);
    delete[] Xh_ptr;
    return 0;
}
#endif

#pragma endregion

/** Template instancing */
#pragma region Template instancing

#if __has_include("povs_cuda_template_instances.gen.inc")
#include "povs_cuda_template_instances.gen.inc"
#else
INSTANTIATE_POVS_CUDA_ALL_TYPES(8, 2, 1)
#endif

#pragma endregion
