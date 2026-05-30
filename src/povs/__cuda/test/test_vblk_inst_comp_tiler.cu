#include <cstdio>
#include <cute/tensor.hpp>

using namespace cute;

// ----------------------------------------------------------------------------
// Print coverage for one tiler variant
// ----------------------------------------------------------------------------
template <int PBlockSize, int VBlockSize, int BlockSize, class Tiler>
void print_partition(const char* name, Tiler tiler)
{
    constexpr int TotalSize = PBlockSize * VBlockSize;

    auto bIs_layout = make_layout(make_shape(Int<PBlockSize>{}, Int<VBlockSize>{}));
    auto bIi = make_identity_tensor(shape(bIs_layout));

    // coverage[flat] = number of times a thread/iter pair targets that flat index
    int coverage[TotalSize] = {};

    printf("\n=== Tiler %s  (PBlockSize=%d, VBlockSize=%d, BlockSize=%d) ===\n", name, PBlockSize, VBlockSize, BlockSize);

    for (int tid = 0; tid < BlockSize; ++tid) {
        auto tIs = local_partition(make_tensor<int>(bIs_layout), tiler, tid);
        auto tIi = local_partition(bIi, tiler, tid);

        const int iters = size(tIs);
        for (int i = 0; i < iters; ++i) {
            // Recover flat index this (tid, iter) pair maps to
            const auto coord = tIi[i];
            const int flat = bIs_layout(coord);
            const bool oob = (flat < 0 || flat >= TotalSize);

            printf(
                "  tid=%2d iter=%d  coord=(%d,%d)  flat=%d%s\n",
                tid,
                i,
                oob ? -1 : (int) get<0>(coord),
                oob ? -1 : (int) get<1>(coord),
                flat,
                oob ? "  *** OOB ***" : ""
            );

            if (!oob) coverage[flat]++;
        }
    }

    // Summary
    int perfect = 0, dup = 0, missed = 0;
    for (int f = 0; f < TotalSize; ++f) {
        if (coverage[f] == 1)
            perfect++;
        else if (coverage[f] > 1) {
            dup++;
            printf("  [coverage] flat=%d hit %d times\n", f, coverage[f]);
        } else {
            missed++;
            printf("  [coverage] flat=%d NEVER covered\n", f);
        }
    }
    printf("  Summary: %d perfect / %d duplicated / %d missed  (total=%d)\n", perfect, dup, missed, TotalSize);
}

// ----------------------------------------------------------------------------
// Host-side helper functions copied from povs.cu
// ----------------------------------------------------------------------------

// Check if GPU thread-block size would satisfy the kernel assertions
template <int BlockSize, int PBlockSize, int VBlockSize>
constexpr bool satisfies_kernel_block_size_reqs()
{
    // clang-format off
    return (BlockSize % VBlockSize == 0)
        && (PBlockSize < BlockSize || PBlockSize % BlockSize == 0)
        && (BlockSize < PBlockSize || BlockSize % PBlockSize == 0);
    // clang-format on
}

// Get GPU thread-block size
// Since the algorithm bottleneck for significant workloads is usually the shared memory, we usually prefer larger
// thread-block sizes, so that the few or single block(s) running in each SM use as many threads as they can.
// However, in order to avoid allocating unnecessary threads for small workloads we first try PBlockSize * VBlockSize.
// Limitation: returns -1 (causing a downstream kernel assertion) if no option satisfies the kernel requirements.
template <int CudaArch, int PBlockSize, int VBlockSize>
constexpr int get_block_size()
{
    // clang-format off
    constexpr int total = PBlockSize * VBlockSize;
    if constexpr (total <= 1024) return total >= 32 ? total : 32;
    if constexpr (satisfies_kernel_block_size_reqs<1024, PBlockSize, VBlockSize>()) return 1024;
    if constexpr (satisfies_kernel_block_size_reqs< 512, PBlockSize, VBlockSize>()) return  512;
    if constexpr (satisfies_kernel_block_size_reqs< 256, PBlockSize, VBlockSize>()) return  256;
    if constexpr (satisfies_kernel_block_size_reqs< 128, PBlockSize, VBlockSize>()) return  128;
    if constexpr (satisfies_kernel_block_size_reqs<  64, PBlockSize, VBlockSize>()) return   64;
    if constexpr (satisfies_kernel_block_size_reqs<  32, PBlockSize, VBlockSize>()) return   32;
    return -1;
    // clang-format on
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main()
{
    {
        // BlockSize is larger than PBlockSize, but divisible by it
        constexpr int PBlockSize = 8;
        constexpr int VBlockSize = 2;
        constexpr int BlockSize = get_block_size<0, PBlockSize, VBlockSize>();
        {
            auto tiler = make_layout(make_shape(Int<BlockSize / VBlockSize>{}, Int<VBlockSize>{}));
            print_partition<PBlockSize, VBlockSize, BlockSize>("D: make_layout(VBlockSize, BlockSize/VBlockSize)", tiler);
        }
    }

    return 0;
}
