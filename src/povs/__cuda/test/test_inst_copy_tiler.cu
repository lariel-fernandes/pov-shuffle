#include <cstdio>
#include <cute/tensor.hpp>

using namespace cute;

// Host copy of get_copy_width from povs.cu (no __device__ qualifier)
template <typename DType, int InstanceSize>
constexpr int get_copy_width()
{
    constexpr int InstanceBits = InstanceSize * static_cast<int>(sizeof(DType)) * 8;
    // clang-format off
    constexpr int BitWidth = InstanceBits % 128 == 0 ? 128 : InstanceBits % 64 == 0 ? 64 : InstanceBits % 32 == 0 ? 32 : 16;
    // clang-format on
    return BitWidth / 8 / static_cast<int>(sizeof(DType));
}

// Host copy of get_inst_copy_tiler from povs.cu (no __device__ qualifier)
template <typename DType, int CudaArch, int InstanceSize>
auto get_inst_copy_tiler()
{
    constexpr int CopyWidth = get_copy_width<DType, InstanceSize>();
    constexpr int BitWidth = CopyWidth * 8 * static_cast<int>(sizeof(DType));
    constexpr auto CopyAtom = Copy_Atom<UniversalCopy<uint_bit_t<BitWidth>>, DType>{};
    auto value_layout = make_layout(make_shape(Int<CopyWidth>{}));
    auto thread_layout = make_layout(make_shape(Int<1>{}));
    return make_tiled_copy(CopyAtom, thread_layout, value_layout);
}

// ----------------------------------------------------------------------------
// Print coverage for inst_copy_tiler over instance_layout
// ----------------------------------------------------------------------------
// inst_copy_tiler has thread_layout of size 1, so only thread 0 is used.
// Coverage iterates over copy-atom iterations rather than threads.
// ----------------------------------------------------------------------------
template <typename DType, int CudaArch, int InstanceSize>
void print_partition(const char* dtype_name)
{
    constexpr int CopyWidth = get_copy_width<DType, InstanceSize>();

    printf("\n=== inst_copy_tiler  (dtype=%s, InstanceSize=%d, CopyWidth=%d) ===\n", dtype_name, InstanceSize, CopyWidth);

    auto instance_layout = make_layout(make_shape(Int<InstanceSize>{}));
    auto bIi = make_identity_tensor(shape(instance_layout));

    int coverage[InstanceSize] = {};

    // thread_layout has size 1, so only thread 0 participates
    auto tiler = get_inst_copy_tiler<DType, CudaArch, InstanceSize>();
    auto tIi = tiler.get_slice(0).partition_S(bIi);

    // tIi is linearized as (v, invoke) column-major: first CopyWidth indices = invoke 0, next = invoke 1, etc.
    constexpr int NumInvocations = InstanceSize / CopyWidth;
    for (int invoke = 0; invoke < NumInvocations; ++invoke) {
        printf("  invoke=%d  elements=[", invoke);
        for (int v = 0; v < CopyWidth; ++v) {
            const auto coord = tIi[invoke * CopyWidth + v];
            const int flat = instance_layout(coord);
            const bool oob = (flat < 0 || flat >= InstanceSize);

            if (v > 0) printf(", ");
            printf("%d%s", flat, oob ? "!" : "");

            if (!oob) coverage[flat]++;
        }
        printf("]\n");
    }

    int perfect = 0, dup = 0, missed = 0;
    for (int f = 0; f < InstanceSize; ++f) {
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
    printf("  Summary: %d perfect / %d duplicated / %d missed  (total=%d)\n", perfect, dup, missed, InstanceSize);
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main()
{
    // float, InstanceSize=1: BitWidth=32, CopyWidth=1 -> 1 iteration of 1 element
    print_partition<float, 700, 1>("float");

    // float, InstanceSize=4: BitWidth=128, CopyWidth=4 -> 1 iteration of 4 elements
    print_partition<float, 700, 4>("float");

    // float, InstanceSize=8: BitWidth=128, CopyWidth=4 -> 2 iterations of 4 elements each
    print_partition<float, 700, 8>("float");

    // int16_t, InstanceSize=8: BitWidth=128, CopyWidth=8 -> 1 iteration of 8 elements
    print_partition<int16_t, 700, 8>("int16_t");

    return 0;
}
