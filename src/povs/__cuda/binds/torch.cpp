#include "torch.h"

#include <torch/torch.h>

#include "../lib/povs.h"

// clang-format off
#ifndef PBLOCK_SIZE_CASES
#define PBLOCK_SIZE_CASES(lambda) \
    case 8: { constexpr int PBLOCK_SIZE = 8; return lambda(); }
#endif

#ifndef VBLOCK_SIZE_CASES
#define VBLOCK_SIZE_CASES(lambda) \
    case 2: { constexpr int VBLOCK_SIZE = 2; return lambda(); }
#endif
// clang-format on

#define PBLOCK_SIZE_DISPATCH(x, lambda)                            \
    [&]() {                                                                    \
        switch (x) {                                                   \
            PBLOCK_SIZE_CASES(lambda)                                      \
            default: TORCH_CHECK(false, "Unsupported PBLOCK_SIZE: ", x); \
        }                                                                      \
    }()

#define VBLOCK_SIZE_DISPATCH(x, lambda)                            \
[&]() {                                                                    \
switch (x) {                                                   \
VBLOCK_SIZE_CASES(lambda)                                      \
default: TORCH_CHECK(false, "Unsupported VBLOCK_SIZE: ", x); \
}                                                                      \
}()

torch::Tensor torch_povs(
    torch::Tensor X
)
{

}
