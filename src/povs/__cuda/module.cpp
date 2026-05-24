#include <torch/extension.h>

#include "./binds/torch.h"

PYBIND11_MODULE(_cuda, m)
{
    m.def("torch_binding", &torch_povs, "PyTorch binding for POV Shuffle in CUDA");
}
