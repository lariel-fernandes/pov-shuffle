#ifndef TORCH_H
#define TORCH_H

#include <torch/torch.h>

void torch_povs(torch::Tensor X, torch::Tensor O, int iterations, int pblock_size, int vblock_size, int seed);

#endif
