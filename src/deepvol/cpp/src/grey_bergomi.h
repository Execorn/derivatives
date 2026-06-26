#pragma once
#include <torch/extension.h>
#include <vector>

torch::Tensor mittag_leffler_cuda(
    torch::Tensor z,
    double beta,
    int max_iter,
    double tol
);

std::vector<torch::Tensor> generate_grey_paths_cuda(
    torch::Tensor params,
    int steps,
    int paths,
    double T,
    double dt
);
