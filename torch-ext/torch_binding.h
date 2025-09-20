#pragma once

#include <torch/torch.h>

// 2D Gaussian projection function
// Projects 2D Gaussians from normalized coordinates to screen space
// Returns (xys, conics) where:
// - xys: projected centers in pixel coordinates [N, 2]
// - conics: inverse covariance matrices [N, 3] (upper triangular)
std::tuple<torch::Tensor, torch::Tensor> project_gaussians_2d(
    torch::Tensor &means2d,    // [N, 2] positions in [0,1] coordinates
    torch::Tensor &scales2d,   // [N, 2] scale factors (will be inverted)
    torch::Tensor &rotation,   // [N] rotation angles in radians
    int64_t img_height,        // output image height
    int64_t img_width          // output image width
);

// 2D Gaussian rasterization function  
// Renders 2D Gaussians to an image using top-K normalization
// Returns rendered image [H, W, C]
torch::Tensor rasterize_gaussians_2d(
    torch::Tensor &xys,        // [N, 2] projected centers in pixel coordinates
    torch::Tensor &conics,     // [N, 3] inverse covariance matrices
    torch::Tensor &colors,     // [N, C] colors/features
    int64_t img_height,        // output image height
    int64_t img_width          // output image width
);
