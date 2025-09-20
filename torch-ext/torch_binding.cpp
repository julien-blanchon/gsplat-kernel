#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // 2D Gaussian projection operation
  ops.def("project_gaussians_2d(Tensor means2d, Tensor scales2d, Tensor rotation, int img_height, int img_width) -> (Tensor, Tensor)");
  ops.impl("project_gaussians_2d", torch::kCUDA, &project_gaussians_2d);

  // 2D Gaussian rasterization operation
  ops.def("rasterize_gaussians_2d(Tensor xys, Tensor conics, Tensor colors, int img_height, int img_width) -> Tensor");
  ops.impl("rasterize_gaussians_2d", torch::kCUDA, &rasterize_gaussians_2d);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
