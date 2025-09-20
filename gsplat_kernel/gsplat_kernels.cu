#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cmath>

#include "gsplat_math.h"
#include "../torch-ext/torch_binding.h"

namespace cg = cooperative_groups;

// ============================================================================
// CUDA Kernels for 2D Gaussian Splatting Pipeline
// ============================================================================

/**
 * CUDA Kernel: Project 2D Gaussians to screen space
 * Always uses inverse scale, no quantization
 */
__global__ void project_gaussians_2d_kernel(
    const int num_points,
    const float2* __restrict__ means2d,
    const float2* __restrict__ scales2d,
    const float* __restrict__ rotation,
    const int img_height,
    const int img_width,
    float2* __restrict__ xys,
    float3* __restrict__ conics
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) {
        return;
    }

    float2 center = {means2d[idx].x * img_width, means2d[idx].y * img_height};

    // Compute 2D covariance matrix directly (no matrix struct needed)
    float3 cov2d = compute_cov2d_from_scale_rot(scales2d[idx], rotation[idx]);
    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok){
        // Set invalid marker
        conics[idx] = make_float3(0.0f, 0.0f, 0.0f);
        xys[idx] = make_float2(-1.0f, -1.0f);
        return;
    }
    conics[idx] = conic;
    xys[idx] = center;
}

/**
 * CUDA Kernel: Rasterize 2D Gaussians with Top-K normalization
 * Simplified version without tiling, always uses top-K normalization
 */
__global__ void rasterize_gaussians_2d_kernel(
    const int img_height,
    const int img_width,
    const unsigned channels,
    const unsigned num_points,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ colors,
    float* __restrict__ out_img
) {
    auto block = cg::this_thread_block();
    unsigned i = block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j = block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_width + j;
    bool inside = (i < img_height && j < img_width);

    if (!inside) return;

    float pix_out[MAX_CHANNELS] = {0.f};
    
    // Top-K Gaussian ids (always enabled)
    int32_t topk[TOP_K];
    float topk_vals[TOP_K] = {0.f};
    for (int k = 0; k < TOP_K; ++k)
        topk[k] = -1;

    // Process all Gaussians for this pixel
    for (int g = 0; g < num_points; ++g) {
        const float3 conic = conics[g];
        const float2 xy = xys[g];
        
        // Skip invalid Gaussians
        if (xy.x < 0.0f || xy.y < 0.0f) continue;
        if (conic.x == 0.0f && conic.y == 0.0f && conic.z == 0.0f) continue;
        
        const float2 delta = {xy.x - px, xy.y - py};
        const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
        
        if (sigma < 0.f || isnan(sigma) || isinf(sigma)) {
            continue;
        }
        
        const float alpha = __expf(-sigma);
        int32_t g_id = g;

        // Find the minimum value in topk for replacement
        int32_t min_topk_id = -1;
        float min_topk_val = 1e30f;
        for (int32_t k = 0; k < TOP_K; ++k) {
            if (topk[k] < 0) {
                min_topk_id = k;
                min_topk_val = -1.0f;
                break;
            } else if (topk_vals[k] < min_topk_val) {
                min_topk_id = k;
                min_topk_val = topk_vals[k];
            }
        }
        if (alpha > min_topk_val) {
            topk[min_topk_id] = g_id;
            topk_vals[min_topk_id] = alpha;
        }
    }

    // Always use top-K normalization
    for (int c = 0; c < channels; ++c) {
        float sum_val = 0.f;
        for (int k = 0; k < TOP_K; ++k) {
            if (topk[k] < 0) continue;
            sum_val += topk_vals[k];
        }
        for (int k = 0; k < TOP_K; ++k) {
            int32_t g = topk[k];
            if (g < 0) continue;
            float vis = topk_vals[k] / (sum_val + EPS);
            pix_out[c] += colors[g * channels + c] * vis;
        }
    }

    // Write output
    for (int c = 0; c < channels; ++c) {
        out_img[pix_id * channels + c] = pix_out[c];
    }
}

// ============================================================================
// Tensor wrapper functions (following kernel-builder patterns)
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> project_gaussians_2d(
    torch::Tensor &means2d,
    torch::Tensor &scales2d,
    torch::Tensor &rotation,
    int64_t img_height,
    int64_t img_width
) {
    const int num_points = means2d.size(0);
    
    // Input validation using PyTorch tensor checks
    TORCH_CHECK(means2d.is_cuda(), "means2d must be a CUDA tensor");
    TORCH_CHECK(scales2d.is_cuda(), "scales2d must be a CUDA tensor");
    TORCH_CHECK(rotation.is_cuda(), "rotation must be a CUDA tensor");
    TORCH_CHECK(means2d.size(0) == num_points, "means2d size mismatch");
    TORCH_CHECK(scales2d.size(0) == num_points, "scales2d size mismatch");
    TORCH_CHECK(rotation.size(0) == num_points, "rotation size mismatch");
    TORCH_CHECK(means2d.size(1) == 2, "means2d must have 2 dimensions");
    TORCH_CHECK(scales2d.size(1) == 2, "scales2d must have 2 dimensions");
    
    // Create output tensors using PyTorch tensor creation
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(means2d.device())
        .requires_grad(false);
        
    torch::Tensor xys = torch::empty({num_points, 2}, options);
    torch::Tensor conics = torch::empty({num_points, 3}, options);

    if (num_points == 0) {
        return std::make_tuple(xys, conics);
    }

    const int blocks = (num_points + N_THREADS - 1) / N_THREADS;
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel with proper CUDA stream handling
    project_gaussians_2d_kernel<<<blocks, N_THREADS, 0, stream>>>(
        num_points,
        (float2*)means2d.data_ptr<float>(),
        (float2*)scales2d.data_ptr<float>(),
        rotation.data_ptr<float>(),
        img_height,
        img_width,
        (float2*)xys.data_ptr<float>(),
        (float3*)conics.data_ptr<float>()
    );

    return std::make_tuple(xys, conics);
}

torch::Tensor rasterize_gaussians_2d(
    torch::Tensor &xys,
    torch::Tensor &conics,
    torch::Tensor &colors,
    int64_t img_height,
    int64_t img_width
) {
    const int num_points = xys.size(0);
    
    // Input validation using PyTorch tensor checks
    TORCH_CHECK(xys.is_cuda(), "xys must be a CUDA tensor");
    TORCH_CHECK(conics.is_cuda(), "conics must be a CUDA tensor");
    TORCH_CHECK(colors.is_cuda(), "colors must be a CUDA tensor");
    TORCH_CHECK(xys.size(0) == num_points, "xys size mismatch");
    TORCH_CHECK(conics.size(0) == num_points, "conics size mismatch");
    TORCH_CHECK(colors.size(0) == num_points, "colors size mismatch");
    TORCH_CHECK(xys.size(1) == 2, "xys must have 2 dimensions");
    TORCH_CHECK(conics.size(1) == 3, "conics must have 3 dimensions");
    
    const int channels = colors.size(1);

    // Create output tensor using PyTorch with same options as input
    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, 
        colors.options()
    );

    if (num_points == 0) {
        return out_img;
    }

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((img_width + block.x - 1) / block.x,
                    (img_height + block.y - 1) / block.y);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(xys));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel with proper CUDA stream handling
    rasterize_gaussians_2d_kernel<<<grid, block, 0, stream>>>(
        img_height,
        img_width,
        channels,
        num_points,
        (float2*)xys.data_ptr<float>(),
        (float3*)conics.data_ptr<float>(),
        colors.data_ptr<float>(),
        out_img.data_ptr<float>()
    );

    return out_img;
}
