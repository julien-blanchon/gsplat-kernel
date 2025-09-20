#pragma once

#include <cuda_runtime.h>
#include <cmath>

// Configuration constants
#define BLOCK_X 16              // Tile width in pixels
#define BLOCK_Y 16              // Tile height in pixels  
#define N_THREADS 256           // Threads for utility kernels
#define MAX_CHANNELS 12         // Maximum supported channels
#define TOP_K 10               // Top-K Gaussians per pixel
#define EPS 1e-7               // Epsilon for numerical stability

/**
 * Compute inverse covariance (conic) and radius from 2D covariance matrix
 * Returns false if covariance is degenerate (zero determinant)
 */
inline __device__ bool
compute_cov2d_bounds(const float3 cov2d, float3 &conic, float &radius) {
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

/**
 * Compute 2D covariance matrix directly from scale and rotation
 * Replaces matrix struct with direct computation: Σ = R * S * S^T * R^T
 */
inline __device__ float3 compute_cov2d_from_scale_rot(const float2 scale, const float rot) {
    // Inverse scale (always applied in simplified version)
    float sx = 1.0f / scale.x;
    float sy = 1.0f / scale.y;
    
    // Rotation matrix elements (column-major like GLM)
    float cosr = cosf(rot);
    float sinr = sinf(rot);
    
    // R = [[cosr, sinr], [-sinr, cosr]]
    // S = [[sx, 0], [0, sy]]
    // M = R * S = [[cosr*sx, sinr*sy], [-sinr*sx, cosr*sy]]
    float m00 = cosr * sx;
    float m01 = sinr * sy;
    float m10 = -sinr * sx;
    float m11 = cosr * sy;
    
    // Compute M * M^T directly
    // cov2d = [[m00*m00 + m01*m01, m00*m10 + m01*m11], 
    //          [m10*m00 + m11*m01, m10*m10 + m11*m11]]
    float cov_00 = m00 * m00 + m01 * m01;
    float cov_01 = m00 * m10 + m01 * m11;
    float cov_11 = m10 * m10 + m11 * m11;
    
    return make_float3(cov_00, cov_01, cov_11);
}
