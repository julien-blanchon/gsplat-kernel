// WGSL shaders reproducing CUDA kernels in gsplat_kernel (compat variant)
// - project: compute xys (pixel centers) and conics (inverse covariance)
// - rasterize: brute-force top-K blending per pixel

struct Params {
  img_width  : u32,
  img_height : u32,
  num_points : u32,
  channels   : u32
};

@group(0) @binding(0) var<uniform> params : Params;

// Flattened storage buffers to avoid alignment pitfalls
@group(0) @binding(1) var<storage, read> means2d : array<f32>;  // length = 2*N
@group(0) @binding(2) var<storage, read> scales2d : array<f32>; // length = 2*N
@group(0) @binding(3) var<storage, read> rotation : array<f32>; // length = 1*N

@group(0) @binding(4) var<storage, read_write> xys    : array<f32>; // length = 2*N
@group(0) @binding(5) var<storage, read_write> conics : array<f32>; // length = 3*N

// Used only by rasterize
@group(0) @binding(6) var<storage, read> colors : array<f32>;  // length = N*C
@group(0) @binding(7) var<storage, read_write> out_img : array<f32>; // length = H*W*C

const TOP_K : u32 = 10u;
const EPS   : f32 = 1e-7;

fn compute_cov2d_from_scale_rot(scale_x : f32, scale_y : f32, rot : f32) -> vec3<f32> {
  // Inverse scale (always applied in simplified version)
  let sx = 1.0 / scale_x;
  let sy = 1.0 / scale_y;

  // Rotation matrix elements (column-major like GLM)
  let cosr = cos(rot);
  let sinr = sin(rot);

  // R = [[cosr, sinr], [-sinr, cosr]]
  // S = [[sx, 0], [0, sy]]
  // M = R * S = [[cosr*sx, sinr*sy], [-sinr*sx, cosr*sy]]
  let m00 = cosr * sx;
  let m01 = sinr * sy;
  let m10 = -sinr * sx;
  let m11 = cosr * sy;

  // cov2d = M * M^T
  let cov_00 = m00 * m00 + m01 * m01;
  let cov_01 = m00 * m10 + m01 * m11;
  let cov_11 = m10 * m10 + m11 * m11;
  return vec3<f32>(cov_00, cov_01, cov_11);
}

@compute @workgroup_size(256)
fn project(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.num_points) {
    return;
  }

  let mx = means2d[2u * idx + 0u];
  let my = means2d[2u * idx + 1u];
  // Convert normalized [0,1] to pixel coordinates
  let center_x = mx * f32(params.img_width);
  let center_y = my * f32(params.img_height);

  let sx = scales2d[2u * idx + 0u];
  let sy = scales2d[2u * idx + 1u];
  let rot = rotation[idx];

  let cov = compute_cov2d_from_scale_rot(sx, sy, rot);
  let det = cov.x * cov.z - cov.y * cov.y;

  if (det == 0.0) {
    // Invalid marker: conics = 0, xys = (-1,-1)
    conics[3u * idx + 0u] = 0.0;
    conics[3u * idx + 1u] = 0.0;
    conics[3u * idx + 2u] = 0.0;
    xys[2u * idx + 0u] = -1.0;
    xys[2u * idx + 1u] = -1.0;
    return;
  }

  let inv_det = 1.0 / det;
  // Inverse covariance (conic)
  let c0 = cov.z * inv_det;
  let c1 = -cov.y * inv_det;
  let c2 = cov.x * inv_det;

  conics[3u * idx + 0u] = c0;
  conics[3u * idx + 1u] = c1;
  conics[3u * idx + 2u] = c2;
  xys[2u * idx + 0u] = center_x;
  xys[2u * idx + 1u] = center_y;
}

@compute @workgroup_size(16, 16, 1)
fn rasterize(@builtin(global_invocation_id) gid : vec3<u32>) {
  let j = gid.x; // x (column)
  let i = gid.y; // y (row)
  if (i >= params.img_height || j >= params.img_width) {
    return;
  }

  let px = f32(j);
  let py = f32(i);
  let img_w = params.img_width;
  let pix_id : u32 = i * img_w + j;

  // Top-K arrays
  var topk_id : array<i32, 10>;
  var topk_val : array<f32, 10>;
  for (var k : u32 = 0u; k < TOP_K; k = k + 1u) {
    topk_id[k] = -1;
    topk_val[k] = 0.0;
  }

  // Brute force over all Gaussians
  for (var g : u32 = 0u; g < params.num_points; g = g + 1u) {
    let con0 = conics[3u * g + 0u];
    let con1 = conics[3u * g + 1u];
    let con2 = conics[3u * g + 2u];
    let xyx = xys[2u * g + 0u];
    let xyy = xys[2u * g + 1u];

    // Skip invalid
    if (xyx < 0.0 || xyy < 0.0) { continue; }
    if (con0 == 0.0 && con1 == 0.0 && con2 == 0.0) { continue; }

    let dx = xyx - px;
    let dy = xyy - py;
    let sigma = 0.5 * (con0 * dx * dx + con2 * dy * dy) + con1 * dx * dy;
    // Portable validity check: skip if negative, NaN, or too large (proxy for inf)
    if (sigma < 0.0 || !(sigma == sigma) || abs(sigma) > 1e30) { continue; }
    let alpha = exp(-sigma);

    // Replace min in top-K
    var min_idx : i32 = -1;
    var min_val : f32 = 1e30;
    for (var k : u32 = 0u; k < TOP_K; k = k + 1u) {
      if (topk_id[k] < 0) {
        min_idx = i32(k);
        min_val = -1.0;
        break;
      } else if (topk_val[k] < min_val) {
        min_idx = i32(k);
        min_val = topk_val[k];
      }
    }
    if (alpha > min_val) {
      topk_id[u32(min_idx)] = i32(g);
      topk_val[u32(min_idx)] = alpha;
    }
  }

  // Normalization sum
  var sum_val : f32 = 0.0;
  for (var k : u32 = 0u; k < TOP_K; k = k + 1u) {
    if (topk_id[k] < 0) { continue; }
    sum_val = sum_val + topk_val[k];
  }

  // For each channel, accumulate and write
  for (var c : u32 = 0u; c < params.channels; c = c + 1u) {
    var acc : f32 = 0.0;
    for (var k : u32 = 0u; k < TOP_K; k = k + 1u) {
      let g = topk_id[k];
      if (g < 0) { continue; }
      let vis = topk_val[k] / (sum_val + EPS);
      let color = colors[u32(g) * params.channels + c];
      acc = acc + color * vis;
    }
    out_img[pix_id * params.channels + c] = acc;
  }
}


