# GSplat WebGPU Minimal

Minimal WebGPU reproduction of the CUDA/PyTorch Gaussian Splatting kernel in this repo.

- Project Gaussians: matches `project_gaussians_2d` from CUDA
- Rasterize with Top-K (K=10) and normalization: matches simplified CUDA kernel
- Inputs: `.g2d` files as described in `gsplat-kernel/render.py`

## Run

Serve this directory with any static server (WebGPU requires secure context):

```bash
# From repo root
cd gsplat-kernel/web
python3 -m http.server 8000
```

Open `http://localhost:8000` in a browser with WebGPU (Chrome 128+, Edge, or recent Chromium with `--enable-unsafe-webgpu`).

Load a `.g2d` file and click Render.

## Files

- `index.html`: Minimal UI
- `main.js`: WebGPU pipeline setup, dispatch, readback
- `g2d-worker.js`: Parses `.g2d` into typed arrays inside a Web Worker
- `shaders.wgsl`: WGSL compute shaders matching CUDA math in:
  - `gsplat-kernel/gsplat_kernel/gsplat_kernels.cu`
  - `gsplat-kernel/gsplat_kernel/gsplat_math.h`

## Notes

- Colors/features are assumed float32 in [0,1]. If your `.g2d` stores `uint8`, convert to float before writing the file (CUDA path also normalizes if `uint8`).
- Output buffer layout: `[H, W, C]` contiguous in row-major; UI converts to RGBA for display.
- Workgroup sizes mirror CUDA: `256` for project, `16x16` for raster.
- Top-K is K=10 with replacement via minimum score, matching CUDA.
