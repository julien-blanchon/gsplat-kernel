# GSplat Kernel

2D Gaussian Splatting kernels for PyTorch, built with the HuggingFace kernel-builder system.

## Features

- **Simplified 2D Gaussian Splatting**: Minimal implementation with fixed parameters
- **No quantization**: Always uses full 32-bit precision
- **Always inverse scale**: Scale parameters are automatically inverted
- **No tiles**: Direct pixel-by-pixel rendering (simpler but slower)
- **Always top-K normalization**: Uses top-10 Gaussians per pixel with normalization
- **Zero external dependencies**: Pure CUDA + PyTorch implementation
- **Kernel-builder compliant**: Follows HuggingFace kernel-builder best practices

## Usage

```python
import torch
import gsplat

# Create test data
num_gaussians = 1000
means2d = torch.rand(num_gaussians, 2, device='cuda')  # [0,1] coordinates
scales2d = torch.rand(num_gaussians, 2, device='cuda') * 0.1 + 0.01
rotation = torch.rand(num_gaussians, device='cuda') * 2 * 3.14159
colors = torch.rand(num_gaussians, 3, device='cuda')  # RGB

# Render image
img = gsplat.render_gsplat(means2d, scales2d, rotation, colors, 512, 512)
print(f"Rendered image shape: {img.shape}")  # [512, 512, 3]
```

## Building

This kernel is built using the HuggingFace kernel-builder system:

```bash
# Install Nix and enable cache
nix run nixpkgs#cachix -- use huggingface

# Build the kernel
cd gsplat-kernel
nix run .#build-and-copy -L
```

## Testing

```bash
# Run tests
nix develop -L .#test
python -m pytest tests -v
```
