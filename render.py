#!/usr/bin/env python3
# /// script
# dependencies = [
#   "torch>=2.5",
#   "numpy>=1.20.0",
#   "pillow>=10.0.0",
#   "kernels>=0.10.0",
# ]
# ///
"""
GSplat Kernel Demo with HuggingFace Kernels Package

Simple demo showing GSplat 2D Gaussian Splatting using the HuggingFace kernels
package for robust kernel loading.

Usage:
    uv run render_with_kernels.py checkpoint.g2d --height 512 --width 512
"""

import argparse
import os
import struct
from pathlib import Path
from time import perf_counter
from typing import Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image

# Import HuggingFace kernels package
from kernels.utils import get_local_kernel


def load_g2d_file(
    g2d_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load G2D binary file containing 2D Gaussian splatting data."""
    print(f"Loading G2D file from {g2d_path}")

    if not os.path.exists(g2d_path):
        raise FileNotFoundError(f"G2D file not found: {g2d_path}")

    with open(g2d_path, "rb") as f:
        # Parse 32-byte header
        magic = f.read(4)
        if magic != b"G2D\0":
            raise ValueError(f"Invalid G2D magic number: {magic}")

        version = struct.unpack("<I", f.read(4))[0]
        num_gaussians = struct.unpack("<I", f.read(4))[0]
        img_width = struct.unpack("<I", f.read(4))[0]
        img_height = struct.unpack("<I", f.read(4))[0]
        feat_channels = struct.unpack("<I", f.read(4))[0]
        quantization_bits = struct.unpack("<I", f.read(4))[0]
        flags = struct.unpack("<I", f.read(4))[0]

        print("G2D Header:")
        print(f"  Version: {version}")
        print(f"  Gaussians: {num_gaussians}")
        print(f"  Image size: {img_width}×{img_height}")
        print(f"  Feature channels: {feat_channels}")

        # Validate file size
        expected_data_size = num_gaussians * (2 + 2 + 1 + feat_channels) * 4
        expected_total_size = 32 + expected_data_size

        # Get actual file size
        current_pos = f.tell()
        f.seek(0, 2)  # Seek to end
        actual_size = f.tell()
        f.seek(current_pos)  # Seek back

        if actual_size != expected_total_size:
            raise ValueError(
                f"Invalid G2D file size. Expected {expected_total_size} bytes, got {actual_size} bytes"
            )

        # Read Gaussian data
        xy = np.frombuffer(f.read(num_gaussians * 2 * 4), dtype=np.float32).reshape(
            num_gaussians, 2
        )
        scale = np.frombuffer(f.read(num_gaussians * 2 * 4), dtype=np.float32).reshape(
            num_gaussians, 2
        )
        rot = np.frombuffer(f.read(num_gaussians * 1 * 4), dtype=np.float32).reshape(
            num_gaussians, 1
        )
        feat = np.frombuffer(
            f.read(num_gaussians * feat_channels * 4), dtype=np.float32
        ).reshape(num_gaussians, feat_channels)

        # Create metadata dictionary
        metadata = {
            "version": version,
            "num_gaussians": num_gaussians,
            "img_width": img_width,
            "img_height": img_height,
            "feat_channels": feat_channels,
        }

        print(f"Position range: [{xy.min():.6f}, {xy.max():.6f}]")
        print(f"Scale range: [{scale.min():.6f}, {scale.max():.6f}]")
        print(f"Rotation range: [{rot.min():.6f}, {rot.max():.6f}]")
        print(f"Feature range: [{feat.min():.6f}, {feat.max():.6f}]")

        return xy, scale, rot, feat, metadata


class GSplatKernelsRenderer:
    """Simple GSplat renderer using HuggingFace kernels package."""

    def __init__(self, g2d_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)

        # Load GSplat kernel using HuggingFace kernels package
        print("Loading GSplat kernel...")
        self.gsplat = get_local_kernel(Path(".") / "build", "gsplat")
        print("✅ GSplat kernel loaded!")

        # Load G2D file
        self._load_g2d(g2d_path)

    def _load_g2d(self, g2d_path: str):
        """Load G2D binary file."""
        xy, scale, rot, feat, metadata = load_g2d_file(g2d_path)

        # Store original image dimensions
        self.original_img_width = metadata["img_width"]
        self.original_img_height = metadata["img_height"]

        # Convert to PyTorch tensors
        self.xy = torch.from_numpy(xy.copy()).to(
            dtype=torch.float32, device=self.device
        )
        self.scale = torch.from_numpy(scale.copy()).to(
            dtype=torch.float32, device=self.device
        )
        self.rot = torch.from_numpy(rot.copy().squeeze()).to(
            dtype=torch.float32, device=self.device
        )
        self.feat = torch.from_numpy(feat.copy()).to(
            dtype=torch.float32, device=self.device
        )

        print(f"Loaded {self.xy.shape[0]} Gaussians with {self.feat.shape[1]} channels")
        print(f"Image size: {self.original_img_width}×{self.original_img_height}")

    def render(self, img_h: int, img_w: int) -> torch.Tensor:
        """Render the Gaussians to an image."""
        return self.gsplat.render_gsplat(
            self.xy, self.scale, self.rot, self.feat, img_h, img_w
        )


def save_image(image: torch.Tensor, save_path: str):
    """Save a tensor image to file."""
    # Convert to numpy and clamp to [0,1]
    image = torch.clamp(image, 0.0, 1.0).detach().cpu().numpy()

    # Convert to uint8 and save
    image = (255.0 * image).astype(np.uint8)
    Image.fromarray(image).save(save_path)
    print(f"Saved image to {save_path}")


def main():
    """Simple GSplat rendering demo."""
    parser = argparse.ArgumentParser(description="GSplat Demo with HuggingFace Kernels")
    parser.add_argument("g2d_file", help="Path to G2D file")
    parser.add_argument(
        "--output", "-o", default="output.png", help="Output image path"
    )
    parser.add_argument("--height", type=int, help="Output height (default: original)")
    parser.add_argument("--width", type=int, help="Output width (default: original)")
    parser.add_argument("--device", default="cuda:0", help="Device to use")

    args = parser.parse_args()

    print("=== GSplat HuggingFace Kernels Demo ===")

    # Load renderer
    renderer = GSplatKernelsRenderer(args.g2d_file, args.device)

    # Determine output dimensions
    if args.height and args.width:
        img_h, img_w = args.height, args.width
    elif args.height:
        img_h = args.height
        img_w = int(img_h * renderer.original_img_width / renderer.original_img_height)
    elif args.width:
        img_w = args.width
        img_h = int(img_w * renderer.original_img_height / renderer.original_img_width)
    else:
        img_h, img_w = renderer.original_img_height, renderer.original_img_width

    print(f"Rendering {img_w}×{img_h} image...")

    # Render image
    start_time = perf_counter()
    with torch.no_grad():
        image = renderer.render(img_h, img_w)
    render_time = perf_counter() - start_time

    print(f"Rendered in {render_time:.4f}s")

    # Save result
    save_image(image, args.output)
    print("✅ Done!")


if __name__ == "__main__":
    main()
