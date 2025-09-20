import torch
from typing import Tuple, Optional

try:
    from ._ops import ops
except ImportError as e:
    # Fallback for local development
    try:
        import _gsplat
        ops = torch.ops._gsplat
    except ImportError:
        raise e


def project_gaussians_2d(
    means2d: torch.Tensor,     # [N, 2] positions in [0,1] coordinates
    scales2d: torch.Tensor,    # [N, 2] scale factors (will be inverted)
    rotation: torch.Tensor,    # [N] rotation angles in radians
    img_height: int,           # output image height
    img_width: int             # output image width
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 2D Gaussians from normalized coordinates to screen space.
    
    Always uses inverse scale, no quantization, no tile computation.
    
    Args:
        means2d: Gaussian centers in [0,1] coordinates
        scales2d: Gaussian scales (will be inverted automatically)
        rotation: Rotation angles in radians
        img_height: Output image height
        img_width: Output image width
        
    Returns:
        Tuple of (xys, conics) where:
        - xys: Projected centers in pixel coordinates [N, 2]
        - conics: Inverse covariance matrices [N, 3] (upper triangular)
    """
    return ops.project_gaussians_2d(means2d, scales2d, rotation, img_height, img_width)


def rasterize_gaussians_2d(
    xys: torch.Tensor,         # [N, 2] projected centers in pixel coordinates
    conics: torch.Tensor,      # [N, 3] inverse covariance matrices
    colors: torch.Tensor,      # [N, C] colors/features
    img_height: int,           # output image height
    img_width: int             # output image width
) -> torch.Tensor:
    """
    Rasterize 2D Gaussians to an image.
    
    Always uses top-K normalization, no tiles, no quantization.
    
    Args:
        xys: Projected Gaussian centers in pixel coordinates
        conics: Inverse covariance matrices (upper triangular)
        colors: Gaussian colors/features
        img_height: Output image height
        img_width: Output image width
        
    Returns:
        Rendered image tensor [H, W, C]
    """
    if colors.dtype == torch.uint8:
        # Ensure colors are float [0,1]
        colors = colors.float() / 255

    return ops.rasterize_gaussians_2d(xys, conics, colors, img_height, img_width)


def render_gsplat(
    means2d: torch.Tensor,     # [N, 2] positions in [0,1] coordinates
    scales2d: torch.Tensor,    # [N, 2] scale factors
    rotation: torch.Tensor,    # [N] rotation angles in radians
    colors: torch.Tensor,      # [N, C] colors/features
    img_height: int,           # output image height
    img_width: int             # output image width
) -> torch.Tensor:
    """
    Complete 2D Gaussian Splatting rendering pipeline.
    
    Combines projection and rasterization in a single convenient function.
    
    Args:
        means2d: Gaussian centers in [0,1] coordinates
        scales2d: Gaussian scales (will be inverted automatically)
        rotation: Rotation angles in radians
        colors: Gaussian colors/features
        img_height: Output image height
        img_width: Output image width
        
    Returns:
        Rendered image tensor [H, W, C]
    """
    # Step 1: Project Gaussians to screen space
    xys, conics = project_gaussians_2d(means2d, scales2d, rotation, img_height, img_width)
    
    # Step 2: Rasterize to image
    return rasterize_gaussians_2d(xys, conics, colors, img_height, img_width)
