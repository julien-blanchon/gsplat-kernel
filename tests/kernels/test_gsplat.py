import pytest
import torch
import gsplat

DTYPES = [torch.float32]  # Start with float32 only for simplicity
NUM_GAUSSIANS = [10, 100, 1000]  # Different numbers of Gaussians
IMG_SIZES = [(64, 64), (128, 128), (256, 256)]  # Different image sizes
CHANNELS = [3]  # RGB channels
SEEDS = [0, 42]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]


@pytest.mark.parametrize("num_gaussians", NUM_GAUSSIANS)
@pytest.mark.parametrize("img_size", IMG_SIZES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_project_gaussians_2d(
    num_gaussians: int,
    img_size: tuple,
    channels: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)

    img_height, img_width = img_size

    # Create test tensors
    means2d = torch.rand(num_gaussians, 2, dtype=dtype)  # [0,1] coordinates
    scales2d = (
        torch.rand(num_gaussians, 2, dtype=dtype) * 0.1 + 0.01
    )  # Small positive scales
    rotation = torch.rand(num_gaussians, dtype=dtype) * 2 * 3.14159  # [0, 2π] radians

    # Test projection
    xys, conics = gsplat.project_gaussians_2d(
        means2d, scales2d, rotation, img_height, img_width
    )

    # Validate output shapes
    assert xys.shape == (num_gaussians, 2), (
        f"Expected xys shape {(num_gaussians, 2)}, got {xys.shape}"
    )
    assert conics.shape == (num_gaussians, 3), (
        f"Expected conics shape {(num_gaussians, 3)}, got {conics.shape}"
    )

    # Validate output ranges
    assert xys[:, 0].min() >= -100, (
        "X coordinates too negative"
    )  # Allow some margin for edge cases
    assert xys[:, 0].max() <= img_width + 100, "X coordinates too large"
    assert xys[:, 1].min() >= -100, "Y coordinates too negative"
    assert xys[:, 1].max() <= img_height + 100, "Y coordinates too large"

    # Validate that conics are finite
    assert torch.all(torch.isfinite(conics) | (conics == 0)), (
        "Conics should be finite or zero (for invalid Gaussians)"
    )


@pytest.mark.parametrize("num_gaussians", NUM_GAUSSIANS)
@pytest.mark.parametrize("img_size", IMG_SIZES)
@pytest.mark.parametrize("channels", CHANNELS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rasterize_gaussians_2d(
    num_gaussians: int,
    img_size: tuple,
    channels: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.manual_seed(seed)
    torch.set_default_device(device)

    img_height, img_width = img_size

    # Create test tensors
    xys = torch.rand(num_gaussians, 2, dtype=dtype) * torch.tensor(
        [img_width, img_height], dtype=dtype
    )
    conics = (
        torch.rand(num_gaussians, 3, dtype=dtype) * 0.1 + 0.001
    )  # Small positive values
    colors = torch.rand(num_gaussians, channels, dtype=dtype)

    # Test rasterization
    out_img = gsplat.rasterize_gaussians_2d(xys, conics, colors, img_height, img_width)

    # Validate output shape
    expected_shape = (img_height, img_width, channels)
    assert out_img.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {out_img.shape}"
    )

    # Validate output range [0, 1] approximately
    assert out_img.min() >= -0.1, "Output values too negative"  # Allow small margin
    assert out_img.max() <= 1.1, "Output values too large"  # Allow small margin

    # Validate that output is finite
    assert torch.all(torch.isfinite(out_img)), "Output should be finite"


@pytest.mark.parametrize("num_gaussians", [100])
@pytest.mark.parametrize("img_size", [(128, 128)])
@pytest.mark.parametrize("channels", [3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_render_gsplat_complete(
    num_gaussians: int,
    img_size: tuple,
    channels: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test the complete rendering pipeline"""
    torch.manual_seed(seed)
    torch.set_default_device(device)

    img_height, img_width = img_size

    # Create test tensors
    means2d = torch.rand(num_gaussians, 2, dtype=dtype)  # [0,1] coordinates
    scales2d = (
        torch.rand(num_gaussians, 2, dtype=dtype) * 0.1 + 0.01
    )  # Small positive scales
    rotation = torch.rand(num_gaussians, dtype=dtype) * 2 * 3.14159  # [0, 2π] radians
    colors = torch.rand(num_gaussians, channels, dtype=dtype)

    # Test complete rendering pipeline
    out_img = gsplat.render_gsplat(
        means2d, scales2d, rotation, colors, img_height, img_width
    )

    # Validate output
    expected_shape = (img_height, img_width, channels)
    assert out_img.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {out_img.shape}"
    )
    assert torch.all(torch.isfinite(out_img)), "Output should be finite"
    assert out_img.min() >= -0.1, "Output values too negative"
    assert out_img.max() <= 1.1, "Output values too large"
