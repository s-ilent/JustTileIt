import math
import torch
import torch.nn.functional as F

def get_device():
    """
    Selects the most performant available hardware accelerator (CUDA or MPS) or falls back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _clamp01(x: torch.Tensor) -> torch.Tensor:
    """
    Constrains tensor element values to remain within the [0.0, 1.0] intensity range.
    """
    return torch.clamp(x, 0.0, 1.0)

def _to_chw(img_hwc: torch.Tensor) -> torch.Tensor:
    """
    Reorders tensor dimensions from [Height, Width, Channels] to [Channels, Height, Width]
    for PyTorch processing requirements.
    """
    return img_hwc.permute(2, 0, 1)

def _to_hwc(img_chw: torch.Tensor) -> torch.Tensor:
    """
    Reorders tensor dimensions from [Channels, Height, Width] to [Height, Width, Channels]
    for standard image visualization or saving.
    """
    return img_chw.permute(1, 2, 0)

def _roll(img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """
    Wraps the image spatially by a specified pixel offset in the horizontal and vertical directions.
    """
    return torch.roll(img, shifts=(dy, dx), dims=(1, 2))

def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    """
    Generates a normalized 1D kernel to define the spread of a blurring operation.
    """
    if sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / torch.sum(k)
    return k

def _gaussian_blur_periodic(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Smooths image data by convolving with a Gaussian kernel, utilizing circular padding
    to maintain seamless transitions across image boundaries.
    """
    if sigma <= 0:
        return img
    k = _gaussian_kernel_1d(sigma, img.device, img.dtype)
    radius = (k.numel() - 1) // 2

    # Horizontal pass
    w_h = k.view(1, 1, 1, -1).repeat(img.shape[0], 1, 1, 1)
    x = img.unsqueeze(0)
    x = F.pad(x, (radius, radius, 0, 0), mode="circular")
    x = F.conv2d(x, w_h, groups=img.shape[0])

    # Vertical pass
    w_v = k.view(1, 1, -1, 1).repeat(img.shape[0], 1, 1, 1)
    x = F.pad(x, (0, 0, radius, radius), mode="circular")
    x = F.conv2d(x, w_v, groups=img.shape[0])

    return x.squeeze(0)

def _pre_average_lighting(
    img: torch.Tensor, intensity: int, radius: int
) -> torch.Tensor:
    """
    Balances luminance by localizing brightness variations and applying a global 
    adjustment factor based on a blurred luminance mask.
    """
    intensity = int(max(0, min(100, intensity)))
    if intensity == 0:
        return img

    radius = int(max(1, min(20, radius)))
    c, h, w = img.shape
    if c < 3:
        return img

    rgb = img[:3]
    y = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).unsqueeze(0)
    sigma = (min(h, w) / 256.0) * float(radius)
    sigma = max(0.5, float(sigma))

    y_blur = _gaussian_blur_periodic(y, sigma=sigma)
    mean_y = torch.mean(y)

    eps = 1e-6
    scale = (mean_y / (y_blur + eps)).clamp(0.25, 4.0)

    corrected = _clamp01(rgb * scale)
    a = intensity / 100.0
    out_rgb = rgb * (1.0 - a) + corrected * a

    out = img.clone()
    out[:3] = out_rgb
    return out

def _apply_blend_curve(t: torch.Tensor, curve: str) -> torch.Tensor:
    """
    Maps a normalized linear input [0.0, 1.0] to an output based on the specified easing function.
    """
    curve = curve.lower()
    if curve == "linear":
        return t
    elif curve == "cosine":
        return 0.5 - 0.5 * torch.cos(math.pi * t)
    elif curve == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    elif curve == "smootherstep":
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    elif curve == "quadratic":
        return torch.where(t < 0.5, 2.0 * t * t, 1.0 - 2.0 * (1.0 - t) ** 2)
    elif curve == "cubic":
        return torch.where(t < 0.5, 4.0 * t * t * t, 1.0 - 4.0 * (1.0 - t) ** 3)
    return 0.5 - 0.5 * torch.cos(math.pi * t)
