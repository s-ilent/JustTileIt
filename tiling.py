import math
import torch
from utils import _roll, _apply_blend_curve, _clamp01, _gaussian_blur_periodic

def make_seamless_substance(
    img: torch.Tensor,
    threshold: float = 0.5,
    smoothness: float = 0.2,
    contrast: float = 1.0,
    lr_source: float = 0.0,
    tb_source: float = 0.0,
    mask_invert: bool = False,
    blend_curve: str = "cosine",
) -> torch.Tensor:
    """Substance-style layered Make it Tile algorithm."""
    c, h, w = img.shape
    device = img.device
    dtype = img.dtype

    # Layers
    layer_shifted = _roll(img, dx=w // 2, dy=h // 2)

    dx_lr = int(lr_source * (w // 2))
    layer_lr = _roll(img, dx=dx_lr, dy=0)

    dy_tb = int(tb_source * (h // 2))
    layer_tb = _roll(img, dx=0, dy=dy_tb)

    layer_center = img

    # Coordinate grids
    ys = torch.arange(h, device=device, dtype=dtype).view(h, 1)
    xs = torch.arange(w, device=device, dtype=dtype).view(1, w)
    ny = (ys - h / 2.0) / (h / 2.0)
    nx = (xs - w / 2.0) / (w / 2.0)

    # Function to construct smooth patch masks
    def get_mask(d_field, th, sm, curve):
        sm = max(0.001, sm)
        r_in = th - sm / 2.0
        r_out = th + sm / 2.0
        t = (d_field - r_in) / (r_out - r_in)
        t = torch.clamp(t, 0.0, 1.0)
        t_curved = _apply_blend_curve(t, curve)
        return 1.0 - t_curved

    # Distance Fields
    d_center = torch.max(torch.abs(nx), torch.abs(ny))
    d_left = torch.sqrt((nx + 1.0) ** 2 + ny**2)
    d_right = torch.sqrt((nx - 1.0) ** 2 + ny**2)
    d_top = torch.sqrt(nx**2 + (ny + 1.0) ** 2)
    d_bottom = torch.sqrt(nx**2 + (ny - 1.0) ** 2)

    r_circle = threshold * 0.8
    m_center = get_mask(d_center, threshold, smoothness, blend_curve)
    m_left = get_mask(d_left, r_circle, smoothness, blend_curve)
    m_right = get_mask(d_right, r_circle, smoothness, blend_curve)
    m_top = get_mask(d_top, r_circle, smoothness, blend_curve)
    m_bottom = get_mask(d_bottom, r_circle, smoothness, blend_curve)

    if contrast != 1.0:

        def apply_contrast(m):
            return torch.clamp((m - 0.5) * contrast + 0.5, 0.0, 1.0)

        m_center = apply_contrast(m_center)
        m_left = apply_contrast(m_left)
        m_right = apply_contrast(m_right)
        m_top = apply_contrast(m_top)
        m_bottom = apply_contrast(m_bottom)

    m_lr = torch.max(m_left, m_right)
    m_tb = torch.max(m_top, m_bottom)

    if mask_invert:
        m_center = 1.0 - m_center
        m_lr = 1.0 - m_lr
        m_tb = 1.0 - m_tb

    # Reshape for broadcasting
    m_center = m_center.unsqueeze(0)
    m_lr = m_lr.unsqueeze(0)
    m_tb = m_tb.unsqueeze(0)

    # Blend Composite
    out = layer_shifted
    out = out * (1.0 - m_tb) + layer_tb * m_tb
    out = out * (1.0 - m_lr) + layer_lr * m_lr
    out = out * (1.0 - m_center) + layer_center * m_center

    return _clamp01(out)

def make_seamless_radial(
    img: torch.Tensor,
    inner_radius: float = 0.85,
    outer_radius: float = 1.0,
    scatter_strength: float = 0.0,
    blend_curve: str = "cosine",
) -> torch.Tensor:
    c, h, w = img.shape
    shifted = _roll(img, dx=w // 2, dy=h // 2)

    ys = torch.arange(h, device=img.device, dtype=img.dtype).view(h, 1)
    xs = torch.arange(w, device=img.device, dtype=img.dtype).view(1, w)
    ny = (ys - h / 2.0) / (h / 2.0)
    nx = (xs - w / 2.0) / (w / 2.0)

    dist = torch.sqrt(nx**2 + ny**2)

    if scatter_strength > 0:
        angle = torch.atan2(ny, nx)
        scatter = (
            0.08 * torch.sin(angle * 5)
            + 0.05 * torch.sin(angle * 11)
            + 0.03 * torch.sin(angle * 17)
        ) * scatter_strength
        dist = dist + scatter

    t = (dist - inner_radius) / max(0.001, outer_radius - inner_radius)
    t = torch.clamp(t, 0.0, 1.0)

    mask = _apply_blend_curve(t, blend_curve).unsqueeze(0)
    out = img * (1.0 - mask) + shifted * mask
    return _clamp01(out)

def make_seamless_half_shift(
    img: torch.Tensor,
    inner_radius: float = 0.7,
    outer_radius: float = 1.0,
    blend_curve: str = "linear",
    orientation: str = "both",
) -> torch.Tensor:
    c, h, w = img.shape
    dx = w // 2 if orientation in ("both", "horizontal") else 0
    dy = h // 2 if orientation in ("both", "vertical") else 0
    shifted = _roll(img, dx=dx, dy=dy)

    ys = torch.arange(h, device=img.device, dtype=img.dtype).view(h, 1)
    xs = torch.arange(w, device=img.device, dtype=img.dtype).view(1, w)
    ny = torch.abs((ys - h / 2.0) / (h / 2.0))
    nx = torch.abs((xs - w / 2.0) / (w / 2.0))

    if orientation == "horizontal":
        dist = nx.expand(h, w)
    elif orientation == "vertical":
        dist = ny.expand(h, w)
    else:
        dist = torch.max(nx, ny)

    t = (dist - inner_radius) / max(0.001, outer_radius - inner_radius)
    t = torch.clamp(t, 0.0, 1.0)

    mask = _apply_blend_curve(t, blend_curve).unsqueeze(0)
    out = img * (1.0 - mask) + shifted * mask
    return _clamp01(out)

def make_seamless_collage(
    img: torch.Tensor, blend_curve: str = "cosine"
) -> torch.Tensor:
    c, h, w = img.shape
    a = img
    b = torch.flip(img, dims=(2,))
    c1 = torch.flip(img, dims=(1,))
    d = torch.flip(img, dims=(1, 2))

    top = torch.cat([a, b], dim=2)
    bottom = torch.cat([c1, d], dim=2)
    big = torch.cat([top, bottom], dim=1)

    y0, x0 = h // 2, w // 2
    out = big[:, y0 : y0 + h, x0 : x0 + w]

    band = max(6, min(h, w) // 12)
    xs = torch.arange(w, device=img.device, dtype=img.dtype).view(1, 1, w)
    ys = torch.arange(h, device=img.device, dtype=img.dtype).view(1, h, 1)

    dx = torch.abs(xs - float(w // 2))
    dy = torch.abs(ys - float(h // 2))

    def band_mask(d_t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(1.0 - d_t / band, 0.0, 1.0)
        return _apply_blend_curve(t, blend_curve)

    mv = band_mask(dx)
    mh = band_mask(dy)
    mask = torch.clamp(mv + mh, 0.0, 1.0)

    blurred = _gaussian_blur_periodic(out, sigma=max(0.8, min(h, w) / 768.0))
    out = out * (1.0 - mask) + blurred * mask
    return _clamp01(out)
