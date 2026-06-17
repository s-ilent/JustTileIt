import math
import os
import re
import sys
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Import Qt with fallbacks for maximum compatibility
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    PYQT_VERSION = 6
except ImportError:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        PYQT_VERSION = 6
    except ImportError:
        from PyQt5 import QtCore, QtGui, QtWidgets
        from PyQt5.QtWidgets import QFileDialog, QMessageBox

        PYQT_VERSION = 5


# Auto-detect computation device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# -------------------------------------------------------------
# Core Image Processing Utilities (PyTorch)
# -------------------------------------------------------------


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _to_chw(img_hwc: torch.Tensor) -> torch.Tensor:
    return img_hwc.permute(2, 0, 1)


def _to_hwc(img_chw: torch.Tensor) -> torch.Tensor:
    return img_chw.permute(1, 2, 0)


def _roll(img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    return torch.roll(img, shifts=(dy, dx), dims=(1, 2))


def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / torch.sum(k)
    return k


def _gaussian_blur_periodic(img: torch.Tensor, sigma: float) -> torch.Tensor:
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


# -------------------------------------------------------------
# Tiling Algorithms Implementation
# -------------------------------------------------------------


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


# -------------------------------------------------------------
# Seam Marker & Preview Rendering
# -------------------------------------------------------------


def _parse_color(color_str: str) -> Optional[Tuple[float, float, float, float]]:
    if not color_str or not color_str.strip():
        return None
    color_str = color_str.strip().lower()
    named_colors = {
        "red": (1.0, 0.0, 0.0, 1.0),
        "green": (0.0, 1.0, 0.0, 1.0),
        "blue": (0.0, 0.0, 1.0, 1.0),
        "white": (1.0, 1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0, 1.0),
        "cyan": (0.0, 1.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0, 1.0),
    }
    if color_str in named_colors:
        return named_colors[color_str]
    if color_str.startswith("#"):
        hex_str = color_str[1:]
        try:
            if len(hex_str) == 3:
                return (
                    int(hex_str[0], 16) / 15.0,
                    int(hex_str[1], 16) / 15.0,
                    int(hex_str[2], 16) / 15.0,
                    1.0,
                )
            elif len(hex_str) == 6:
                return (
                    int(hex_str[0:2], 16) / 255.0,
                    int(hex_str[2:4], 16) / 255.0,
                    int(hex_str[4:6], 16) / 255.0,
                    1.0,
                )
        except ValueError:
            return None
    return None


def _mark_seams(
    tile: torch.Tensor, single_w: int, single_h: int, tx: int, ty: int, color_str: str
) -> torch.Tensor:
    rgba = _parse_color(color_str)
    if rgba is None:
        return tile
    r, g, b, a = rgba
    out = tile.clone()
    c, h, w = out.shape
    thickness = max(1, int(round(min(single_w, single_h) * 0.01)))
    half_t = thickness // 2

    def paint_v(x_center: int):
        x0 = max(0, x_center - half_t)
        x1 = min(w, x_center - half_t + thickness)
        if x1 > x0:
            out[0, :, x0:x1] = out[0, :, x0:x1] * (1 - a) + r * a
            if c > 1:
                out[1, :, x0:x1] = out[1, :, x0:x1] * (1 - a) + g * a
            if c > 2:
                out[2, :, x0:x1] = out[2, :, x0:x1] * (1 - a) + b * a

    def paint_h(y_center: int):
        y0 = max(0, y_center - half_t)
        y1 = min(h, y_center - half_t + thickness)
        if y1 > y0:
            out[0, y0:y1, :] = out[0, y0:y1, :] * (1 - a) + r * a
            if c > 1:
                out[1, y0:y1, :] = out[1, y0:y1, :] * (1 - a) + g * a
            if c > 2:
                out[2, y0:y1, :] = out[2, y0:y1, :] * (1 - a) + b * a

    for i in range(1, tx):
        paint_v(i * single_w)
    for j in range(1, ty):
        paint_h(j * single_h)
    return out


# -------------------------------------------------------------
# PyQt6 GUI Implementation
# -------------------------------------------------------------


class ImagePreviewLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(200, 200)
        self._pixmap = None

    def set_pixmap(self, pixmap: QtGui.QPixmap):
        self._pixmap = pixmap
        self.update_view()

    def update_view(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_view()


class CollapsibleGroupBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)

        self.toggle_btn = QtWidgets.QPushButton(f"▼ {title}")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(True)
        self.toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; padding: 4px; }"
        )
        self.toggle_btn.clicked.connect(self.toggle)

        self.content_area = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(8, 4, 8, 4)

        self.layout.addWidget(self.toggle_btn)
        self.layout.addWidget(self.content_area)

    def toggle(self):
        if self.toggle_btn.isChecked():
            self.toggle_btn.setText(self.toggle_btn.text().replace("▶", "▼"))
            self.content_area.show()
        else:
            self.toggle_btn.setText(self.toggle_btn.text().replace("▼", "▶"))
            self.content_area.hide()

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Substance Make-It-Tile & Seamless Compositor")
        self.resize(1100, 750)

        self.original_pil: Optional[Image.Image] = None
        self.processed_tensor: Optional[torch.Tensor] = None

        # Coalescing processing updates to preserve real-time UI feel
        self.update_timer = QtCore.QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(80)  # ms delay
        self.update_timer.timeout.connect(self.process_image)

        # Allow image files to be dropped into the window
        self.setAcceptDrops(True)

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        # ------------------ LEFT SIDE PANEL (Controls) ------------------
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(340)

        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(12)

        # File Operations
        file_box = CollapsibleGroupBox("File Settings")

        self.load_btn = QtWidgets.QPushButton("Open Image...")
        self.load_btn.clicked.connect(self.open_file)

        file_buttons_layout = QtWidgets.QHBoxLayout()
        file_buttons_layout.setContentsMargins(0, 0, 0, 0)
        file_buttons_layout.setSpacing(4)

        self.save_btn = QtWidgets.QPushButton("Export Image...")
        self.save_btn.clicked.connect(self.save_file)
        self.save_btn.setEnabled(False)

        self.copy_btn = QtWidgets.QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        self.copy_btn.setEnabled(False)

        file_buttons_layout.addWidget(self.save_btn)
        file_buttons_layout.addWidget(self.copy_btn)

        file_buttons_widget = QtWidgets.QWidget()
        file_buttons_widget.setLayout(file_buttons_layout)

        file_box.addWidget(self.load_btn)
        file_box.addWidget(file_buttons_widget)
        panel_layout.addWidget(file_box)

        # Tiling Algorithm Picker
        method_box = QtWidgets.QGroupBox("Tiling Algorithm")
        method_layout = QtWidgets.QVBoxLayout(method_box)
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(
            [
                "Make it Tile (Substance Sampler)",
                "Radial Mask Offset & Blend",
                "Half Shift (Axis Aligned)",
                "Mirrored Collage Grid",
            ]
        )
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.method_combo)
        panel_layout.addWidget(method_box)

        # Dynamic Controls Group (Stacked Layout based on method selection)
        self.stack_box = QtWidgets.QGroupBox("Algorithm Options")
        self.stack_layout = QtWidgets.QStackedLayout(self.stack_box)

        # Build options for each stack
        self.build_substance_controls()
        self.build_radial_controls()
        self.build_half_shift_controls()
        self.build_mirrored_controls()

        panel_layout.addWidget(self.stack_box)

        # Pre-Processor Settings
        pre_box = CollapsibleGroupBox("Pre-Processor Lighting / Crop")

        crop_layout = QtWidgets.QGridLayout()
        self.crop_t = QtWidgets.QSpinBox()
        self.crop_t.setRange(0, 4096)
        self.crop_t.setPrefix("Top: ")
        self.crop_b = QtWidgets.QSpinBox()
        self.crop_b.setRange(0, 4096)
        self.crop_b.setPrefix("Bottom: ")
        self.crop_l = QtWidgets.QSpinBox()
        self.crop_l.setRange(0, 4096)
        self.crop_l.setPrefix("Left: ")
        self.crop_r = QtWidgets.QSpinBox()
        self.crop_r.setRange(0, 4096)
        self.crop_r.setPrefix("Right: ")
        crop_layout.addWidget(self.crop_t, 0, 1)
        crop_layout.addWidget(self.crop_l, 1, 0)
        crop_layout.addWidget(self.crop_r, 1, 2)
        crop_layout.addWidget(self.crop_b, 2, 1)
        for w in (self.crop_t, self.crop_b, self.crop_l, self.crop_r):
            w.valueChanged.connect(self.trigger_update)

        pre_box.addWidget(QtWidgets.QLabel("Pre-Crop Borders:"))
        crop_container = QtWidgets.QWidget()
        crop_container.setLayout(crop_layout)
        pre_box.addWidget(crop_container)

        self.eq_intensity = self.create_slider(
            0, 100, 65, "Lighting Correction Intensity: {}%"
        )
        self.eq_radius = self.create_slider(1, 20, 5, "Lighting Correction Radius: {}")
        pre_box.addWidget(self.eq_intensity[0])
        pre_box.addWidget(self.eq_radius[0])
        panel_layout.addWidget(pre_box)

        # Preview Configuration
        preview_cfg = CollapsibleGroupBox("Preview Options")
        self.grid_x = QtWidgets.QSpinBox()
        self.grid_x.setRange(1, 10)
        self.grid_x.setValue(3)
        self.grid_x.setPrefix("Columns: ")
        self.grid_y = QtWidgets.QSpinBox()
        self.grid_y.setRange(1, 10)
        self.grid_y.setValue(3)
        self.grid_y.setPrefix("Rows: ")
        self.grid_x.valueChanged.connect(self.trigger_update)
        self.grid_y.valueChanged.connect(self.trigger_update)

        self.seam_color = QtWidgets.QLineEdit()
        self.seam_color.setPlaceholderText("e.g. red, #FF0000 (empty to hide)")
        self.seam_color.textChanged.connect(self.trigger_update)

        preview_cfg.addWidget(QtWidgets.QLabel("Tiled Preview Grid:"))
        grid_container = QtWidgets.QHBoxLayout()
        grid_container.addWidget(self.grid_x)
        grid_container.addWidget(self.grid_y)
        grid_w = QtWidgets.QWidget()
        grid_w.setLayout(grid_container)
        preview_cfg.addWidget(grid_w)
        preview_cfg.addWidget(QtWidgets.QLabel("Seam Marker Color:"))
        preview_cfg.addWidget(self.seam_color)
        panel_layout.addWidget(preview_cfg)

        panel_layout.addStretch()
        scroll_area.setWidget(panel)
        main_layout.addWidget(scroll_area)

        # ------------------ RIGHT SIDE DISPLAY (Tabs) ------------------
        self.tabs = QtWidgets.QTabWidget()

        # Seamless View Tab
        self.seamless_view = ImagePreviewLabel()
        self.tabs.addTab(self.seamless_view, "Seamless Output")

        # Tiled Grid Tab
        self.tiled_view = ImagePreviewLabel()
        self.tabs.addTab(self.tiled_view, "Tiled Pattern Verification")

        # Split Original View Tab
        self.split_view = ImagePreviewLabel()
        self.tabs.addTab(self.split_view, "Before / After")

        main_layout.addWidget(self.tabs, stretch=1)

    # -------------------------------------------------------------
    # Drag and Drop Events
    # -------------------------------------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            # Check if any dropped URL matches image file patterns
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tiff", ".webp"]:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.exists(file_path):
                self.load_image_from_path(file_path)

    # -------------------------------------------------------------
    # Controls Builder Functions
    # -------------------------------------------------------------

    def create_slider(
        self,
        min_val: int,
        max_val: int,
        default: int,
        label_fmt: str,
        float_scale: float = 1.0,
    ) -> Tuple[QtWidgets.QWidget, QtWidgets.QSlider, QtWidgets.QLabel]:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        lbl = QtWidgets.QLabel(
            label_fmt.format(default / float_scale if float_scale != 1.0 else default)
        )
        sld = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        sld.setRange(min_val, max_val)
        sld.setValue(default)

        def on_val_changed(val):
            disp = val / float_scale if float_scale != 1.0 else val
            lbl.setText(label_fmt.format(disp))
            self.trigger_update()

        sld.valueChanged.connect(on_val_changed)
        layout.addWidget(lbl)
        layout.addWidget(sld)
        return container, sld, lbl

    def build_substance_controls(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.sub_threshold = self.create_slider(
            5, 100, 50, "Patch Size Threshold: {}", 100.0
        )
        self.sub_smoothness = self.create_slider(
            1, 100, 25, "Seam Smoothness: {}", 100.0
        )
        self.sub_contrast = self.create_slider(5, 50, 10, "Seam Contrast: {}", 10.0)
        self.sub_lr_source = self.create_slider(
            -100, 100, 0, "L/R Patch Source: {}", 100.0
        )
        self.sub_tb_source = self.create_slider(
            -100, 100, 0, "T/B Patch Source: {}", 100.0
        )

        self.sub_curve = QtWidgets.QComboBox()
        self.sub_curve.addItems(
            ["cubic", "cosine", "smoothstep", "smootherstep", "linear"]
        )
        self.sub_curve.currentIndexChanged.connect(self.trigger_update)

        self.sub_invert = QtWidgets.QCheckBox("Invert Blend Mask")
        self.sub_invert.stateChanged.connect(self.trigger_update)

        layout.addWidget(self.sub_threshold[0])
        layout.addWidget(self.sub_smoothness[0])
        layout.addWidget(self.sub_contrast[0])
        layout.addWidget(self.sub_lr_source[0])
        layout.addWidget(self.sub_tb_source[0])
        layout.addWidget(QtWidgets.QLabel("Transition Easing:"))
        layout.addWidget(self.sub_curve)
        layout.addWidget(self.sub_invert)
        self.stack_layout.addWidget(widget)

    def build_radial_controls(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.rad_inner = self.create_slider(
            0, 150, 85, "Inner Smooth Radius: {}", 100.0
        )
        self.rad_outer = self.create_slider(
            10, 200, 100, "Outer Smooth Radius: {}", 100.0
        )
        self.rad_scatter = self.create_slider(
            0, 200, 50, "Scatter/Wave Noise: {}", 100.0
        )

        self.rad_curve = QtWidgets.QComboBox()
        self.rad_curve.addItems(
            ["cosine", "linear", "smoothstep", "smootherstep", "quadratic", "cubic"]
        )
        self.rad_curve.currentIndexChanged.connect(self.trigger_update)

        layout.addWidget(self.rad_inner[0])
        layout.addWidget(self.rad_outer[0])
        layout.addWidget(self.rad_scatter[0])
        layout.addWidget(QtWidgets.QLabel("Transition Easing:"))
        layout.addWidget(self.rad_curve)
        self.stack_layout.addWidget(widget)

    def build_half_shift_controls(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.hs_inner = self.create_slider(0, 150, 70, "Inner Blend Limit: {}", 100.0)
        self.hs_outer = self.create_slider(10, 200, 100, "Outer Blend Limit: {}", 100.0)

        self.hs_curve = QtWidgets.QComboBox()
        self.hs_curve.addItems(
            ["linear", "cosine", "smoothstep", "smootherstep", "quadratic", "cubic"]
        )
        self.hs_curve.currentIndexChanged.connect(self.trigger_update)

        self.hs_orientation = QtWidgets.QComboBox()
        self.hs_orientation.addItems(["both", "horizontal", "vertical"])
        self.hs_orientation.currentIndexChanged.connect(self.trigger_update)

        layout.addWidget(self.hs_inner[0])
        layout.addWidget(self.hs_outer[0])
        layout.addWidget(QtWidgets.QLabel("Orientation:"))
        layout.addWidget(self.hs_orientation)
        layout.addWidget(QtWidgets.QLabel("Transition Easing:"))
        layout.addWidget(self.hs_curve)
        self.stack_layout.addWidget(widget)

    def build_mirrored_controls(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.col_curve = QtWidgets.QComboBox()
        self.col_curve.addItems(
            ["cosine", "linear", "smoothstep", "smootherstep", "quadratic", "cubic"]
        )
        self.col_curve.currentIndexChanged.connect(self.trigger_update)

        layout.addWidget(QtWidgets.QLabel("Local Seam Smooth Curve:"))
        layout.addWidget(self.col_curve)
        self.stack_layout.addWidget(widget)

    # -------------------------------------------------------------
    # Event Handlers & Calculations
    # -------------------------------------------------------------

    def on_method_changed(self, idx: int):
        self.stack_layout.setCurrentIndex(idx)
        self.trigger_update()

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Source Texture Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tga *.tiff *.webp)",
        )
        if file_path:
            self.load_image_from_path(file_path)

    def load_image_from_path(self, file_path: str):
        try:
            self.original_pil = Image.open(file_path).convert("RGBA")
            self.save_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)
            self.trigger_update()
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error", f"Failed to load image file:\n{str(e)}"
            )

    def trigger_update(self):
        self.update_timer.start()

    def process_image(self):
        if self.original_pil is None:
            return

        # 1. Convert Image to PyTorch Tensor on target computing hardware
        np_arr = np.array(self.original_pil).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_arr).to(DEVICE)

        # 2. Extract Pre-crop values
        crop_l = self.crop_l.value()
        crop_r = self.crop_r.value()
        crop_t = self.crop_t.value()
        crop_b = self.crop_b.value()

        h, w, c = tensor.shape
        # Handle crop safety boundary limits
        x0 = min(w, crop_l)
        y0 = min(h, crop_t)
        x1 = max(x0 + 2, w - crop_r)
        y1 = max(y0 + 2, h - crop_b)
        x1 = min(w, x1)
        y1 = min(h, y1)

        cropped_tensor = tensor[y0:y1, x0:x1, :]
        chw = _to_chw(cropped_tensor)

        # 3. Apply Lighting Equalizer before building the seamless blend
        eq_intensity = self.eq_intensity[1].value()
        eq_radius = self.eq_radius[1].value()
        if eq_intensity > 0:
            chw = _pre_average_lighting(chw, eq_intensity, eq_radius)

        # 4. Process Tiling Method
        method_idx = self.method_combo.currentIndex()
        if method_idx == 0:  # Substance Make-It-Tile
            chw = make_seamless_substance(
                chw,
                threshold=self.sub_threshold[1].value() / 100.0,
                smoothness=self.sub_smoothness[1].value() / 100.0,
                contrast=self.sub_contrast[1].value() / 10.0,
                lr_source=self.sub_lr_source[1].value() / 100.0,
                tb_source=self.sub_tb_source[1].value() / 100.0,
                mask_invert=self.sub_invert.isChecked(),
                blend_curve=self.sub_curve.currentText(),
            )
        elif method_idx == 1:  # Radial Mask Offset
            chw = make_seamless_radial(
                chw,
                inner_radius=self.rad_inner[1].value() / 100.0,
                outer_radius=self.rad_outer[1].value() / 100.0,
                scatter_strength=self.rad_scatter[1].value() / 100.0,
                blend_curve=self.rad_curve.currentText(),
            )
        elif method_idx == 2:  # Half Shift
            chw = make_seamless_half_shift(
                chw,
                inner_radius=self.hs_inner[1].value() / 100.0,
                outer_radius=self.hs_outer[1].value() / 100.0,
                blend_curve=self.hs_curve.currentText(),
                orientation=self.hs_orientation.currentText(),
            )
        elif method_idx == 3:  # Collage
            chw = make_seamless_collage(chw, blend_curve=self.col_curve.currentText())

        self.processed_tensor = chw.clone()

        # 5. Populate Display tabs
        # Seamless Preview Tab
        qpix_seamless = self.tensor_to_qpixmap(chw)
        self.seamless_view.set_pixmap(qpix_seamless)

        # Tiled Pattern Verification Tab
        tx = self.grid_x.value()
        ty = self.grid_y.value()
        tiled_chw = torch.tile(chw, (1, ty, tx))
        tiled_chw = _mark_seams(
            tiled_chw,
            single_w=chw.shape[2],
            single_h=chw.shape[1],
            tx=tx,
            ty=ty,
            color_str=self.seam_color.text(),
        )
        qpix_tiled = self.tensor_to_qpixmap(tiled_chw)
        self.tiled_view.set_pixmap(qpix_tiled)

        # Before / After Comparison Tab
        orig_chw = _to_chw(tensor)
        # Match dimensions of current processed outputs
        orig_resized = F.interpolate(
            orig_chw.unsqueeze(0),
            size=(chw.shape[1], chw.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Generate a side-by-side composite
        split_chw = torch.cat([orig_resized, chw], dim=2)
        qpix_split = self.tensor_to_qpixmap(split_chw)
        self.split_view.set_pixmap(qpix_split)

    def tensor_to_qpixmap(self, tensor: torch.Tensor) -> QtGui.QPixmap:
        """Helper to convert float PyTorch tensor [C, H, W] to QPixmap."""
        cpu_tensor = tensor.detach().cpu().clamp(0.0, 1.0)
        c, h, w = cpu_tensor.shape

        # Convert to [H, W, C] uint8 NumPy array
        arr = (cpu_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        arr = np.ascontiguousarray(arr)

        # Convert raw NumPy values directly to Python bytes to comply with PyQt overloads
        data_bytes = arr.tobytes()
        bytes_per_line = w * c

        if PYQT_VERSION == 6:
            format_rgb = QtGui.QImage.Format.Format_RGB888
            format_rgba = QtGui.QImage.Format.Format_RGBA8888
        else:
            format_rgb = QtGui.QImage.Format_RGB888
            format_rgba = QtGui.QImage.Format_RGBA8888

        if c == 3:
            qimg = QtGui.QImage(data_bytes, w, h, bytes_per_line, format_rgb)
        elif c == 4:
            qimg = QtGui.QImage(data_bytes, w, h, bytes_per_line, format_rgba)
        else:
            # Duplicate channels to translate single channel/grayscale safely
            arr = np.repeat(arr, 3, axis=-1)
            data_bytes = arr.tobytes()
            bytes_per_line = w * 3
            qimg = QtGui.QImage(data_bytes, w, h, bytes_per_line, format_rgb)

        return QtGui.QPixmap.fromImage(qimg.copy())

    def save_file(self):
        if self.processed_tensor is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Seamless Texture",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;Targa Files (*.tga)",
        )
        if file_path:
            try:
                cpu_tensor = self.processed_tensor.detach().cpu().clamp(0.0, 1.0)
                c, h, w = cpu_tensor.shape
                arr = (cpu_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

                # Check for transparency
                mode = "RGBA" if c == 4 else "RGB"
                if c == 1:
                    arr = arr.squeeze(-1)
                    mode = "L"

                pil_out = Image.fromarray(arr, mode=mode)
                pil_out.save(file_path)
                QMessageBox.information(
                    self, "Export Successful", f"Saved seamless file to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", f"Failed to save image file:\n{str(e)}"
                )

    def copy_to_clipboard(self):
        """Copies the currently processed seamless output image directly to the system clipboard."""
        if self.processed_tensor is None:
            return
        try:
            pixmap = self.seamless_view._pixmap
            if pixmap:
                clipboard = QtWidgets.QApplication.clipboard()
                clipboard.setPixmap(pixmap)

                # Visual verification: change button text temporarily
                self.copy_btn.setText("Copied!")
                QtCore.QTimer.singleShot(
                    1500, lambda: self.copy_btn.setText("Copy to Clipboard")
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Clipboard Error", f"Failed to copy image to clipboard:\n{str(e)}"
            )

    def apply_theme(self):
        # Set dark-mode stylings (similar to Substance and ComfyUI dark palette)
        dark_stylesheet = """
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #1a1a1a;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #2e2e2e;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 18px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border-color: #5c5c5c;
            }
            QPushButton:pressed {
                background-color: #1f1f1f;
            }
            QPushButton:disabled {
                color: #666666;
                background-color: #1a1a1a;
                border-color: #2b2b2b;
            }
            QComboBox {
                background-color: #2e2e2e;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 4px;
                min-width: 6em;
            }
            QComboBox::drop-down {
                border: none;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3d3d3d;
                height: 4px;
                background: #2e2e2e;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #007acc;
                border: 1px solid #005999;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #148cdb;
            }
            QSpinBox, QLineEdit {
                background-color: #2e2e2e;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background: #141414;
            }
            QTabBar::tab {
                background: #242424;
                border: 1px solid #3d3d3d;
                border-bottom-color: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 16px;
            }
            QTabBar::tab:selected {
                background: #141414;
                border-bottom-color: #141414;
            }
            QTabBar::tab:hover {
                background: #2d2d2d;
            }
            QScrollArea {
                border: none;
                background-color: #1a1a1a;
            }
        """
        self.setStyleSheet(dark_stylesheet)


# -------------------------------------------------------------
# App Entrypoint
# -------------------------------------------------------------


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
