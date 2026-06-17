# Just Tile It!

A desktop application for generating seamless textures from source images, featuring multiple tiling algorithms and real-time previews.

## Features
- **Seamless Tiling Algorithms:** Substance-style layered tiling, radial masks, half-shifts, and mirrored collages.
- **Pre-processing:** Automated lighting equalization and crop controls.
- **Interactive GUI:** Real-time preview with seam marker visualization.
- **Export/Copy:** Export processed textures or copy directly to the clipboard.

## Requirements
- Python 3.x
- Dependencies: `torch`, `numpy`, `Pillow`, `PyQt6` (or `PySide6`/`PyQt5`)

## Installation
Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage
Run the application using `uv`:

```bash
uv run python app.py
```

## License
MIT license.
Made with ✨Gemini 3.5.
