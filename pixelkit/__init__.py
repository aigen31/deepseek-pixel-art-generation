"""pixelkit - deterministic pixel-art analysis, rendering and verification.

The point of this package is that a sprite is data, not a picture: a grid of
characters plus a palette.  Text grids can be written, diffed, reviewed and
regenerated exactly; raster images cannot.
"""

from .analyze import analyze, analyze_from_array, conform_to_reference, quantize_reference
from .core import (
    TRANSPARENT,
    ascii_to_grid,
    detect_pixel_size,
    grid_to_array,
    grid_to_ascii,
    image_to_grid,
    load_rgba,
    quantize_rgba,
    save_rgba,
    upscale,
)
from .pixelate import (
    block_histogram,
    block_map,
    build_palette,
    median_cut,
    pixelate,
    redmean,
    silhouette,
    snap,
)
from .render import (
    flip_h,
    flip_v,
    outline,
    pad_canvas,
    recolor,
    resize,
    sheet,
    shift,
    to_png,
    to_preview,
)
from .verify import (
    compare,
    diff_image,
    grid_diff_report,
    palette_conformance,
    silhouette_check,
    silhouette_iou,
)

__all__ = [
    "TRANSPARENT",
    "analyze",
    "analyze_from_array",
    "block_histogram",
    "block_map",
    "build_palette",
    "ascii_to_grid",
    "compare",
    "conform_to_reference",
    "detect_pixel_size",
    "diff_image",
    "flip_h",
    "flip_v",
    "grid_diff_report",
    "grid_to_array",
    "grid_to_ascii",
    "image_to_grid",
    "load_rgba",
    "median_cut",
    "outline",
    "pad_canvas",
    "palette_conformance",
    "pixelate",
    "quantize_reference",
    "quantize_rgba",
    "recolor",
    "redmean",
    "resize",
    "save_rgba",
    "sheet",
    "shift",
    "silhouette",
    "silhouette_check",
    "silhouette_iou",
    "snap",
    "to_png",
    "to_preview",
    "upscale",
]

__version__ = "0.1.0"
