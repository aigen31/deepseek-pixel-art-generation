"""Reference analysis: turn one pixel-art image into an explicit, reusable style spec.

The output is the contract that lets a variant be *redrawn to requirements*
instead of merely resized: which grid, which palette, which colour plays which
role, where the silhouette sits, and what the line weight is.
"""

from __future__ import annotations

import numpy as np

from .core import TRANSPARENT, detect_pixel_size, image_to_grid, load_rgba, quantize_rgba

MAX_CLEAN_COLORS = 32


def _luminance(rgb) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(float(int(value[i : i + 2], 16)) for i in (0, 2, 4))


def analyze_from_array(arr: np.ndarray, cell: int | None = None, source: str = "<array>") -> dict:
    """Describe a sprite array: grid, palette with roles, silhouette, line weight."""
    warnings: list[str] = []
    detected = detect_pixel_size(arr)
    if detected == 1 and min(arr.shape[:2]) > 32:
        warnings.append(
            "no integer pixel grid detected: the source looks anti-aliased or "
            "photo-like. Quantise it (pixelkit.cli quantize) before using it as a "
            "style reference, otherwise every pixel becomes its own colour."
        )
    chosen = cell or detected

    grid, palette = image_to_grid(arr, chosen)
    h = len(grid)
    w = max(len(r) for r in grid)
    flat = "".join(row.ljust(w, TRANSPARENT) for row in grid)
    total = h * w

    opaque = np.array([[c != TRANSPARENT for c in row.ljust(w, TRANSPARENT)] for row in grid])
    padded = np.pad(opaque, 1, constant_values=False)
    edge = opaque & ~(
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    grid_chars = np.array([list(row.ljust(w, TRANSPARENT)) for row in grid])

    entries: list[dict] = []
    for char, value in palette.items():
        count = flat.count(char)
        if char == TRANSPARENT:
            entries.append(
                {
                    "char": char,
                    "hex": "transparent",
                    "count": count,
                    "share": round(count / total, 4),
                    "role": "transparent",
                }
            )
            continue
        rgb = _hex_to_rgb(value)
        entries.append(
            {
                "char": char,
                "hex": value,
                "count": count,
                "share": round(count / total, 4),
                "luminance": round(_luminance(rgb), 1),
                "edge_pixels": int(np.sum(edge & (grid_chars == char))),
            }
        )
    entries.sort(key=lambda e: -e["count"])

    border = [grid_chars[y][x] for y in range(h) for x in (0, w - 1)] + [
        grid_chars[y][x] for x in range(w) for y in (0, h - 1)
    ]
    bg_char = max(set(border), key=border.count) if border else TRANSPARENT
    edge_chars = [grid_chars[y][x] for y in range(h) for x in range(w) if edge[y, x]]
    outline_char = max(set(edge_chars), key=edge_chars.count) if edge_chars else None

    opaque_entries = [e for e in entries if e["char"] != TRANSPARENT]
    by_lum = sorted(opaque_entries, key=lambda e: e["luminance"])
    for e in entries:
        if e["char"] == TRANSPARENT:
            continue
        if e["char"] == bg_char:
            e["role"] = "background"
        elif e["char"] == outline_char:
            e["role"] = "outline"
        elif by_lum and e is by_lum[-1]:
            e["role"] = "highlight"
        elif by_lum and e is by_lum[0] and len(by_lum) > 1:
            e["role"] = "shadow"
        else:
            e["role"] = "mid"

    if bg_char == TRANSPARENT:
        for e in entries:
            if e["char"] == TRANSPARENT:
                e["role"] = "transparent"

    ys, xs = np.nonzero(opaque)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else [0, 0, 0, 0]

    if len(opaque_entries) > MAX_CLEAN_COLORS:
        warnings.append(
            f"{len(opaque_entries)} distinct colours: too many for deliberate pixel art; "
            "expect shading noise rather than an intended palette."
        )
    if chosen == 1 and min(arr.shape[:2]) > 32:
        warnings.append("pixel_size 1 at this canvas size - check the reference before trusting it.")

    return {
        "source": source,
        "pixel_size": chosen,
        "autodetected_pixel_size": detected,
        "native_size": [w, h],
        "export_scale": [arr.shape[1] // chosen, arr.shape[0] // chosen],
        "pixel_count": total,
        "opaque_share": round(float(opaque.mean()), 4),
        "bbox": bbox,
        "palette": entries,
        "palette_size": len(opaque_entries),
        "background_char": bg_char,
        "outline_char": outline_char,
        "grid": grid,
        "warnings": warnings,
    }


def analyze(path, cell: int | None = None) -> dict:
    """Analyse a reference sprite file."""
    return analyze_from_array(load_rgba(path), cell=cell, source=str(path))


def quantize_reference(path, colors: int = 24, cell: int | None = None) -> dict:
    """Analyse a noisy (anti-aliased / AI-generated) reference after quantising it."""
    return analyze_from_array(quantize_rgba(load_rgba(path), colors=colors), cell=cell, source=str(path))


def conform_to_reference(arr: np.ndarray, reference_palette: list[str]) -> np.ndarray:
    """Snap every pixel to the nearest colour of a reference palette (no dithering).

    Used on AI-generated or resized candidates before they are accepted: this is
    what turns a soft image into honest pixel art wearing the reference palette.
    Images much larger than the target grid should be downscaled first.
    """
    ref = np.array(
        [[int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)] for h in reference_palette],
        dtype=np.int16,
    )
    alpha = arr[..., 3]
    rgb = arr[..., :3].astype(np.int16)
    dist = np.linalg.norm(rgb[:, :, None, :].astype(np.float32) - ref[None, None, :, :], axis=3)
    nearest = ref[np.argmin(dist, axis=2)]
    return np.dstack([nearest, np.where(alpha < 128, 0, 255).astype(np.int16)]).astype(np.uint8)
