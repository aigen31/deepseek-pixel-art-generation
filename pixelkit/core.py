"""Core primitives: pixel-art images <-> character grids, palettes, grid detection.

A "grid" is the canonical intermediate representation of every sprite in this
project: a list of equal-length strings, one character per pixel.  Characters
index into a palette table ``{char: "#rrggbb" | "transparent"}``.  Text is the
cheapest, most reviewable representation an agent can write and diff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

TRANSPARENT = "."
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --------------------------------------------------------------------------- io


def load_rgba(path: str | Path) -> np.ndarray:
    """Load an image as an (h, w, 4) uint8 RGBA array."""
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"), dtype=np.uint8)


def save_rgba(arr: np.ndarray, path: str | Path) -> Path:
    """Write an (h, w, 4) uint8 array to PNG, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGBA").save(path)
    return path


# ------------------------------------------------------------------ pixel grid


def detect_pixel_size(arr: np.ndarray, tol: int = 1) -> int:
    """Return the largest block size at which every block is a solid colour.

    This recovers the *native* resolution of an upscaled sprite: a 16x16 sprite
    exported at 4x (64x64) reports 4.  Returns 1 when the image is not clean
    pixel art (anti-aliased, photographic, JPEG-noisy).
    """
    h, w = arr.shape[:2]
    best = 1
    limit = min(h, w)
    for cell in range(2, limit + 1):
        if h % cell or w % cell:
            continue
        blocks = arr.reshape(h // cell, cell, w // cell, cell, 4).astype(np.int16)
        ref = blocks[:, 0, :, 0, :]
        dev = np.abs(blocks - ref[:, None, :, None, :]).max()
        if dev <= tol:
            best = cell
    return best


def _block_colours(arr: np.ndarray, cell: int) -> np.ndarray:
    """Reduce each cell x cell block to one representative colour (median)."""
    h, w = arr.shape[:2]
    blocks = arr.reshape(h // cell, cell, w // cell, cell, 4)
    return np.median(blocks, axis=(1, 3)).astype(np.uint8)


def quantize_rgba(arr: np.ndarray, colors: int = 32) -> np.ndarray:
    """Snap an image to at most `colors` colours, preserving a binary alpha."""
    alpha = arr[..., 3]
    rgb = Image.fromarray(arr[..., :3], mode="RGB").quantize(colors=colors, method=Image.MEDIANCUT)
    out = np.array(rgb.convert("RGBA"), dtype=np.uint8)
    out[..., 3] = np.where(alpha < 128, 0, 255).astype(np.uint8)
    return out


def image_to_grid(arr: np.ndarray, cell: int | None = None) -> tuple[list[str], dict[str, str]]:
    """Convert an image to (grid, palette).  `cell` defaults to autodetection."""
    if cell is None:
        cell = detect_pixel_size(arr)
    small = _block_colours(arr, cell)
    h, w = small.shape[:2]

    keys = small.reshape(-1, 4)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)  # most frequent colour first, stable for agents
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))

    palette: dict[str, str] = {}
    chars = [""] * len(uniq)
    next_letter = 0
    for idx in sorted(range(len(uniq)), key=lambda i: rank[i]):
        colour = uniq[idx]
        if colour[3] == 0:
            chars[idx] = TRANSPARENT
            palette[TRANSPARENT] = "transparent"
        else:
            if next_letter >= len(ALPHABET):
                raise ValueError("more than 62 distinct colours: quantise the source first")
            chars[idx] = ALPHABET[next_letter]
            palette[ALPHABET[next_letter]] = "#%02x%02x%02x" % tuple(int(c) for c in colour[:3])
            next_letter += 1

    ids = np.array([chars[i] for i in inverse], dtype="<U1").reshape(h, w)
    return ["".join(row) for row in ids], palette


def _hex_to_rgba(value: str) -> tuple[int, int, int, int]:
    if value == "transparent":
        return (0, 0, 0, 0)
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), 255)


def grid_to_array(grid: list[str], palette: dict[str, str]) -> np.ndarray:
    """Rasterise a grid to an (h, w, 4) uint8 array at 1:1 scale."""
    lookup = {c: _hex_to_rgba(v) for c, v in palette.items()}
    h = len(grid)
    w = max(len(r) for r in grid)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            out[y, x] = lookup.get(char, (0, 0, 0, 0))
    return out


def upscale(arr: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour integer upscale (the only honest way to enlarge pixels)."""
    if scale == 1:
        return arr
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def ascii_to_grid(text: str) -> tuple[list[str], dict[str, str]]:
    """Parse a text block with a ``char = #rrggbb`` legend into (grid, palette)."""
    palette: dict[str, str] = {}
    rows: list[str] = []
    started = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not started and "=" in line and len(line.split("=", 1)[0].strip()) == 1:
            char, value = line.split("=", 1)
            palette[char.strip()] = value.strip()
            continue
        started = True
        rows.append(line)
    if not started:
        raise ValueError("no grid rows found in text block")
    return rows, palette


def grid_to_ascii(grid: list[str], palette: dict[str, str], header: str = "") -> str:
    """Serialise a grid back into the legend + rows text format."""
    lines = [f"# {header}".rstrip(), "# palette"]
    for char, value in palette.items():
        lines.append(f"{char} = {value}")
    lines.append("# grid")
    lines.extend(grid)
    return "\n".join(lines) + "\n"


def dump_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
