"""Rendering operations on grids: the vocabulary in which requirements are written.

Every function takes and returns the same (grid, palette) shape, so operations
compose::

    grid, pal = render.resize(grid, pal, 32, 32)
    grid, pal = render.recolor(grid, pal, {"0": "#3a1f5c"})
    grid, pal = render.outline(grid, pal, color="#12081f")

Grid invariant: rows may have different lengths while being edited; helpers
normalise to the widest row and pad with transparent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core import TRANSPARENT, grid_to_array, save_rgba, upscale


def normalize(grid: list[str]) -> list[str]:
    width = max(len(r) for r in grid) if grid else 0
    return [r.ljust(width, TRANSPARENT) for r in grid]


def blit(
    canvas: list[str], part: list[str], x: int, y: int, keep_transparent: bool = True
) -> list[str]:
    """Paste a part onto a canvas at (x, y), growing the canvas if needed.

    ``keep_transparent`` copies only the part's opaque pixels (alpha compositing),
    which is how a head, an arm or a prop is laid over a body.
    """
    part = [r for r in part]
    rows = [list(r) for r in normalize(canvas)]
    ph = len(part)
    pw = max((len(r) for r in part), default=0)
    need_h = max(len(rows), y + ph)
    need_w = max(len(rows[0]) if rows else 0, x + pw)
    for row in rows:
        row.extend(TRANSPARENT * (need_w - len(row)))
    while len(rows) < need_h:
        rows.append([TRANSPARENT] * need_w)
    for dy, prow in enumerate(part):
        ty = y + dy
        if ty < 0:
            continue
        for dx, ch in enumerate(prow):
            tx = x + dx
            if tx < 0 or tx >= need_w:
                continue
            if keep_transparent and ch == TRANSPARENT:
                continue
            rows[ty][tx] = ch
    return ["".join(r) for r in rows]


def cut(grid: list[str], box: tuple[int, int, int, int]) -> list[str]:
    """Extract an inclusive (x0, y0, x1, y1) region as its own grid."""
    x0, y0, x1, y1 = box
    return [r[x0 : x1 + 1] for r in normalize(grid)[y0 : y1 + 1]]


def apply_patches(
    grid: list[str], patches: list[dict], palette: dict[str, str] | None = None
) -> list[str]:
    """Replace exact pixel runs, e.g. ``{"y": 44, "x": 28, "row": "333"}``.

    Patching is how a pose variant is authored: everything outside the named runs
    stays byte-identical to the reference, so the style cannot drift. Characters
    are validated against the palette so a typo cannot invent a colour.
    """
    rows = [list(r) for r in normalize(grid)]
    for patch in patches:
        y = patch["y"]
        x = patch["x"]
        text = patch["row"]
        if palette:
            unknown = sorted({c for c in text if c not in palette})
            if unknown:
                raise ValueError(f"patch at y={y} x={x} uses unknown palette chars {unknown}")
        if y >= len(rows):
            raise ValueError(f"patch row {y} is outside the {len(rows)}-row grid")
        width = len(rows[y])
        if x + len(text) > width:
            raise ValueError(f"patch y={y} x={x} len={len(text)} exceeds width {width}")
        rows[y][x : x + len(text)] = list(text)
    return ["".join(r) for r in rows]


def size(grid: list[str]) -> tuple[int, int]:
    grid = normalize(grid)
    return (len(grid[0]) if grid else 0, len(grid))


# ------------------------------------------------------------------- geometry


def resize(grid: list[str], palette: dict[str, str], width: int, height: int) -> list[str]:
    """Resample to an exact grid. Downscaling is area-based, upscaling is nearest.

    Returns rows only; the palette is unchanged and passed for signature symmetry.
    """
    arr = grid_to_array(normalize(grid), palette)
    im = Image.fromarray(arr, mode="RGBA").resize((width, height), Image.NEAREST)
    rgba = np.array(im, dtype=np.uint8)
    return _array_to_grid_letters(rgba, palette)


def _array_to_grid_letters(arr: np.ndarray, palette: dict[str, str]) -> list[str]:
    """Map an array back onto the existing palette characters (nearest colour)."""
    lookup = []
    for char, value in palette.items():
        if value == "transparent":
            lookup.append((char, (0, 0, 0, 0)))
        else:
            v = value.lstrip("#")
            lookup.append((char, (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16), 255)))
    ref = np.array([c for _, c in lookup], dtype=np.int16)
    dist = np.linalg.norm(arr[:, :, None, :].astype(np.float32) - ref[None, None, :, :], axis=3)
    idx = np.argmin(dist, axis=2)
    chars = np.array([ch for ch, _ in lookup], dtype="<U1")
    return ["".join(row) for row in chars[idx]]


def pad_canvas(
    grid: list[str], palette: dict[str, str], left=0, top=0, right=0, bottom=0
) -> list[str]:
    grid = normalize(grid)
    width = len(grid[0]) if grid else 0
    blank = TRANSPARENT * (width + left + right)
    rows = [blank] * top + [TRANSPARENT * left + r + TRANSPARENT * right for r in grid]
    rows += [blank] * bottom
    return rows


def shift(grid: list[str], palette: dict[str, str], dx: int, dy: int) -> list[str]:
    grid = normalize(grid)
    h, w = len(grid), len(grid[0]) if grid else 0
    out = [[TRANSPARENT] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                out[ny][nx] = grid[y][x]
    return ["".join(r) for r in out]


def flip_h(grid: list[str], palette=None) -> list[str]:
    return [r[::-1] for r in normalize(grid)]


def flip_v(grid: list[str], palette=None) -> list[str]:
    return list(reversed(normalize(grid)))


def crop(grid: list[str], box: tuple[int, int, int, int]) -> list[str]:
    x0, y0, x1, y1 = box
    return [r[x0 : x1 + 1] for r in normalize(grid)[y0 : y1 + 1]]


# --------------------------------------------------------------------- colour


def recolor(grid: list[str], palette: dict[str, str], mapping: dict[str, str]) -> dict[str, str]:
    """Return a new palette with characters remapped to new hex values."""
    out = dict(palette)
    for char, value in mapping.items():
        out[char] = value
    return out


def swap(grid: list[str], a: str, b: str) -> list[str]:
    """Exchange two characters (two materials/colours) inside the grid."""
    table = str.maketrans({a: b, b: a})
    return [r.translate(table) for r in grid]


def replace(grid: list[str], a: str, b: str) -> list[str]:
    return [r.replace(a, b) for r in grid]


def ramp_shift(grid: list[str], palette: dict[str, str], order: list[str], steps: int = 1) -> list[str]:
    """Move every pixel along a shading ramp, e.g. ``order=['2','1','0']``.

    Positive ``steps`` darkens when the ramp is written dark-first, which is the
    usual way to state "same sprite, night variant".
    """
    index = {c: i for i, c in enumerate(order)}
    table = {}
    for char, i in index.items():
        j = min(max(i + steps, 0), len(order) - 1)
        table[char] = order[j]
    return [r.translate(str.maketrans(table)) for r in normalize(grid)]


# ------------------------------------------------------------------ silhouette


def outline(
    grid: list[str],
    palette: dict[str, str],
    color: str = "#000000",
    char: str = "o",
    diagonal: bool = False,
    replace_existing: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Add a one-pixel outline around the opaque silhouette.

    Grows the canvas by one pixel on each side so the outline is never clipped.
    """
    grid = normalize(grid)
    h, w = len(grid), len(grid[0]) if grid else 0
    opaque = np.array([[c != TRANSPARENT for c in row] for row in grid])
    if diagonal:
        nb = np.zeros_like(opaque)
        for dy in (-1, 1):
            for dx in (-1, 1):
                nb |= np.roll(np.roll(opaque, dy, 0), dx, 1)
        ring = nb & ~opaque
    else:
        up = np.roll(opaque, 1, 0); up[0, :] = False
        down = np.roll(opaque, -1, 0); down[-1, :] = False
        left = np.roll(opaque, 1, 1); left[:, 0] = False
        right = np.roll(opaque, -1, 1); right[:, -1] = False
        ring = (up | down | left | right) & ~opaque

    padded_rows = [TRANSPARENT + r + TRANSPARENT for r in grid]
    padded_rows = [TRANSPARENT * (w + 2)] + padded_rows + [TRANSPARENT * (w + 2)]
    ring_p = np.pad(ring, 1, constant_values=False)

    out = [[char if ring_p[y, x] else padded_rows[y][x] for x in range(w + 2)] for y in range(h + 2)]
    if replace_existing:
        out = [[char if ring_p[y, x] else padded_rows[y][x] for x in range(w + 2)] for y in range(h + 2)]
    pal = dict(palette)
    pal[char] = color
    return ["".join(r) for r in out], pal


def silhouette(grid: list[str]) -> np.ndarray:
    return np.array([[c != TRANSPARENT for c in r] for r in normalize(grid)])


# ---------------------------------------------------------------------- output


def to_png(grid: list[str], palette: dict[str, str], path, scale: int = 1) -> Path:
    arr = upscale(grid_to_array(normalize(grid), palette), scale)
    return save_rgba(arr, path)


def to_preview(
    grid: list[str], palette: dict[str, str], path, scale: int = 8, background: str = "#20242c"
) -> Path:
    """Render a scaled PNG on a solid backdrop, for eyeballing a sprite in a chat."""
    arr = upscale(grid_to_array(normalize(grid), palette), scale)
    bg = np.array([int(background.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)] + [255], dtype=np.uint8)
    alpha = arr[..., 3:4].astype(np.float32) / 255.0
    flat = (arr[..., :3].astype(np.float32) * alpha + bg[:3] * (1 - alpha)).astype(np.uint8)
    return save_rgba(np.dstack([flat, np.full(arr.shape[:2], 255, dtype=np.uint8)]), path)


def sheet(
    frames: list[tuple[str, list[str]]],
    palette: dict[str, str],
    path,
    columns: int = 4,
    scale: int = 1,
    padding: int = 0,
) -> tuple[Path, Path]:
    """Pack named frames into a uniform-cell spritesheet plus a JSON atlas."""
    if not frames:
        raise ValueError("no frames to pack")
    cw = max(len(normalize(g)[0]) for _, g in frames) + padding * 2
    ch = max(len(normalize(g)) for _, g in frames) + padding * 2
    rows = (len(frames) + columns - 1) // columns
    canvas = np.zeros((ch * rows, cw * columns, 4), dtype=np.uint8)
    atlas = {"cell": [cw, ch], "columns": columns, "scale": scale, "frames": {}}
    for i, (name, grid) in enumerate(frames):
        g = pad_canvas(grid, palette, padding, padding, padding, padding)
        cell = grid_to_array(g, palette)
        y, x = divmod(i, columns)
        canvas[y * ch : y * ch + cell.shape[0], x * cw : x * cw + cell.shape[1]] = cell
        atlas["frames"][name] = {"x": x * cw, "y": y * ch, "w": cell.shape[1], "h": cell.shape[0]}
    png = to_png_array(canvas, path, scale)
    atlas_path = Path(path).with_suffix(".json")
    atlas_path.write_text(json.dumps(atlas, indent=2) + "\n", encoding="utf-8")
    return png, atlas_path


def to_png_array(canvas: np.ndarray, path, scale: int = 1) -> Path:
    return save_rgba(upscale(canvas, scale), path)
