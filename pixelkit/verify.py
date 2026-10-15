"""Verification: prove a redraw actually satisfies the requirements.

Agents (and people) are bad at eyeballing pixel art.  These checks are cheap and
mechanical: palette conformance catches anti-aliasing, silhouette IoU catches a
warped pose, and the diff image shows exactly which pixels moved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import TRANSPARENT, grid_to_array, load_rgba, save_rgba, upscale
from .render import normalize, silhouette


def _as_array(source) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, tuple):  # (grid, palette)
        grid, palette = source
        return grid_to_array(normalize(grid), palette)
    return load_rgba(source)


def palette_conformance(arr: np.ndarray, palette: list[str], tol: int = 0) -> float:
    """Share of opaque pixels that sit exactly on one of the allowed colours."""
    ref = np.array(
        [[int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)] for h in palette], dtype=np.int16
    )
    opaque = arr[..., 3] > 0
    if not opaque.any():
        return 1.0
    rgb = arr[..., :3].astype(np.int16)[opaque]
    dist = np.abs(rgb[:, None, :] - ref[None, :, :]).max(axis=2)
    return float((dist.min(axis=1) <= tol).mean())


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of the opaque silhouettes (pose/shape agreement)."""
    ma = a[..., 3] > 0
    mb = b[..., 3] > 0
    if ma.shape != mb.shape:
        h = max(ma.shape[0], mb.shape[0])
        w = max(ma.shape[1], mb.shape[1])
        ma = np.pad(ma, ((0, h - ma.shape[0]), (0, w - ma.shape[1])))
        mb = np.pad(mb, ((0, h - mb.shape[0]), (0, w - mb.shape[1])))
    union = np.logical_or(ma, mb).sum()
    return 1.0 if union == 0 else float(np.logical_and(ma, mb).sum() / union)


def compare(reference, candidate, palette: list[str] | None = None) -> dict:
    """Compare a candidate against a reference; returns a pass/fail metric bundle."""
    a = _as_array(reference)
    b = _as_array(candidate)
    same_shape = a.shape == b.shape
    h0 = min(a.shape[0], b.shape[0])
    w0 = min(a.shape[1], b.shape[1])
    overlap = np.abs(
        a[:h0, :w0, :3].astype(int) - b[:h0, :w0, :3].astype(int)
    ).mean()
    result = {
        "same_shape": same_shape,
        "reference_size": [a.shape[1], a.shape[0]],
        "candidate_size": [b.shape[1], b.shape[0]],
        "pixel_mismatch_share": (
            None if not same_shape else round(float(np.any(a != b, axis=2).mean()), 4)
        ),
        "identical": bool(same_shape and np.array_equal(a, b)),
        "silhouette_iou": round(silhouette_iou(a, b), 4),
        "mean_color_distance_over_overlap": round(float(overlap), 2),
    }
    if palette:
        result["palette_conformance"] = round(palette_conformance(b, palette), 4)
        result["palette_ok"] = result["palette_conformance"] >= 0.999
    return result


def diff_image(reference, candidate, path, scale: int = 1) -> Path:
    """Write a PNG highlighting mismatched pixels in magenta over the candidate."""
    a = _as_array(reference)
    b = _as_array(candidate)
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])

    def fit(x):
        return np.pad(x, ((0, h - x.shape[0]), (0, w - x.shape[1]), (0, 0)))

    a, b = fit(a), fit(b)
    bad = np.any(a != b, axis=2)
    out = b.copy()
    out[bad] = (255, 0, 200, 255)
    return save_rgba(upscale(out, scale), path)


def grid_diff_report(reference: list[str], candidate: list[str]) -> dict:
    """Row-level structural diff of two grids, for iteration without rendering."""
    a, b = normalize(reference), normalize(candidate)
    h = max(len(a), len(b))
    w = max(len(a[0]), len(b[0]))
    a = [r.ljust(w, TRANSPARENT) for r in a] + [TRANSPARENT * w] * (h - len(a))
    b = [r.ljust(w, TRANSPARENT) for r in b] + [TRANSPARENT * w] * (h - len(b))
    rows = []
    for y in range(h):
        diffs = [x for x in range(w) if a[y][x] != b[y][x]]
        if diffs:
            rows.append({"y": y, "pixels": len(diffs), "first_x": diffs[0], "last_x": diffs[-1]})
    total = sum(r["pixels"] for r in rows)
    return {
        "changed_pixels": total,
        "changed_share": round(total / (w * h), 4),
        "rows": rows[:40],
    }


def silhouette_check(grid: list[str], min_coverage: float = 0.05) -> dict:
    """Sanity-check a grid: non-empty, single connected blob, sane coverage."""
    mask = silhouette(grid)
    coverage = float(mask.mean())
    labels = _connected_components(mask)
    return {
        "empty": bool(mask.sum() == 0),
        "coverage": round(coverage, 4),
        "components": labels,
        "too_small": coverage < min_coverage,
        "fragmented": labels > 1,
    }


def _connected_components(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            count += 1
            stack = [(sy, sx)]
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return count
