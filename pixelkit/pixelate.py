"""Multi-scale pixel reduction: fake pixel art in, real pixel art out.

An image model asked for "pixel art" returns a *picture of* pixel art: hundreds of
near-duplicate colours, anti-aliased edges, and blocks whose size drifts because
the block grid was never real.  This module turns such a draft into an honest
sprite while keeping the one thing the drafts get right — **pixels of different
sizes**.  Flat areas (a hat crown, a coat) collapse into large squares; fine
features (blush stripes, boot laces, the buckle rim) stay one cell wide.

How it works, in order:

1. ``cell`` source pixels -> one cell, taking the per-channel **median** (the
   median survives anti-aliasing and WebP ringing better than the mean).
2. A **shared palette** is cut from the pooled colour histogram of every draft in
   the set (median cut, then near-duplicate merging, then a hard cap so the grid
   still fits in the 62-character text alphabet).
3. Every cell snaps to that palette: one colour per pixel, by construction.
4. **Despeckle** — cells that are locally flat, differ from all eight neighbours
   and have no support become the neighbourhood mode.
5. **Silhouette** — background is flood-filled from the border (so the white eye
   highlights, which are enclosed, survive) and the light anti-aliased rim left
   by the generator is peeled off.
6. **Quadtree collapse** records which square blocks are uniform; that block map
   is the evidence for "mixed pixel sizes" and is reported as a size histogram.

The output is an ordinary RGBA array on the ``cell`` lattice, plus a character
grid and palette, so everything downstream (verify, render, sheet) applies.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .core import ALPHABET, TRANSPARENT

RGB = np.ndarray  # (h, w, 3) float32


# ------------------------------------------------------------------ colour maths


def redmean(a, b) -> np.ndarray:
    """Perceptually weighted RGB distance ("redmean"), the cheap Lab stand-in.

    Plain Euclidean RGB distance over-weights blue and under-weights green; the
    redmean weights are accurate to a few percent of CIEDE2000 at a fraction of
    the cost, which matters when snapping ~10^5 cells to ~10^2 palette entries.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    r1, g1, b1 = a[..., 0], a[..., 1], a[..., 2]
    r2, g2, b2 = b[..., 0], b[..., 1], b[..., 2]
    rm = (r1 + r2) * 0.5
    dr, dg, db = r1 - r2, g1 - g2, b1 - b2
    return np.sqrt((2 + rm / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rm) / 256) * db * db)


# ------------------------------------------------------------------- reduction


def cell_blocks(arr: np.ndarray, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """Group the image into ``cell`` x ``cell`` blocks.

    Returns ``(cells, blocks)`` where ``cells`` is the per-channel median colour of
    each block and ``blocks`` keeps the raw pixels as ``(h//cell, w//cell, cell**2, 3)``.
    """
    h, w = arr.shape[:2]
    pad_h, pad_w = (-h) % cell, (-w) % cell
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    ny, nx = arr.shape[0] // cell, arr.shape[1] // cell
    blocks = (
        arr.reshape(ny, cell, nx, cell, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(ny, nx, cell * cell, 3)
        .astype(np.float32)
    )
    return np.median(blocks, axis=2), blocks


def histogram(px: np.ndarray, bits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Collapse pixels into a weighted histogram of 2**bits per channel buckets."""
    levels = 1 << bits
    shift = 8 - bits
    q = np.clip(px // (1 << shift), 0, levels - 1).astype(np.int64)
    key = (q[:, 0] << (2 * bits)) + (q[:, 1] << bits) + q[:, 2]
    uniq, counts = np.unique(key, return_counts=True)
    centre = (1 << shift) // 2
    colours = np.stack(
        [(uniq >> (2 * bits)) & (levels - 1), (uniq >> bits) & (levels - 1), uniq & (levels - 1)], 1
    ).astype(np.float32)
    return colours * (1 << shift) + centre, counts.astype(np.float64)


def median_cut(colours: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """Classic median cut: split the widest, heaviest box until there are ``k``.

    Median cut is preferred over k-means here because it keeps *small distinct*
    clusters — the green iris and the gold buckle are a fraction of a percent of
    the pixels and plain k-means spends every centroid on the dress instead.
    """
    boxes = [np.arange(len(colours))]

    def cut_value(idx: np.ndarray) -> float:
        if len(idx) < 2:
            return -1.0
        c, w = colours[idx], weights[idx]
        spread = ((c - c.mean(0)) ** 2 * w[:, None]).sum() / w.sum()
        return float(spread.sum() * np.sqrt(w.sum()))

    while len(boxes) < k:
        best = None
        for i, box in enumerate(boxes):
            value = cut_value(box)
            if value > 0 and (best is None or value > best[0]):
                best = (value, i)
        if best is None:
            break
        _, i = best
        box = boxes.pop(i)
        axis = int(np.argmax(colours[box].max(0) - colours[box].min(0)))
        order = box[np.argsort(colours[box][:, axis], kind="stable")]
        cumulative = np.cumsum(weights[order])
        split = int(np.searchsorted(cumulative, cumulative[-1] / 2)) + 1
        split = min(max(split, 1), len(order) - 1)
        boxes.append(order[:split])
        boxes.append(order[split:])

    return np.array(
        [(colours[b] * weights[b, None]).sum(0) / weights[b].sum() for b in boxes], dtype=np.float32
    )


def merge_near(palette: np.ndarray, mass: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop palette entries that sit within ``threshold`` of a heavier entry."""
    keep: list[int] = []
    for i in np.argsort(-mass):
        if all(redmean(palette[i], palette[j]) > threshold for j in keep):
            keep.append(int(i))
    keep_arr = np.array(keep)
    return palette[keep_arr], mass[keep_arr]


def cap_palette(palette: np.ndarray, mass: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedily merge the closest pair until the palette fits in ``cap`` entries.

    The cap matters: a grid is stored as characters, and the alphabet only has 62.
    """
    palette, mass = palette.copy(), mass.copy()
    while len(palette) > cap:
        d = redmean(palette[:, None, :], palette[None, :, :])
        d[np.triu_indices(len(palette), 0)] = np.inf
        i, j = np.unravel_index(int(d.argmin()), d.shape)
        total = mass[i] + mass[j]
        palette[i] = (palette[i] * mass[i] + palette[j] * mass[j]) / total
        mass[i] = total
        palette = np.delete(palette, j, axis=0)
        mass = np.delete(mass, j)
    return palette, mass


def build_palette(
    arrays: list[np.ndarray],
    *,
    cell: int = 4,
    colors: int = 56,
    oversample: int = 2,
    merge_threshold: float = 11.0,
    base: np.ndarray | None = None,
    residual: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One palette for a whole *set* of drafts, so every sprite shares a style.

    Cutting from the pooled histogram means an expression that only appears in one
    frame (the smirk, the open mouth) still gets its colour, and the frames cannot
    drift apart in hue.

    ``base`` freezes a palette that is already in use and only *extends* it: the
    colours that the base cannot represent within ``residual`` get the free slots.
    That is how a new frame joins an existing set — the old sprites keep every one
    of their colours, and the newcomer brings only the accents it introduces.
    """
    colours, weights = [], []
    for arr in arrays:
        cells, _ = cell_blocks(arr, cell)
        bg = background_colour(arr)
        U, W = histogram(cells[redmean(cells, bg) > 40])
        colours.append(U)
        weights.append(W)
    U = np.concatenate(colours)
    W = np.concatenate(weights)

    if base is not None:
        base = np.asarray(base, dtype=np.float32)
        room = min(colors, len(ALPHABET)) - len(base)
        if room <= 0:
            return base, np.zeros(len(base))
        gap = redmean(U[:, None, :], base[None, :, :]).min(1)
        keep = gap > residual
        if not keep.any():
            return base, np.zeros(len(base))
        extra = median_cut(U[keep], W[keep], min(room * oversample, len(ALPHABET)))
        label = np.stack([redmean(U[keep], c) for c in extra]).argmin(0)
        mass = np.array([W[keep][label == i].sum() for i in range(len(extra))])
        extra, _ = merge_near(extra, mass, merge_threshold)
        # Drop anything the frozen palette already covers just as well.
        extra = np.array([c for c in extra if redmean(base, c).min() > residual], dtype=np.float32)
        if not len(extra):
            return base, np.zeros(len(base))
        palette = np.vstack([base, extra])[: min(colors, len(ALPHABET))]
        return palette, np.concatenate(
            [np.zeros(len(base)), np.ones(len(palette) - len(base))]
        )

    palette = median_cut(U, W, min(colors * oversample, 160))
    label = np.stack([redmean(U, c) for c in palette]).argmin(0)
    mass = np.array([W[label == i].sum() for i in range(len(palette))])
    palette, mass = merge_near(palette, mass, merge_threshold)
    palette, mass = cap_palette(palette, mass, min(colors, len(ALPHABET)))
    return palette, mass


def snap(cells: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Nearest palette entry for every cell — the "one colour per pixel" gate."""
    return np.stack([redmean(cells, c) for c in palette]).argmin(0).astype(np.int32)


# ---------------------------------------------------------------------- cleanup


def despeckle(idx: np.ndarray, allowed: np.ndarray, passes: int = 2) -> np.ndarray:
    """Remove isolated single cells that differ from every neighbour.

    Only cells inside ``allowed`` (locally flat, inside the silhouette) are touched,
    so real one-pixel features — a highlight, a lace knot — are never eaten.
    """
    idx = idx.copy()
    h, w = idx.shape
    for _ in range(passes):
        neighbours = np.stack(
            [
                np.roll(np.roll(idx, dy, 0), dx, 1)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if (dy, dx) != (0, 0)
            ]
        )
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if not allowed[y, x]:
                    continue
                values = neighbours[:, y, x]
                if np.any(values == idx[y, x]):
                    continue
                uniq, counts = np.unique(values, return_counts=True)
                idx[y, x] = uniq[counts.argmax()]
    return idx


def background_colour(arr: np.ndarray, band: int = 8) -> np.ndarray:
    """The flat backdrop colour, read from the image border."""
    strips = [
        arr[:band].reshape(-1, 3),
        arr[-band:].reshape(-1, 3),
        arr[:, :band].reshape(-1, 3),
        arr[:, -band:].reshape(-1, 3),
    ]
    return np.median(np.concatenate(strips).astype(np.float32), axis=0)


# ------------------------------------------------------------------- haze & shadow


def _components(flag: np.ndarray, min_area: int = 1) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the 4-connected components of ``flag``, largest first."""
    h, w = flag.shape
    seen = np.zeros((h, w), bool)
    boxes = []
    for sy in range(h):
        for sx in range(w):
            if not flag[sy, sx] or seen[sy, sx]:
                continue
            seen[sy, sx] = True
            stack = [(sy, sx)]
            y0 = y1 = sy
            x0 = x1 = sx
            area = 0
            while stack:
                y, x = stack.pop()
                area += 1
                y0, y1 = min(y0, y), max(y1, y)
                x0, x1 = min(x0, x), max(x1, x)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and flag[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if area >= min_area:
                boxes.append((y0, x0, y1, x1))
    return boxes


def skin_regions(rgb: np.ndarray, background: np.ndarray, *, min_area: int = 40) -> list[tuple]:
    """Bounding boxes of the warm, light skin areas (face, hands, knees)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    skin = (~background) & (r > 180) & (r - b > 30) & (r - g > 15) & (r - g < 90)
    return _components(skin, min_area)


def eye_regions(
    rgb: np.ndarray,
    background: np.ndarray,
    *,
    pad: int = 3,
    min_area: int = 2,
    face_pad: int = 2,
) -> np.ndarray:
    """Protect the eyes: the iris is the only saturated green in the sprite.

    Everything else the drafts use is black, brown, tan or skin, so a green-dominant
    test finds the irises without any hand-tuned coordinates.  Their bounding boxes,
    grown by ``pad`` pixels, cover the sclera and the specular highlight too —
    which matters, because the white we *keep* and the white we *delete* look
    identical pixel for pixel.

    Closed-eye frames have no iris left to find (a drawn ``^^`` is a dark line), so
    the topmost skin region — the face — is protected as a whole.  That is what
    keeps the highlights of ``ухмылка`` and ``счастлива`` alive.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    green = (g - np.maximum(r, b) > 25) & ~background
    h, w = background.shape
    protect = np.zeros((h, w), bool)
    for y0, x0, y1, x1 in _components(green, min_area):
        protect[
            max(0, y0 - pad) : min(h, y1 + pad + 1), max(0, x0 - pad) : min(w, x1 + pad + 1)
        ] = True
    faces = skin_regions(rgb, background)
    if faces:
        y0, x0, y1, x1 = min(faces, key=lambda box: (box[0], -(box[2] - box[0]) * (box[3] - box[1])))
        protect[
            max(0, y0 - face_pad) : min(h, y1 + face_pad + 1),
            max(0, x0 - face_pad) : min(w, x1 + face_pad + 1),
        ] = True
    return protect


def repaint(idx: np.ndarray, target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Flood ``source`` colours into ``target`` cells, nearest cell wins.

    Used to turn a light hair streak into hair instead of into a hole: deleting the
    pixels outright leaves the backdrop showing through the middle of the hair,
    which looks worse than the streak did.
    """
    out = idx.copy()
    known = source.copy() & ~target
    pending = target.copy()
    for _ in range(16):
        if not pending.any():
            break
        progress = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy, dx) == (0, 0):
                    continue
                shifted_known = np.roll(np.roll(known, dy, 0), dx, 1)
                take = pending & shifted_known
                if not take.any():
                    continue
                out[take] = np.roll(np.roll(out, dy, 0), dx, 1)[take]
                known |= take
                pending &= ~take
                progress = True
        if not progress:
            break
    return out


def clear_haze(
    idx: np.ndarray,
    palette: np.ndarray,
    background: np.ndarray,
    *,
    luminance: float = 112.0,
    chroma: float = 30.0,
    protect: np.ndarray | None = None,
    enclosure: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Clear the light neutral pixels: the generator's glow, hair streaks and shadow.

    The drafts paint a white halo along the hair, light gloss streaks inside it and a
    soft grey ground shadow.  All three are low-chroma and light, so one test finds
    them; ``protect`` (see :func:`eye_regions`) is subtracted first so the sclera and
    the highlights survive.

    What happens next depends on where the pixel sits.  A pixel the surroundings
    mostly *don't* cover is part of the outer glow or the shadow, and becomes
    transparent.  A pixel walled in by dark hair is a streak: it is repainted with
    the neighbouring hair colour.  Deleting those instead would punch the backdrop
    through the middle of the hair, which is a worse artefact than the streak.

    Returns ``(idx, background)``.
    """
    rgb = palette[idx].astype(np.float32)
    light = (rgb.mean(-1) > luminance) & ((rgb.max(-1) - rgb.min(-1)) < chroma) & ~background
    if protect is not None:
        light &= ~protect
    if not light.any():
        return idx, background

    neighbours = np.zeros(light.shape, np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if (dy, dx) == (0, 0):
                continue
            neighbours += np.roll(np.roll(light | background, dy, 0), dx, 1)
    free_ratio = neighbours / 24.0

    glow = light & (free_ratio >= enclosure)  # wide open space: outer glow or shadow
    streak = light & ~glow  # walled in by dark hair: a gloss streak
    if streak.any():
        idx = repaint(idx, streak, (~light) & (~background))
    return idx, background | glow


def silhouette(
    blocks: np.ndarray,
    bg: np.ndarray,
    *,
    loose: float = 72.0,
    spread: float = 16.0,
    peel: int = 3,
) -> np.ndarray:
    """Boolean background mask for the cell grid.

    Background is flood-filled from the border rather than thresholded globally:
    the eye highlight is nearly white and would otherwise be punched out of the
    face.  The generator's soft glow around the sprite is then peeled off, because
    a light rim is a giveaway that the art was not drawn on a grid.
    """
    h, w = blocks.shape[:2]
    dist = redmean(blocks, bg)
    chroma = (blocks.max(3) - blocks.min(3)).mean(2)
    bg_like = ((dist < loose).mean(2) > 0.55) & (chroma < spread)

    mask = np.zeros((h, w), bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(w):
        for y in (0, h - 1):
            if bg_like[y, x] and not mask[y, x]:
                mask[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if bg_like[y, x] and not mask[y, x]:
                mask[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and bg_like[ny, nx] and not mask[ny, nx]:
                mask[ny, nx] = True
                queue.append((ny, nx))

    cells = blocks.mean(2)  # mean colour of each cell, for the rim comparison
    for _ in range(peel):
        rim = np.zeros((h, w), bool)
        rim[1:, :] |= mask[:-1, :]
        rim[:-1, :] |= mask[1:, :]
        rim[:, 1:] |= mask[:, :-1]
        rim[:, :-1] |= mask[:, 1:]
        rim &= ~mask
        if not rim.any():
            break
        # Compare each rim cell against the *interior* colours it touches, not
        # against the rest of the rim, otherwise a two-cell glow peels itself.
        interior = ~mask & ~rim
        acc = np.zeros((h, w, 3), np.float32)
        cnt = np.zeros((h, w), np.float32)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ys0, ys1 = max(0, dy), min(h, h + dy)
                xs0, xs1 = max(0, dx), min(w, w + dx)
                src_i = interior[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
                src_c = cells[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
                acc[ys0:ys1, xs0:xs1] += np.where(src_i[..., None], src_c, 0)
                cnt[ys0:ys1, xs0:xs1] += src_i
        inner = np.divide(acc, np.maximum(cnt, 1)[..., None])
        has_inner = cnt > 0
        kill = (
            rim
            & has_inner
            & (redmean(cells, bg) < redmean(inner, bg) * 0.8)
            & (redmean(cells, inner) > 20)
        )
        if not kill.any():
            break
        mask |= kill
    return mask


def block_map(idx: np.ndarray, max_block: int = 8) -> tuple[np.ndarray, int]:
    """Quadtree: label every maximal uniform square block, largest first.

    A flat 8x8 region becomes one block id; a lone detailed pixel stays 1x1.  The
    histogram of block sizes over a sprite *is* the "pixels are not all the same
    size" requirement, made countable.
    """
    h, w = idx.shape
    size = 1
    while size * 2 <= max_block:
        size *= 2
    # Pad to a whole number of blocks with a sentinel that matches nothing, so the
    # tiling is exact and no two quadrants can ever overlap.
    padded = np.full((h + (-h) % size, w + (-w) % size), -2, idx.dtype)
    padded[:h, :w] = idx
    ph, pw = padded.shape
    labels = -np.ones((ph, pw), np.int32)
    counter = 0

    def visit(x: int, y: int, span: int) -> None:
        nonlocal counter
        if span == 1 or np.all(padded[y : y + span, x : x + span] == padded[y, x]):
            labels[y : y + span, x : x + span] = counter
            counter += 1
            return
        half = span // 2
        for dy in (0, half):
            for dx in (0, half):
                visit(x + dx, y + dy, half)

    for y in range(0, ph, size):
        for x in range(0, pw, size):
            visit(x, y, size)
    return labels[:h, :w], size


def block_histogram(labels: np.ndarray) -> dict:
    """Count blocks by edge length, and how much area the large ones cover."""
    present = labels[labels >= 0]
    if not present.size:
        return {}
    counts = np.bincount(present)
    area = counts[counts > 0].astype(np.int64)
    sides = np.rint(np.sqrt(area)).astype(int)
    total = labels.size
    hist: dict[int, dict] = {}
    for n in sorted(set(sides.tolist())):
        pick = sides == n
        hist[n] = {
            "blocks": int(pick.sum()),
            "share_of_canvas": round(float(area[pick].sum()) / total, 4),
        }
    return hist


# ------------------------------------------------------------------- silhouette


def _trim(arr: np.ndarray, bg: np.ndarray, cell: int, margin: int) -> tuple[np.ndarray, tuple[int, int]]:
    far = redmean(arr, bg) > 40
    ys, xs = np.where(far)
    if not len(ys):
        return arr, (0, 0)
    pad = cell * margin
    y0 = max(0, ys.min() - pad)
    x0 = max(0, xs.min() - pad)
    return arr[y0 : ys.max() + 1 + pad, x0 : xs.max() + 1 + pad], (int(y0), int(x0))


def grid_from_array(
    idx: np.ndarray, palette: np.ndarray, background: np.ndarray
) -> tuple[list[str], dict[str, str]]:
    """Rasterise indices + background mask into the project's (grid, palette) text form."""
    used = np.unique(idx)
    counts = {int(i): int((idx[~background] == i).sum()) for i in used}
    order = sorted(used.tolist(), key=lambda i: (-counts[i], i))
    if len(order) > len(ALPHABET):
        raise ValueError(f"palette has {len(order)} colours, the text alphabet holds {len(ALPHABET)}")
    chars = {int(i): ALPHABET[n] for n, i in enumerate(order)}
    table = {chars[int(i)]: "#%02x%02x%02x" % tuple(int(v) for v in palette[int(i)]) for i in order}
    table[TRANSPARENT] = "transparent"
    rows = []
    for y in range(idx.shape[0]):
        rows.append(
            "".join(
                TRANSPARENT if background[y, x] else chars[int(idx[y, x])]
                for x in range(idx.shape[1])
            )
        )
    return rows, table


# -------------------------------------------------------------------- the whole


def pixelate(
    arr: np.ndarray,
    *,
    cell: int = 4,
    palette: np.ndarray | None = None,
    colors: int = 56,
    max_block: int = 8,
    despeckle_passes: int = 2,
    peel: int = 3,
    margin: int = 2,
    clear_light_haze: bool = True,
    protect_eyes: bool = True,
    haze_luminance: float = 112.0,
    haze_chroma: float = 30.0,
    eye_pad: int = 3,
    haze_enclosure: float = 0.75,
) -> dict:
    """Turn one anti-aliased draft into a real sprite on a ``cell``-pixel lattice.

    ``palette`` lets a whole set share one palette; omit it and a per-image one is
    cut.  Returns the sprite plus the evidence needed to verify it.

    With ``clear_light_haze`` the generator's white glow on the hair and the soft
    grey ground shadow are deleted, while ``protect_eyes`` keeps the sclera and the
    specular highlights (see :func:`eye_regions`).
    """
    arr = arr[..., :3].astype(np.float32)
    bg = background_colour(arr)
    arr, origin = _trim(arr, bg, cell, margin)
    cells, blocks = cell_blocks(arr, cell)

    if palette is None:
        U, W = histogram(cells[redmean(cells, bg) > 40])
        palette = median_cut(U, W, min(colors * 2, 160))
        label = np.stack([redmean(U, c) for c in palette]).argmin(0)
        mass = np.array([W[label == i].sum() for i in range(len(palette))])
        palette, mass = merge_near(palette, mass, 11.0)
        palette, _ = cap_palette(palette, mass, min(colors, len(ALPHABET)))

    background = silhouette(blocks, bg, peel=peel)
    flat = blocks.std(2).mean(-1) < 8
    idx = snap(cells, palette)
    idx = despeckle(idx, flat & ~background, despeckle_passes)

    eyes = None
    haze_removed = 0
    if clear_light_haze:
        before = background.copy()
        eyes = (
            eye_regions(palette[idx].astype(np.float32), background, pad=eye_pad)
            if protect_eyes
            else None
        )
        idx, background = clear_haze(
            idx,
            palette,
            background,
            luminance=haze_luminance,
            chroma=haze_chroma,
            protect=eyes,
            enclosure=haze_enclosure,
        )
        haze_removed = int((background & ~before).sum())

    labels, base = block_map(np.where(background, -1, idx), max_block)
    rgba = np.dstack(
        [palette[idx].astype(np.uint8), np.where(background, 0, 255).astype(np.uint8)]
    )
    rgba[background] = 0  # keep RGB clean under transparency, so the grid round-trips
    grid, table = grid_from_array(idx, palette, background)
    return {
        "rgba": rgba,
        "palette": palette,
        "grid": grid,
        "table": table,
        "background": background,
        "blocks": labels,
        "block_base": base,
        "cell": cell,
        "eyes": eyes,
        "haze_removed": haze_removed,
        "source": arr,
        "origin": origin,
        "source_crop": arr.shape[:2],
    }
