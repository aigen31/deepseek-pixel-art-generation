"""Self-checking tests for `pixelkit.pixelate` — the mixed-pixel-size reducer.

The requirement being tested is unusual, so it is worth restating: the output must
be *real* pixel art (one colour per pixel, binary alpha, no anti-aliasing) while
its pixels are deliberately **not all the same size** — flat areas collapse into
large squares, fine features stay one pixel wide.

Run with:  ./.venv/bin/python examples/test_pixelate.py

Everything is synthesised here, so the test needs no draft images and runs in a
couple of seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixelkit import verify  # noqa: E402
from pixelkit.core import grid_to_array, upscale  # noqa: E402
from pixelkit.pixelate import (  # noqa: E402
    clear_haze,
    cell_blocks,
    block_histogram,
    block_map,
    build_palette,
    pixelate,
    redmean,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def make_draft(seed: int = 3) -> np.ndarray:
    """A synthetic 'AI pixel art' draft: blocky subject, big flat areas, soft edges.

    Reproduces what the real drafts do wrong — anti-aliased edges, a colour cast
    that is not a palette, and haze — while containing features at three scales: a
    16px flat crown, 8px cheek blocks and 2px stripes.
    """
    rng = np.random.default_rng(seed)
    size = 256
    img = np.full((size, size, 3), 247.0, np.float32)  # the near-white backdrop
    img += rng.normal(0, 1.2, img.shape)  # sensor-ish haze

    # big flat body: 128px square, one colour — the "large pixels" case
    img[64:192, 64:192] = (30, 26, 38)
    # medium blocks: an 8px checker band along the bottom edge, leaving the rest of
    # the body genuinely flat so large squares can actually form
    band = img[176:192, 64:192].reshape(2, 8, 16, 8, 3)
    band[::2, ::2] = (52, 46, 60)
    # a face patch with 2px blush stripes on 8px skin blocks
    img[96:144, 96:144] = (252, 211, 184)
    for y in range(104, 136, 8):
        img[y : y + 2, 112:136] = (224, 147, 125)
    # small dark eyes
    img[104:120, 100:112] = (26, 34, 30)
    img[104:120, 128:140] = (26, 34, 30)

    # anti-alias every edge, as a generator would
    blurred = img.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rolled = np.roll(np.roll(img, dy, 0), dx, 1)
        edge = redmean(img, rolled) > 20
        blurred[edge] = (img[edge] + rolled[edge]) / 2
    return blurred


def main() -> int:
    draft = make_draft()
    print("1. intake")
    check("draft has no flat palette (it is a picture, not data)",
          len(np.unique(draft.reshape(-1, 3).astype(np.uint8), axis=0)) > 500,
          f"{len(np.unique(draft.reshape(-1, 3).astype(np.uint8), axis=0))} colours")

    print("\n2. reduce")
    result = pixelate(draft, cell=4, colors=24)
    rgba = result["rgba"]
    palette = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in result["palette"]]
    check("output is smaller than the draft", rgba.shape[0] < draft.shape[0],
          f"{rgba.shape[1]}x{rgba.shape[0]} from {draft.shape[1]}x{draft.shape[0]}")

    print("\n3. it is real pixel art")
    check("palette conformance is exactly 1.0",
          verify.palette_conformance(rgba, palette) == 1.0)
    with_grid = grid_to_array(result["grid"], result["table"])
    check("the text grid rasterises back to the same image", np.array_equal(with_grid, rgba))
    alpha = np.unique(rgba[..., 3]).tolist()
    check("alpha is binary", alpha in ([0], [0, 255], [255]), f"alpha values {alpha}")
    check("colour count is inside the palette",
          len(np.unique(rgba[rgba[..., 3] > 0][:, :3], axis=0)) <= len(palette))
    scaled = upscale(rgba, 4)
    check("integer scaling adds no new colours",
          len(np.unique(scaled.reshape(-1, 4), axis=0)) == len(np.unique(rgba.reshape(-1, 4), axis=0)))
    check("no anti-aliasing survives (grid has no half-tones)",
          verify.palette_conformance(scaled, palette) == 1.0)

    print("\n4. pixels are deliberately NOT one size")
    hist = block_histogram(result["blocks"])
    sizes = sorted(int(k) for k in hist)
    check("more than one block size is present", len(sizes) > 1, f"sizes {sizes}")
    check("large blocks exist (flat areas collapsed)", max(sizes) >= 4,
          f"largest {max(sizes)}x{max(sizes)}")
    check("small blocks exist (fine detail kept)", min(sizes) == 1)
    fine = hist.get(1, {}).get("share_of_canvas", 0)
    coarse = sum(v["share_of_canvas"] for k, v in hist.items() if int(k) >= 4)
    check("fine and coarse both carry real area", fine > 0.05 and coarse > 0.2,
          f"1x1 {fine:.0%}, >=4x4 {coarse:.0%}")

    print("\n5. the quadtree itself is sound")
    labels, base = block_map(result["blocks"] * 0 + np.arange(
        result["blocks"].shape[0] * result["blocks"].shape[1]).reshape(result["blocks"].shape) % 3,
        max_block=4)
    ok = True
    for value in np.unique(labels):
        ys, xs = np.where(labels == value)
        region = labels[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        ok &= bool((region == value).all()) and region.shape[0] == region.shape[1]
    check("every block is a filled square", ok, f"block base {base}")

    print("\n6. detail survives")
    src = result["source"]
    cells, _ = cell_blocks(src, 4)
    fg = ~result["background"]
    cells = cells[: fg.shape[0], : fg.shape[1]]
    out_rgb = rgba[..., :3].astype(np.float32)[: cells.shape[0], : cells.shape[1]]
    live = fg[: cells.shape[0], : cells.shape[1]]
    hit = miss = 0
    for axis in (0, 1):
        s = redmean(cells[1:, :], cells[:-1, :]) if axis == 0 else redmean(cells[:, 1:], cells[:, :-1])
        o = redmean(out_rgb[1:, :], out_rgb[:-1, :]) if axis == 0 else redmean(out_rgb[:, 1:], out_rgb[:, :-1])
        pair = (live[1:, :] & live[:-1, :]) if axis == 0 else (live[:, 1:] & live[:, :-1])
        hit += int(((s > 40) & (o > 40) & pair).sum())
        miss += int(((s > 40) & ~(o > 40) & pair).sum())
    recall = hit / max(hit + miss, 1)
    check("strong source edges are reproduced (recall > 0.8)", recall > 0.8, f"recall {recall:.2f}")

    print("\n7. shared palette keeps a set consistent")
    second = make_draft(seed=11)
    palette_set, _ = build_palette([draft, second], cell=4, colors=24)
    a = pixelate(draft, cell=4, palette=palette_set)
    b = pixelate(second, cell=4, palette=palette_set)
    check("both frames use only the shared palette",
          all(verify.palette_conformance(r["rgba"], palette) == 1.0 for r in (a, b)))
    frozen, _ = build_palette([draft], cell=4, colors=16)
    widened, _ = build_palette([draft, second], cell=4, colors=24, base=frozen)
    check("a frozen palette is kept verbatim and only extended",
          len(widened) >= len(frozen) and np.allclose(widened[: len(frozen)], frozen),
          f"{len(frozen)} -> {len(widened)} colours")

    print("\n8. determinism")
    again = pixelate(draft, cell=4, colors=24)
    check("same input, same bytes", np.array_equal(again["rgba"], rgba))

    print("\n9. glow, hair streaks, shadow and the eyes")
    # A hand-built 20x20 case: palette 0 = hair, 1 = white, 2 = grey glow, 3 = skin.
    pal = np.array([[20, 18, 26], [240, 240, 240], [150, 150, 152], [252, 211, 184]], np.float32)
    idx = np.zeros((20, 20), int)
    bg = np.zeros((20, 20), bool)
    bg[:, :4] = True  # open background on the left
    idx[6:14, 4:16] = 0  # hair block
    idx[9, 10] = 1  # a white gloss streak, walled in by hair
    bg[2:5, 2:5] = True  # a wide open area (the ground shadow lives here)
    idx[3, 3] = 2  # a grey pixel alone in that open area
    idx[16:20, 4:12] = 2  # a soft grey ground shadow...
    bg[16:20, 0:4] = True  # ...touching the open background on the left
    protect = np.zeros((20, 20), bool)
    protect[15:19, 14:19] = True
    idx[16, 16] = 1  # the eye highlight inside the protected box

    cleaned, background = clear_haze(idx, pal, bg, protect=protect)
    check("a streak walled in by hair is repainted, not punched out",
          cleaned[9, 10] == 0 and not background[9, 10], f"cell -> {cleaned[9, 10]}")
    check("a grey pixel standing in the open becomes transparent", background[3, 3])
    check("the body of the ground shadow is cleared", bool(background[17:19, 5:9].all()))
    check("the protected eye highlight is untouched",
          not background[16, 16] and cleaned[16, 16] == 1)
    check("no opaque light pixel is left outside the protected box",
          not any(
              pal[cleaned[y, x]].mean() > 112
              and pal[cleaned[y, x]].max() - pal[cleaned[y, x]].min() < 30
              and not background[y, x]
              for y in range(20)
              for x in range(20)
              if not protect[y, x]
          ))
    real = pixelate(draft, cell=4, colors=24)
    check("a clean draft loses nothing (there is no glow to clear)",
          real["haze_removed"] == 0, f"{real['haze_removed']} cells cleared")
    check("pixelate reports the protected eye region", real["eyes"] is not None
          and bool(real["eyes"].any()))
    with_haze = pixelate(draft, cell=4, colors=24, clear_light_haze=False)
    box = real["eyes"]
    check("nothing inside the eye region changed",
          int(((~with_haze["background"]) & box).sum()) == int(((~real["background"]) & box).sum()))

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
