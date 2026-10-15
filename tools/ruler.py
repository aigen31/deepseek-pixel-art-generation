"""Authoring aid: render a crop of a grid at high zoom with a pixel ruler.

Use it to look at a region you are about to edit and to read the exact rows.

    ./.venv/bin/python tools/ruler.py reference/mascot-witch.png --box 0,60,83,122 --scale 6 \
        --out out/ruler_lower.png --text out/ruler_lower.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixelkit.analyze import analyze_from_array  # noqa: E402
from pixelkit.core import grid_to_array, load_rgba  # noqa: E402

PAPER = (245, 245, 250)
INK = (60, 60, 80)
GRID = (210, 210, 225)
LABEL_EVERY = 5


def build(image: str, box: tuple[int, int, int, int] | None, scale: int, grid_step: int):
    arr = load_rgba(image)
    if box:
        x0, y0, x1, y1 = box
        arr = arr[y0 : y1 + 1, x0 : x1 + 1]
    else:
        x0 = y0 = 0
    h, w = arr.shape[:2]
    margin = 18
    canvas = Image.new("RGB", (margin + w * scale, margin + h * scale), PAPER)
    sprite = Image.fromarray(arr, "RGBA").resize((w * scale, h * scale), Image.NEAREST)
    canvas.paste(sprite, (margin, margin), sprite)

    d = ImageDraw.Draw(canvas)
    for x in range(0, w + 1, grid_step):
        px = margin + x * scale
        d.line([(px, margin), (px, margin + h * scale)], fill=GRID)
        if (x + x0) % LABEL_EVERY == 0:
            d.text((px + 1, 2), str(x + x0), fill=INK)
    for y in range(0, h + 1, grid_step):
        py = margin + y * scale
        d.line([(margin, py), (margin + w * scale, py)], fill=GRID)
        if (y + y0) % LABEL_EVERY == 0:
            d.text((1, py + 1), str(y + y0), fill=INK)
    return canvas, arr, (x0, y0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--box", default=None, help="x0,y0,x1,y1 inclusive; default whole image")
    p.add_argument("--scale", type=int, default=6)
    p.add_argument("--step", type=int, default=1, help="ruler line spacing in pixels")
    p.add_argument("--out", required=True)
    p.add_argument("--text", default=None, help="also write the rows as a text grid")
    args = p.parse_args(argv)

    box = tuple(int(v) for v in args.box.split(",")) if args.box else None
    canvas, arr, (x0, y0) = build(args.image, box, args.scale, args.step)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(args.out, canvas.size)

    if args.text:
        spec = analyze_from_array(arr)
        pal = {e["char"]: e["hex"] for e in spec["palette"]}
        lines = [f"# crop x={x0} y={y0} size={arr.shape[1]}x{arr.shape[0]}", "# palette"]
        lines += [f"{c} = {v}" for c, v in pal.items()]
        lines.append("# grid")
        lines += [f"{y0 + i:>3} | {row}" for i, row in enumerate(spec["grid"])]
        Path(args.text).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(args.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
