"""Redraw every draft in ``in/`` as real pixel art on one shared palette.

The drafts are AI "pixel art": anti-aliased, off-grid, thousands of near-duplicate
colours, and — the part worth keeping — blocks of visibly different sizes.  This
driver reduces each one to a crisp ``cell``-pixel lattice using
``pixelkit.pixelate``, so the large shapes collapse into big squares while the
blush stripes, boot laces and buckle rim stay one pixel wide.

Run:  ./.venv/bin/python sprites/redraw_drafts.py --colors 62 --palette sprites/palette.json
      ./.venv/bin/python sprites/redraw_drafts.py --cell 5   # coarser lattice

The first form is the shipped set: it freezes the palette that set is defined by.
Omitting ``--palette`` re-cuts one from the drafts, which shifts every colour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixelkit import render, verify  # noqa: E402
from pixelkit.core import dump_json, grid_to_ascii, load_rgba, save_rgba, upscale  # noqa: E402
from pixelkit.pixelate import (  # noqa: E402
    background_colour,
    block_histogram,
    build_palette,
    cell_blocks,
    pixelate,
    redmean,
)

IN = ROOT / "in"
OUT = ROOT / "out" / "pixel"
BACKDROP = (233, 230, 255)

# Draft order is the order the poses were supplied in: the sitting expressions
# first, then the two broom frames. ``Qwen-Image-2512_00029_`` keeps the name it
# arrived with; it is the eleventh frame and sits with the other sitting poses.
DRAFTS = [
    "счастлива",
    "радуется",
    "довольна",
    "ухмылка",
    "взволнована",
    "ждёт",
    "задумалась",
    "недовольна",
    "Qwen-Image-2512_00029_",
    "read",
    "на_метле_влево",
    "на_метле_вправо",
]


def draft_path(stem: str) -> Path:
    """The draft file for a stem, whatever extension it arrived with."""
    hits = sorted(IN.glob(f"{stem}.*"))
    if not hits:
        raise FileNotFoundError(stem)
    return hits[0]


def load_palette(path: str | Path | None) -> np.ndarray | None:
    """Read a frozen palette: either a run report or a bare list of hex colours."""
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    hexes = data["palette"] if isinstance(data, dict) else data
    hexes = [h for h in hexes if isinstance(h, str) and h.startswith("#")]
    return np.array([[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in hexes], dtype=np.float32)


def truth_mask(arr: np.ndarray, cell: int) -> np.ndarray:
    """Independent silhouette: per-pixel background test, then a majority vote.

    Deliberately does not use the pipeline's flood fill, so the IoU below is a
    real check rather than a tautology.
    """
    bg = background_colour(arr)
    far = redmean(arr[..., :3].astype(np.float32), bg) > 60
    h, w = far.shape
    pad_h, pad_w = (-h) % cell, (-w) % cell
    far = np.pad(far, ((0, pad_h), (0, pad_w)))
    ny, nx = far.shape[0] // cell, far.shape[1] // cell
    return far.reshape(ny, cell, nx, cell).mean((1, 3)) > 0.5


def edge_scores(source: np.ndarray, out: np.ndarray, mask: np.ndarray, cell: int) -> dict:
    """Precision/recall over strong adjacent-cell colour changes ("real edges").

    Recall near 1 means fine structure survived; precision near 1 means the
    reduction did not invent speckle where the draft was flat.
    """
    src, _ = cell_blocks(source, cell)
    h = min(src.shape[0], out.shape[0])
    w = min(src.shape[1], out.shape[1])
    src, out, mask = src[:h, :w], out[:h, :w].astype(np.float32), mask[:h, :w]
    hit = miss = spurious = 0
    for axis in (0, 1):
        d_src = redmean(src[1:, :], src[:-1, :]) if axis == 0 else redmean(src[:, 1:], src[:, :-1])
        d_out = redmean(out[1:, :], out[:-1, :]) if axis == 0 else redmean(out[:, 1:], out[:, :-1])
        live = (
            (mask[1:, :] & mask[:-1, :]) if axis == 0 else (mask[:, 1:] & mask[:, :-1])
        )
        s, o = d_src > 40, d_out > 40
        hit += int((s & o & live).sum())
        miss += int((s & ~o & live).sum())
        spurious += int((~s & o & live).sum())
    recall = hit / max(hit + miss, 1)
    precision = hit / max(hit + spurious, 1)
    return {"edge_recall": round(recall, 4), "edge_precision": round(precision, 4)}


def contact_sheet(tiles: list[tuple[str, Image.Image]], path: Path, zoom: int = 3) -> Path:
    cw = max(t.width for _, t in tiles) * zoom + 24
    ch = max(t.height for _, t in tiles) * zoom + 34
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * ch), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (name, tile) in enumerate(tiles):
        x, y = (i % cols) * cw, (i // cols) * ch
        sheet.paste(tile.resize((tile.width * zoom, tile.height * zoom), Image.NEAREST), (x + 12, y + 28))
        draw.text((x + 12, y + 10), name, fill=(20, 20, 30))
    sheet.save(path)
    return path


def run(
    cell: int,
    colors: int,
    max_block: int,
    out_dir: Path | None = None,
    frozen: np.ndarray | None = None,
) -> int:
    drafts = [(name, draft_path(name)) for name in DRAFTS]
    missing = [str(p) for _, p in drafts if not p.exists()]
    if missing:
        print("missing drafts: " + ", ".join(missing), file=sys.stderr)
        return 2

    sources = [load_rgba(path)[..., :3] for _, path in drafts]
    base = frozen
    palette, _ = build_palette(sources, cell=cell, colors=colors, base=base)
    palette_hex = ["#%02x%02x%02x" % tuple(int(v) for v in c) for c in palette]
    frozen_note = f", extended from {len(base)} frozen" if base is not None else ""
    print(
        f"shared palette: {len(palette_hex)} colours{frozen_note}, "
        f"cell={cell}px, max block={max_block} cells"
    )

    out = out_dir or OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(exist_ok=True)
    (out / "spec").mkdir(exist_ok=True)
    grids = out / "grids"
    grids.mkdir(parents=True, exist_ok=True)

    report, tiles = [], []
    for name, path in drafts:
        src = load_rgba(path)[..., :3]
        result = pixelate(src, cell=cell, palette=palette, max_block=max_block)
        rgba, grid, table = result["rgba"], result["grid"], result["table"]

        native = save_rgba(rgba, out / f"{name}.png")
        scaled = save_rgba(upscale(rgba, cell), out / f"{name}@{cell}x.png")
        backdrop = Image.new("RGBA", (rgba.shape[1], rgba.shape[0]), (*BACKDROP, 255))
        backdrop.alpha_composite(Image.fromarray(rgba, "RGBA"))
        preview = backdrop.convert("RGB").resize(
            (rgba.shape[1] * cell, rgba.shape[0] * cell), Image.NEAREST
        )
        preview.save(out / "preview" / f"{name}.png")
        tiles.append((name, preview))

        (grids / f"{name}.txt").write_text(
            grid_to_ascii(
                grid,
                table,
                header=f"drafts/{name} - pixelkit pixelate, cell={cell}, {len(palette_hex)} colours",
            ),
            encoding="utf-8",
        )
        spec = {
            "name": name,
            "source": str(path.relative_to(ROOT)),
            "cell": cell,
            "native_size": [rgba.shape[1], rgba.shape[0]],
            "export_scale": cell,
            "palette": [{"char": c, "hex": v} for c, v in table.items() if v != "transparent"],
            "grid": grid,
        }
        dump_json(spec, out / "spec" / f"{name}.json")

        reloaded = load_rgba(native)
        conformance = verify.palette_conformance(reloaded, palette_hex)
        unique = len(np.unique(reloaded[reloaded[..., 3] > 0][:, :3], axis=0))
        alpha = np.unique(reloaded[..., 3]).tolist()
        truth = truth_mask(result["source"], cell)
        foreground = ~result["background"]
        h = min(truth.shape[0], foreground.shape[0])
        w = min(truth.shape[1], foreground.shape[1])

        def as_rgba(flag: np.ndarray) -> np.ndarray:
            a = flag[:h, :w].astype(np.uint8) * 255
            return np.dstack([a, a, a, a])

        mask_iou = verify.silhouette_iou(as_rgba(foreground), as_rgba(truth))
        entry = {
            "name": name,
            "native_size": [rgba.shape[1], rgba.shape[0]],
            "source_crop": list(result["source_crop"]),
            "origin": list(result["origin"]),
            "reduction": round(result["source_crop"][0] / rgba.shape[0], 2),
            "palette_conformance": round(conformance, 4),
            "colours_used": unique,
            "alpha_values": alpha,
            "alpha_binary": alpha in ([0], [0, 255], [255]),
            "silhouette_iou_vs_source": round(mask_iou, 4),
            "haze_and_shadow_cells_removed": int(result["haze_removed"]),
            "eye_region_cells_protected": int(result["eyes"].sum()) if result["eyes"] is not None else 0,
            "block_histogram": block_histogram(result["blocks"]),
            **edge_scores(result["source"], rgba[..., :3], foreground, cell),
            "files": {
                "native": str(native.relative_to(ROOT)),
                "export": str(scaled.relative_to(ROOT)),
                "preview": str((out / "preview" / f"{name}.png").relative_to(ROOT)),
                "grid": str((grids / f"{name}.txt").relative_to(ROOT)),
            },
        }
        report.append(entry)
        flag = "OK " if entry["palette_conformance"] == 1.0 else "BAD"
        print(
            f"  [{flag}] {name:<14} {entry['native_size'][0]:>3}x{entry['native_size'][1]:<3} "
            f"colours={unique:>2} iou={entry['silhouette_iou_vs_source']:.3f} "
            f"edge r/p={entry['edge_recall']:.2f}/{entry['edge_precision']:.2f} "
            f"haze-={entry['haze_and_shadow_cells_removed']:<5} eyes={entry['eye_region_cells_protected']}"
        )

    dump_json(
        {
            "cell": cell,
            "max_block": max_block,
            "palette_size": len(palette_hex),
            "palette": palette_hex,
            "sprites": report,
        },
        out / "report.json",
    )
    dump_json({"cell": cell, "palette": palette_hex}, out / "palette.json")
    sheet = contact_sheet(tiles, out / "sheet.png", zoom=1)
    print(f"{len(report)} sprites -> {out.relative_to(ROOT)}  (sheet: {sheet.relative_to(ROOT)})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, default=4, help="source pixels per sprite pixel")
    parser.add_argument("--colors", type=int, default=56, help="palette entries before capping")
    parser.add_argument("--max-block", type=int, default=8, help="largest merged square, in cells")
    parser.add_argument("--out", default=None, help="output directory (default out/pixel)")
    parser.add_argument(
        "--palette",
        default=None,
        help="freeze an existing palette (a report.json or a list of hex) and only extend it",
    )
    args = parser.parse_args(argv)
    return run(
        args.cell,
        args.colors,
        args.max_block,
        Path(args.out).resolve() if args.out else None,
        load_palette(args.palette),
    )


if __name__ == "__main__":
    raise SystemExit(main())
