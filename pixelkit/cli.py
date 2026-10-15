"""Command line interface: python -m pixelkit <command> ..."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from .analyze import analyze as analyze_image
from .analyze import conform_to_reference
from .core import ascii_to_grid, dump_json, grid_to_ascii, load_rgba, quantize_rgba, save_rgba, upscale
from .pixelate import block_histogram, build_palette, pixelate
from .render import sheet as pack_sheet
from .render import to_png, to_preview
from .verify import compare, diff_image


def _as_char_map(palette):
    """Accept ``{char: hex}`` and the ``[{char, hex}, ...]`` an ``analyze`` spec carries.

    Keeping the char keys is what lets a spec go straight from ``analyze`` into
    ``render``: the grid's characters have to resolve to the colours they arrived with.
    """
    if isinstance(palette, dict):
        return palette
    if isinstance(palette, list) and palette and all(isinstance(e, dict) for e in palette):
        return {e["char"]: e["hex"] for e in palette if "char" in e and "hex" in e}
    raise ValueError("palette must be a {char: hex} map or a list of {char, hex} entries")


def _load_spec(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "text" in data:
        grid, palette = ascii_to_grid(data["text"])
    else:
        grid, palette = data["grid"], _as_char_map(data["palette"])
    return grid, palette, data


HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def extract_palette(raw):
    """Accept a style spec, a char->hex map, a list of hex, or a list of palette entries."""
    if isinstance(raw, dict) and "palette" in raw:
        raw = raw["palette"]
    if isinstance(raw, dict):
        values = list(raw.values())
    elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
        values = [e.get("hex") for e in raw]
    elif isinstance(raw, list):
        values = list(raw)
    else:
        values = []
    return [v for v in values if isinstance(v, str) and HEX_RE.match(v)]


def cmd_analyze(args) -> int:
    result = analyze_image(args.image, cell=args.cell)
    if args.json:
        dump_json(result, args.json)
    if args.ascii:
        Path(args.ascii).parent.mkdir(parents=True, exist_ok=True)
        Path(args.ascii).write_text(
            grid_to_ascii(result["grid"], {e["char"]: e["hex"] for e in result["palette"]},
                          header=str(args.image)),
            encoding="utf-8",
        )
    summary = {k: v for k, v in result.items() if k != "grid"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_quantize(args) -> int:
    arr = quantize_rgba(load_rgba(args.image), colors=args.colors)
    print(save_rgba(arr, args.output))
    return 0


def cmd_conform(args) -> int:
    colors = extract_palette(json.loads(Path(args.palette).read_text(encoding="utf-8")))
    if not colors:
        print("no colours found in the palette file", file=sys.stderr)
        return 2
    arr = conform_to_reference(load_rgba(args.image), colors)
    print(save_rgba(arr, args.output))
    return 0


def cmd_render(args) -> int:
    grid, palette, data = _load_spec(args.spec)
    scale = args.scale or data.get("scale", 1)
    out = args.output or data.get("output", "out.png")
    if args.preview:
        print(to_preview(grid, palette, out, scale=scale, background=args.background))
    else:
        print(to_png(grid, palette, out, scale=scale))
    return 0


def cmd_verify(args) -> int:
    palette = None
    if args.palette:
        palette = extract_palette(json.loads(Path(args.palette).read_text(encoding="utf-8"))) or None
    result = compare(args.reference, args.candidate, palette=palette)
    if args.diff:
        diff_image(args.reference, args.candidate, args.diff, scale=args.scale)
        result["diff_image"] = str(args.diff)
    print(json.dumps(result, indent=2))
    return 0


def cmd_sheet(args) -> int:
    data = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if "palette" not in data or "frames" not in data:
        print(
            "sheet spec needs {\"palette\": {char: hex}, \"frames\": [{\"name\": str, "
            "\"grid\": [rows]}]}. An atlas JSON written next to a spritesheet is an "
            "output format and cannot be re-packed directly.",
            file=sys.stderr,
        )
        return 2
    palette = data["palette"]
    frames = [
        (f["name"], ascii_to_grid(f["text"])[0] if "text" in f else f["grid"])
        for f in data["frames"]
    ]
    png, atlas = pack_sheet(
        frames, palette, args.output, columns=args.columns, scale=args.scale, padding=args.padding
    )
    print(json.dumps({"png": str(png), "atlas": str(atlas)}, indent=2))
    return 0


def cmd_pixelate(args) -> int:
    sources = [load_rgba(path)[..., :3] for path in args.image]
    palette, _ = build_palette(sources, cell=args.cell, colors=args.colors)
    summary = []
    for path in args.image:
        result = pixelate(
            load_rgba(path)[..., :3], cell=args.cell, palette=palette, max_block=args.max_block
        )
        stem = Path(path).stem
        out = Path(args.output_dir) / f"{stem}.png"
        save_rgba(result["rgba"], out)
        spec = Path(args.output_dir) / "spec" / f"{stem}.json"
        dump_json(
            {
                "name": stem,
                "source": path,
                "cell": args.cell,
                "native_size": [result["rgba"].shape[1], result["rgba"].shape[0]],
                "palette": [{"char": c, "hex": v} for c, v in result["table"].items()
                            if v != "transparent"],
                "grid": result["grid"],
            },
            spec,
        )
        if args.scale != 1:
            save_rgba(upscale(result["rgba"], args.scale), Path(args.output_dir) / f"{stem}@{args.scale}x.png")
        summary.append(
            {
                "name": stem,
                "png": str(out),
                "spec": str(spec),
                "native_size": [result["rgba"].shape[1], result["rgba"].shape[0]],
                "colours": int(np.unique(result["rgba"][..., :3][result["rgba"][..., 3] > 0],
                                         axis=0).shape[0]),
                "blocks": block_histogram(result["blocks"]),
            }
        )
    print(json.dumps({"cell": args.cell, "palette_size": len(palette), "sprites": summary}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pixelkit", description="pixel-art analysis / render / verify")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="reference image -> style spec JSON")
    a.add_argument("image")
    a.add_argument("--cell", type=int, default=None, help="force the native pixel size")
    a.add_argument("--json", default=None)
    a.add_argument("--ascii", default=None, help="also write the grid as text")
    a.set_defaults(func=cmd_analyze)

    q = sub.add_parser("quantize", help="snap a noisy image onto a clean palette")
    q.add_argument("image")
    q.add_argument("-o", "--output", required=True)
    q.add_argument("--colors", type=int, default=24)
    q.set_defaults(func=cmd_quantize)

    c = sub.add_parser("conform", help="recolour every pixel to a reference palette")
    c.add_argument("image")
    c.add_argument("--palette", required=True, help="JSON spec from `analyze`")
    c.add_argument("-o", "--output", required=True)
    c.set_defaults(func=cmd_conform)

    r = sub.add_parser("render", help="spec JSON -> PNG")
    r.add_argument("spec")
    r.add_argument("-o", "--output", default=None)
    r.add_argument("--scale", type=int, default=None)
    r.add_argument("--preview", action="store_true", help="flatten on a backdrop")
    r.add_argument("--background", default="#20242c")
    r.set_defaults(func=cmd_render)

    v = sub.add_parser("verify", help="candidate vs reference metrics + diff image")
    v.add_argument("reference")
    v.add_argument("candidate")
    v.add_argument("--palette", default=None, help="JSON spec from `analyze`")
    v.add_argument("--diff", default=None)
    v.add_argument("--scale", type=int, default=1)
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("sheet", help="frames JSON -> spritesheet + atlas")
    s.add_argument("spec")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--columns", type=int, default=4)
    s.add_argument("--scale", type=int, default=1)
    s.add_argument("--padding", type=int, default=0)
    s.set_defaults(func=cmd_sheet)

    x = sub.add_parser(
        "pixelate",
        help="anti-aliased AI draft -> real sprite with mixed pixel sizes",
    )
    x.add_argument("image", nargs="+", help="draft images (share one palette)")
    x.add_argument("-o", "--output-dir", default="out/pixel")
    x.add_argument("--cell", type=int, default=4, help="source pixels per sprite pixel")
    x.add_argument("--colors", type=int, default=56, help="palette entries before capping")
    x.add_argument("--max-block", type=int, default=8, help="largest merged square, in cells")
    x.add_argument("--scale", type=int, default=4, help="also write an integer-scaled export")
    x.set_defaults(func=cmd_pixelate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
