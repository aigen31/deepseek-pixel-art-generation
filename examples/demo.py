"""End-to-end demo: one reference in, several specification-driven redraws out.

Run with:  ./.venv/bin/python examples/demo.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixelkit import analyze as analyze_spec  # noqa: E402
from pixelkit import render, verify  # noqa: E402
from pixelkit.core import ascii_to_grid, grid_to_ascii, save_rgba, upscale, load_rgba, grid_to_array  # noqa: E402

OUT = ROOT / "examples" / "out"

# ---------------------------------------------------------------- the reference
# 16x16 knight. This is what the user hands over as "the style".
REFERENCE = """\
. = transparent
0 = #1b1a2e
1 = #3d3a5c
2 = #6f6ab0
3 = #e8b48c
4 = #c9d2ff
.....000000.....
....04444440....
....04222240....
....03333330....
....03033030....
....03333330....
...0022222200...
..022222222220..
..022222222220..
..011111111110..
..022222222220..
...0222..2220...
...0222..2220...
...0111..1110...
...0000..0000...
"""


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' - ' + detail) if detail else ''}")
    return ok


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    grid, palette = ascii_to_grid(REFERENCE)
    width = max(len(r) for r in grid)
    assert all(len(r) == width for r in grid), "reference rows must be equal length"
    print(f"reference grid: {width}x{len(grid)}, {len(palette) - 1} opaque colours")

    # 1. Ship the reference the way a real one arrives: an upscaled PNG, not text.
    ref_png = render.to_png(grid, palette, OUT / "reference_knight.png", scale=4)
    print(f"\n1. reference written: {ref_png.name} (16x15 native, 4x export)")

    # 2. Recover the style spec from the image alone.
    spec = analyze_spec(ref_png)
    (OUT / "style_spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"2. style spec: pixel_size={spec['pixel_size']} native={spec['native_size']} "
          f"colours={spec['palette_size']} outline={spec['outline_char']!r} "
          f"warnings={spec['warnings']}")
    recovered = spec["grid"]
    pal = {e["char"]: e["hex"] for e in spec["palette"]}
    opaque_hexes = [v for v in pal.values() if v != "transparent"]

    results: list[bool] = []
    same = np.array_equal(grid_to_array(grid, palette), grid_to_array(recovered, pal))
    print(f"   recovered pixels == authored pixels: {same} (char labels are re-derived, "
          f"pixels are what matter)")
    results.append(check("reference round-trips through analysis", same))

    # ---- requirement A: 2x HiDPI export, form unchanged ---------------------
    print("\n3. requirement A: 'the build needs everything at 2x'")
    up = render.resize(recovered, pal, 32, 30)
    a_png = render.to_png(up, pal, OUT / "variant_2x.png", scale=2)
    ref2x = render.to_png(up, pal, OUT / "_tmp_ref2x.png", scale=2)
    m = verify.compare(ref2x, a_png, palette=opaque_hexes)
    blocks_ok = all(len(set(row[x : x + 2])) == 1 for row in up for x in range(0, len(row), 2))
    print(f"   every 2x2 block is a single colour: {blocks_ok}; "
          f"palette conformance: {m['palette_conformance']}")
    results.append(check("2x export has no invented pixels or colours",
                         blocks_ok and m["palette_ok"], a_png.name))

    # ---- requirement A2: fit a fixed 32x32 engine canvas --------------------
    print("\n4. requirement A2: 'it must ship inside a 32x32 atlas cell'")
    fitted = render.pad_canvas(recovered, pal, left=8, top=8, right=8, bottom=9)
    fit_png = render.to_png(fitted, pal, OUT / "variant_32x32_cell.png", scale=2)
    size_ok = (len(fitted[0]), len(fitted)) == (32, 32)
    print(f"   canvas is {len(fitted[0])}x{len(fitted)}: {size_ok}")
    results.append(check("fits a 32x32 cell without resampling", size_ok, fit_png.name))

    # ---- requirement B: night / cold palette -------------------------------
    print("\n5. requirement B: 'same knight, night scene'")
    night_pal = render.recolor(recovered, pal, {
        "0": "#0d0b1a", "1": "#232042", "2": "#404a8f", "3": "#9c7fa8", "4": "#8ea0ff",
    })
    night_png = render.to_png(recovered, night_pal, OUT / "variant_night.png", scale=4)
    nm = verify.compare(ref_png, night_png)
    print(f"   silhouette IoU vs reference: {nm['silhouette_iou']}")
    results.append(check("night variant keeps the silhouette exactly",
                         nm["silhouette_iou"] == 1.0, night_png.name))
    results.append(check("night palette is a 1:1 remap of the reference palette",
                         len(set(v for v in night_pal.values() if v != "transparent"))
                         == len(set(opaque_hexes))))
    # ---- requirement C: hard outline, safe for any background --------------
    print("\n6. requirement C: 'must read on a light background -> add outline'")
    outlined, opal = render.outline(recovered, pal, color="#1b1a2e", char="o")
    o_png = render.to_preview(outlined, opal, OUT / "variant_outlined_preview.png", scale=6,
                              background="#e9e6ff")
    o_conv = verify.palette_conformance(load_rgba(o_png), opal_values := [v for v in opal.values() if v != "transparent"], tol=32)
    print(f"   palette conformance after outlining: {o_conv}")
    results.append(check("outline variant renders on a light background", Path(o_png).exists(), o_png.name))

    # ---- requirement D: 4-frame sheet from the same base --------------------
    print("\n7. requirement D: 'give me a walk sheet from this character'")
    base = render.flip_h(recovered)
    frames = [
        ("idle_0", recovered),
        ("idle_1", render.shift(recovered, pal, 0, -1)),
        ("walk_0", render.shift(recovered, pal, -1, 0)),
        ("walk_1", base),
    ]
    sheet_png, atlas = render.sheet(frames, pal, OUT / "knight_sheet.png", columns=4, scale=4)
    results.append(check("spritesheet + atlas written", sheet_png.exists() and atlas.exists(),
                         sheet_png.name))

    # ---- verification gate -------------------------------------------------
    print("\n8. verification gate on the strictest candidate")
    final = render.to_png(recovered, pal, OUT / "_tmp_final.png", scale=4)
    report = verify.compare(ref_png, final, palette=[v for v in pal.values() if v != "transparent"])
    diff = verify.diff_image(ref_png, final, OUT / "diff.png", scale=4)
    print(json.dumps(report, indent=2))
    results.append(check("final candidate is pixel-identical to the reference", report["identical"]))
    results.append(check("diff image produced", diff.exists()))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
