---
name: pixel-art-redraw
description: Redraw arbitrary pixel art to a written specification using one reference image, without a diffusion model. Covers reference analysis (native grid, palette, colour roles, line weight), translating requirements into deterministic grid operations, rendering PNG/spritesheets, and the verification gates that prove the result is still real pixel art. Use when asked to recolour, resize, re-pose, restyle, animate, or reskin pixel art, sprites, tilesets, or icons from an existing reference.
---

# Redrawing pixel art from a reference

Producing pixel art is a data problem, not a painting problem. A sprite is a small
integer grid plus a palette; once both are explicit, "redraw this but ..." becomes
a sequence of reversible operations you can verify. Generated images that merely
*look* like pixel art fail here: they arrive anti-aliased, off-grid, and with
hundreds of near-duplicate colours.

## Invariants (never break these)

1. **One colour per pixel.** No anti-aliasing, no soft edges, no gradients. If a
   final PNG has more colours than the palette, it is wrong.
2. **Integer scaling only.** Enlarge with nearest-neighbour by 2x/3x/4x. Never
   bilinear, never 1.875x.
3. **Colours come from the reference palette.** Variants remap the reference's
   colours 1:1 (background → night background, mid → night mid). Inventing new
   per-pixel colour is how a variant stops matching the reference.
4. **Transparent stays transparent.** Alpha is binary; render previews on a
   backdrop instead of faking transparency.
5. **The grid is the deliverable.** Always keep the text grid (`examples/out/*.txt`
   or JSON spec) next to the PNG. A PNG alone cannot be revised at pixel level.

## Workflow

### 1. Intake — get a reference and a written requirement

Ask for the reference image and the requirement in a form that maps to operations:
canvas size, palette shift, silhouette change, new frames, export scale. Vague
briefs ("make it cooler") must be turned into operations *before* drawing, and the
translation restated in one sentence for confirmation.

### 2. Analyse the reference

```bash
./.venv/bin/python -m pixelkit analyze reference.png --json out/style_spec.json --ascii out/reference.txt
```

The spec gives `pixel_size` (native grid block, e.g. 4 for a 4x export),
`native_size`, the palette with roles (`background`, `outline`, `shadow`, `mid`,
`highlight`), `bbox`, `outline_char`, and warnings. Read the warnings:

- *no integer pixel grid detected* → the reference is anti-aliased or AI-made;
  run `quantize` first, then analyse again.
- *too many colours* → shading noise; posterise to ≤ 24 colours.

Then **read the grid yourself** and confirm it matches what the image looks like.
If the grid is mush, the reference was not clean pixel art and everything
downstream would inherit the problem.

### 3. Write the requirement as operations

| Requirement | Operation |
|---|---|
| another size / HiDPI export | `render.resize` to an exact multiple, then verify blocks are uniform |
| fit a fixed engine cell | `render.pad_canvas` (centre, never resample) |
| palette variant (night, desert, team colour) | `render.recolor` — a 1:1 char→hex map |
| readability on dark/light backgrounds | `render.outline` (grows the canvas by 1px) |
| pose / direction change | edit the grid rows directly; keep the same palette |
| animation frames | derived shifts: 1px bob, leg rows swapped, `flip_h` for a mirror |
| tileset / sheet | `render.sheet` → PNG + JSON atlas |

Prefer editing the **text grid** for anything structural, and operations for
anything mechanical. Hand-editing is legitimate: you are the artist, the toolkit
is the brush.

### 4. Render

```bash
./.venv/bin/python -m pixelkit render spec.json -o out/candidate.png --scale 4
./.venv/bin/python -m pixelkit render spec.json -o out/preview.png --scale 6 --preview --background "#e9e6ff"
```

`spec.json` = `{"palette": {...}, "grid": [...], "scale": 4}`, or `{"text": "<legend + rows>"}`.

### 5. Verify — do not skip this

```bash
./.venv/bin/python -m pixelkit verify reference.png out/candidate.png --palette out/style_spec.json --diff out/diff.png
```

Gates, in order of strictness:

- `palette_conformance == 1.0` — every opaque pixel is exactly a palette colour.
  A value below 1.0 means anti-aliasing crept in: quantise, do not reframe it as
  "smooth shading".
- `silhouette_iou == 1.0` — required when the brief says "same character".
  Below ~0.9 means the shape drifted.
- `identical == true` — required when the brief says "same sprite, different
  palette/size".
- `diff.png` — magenta marks mismatched pixels. Look at it before claiming success.

Also run `verify.silhouette_check` on any hand-written grid: `empty`,
`components > 1` (stray pixels), or `too_small` are all real defects.

### 6. Deliver

Ship the PNG (at the requested export scale), the preview, the grid text/JSON, and
the verification numbers. Name files for the requirement, not for the iteration.

## Working without the reference in hand

If only a description exists, author the grid as text first with a deliberate
palette (4–8 colours: outline, shadow, mid, highlight, skin/prop accents), render,
look at it, iterate. That grid then *becomes* the reference for later variants, so
the pipeline is identical.

## Generative models: where they help and where they hurt

An image model is useful only for the step the grid cannot do: inventing a
silhouette that no existing sprite has. When one is used:

1. Ask for pixel art at an explicit grid and palette (name the colours and the
   cell size in the prompt).
2. Downscale to the target grid with nearest-neighbour, then
   `pixelkit conform --palette style_spec.json` to snap every pixel onto the
   reference palette.
3. Run the verification gates above. Treat a candidate that fails
   `palette_conformance` as noise, not as an artistic choice.
4. Never let the model produce the *deliverable* PNG; it produces a candidate to
   be traced into the grid.

## Companion skill

`pixel-art-sprites` (installed in this workspace) carries the craft side —
silhouette readability, 4-frame animation economy, palette construction, tileset
seams, and the failure catalogue in `references/sharp_edges.md`. Load it alongside
this skill: it decides *what* to draw, this one guarantees *how* the pixels land
and how to prove it. When the two disagree about anti-aliasing or non-integer
scaling, this skill wins — `verify` enforces it mechanically.

## Layout

- `pixelkit/` — the toolkit (`analyze`, `render`, `verify`, `cli`).
- `.venv/` — Pillow + numpy. Recreate with `python3 -m venv .venv && ./.venv/bin/pip install -r pixelkit/requirements.txt`.
- `examples/demo.py` — nine self-checking assertions covering every operation above.
- `examples/out/` — reference, style spec, variants, sheet + atlas, diff.
