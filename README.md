# deepseek-pixel-art-generation

Redraw pixel art to a written specification — from one reference image, or from
an AI draft that only *looks* like pixel art.

The idea: **a sprite is data, not a picture.** Store it as a character grid plus a
palette, and every requirement ("same witch, night palette, 2x, in a 32×32 cell")
becomes a deterministic operation you can verify pixel by pixel. Image models
cannot hold that guarantee: they return a *picture of* pixel art — anti-aliased,
off-grid, and with tens of thousands of distinct colours. Here they are optional
drafters, never the deliverable.

## Quick start

```bash
python3 -m venv .venv
./.venv/bin/pip install -r pixelkit/requirements.txt

./.venv/bin/python examples/demo.py           # 9 self-checking assertions
./.venv/bin/python examples/test_pixelate.py  # 25 self-checking assertions on the draft pipeline
```

Both scripts print one line per check and exit non-zero on the first failure.

## Two pipelines

The repo carries two lineages that share one library. The **reference-driven** one
came first and is where the invariants were worked out; the **draft-driven** one is
what the toolkit grew into.

### From a reference image

```bash
# 1. read the reference: native grid, palette with roles, silhouette, line weight
./.venv/bin/python -m pixelkit analyze reference/mascot-witch.png \
    --json out/witch_spec.json --ascii out/reference.txt

# 2. render a spec (palette + grid, or a text legend+rows block)
./.venv/bin/python -m pixelkit render out/witch_spec.json -o out/variant.png --scale 4

# 3. verify: palette conformance, silhouette IoU, pixel diff image
./.venv/bin/python -m pixelkit verify reference/mascot-witch.png out/variant.png \
    --palette out/witch_spec.json --diff out/diff.png
```

Then `./.venv/bin/python sprites/build.py` turns the analysed reference into six
patched poses under `sprites/grids/` and `out/sprites/`. Poses are explicit
pixel-run patches, so anything not named in a patch stays byte-identical to the
reference and every pose keeps one palette. The driver regenerates
`out/witch_spec.json` if it is missing, so a fresh clone runs it as-is.

`tools/ruler.py` renders a crop at high zoom with a pixel ruler — the authoring aid
for reading exact rows before patching them.

### From an AI draft

`in/` holds twelve "pixel art" drafts from Qwen Image 2.1 (plus one earlier 2512
frame). They look right and are not pixel art: anti-aliased edges, off-grid blocks,
30 000–55 000 distinct colours. What they *do* get right is that their blocks are
not all the same size — big squares across a hat, one-pixel stripes on a cheek.

`pixelkit.pixelate` keeps that property and fixes the rest:

```bash
# the shipped set: twelve frames on one frozen 61-colour palette
./.venv/bin/python sprites/redraw_drafts.py --colors 62 --palette sprites/palette.json

# a quick look at the same reduction, palette re-cut on the spot
./.venv/bin/python -m pixelkit pixelate in/*.webp -o out/pixel --cell 4 --scale 4

# re-cut the palette from scratch instead of freezing it (every colour moves)
./.venv/bin/python sprites/redraw_drafts.py --colors 62
```

`sprites/palette.json` is tracked on purpose. The frozen palette *is* part of the
deliverable — re-cutting it shifts every colour in every frame — so it lives next to
the grids rather than in ignored build output.

| Step | What it does |
|---|---|
| cell median | `cell`×`cell` source pixels → one cell, by per-channel median (survives anti-aliasing) |
| shared palette | median cut over the pooled histogram of the whole set, near-duplicates merged, capped so the grid still fits the 62-character alphabet |
| snap | every cell to a palette entry — one colour per pixel, by construction |
| despeckle | isolated cells in locally flat areas become the neighbourhood mode |
| silhouette | background flood-filled from the border, then the generator's light rim peeled |
| haze & shadow | light low-chroma pixels cleared: the outer glow and the ground shadow become transparent, but a streak walled in by hair is *repainted* with hair colour so no backdrop shows through |
| eyes | the iris is the only saturated green in the sprite; its bounding box (plus the topmost skin region, for closed eyes) is exempt from every clearing rule, so sclera and highlights survive |
| quadtree | maximal uniform squares labelled; the block-size histogram *is* the "mixed pixel sizes" claim |

Passing `--palette` freezes the palette a set already uses and only *extends* it
with the accents a new frame introduces (`build_palette(..., base=...)`). The twelve
frames share one 61-colour palette: the buckle frame added a bright gold and two
iris greens, the reading frame added the ivory of the book's pages, without which
they came out grey.

The palette sits at the ceiling: a grid is stored as characters and the alphabet
holds 62, so the coarse `cell=6` set was re-cut over all twelve frames rather than
extended. A further frame with new colours needs one of the near-duplicate dress
shades merged away — today nothing is closer than 25 units, so that is a real
trade-off and not a free merge.

### Measured on the twelve drafts at `cell=4`

| Metric | Value |
|---|---|
| palette conformance | 1.0 on every frame (with binary alpha) |
| colours per frame | 41–61, against a 61-colour palette |
| native canvas | 217–286 × 235–318 |
| silhouette IoU vs. an independent background test | 0.941–0.978 (mean 0.960) |
| strong-edge recall / precision | 0.873–0.906 / 0.671–0.768 |
| block-size mix, share of canvas | 41.3 % `8×8`, 25.8 % `1×1`, 21.8 % `2×2`, 11.1 % `4×4` |
| glow and shadow cells cleared | 7 894 |
| eye region protected | 20 388 px, 848 of them light enough that the glow rule would have eaten them |

`out/pixel/report.json` carries these per frame; `examples/test_pixelate.py`
asserts the pipeline's own claims, including that clearing the glow changes
**zero** pixels inside the protected eye region.

Outputs per draft: `out/pixel/<name>.png` (native), `<name>@4x.png` (integer
export), `preview/<name>.png` (on a backdrop), `spec/<name>.json`, and a text grid
in `out/pixel/grids/<name>.txt`.

## What each module is for

| Module | Responsibility |
|---|---|
| `pixelkit/core.py` | image ↔ grid, palette extraction, native pixel-size detection |
| `pixelkit/analyze.py` | reference → style spec (grid, palette with roles, bbox, warnings) |
| `pixelkit/pixelate.py` | anti-aliased draft → real sprite with pixels of different sizes |
| `pixelkit/render.py` | operations: `resize`, `recolor`, `ramp_shift`, `outline`, `shift`, `flip_*`, `pad_canvas`, `sheet` |
| `pixelkit/verify.py` | `palette_conformance`, `silhouette_iou`, `compare`, `diff_image`, `silhouette_check` |
| `pixelkit/cli.py` | the `python -m pixelkit <command>` surface |
| `sprites/palette.json` | the frozen palette the shipped sprite set is defined by |
| `.agents/skills/pixel-art-redraw/` | the instructions an agent loads to do this work |

Three more files in `sprites/` are **not wired to any command**: `body.py`,
`faces.py` and `standing.py` build standing and flying poses procedurally, because
the reference is a seated pose that no patch list can turn around. They work —
`standing_poses(grid, palette)` returns nine poses, and the trio resolves its own
imports — but they were written for a set that was never shipped, so no driver
calls them.

## What is generic and what is this witch

`pixelkit` is a general tool, and it is worth being precise about where that stops,
because the shipped set is one character generated by one model.

Generic: the whole surface. Every threshold in `pixelate()` is a keyword argument,
a palette can be passed in or cut per image, `render`/`verify`/`analyze` make no
assumptions about the subject, and the only structural limit — the 62-character
alphabet in `core.py` — comes from storing the grid as text, not from the character.

Tuned, not derived: the *defaults* were picked by hand on these twelve drafts —
`haze_luminance=112`, `haze_chroma=30`, `haze_enclosure=0.75`, the background
distance 40 in `_trim`, the palette merge distance 11.0. Sweep them again for another
model or another cell size.

Character-bound, and this is the real limit: `eye_regions` finds the iris as the only
saturated **green** (`g - max(r, b) > 25`), and its closed-eye fallback needs
`skin_regions` to find a light, warm **skin** tone. A blue- or brown-eyed character,
an animal, a robot or a dark-skinned one matches neither test, gets an empty mask, and
then loses the eye whites to the glow rule. Rewrite both — or pass
`protect_eyes=False` — before reusing the pipeline on someone else. The docstrings say
so at the call site.

## Third-party material

The `pixel-art-sprites` skill is a verbatim copy of
[omer-metin/skills-for-antigravity](https://github.com/omer-metin/skills-for-antigravity)
under the Apache License 2.0, which is why it carries its own license file inside the
directory. Everything else here is MIT. See [THIRD-PARTY.md](THIRD-PARTY.md).

## The rules that make it look like pixel art

1. One colour per pixel — no anti-aliasing, no gradients.
2. Integer scaling only (2x/3x/4x, nearest-neighbour).
3. Variants remap the reference palette 1:1; never invent colours per pixel.
4. Binary alpha; preview on a backdrop instead of faking transparency.
5. Keep the text grid next to every PNG — a PNG alone cannot be revised.

## Working on this repo

Agent instructions live in [`AGENTS.md`](AGENTS.md). Environment notes — including
why the toolkit is pure Python — are in
[`.agents/docs/environment.md`](.agents/docs/environment.md).

## What is not in the repository

Build output and side projects are ignored (see `.gitignore`), because each is
regenerated by a command above:

- `out/` — sprites, specs, reports, contact sheets.
- `examples/out/` — the demo's output.
- `article/`, `wallpaper/`, `work/`, `_research/` — the Habr write-up, the
  Wallpaper Engine project, its scratch plates, and a cloned site used as reference.

Inputs stay tracked: `in/` (the twelve drafts), `reference/` (the witch reference),
and `sprites/palette.json` (the frozen palette the sprite set is defined by) are what
the pipelines read.
