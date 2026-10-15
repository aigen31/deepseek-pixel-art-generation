"""Sprite source for the witch mascot.

Every pose is the analysed reference plus an explicit list of pixel-run patches.
Anything not named in a patch stays byte-identical to the reference, so all states
keep one style and one palette.

Run:  ./.venv/bin/python sprites/build.py            # everything
      ./.venv/bin/python sprites/build.py sitting    # one group
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixelkit import render, verify  # noqa: E402
from pixelkit.core import dump_json, grid_to_ascii, load_rgba, upscale, grid_to_array  # noqa: E402
from pixelkit.core import save_rgba  # noqa: E402
import numpy as np  # noqa: E402

SPEC = ROOT / "out" / "witch_spec.json"
OUT = ROOT / "out" / "sprites"
GRIDS = ROOT / "sprites" / "grids"
EXPORT_SCALE = 4

REFERENCE = ROOT / "reference" / "mascot-witch.png"

EYE_L = (27, 33)   # inclusive pixel range of the left eye
EYE_R = (43, 49)
SKIN = "8"         # #fcd8bf, the dominant face colour - used to erase
LASH = "3"         # #1a1722
MOUTH_DARK = "d"   # #b06b5f
MOUTH_MID = "f"    # #d68b7a
MOUTH_EDGE = "b"   # #f8ceb6


def ensure_spec() -> None:
    """Analyse the reference into ``out/witch_spec.json`` if it is not there yet.

    ``out/`` is build output and is not tracked, so a fresh clone has no spec.
    The analysis is deterministic, so regenerating it costs a second and yields
    the same grid the committed ones were built from.
    """
    if SPEC.exists():
        return
    from pixelkit import analyze as analyse_image

    SPEC.parent.mkdir(parents=True, exist_ok=True)
    analysis = analyse_image(REFERENCE)
    SPEC.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"analysed {REFERENCE.relative_to(ROOT)} -> {SPEC.relative_to(ROOT)}")


def base():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    palette = {e["char"]: e["hex"] for e in spec["palette"]}
    return spec["grid"], palette


S = SKIN = "8"       # #fcd8bf - dominant face colour, used to erase
LASH = LASH = "3"    # #1a1722
MOUTH_DARK = "d"     # #b06b5f - open mouth / mouth line
MOUTH_MID = "f"      # #d68b7a - lip
MOUTH_EDGE = "b"     # #f8ceb6 - lip highlight


def _closed_eye(x: int) -> list[dict]:
    """A content ∩ closed eye inside a 7x4 box (the reference eye's exact box)."""
    S, L = SKIN, LASH
    return [
        {"y": 43, "x": x, "row": S * 2 + L * 3 + S * 2},
        {"y": 44, "x": x, "row": S + L + S * 3 + L + S},
        {"y": 45, "x": x, "row": S * 7},
        {"y": 46, "x": x, "row": S * 7},
    ]


def closed_eyes() -> list[dict]:
    """Left eye box x27..33, right eye box x43..49, rows 43..46."""
    return _closed_eye(27) + _closed_eye(43)


def clear_mouth() -> list[dict]:
    """Erase the reference mouth (x34..41, rows 51..52) to skin."""
    return [{"y": y, "x": 34, "row": SKIN * 8} for y in (51, 52)]


def mouth_small_smile() -> list[dict]:
    S = SKIN
    return clear_mouth() + [
        {"y": 51, "x": 36, "row": MOUTH_MID * 3},
        {"y": 52, "x": 35, "row": MOUTH_EDGE + S * 3 + MOUTH_EDGE},
    ]


def mouth_open_happy() -> list[dict]:
    S = SKIN
    return clear_mouth() + [
        {"y": 50, "x": 36, "row": MOUTH_MID * 5},
        {"y": 51, "x": 35, "row": MOUTH_DARK * 7},
        {"y": 52, "x": 36, "row": MOUTH_EDGE * 5},
    ]


def mouth_frown() -> list[dict]:
    """⌢ - corners up, middle down."""
    S = SKIN
    return clear_mouth() + [
        {"y": 51, "x": 35, "row": MOUTH_DARK + S * 3 + MOUTH_DARK},
        {"y": 52, "x": 36, "row": MOUTH_DARK * 3},
    ]


def mouth_smirk() -> list[dict]:
    """A line falling to the right - one-sided smile."""
    return clear_mouth() + [
        {"y": 51, "x": 35, "row": MOUTH_DARK * 3},
        {"y": 52, "x": 38, "row": MOUTH_DARK},
    ]


def brows_angry() -> list[dict]:
    """Additive only: the inner ends of both brows drop one row."""
    return [
        {"y": 41, "x": 31, "row": LASH * 3},
        {"y": 41, "x": 44, "row": LASH * 2},
    ]



# --------------------------------------------------------------- arm surgery
# The reference rests both hands on the ground. For the "arms spread" pose the
# hand blocks are lifted up and out; the vacated pixels are refilled with the
# dress tone so the silhouette stays intact.

DRESS = "1"   # #14111c - the dress in shadow


def _fill(y: int, x: int, n: int) -> dict:
    return {"y": y, "x": x, "row": DRESS * n}


ERASE_LEFT = [_fill(y, 8, 11) for y in (88, 89, 90, 91, 92, 93, 94)]

ERASE_RIGHT = [
    _fill(89, 61, 11),
    _fill(90, 61, 12),
    _fill(91, 61, 12),
    _fill(92, 61, 13),
    _fill(93, 61, 13),
    _fill(94, 61, 13),
    _fill(95, 61, 12),
]

HAND_LEFT_BOX = (8, 87, 18, 94)      # x0, y0, x1, y1 inclusive
HAND_RIGHT_BOX = (61, 89, 72, 95)
HAND_LEFT_MOVE = (-4, -4)
HAND_RIGHT_MOVE = (4, -4)


def raise_hands(grid, palette):
    """Lift both hands out and up, then lay the smirk on top."""
    left = render.cut(grid, HAND_LEFT_BOX)
    right = render.cut(grid, HAND_RIGHT_BOX)
    out = render.apply_patches(grid, ERASE_LEFT + ERASE_RIGHT + mouth_smirk(), palette)
    out = render.blit(out, left, HAND_LEFT_BOX[0] + HAND_LEFT_MOVE[0], HAND_LEFT_BOX[1] + HAND_LEFT_MOVE[1])
    out = render.blit(out, right, HAND_RIGHT_BOX[0] + HAND_RIGHT_MOVE[0], HAND_RIGHT_BOX[1] + HAND_RIGHT_MOVE[1])
    return out


# --------------------------------------------------------------------- poses
# key -> (group, note, patches)

POSES: dict[str, tuple[str, str, list[dict]]] = {
    "sitting_standard": ("sitting", "reference, unchanged", []),
    "sitting_smirk": (
        "sitting",
        "lopsided smirk only",
        mouth_smirk(),
    ),
    "sitting_smirk_arms_spread": (
        "sitting",
        "both hands lifted out and up - spread-arms shrug with a lopsided smirk",
        raise_hands,
    ),
    "sitting_happy": (
        "sitting",
        "closed happy eyes, open smile",
        closed_eyes() + mouth_open_happy(),
    ),
    "sitting_pleased": (
        "sitting",
        "soft closed eyes, small smile",
        closed_eyes() + mouth_small_smile(),
    ),
    "sitting_displeased": (
        "sitting",
        "angry brows, frown",
        brows_angry() + mouth_frown(),
    ),
}


def build(group: str | None = None) -> int:
    ensure_spec()
    grid, palette = base()
    OUT.mkdir(parents=True, exist_ok=True)
    GRIDS.mkdir(parents=True, exist_ok=True)
    reference = load_rgba(ROOT / "reference" / "mascot-witch.png")
    report = []

    for name, (grp, note, patches) in POSES.items():
        if group and grp != group:
            continue
        pose = patches(grid, palette) if callable(patches) else render.apply_patches(grid, patches, palette)
        (GRIDS / f"{name}.txt").write_text(
            grid_to_ascii(pose, palette, header=f"{name} - {note}"), encoding="utf-8"
        )
        png = render.to_png(pose, palette, OUT / f"{name}.png", scale=EXPORT_SCALE)
        preview = render.to_preview(
            pose, palette, OUT / f"{name}_preview.png", scale=5, background="#e9e6ff"
        )
        opaque = [v for v in palette.values() if v != "transparent"]
        conv = verify.palette_conformance(load_rgba(png), opaque)
        shape = verify.silhouette_check(pose)
        same_silhouette = np.array_equal(
            (grid_to_array(pose, palette)[..., 3] > 0),
            (grid_to_array(grid, palette)[..., 3] > 0),
        )
        report.append(
            {
                "name": name,
                "group": grp,
                "note": note,
                "patches": (patches.__name__ if callable(patches) else len(patches)),
                "palette_conformance": round(conv, 4),
                "silhouette_unchanged": bool(same_silhouette),
                "components": shape["components"],
                "png": str(png.relative_to(ROOT)),
                "preview": str(preview.relative_to(ROOT)),
            }
        )

    dump_json({"poses": report}, OUT / "build_report.json")
    width = max((p["name"] for p in report), key=len, default="")
    for p in report:
        flag = "OK " if p["palette_conformance"] == 1.0 else "BAD"
        print(
            f"  [{flag}] {p['name']:<{len(width)}}  patches={p['patches']:>2} "
            f"palette={p['palette_conformance']:.3f} silhouette_same={p['silhouette_unchanged']} "
            f"components={p['components']}"
        )
    print(f"{len(report)} poses written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build(sys.argv[1] if len(sys.argv) > 1 else None))
