"""Standing witch states: arms, props and the nine requested poses.

Arms are rasterised from a centre-line path (a few points) with a thickness, so a
pose is described by numbers rather than a wall of characters.  Hands are drawn as
separate blobs, which lets a hand land *in front* of the head (chin, chest) while
the arm itself stays behind the hair.
"""

from __future__ import annotations

from .body import (
    DRESS_DARK,
    DRESS_DEEP,
    DRESS_HILITE,
    DRESS_LIT,
    DRESS_MID,
    FRAME_W,
    OUTLINE,
    SKIN,
    SKIN_EDGE,
    SKIN_LIGHT,
    SKIN_SHADE,
    STANDING_H,
    build_arm,
    build_standing_body,
)

HEAD_CUT = 73


# ---------------------------------------------------------------------- arms


def _lerp_path(path: list[tuple[int, float]], y: int) -> float:
    if y <= path[0][0]:
        return path[0][1]
    if y >= path[-1][0]:
        return path[-1][1]
    for (y0, x0), (y1, x1) in zip(path, path[1:]):
        if y0 <= y <= y1:
            t = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
            return x0 + (x1 - x0) * t
    return path[-1][1]


def arm_rows(path: list[tuple[int, float]], width: float) -> list[str]:
    """A limb along a vertical-ish centre line, outlined on both sides."""
    rows = [["."] * FRAME_W for _ in range(STANDING_H)]
    ys = [pt[0] for pt in path]
    y0 = max(0, int(min(ys)))
    y1 = min(STANDING_H - 1, int(max(ys)))
    for y in range(y0, y1 + 1):
        cx = _lerp_path(path, y)
        x0 = int(round(cx - width / 2))
        x1 = int(round(cx + width / 2))
        lo, hi = sorted((x0, x1))
        for x in range(max(0, lo), min(FRAME_W - 1, hi) + 1):
            rows[y][x] = DRESS_LIT
        for x in range(max(0, lo), min(FRAME_W - 1, lo + 1) + 1):
            rows[y][x] = DRESS_HILITE
        rows[y][max(0, hi)] = OUTLINE
        rows[y][max(0, lo)] = OUTLINE
    return ["".join(r) for r in rows]


def hand_blob(x: int, y: int, w: int = 8, h: int = 8) -> list[str]:
    """A mitten hand, drawn after the head so it can sit in front of the face."""
    rows = [["."] * FRAME_W for _ in range(STANDING_H)]
    for dy in range(h):
        # rounded corners
        shrink = 1 if dy in (0, h - 1) else 0
        for dx in range(shrink, w - shrink):
            rows[y + dy][x + dx] = SKIN
    for dy in range(1, h - 1):
        rows[y + dy][x + 1] = SKIN_LIGHT
        rows[y + dy][x + w - 2] = SKIN_SHADE
    for dx in range(w):
        rows[y][x + dx] = OUTLINE
        rows[y + h - 1][x + dx] = OUTLINE
    for dy in range(h):
        rows[y + dy][x] = OUTLINE
        rows[y + dy][x + w - 1] = OUTLINE
    return ["".join(r) for r in rows]


# --------------------------------------------------------------------- props

POTION = """\
....0000....
....0770....
....0770....
....07770...
...0tttt0...
..0tqqqqt0..
.0tqvqqvqt0.
.0tqpppqt0..
0tqpooopqt0.
0tqpooopqt0.
0tqpooopqt0.
0tqpooopqt0.
.0tqpppqt0..
.0tqqqqqt0..
..0ttttt0...
...00000....
"""

# The flask uses only colours that already exist in the reference palette.
POTION_PALETTE: dict[str, str] = {}


def potion_grid() -> list[str]:
    rows = [r for r in POTION.splitlines() if r.strip()]
    width = max(len(r) for r in rows)
    return [r.ljust(width, ".") for r in rows]


# ------------------------------------------------------------------ assemble

# Centre-line paths per arm pose (y, x) for the character's LEFT arm (viewer's left).
ARM_POSES: dict[str, list[tuple[int, float]]] = {
    "down": [(76, 21.0), (100, 18.0), (126, 18.0)],
    "spread": [(78, 22.0), (92, 14.0), (106, 6.0)],
    "up": [(80, 22.0), (86, 13.0), (76, 5.0)],
    "chin": [(78, 22.0), (92, 21.0), (74, 27.0)],
    "forward": [(78, 22.0), (95, 22.0), (104, 30.0)],
    "chest": [(78, 22.0), (95, 21.0), (100, 32.0)],
}
ARM_WIDTH = 10.0


def mirrored_path(path: list[tuple[int, float]]) -> list[tuple[int, float]]:
    return [(y, 2 * 41.0 - x) for y, x in path]


def compose_standing(
    palette: dict[str, str],
    reference_grid: list[str],
    arms: list[str] | None = None,
    hands: list[tuple[int, int]] | None = None,
    props: list[tuple[list[str], dict[str, str], int, int]] | None = None,
    face_patches: list[dict] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Body, then arms, then head, then anything that must be in front."""
    from pixelkit import render

    pal = dict(palette)
    canvas = ["." * FRAME_W] * STANDING_H
    canvas = render.blit(canvas, build_standing_body(), 0, 0)
    for arm in arms or []:
        canvas = render.blit(canvas, arm, 0, 0)

    head = render.cut(reference_grid, (0, 0, FRAME_W - 1, HEAD_CUT))
    if face_patches:
        head = render.apply_patches(head, face_patches, pal)
    canvas = render.blit(canvas, head, 0, 0)

    for x, y in hands or []:
        canvas = render.blit(canvas, hand_blob(x, y), 0, 0)
    for grid, prop_pal, x, y in props or []:
        pal.update(prop_pal)
        canvas = render.blit(canvas, grid, x, y)
    return canvas, pal


# --------------------------------------------------------------- pose registry

def _arms(*poses: str) -> list[str]:
    out = []
    for pose in poses:
        path = ARM_POSES[pose]
        out.append(arm_rows(path, ARM_WIDTH))
    return out


def _mirror(pose: str) -> list[str]:
    return arm_rows(mirrored_path(ARM_POSES[pose]), ARM_WIDTH)


def standing_poses(reference_grid, palette):
    """The nine requested standing states as (note, grid, palette) triples."""
    from .faces import EXPRESSIONS

    potion = potion_grid()
    both_forward = [arm_rows(ARM_POSES["forward"], ARM_WIDTH), _mirror("forward")]
    both_chest = [arm_rows(ARM_POSES["chest"], ARM_WIDTH), _mirror("chest")]

    specs = [
        ("standing_standard", "arms down, neutral", ["down", "down"], [], [], "neutral"),
        ("standing_thinking", "left hand up to the cheek, level mouth",
         ["chin", "down"], [(26, 58)], [], "thinking"),
        ("standing_happy", "both arms up, closed happy eyes, open smile",
         ["up", "up"], [], [], "happy"),
        ("standing_displeased", "arms down, angry brows, frown",
         ["down", "down"], [], [], "displeased"),
        ("standing_pleased", "arms down, soft closed eyes, small smile",
         ["down", "down"], [], [], "pleased"),
        ("standing_hold_item_two_hands", "both hands forward around a potion flask",
         None, [(30, 104), (44, 104)], [(potion, POTION_PALETTE, 35, 92)],
         "neutral"),
        ("standing_hold_item_one_hand", "right hand forward holding a potion flask",
         None, [(46, 100)], [(potion, POTION_PALETTE, 48, 88)], "neutral"),
        ("standing_spread_smirk", "both arms out to the sides, lopsided smirk",
         ["spread", "spread"], [], [], "smirk"),
        ("standing_worried", "hands together at the chest, worried brows and mouth",
         None, [(31, 88), (42, 88)], [], "worried"),
    ]

    result = {}
    for name, note, arm_poses, hands, props, face in specs:
        if arm_poses is None:
            arms = both_forward if "two_hands" in name else (
                both_chest if name.endswith("worried") else
                [arm_rows(ARM_POSES["forward"], ARM_WIDTH), arm_rows(ARM_POSES["down"], ARM_WIDTH)]
            )
        else:
            arms = [arm_rows(ARM_POSES[p], ARM_WIDTH) for p in arm_poses]
        canvas, pal = compose_standing(
            palette, reference_grid, arms=arms, hands=hands, props=props,
            face_patches=EXPRESSIONS[face],
        )
        result[name] = (note, canvas, pal)
    return result
