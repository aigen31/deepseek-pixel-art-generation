"""Standing and flying bodies for the witch, built procedurally in the reference palette.

The reference is a *seated* pose, so no transformation can produce a standing
sprite: the lower body has to be drawn.  Expressing it as a profile table plus a
shading rule keeps the silhouette consistent across the nine standing states and
makes the geometry reviewable as numbers instead of a wall of characters.

Colour roles come straight from the reference analysis:
  0 #08070d outline        1 #14111c dress deep      3 #1a1722 dress mid
  4 #29242d dress lit      5 #1d1e27 dress mid-dark 6 #3e3840 dress highlight
  a #f3b7a0 skin           e #e4ac96 skin mid       8 #fcd8bf skin light
  b #f8ceb6 skin light 2   d #b06b5f skin shade     7 #573c3d hair warm
"""

from __future__ import annotations

FRAME_W = 83
CENTER = 41.0
STANDING_H = 176

OUTLINE = "0"
DRESS_HILITE = "6"
DRESS_LIT = "4"
DRESS_MID = "3"
DRESS_DARK = "1"
DRESS_DEEP = "5"
SKIN = "a"
SKIN_MID = "e"
SKIN_LIGHT = "8"
SKIN_SHADE = "d"
SKIN_EDGE = "b"

NECK_TOP = 70
HEM_Y = 124
BOOT_TOP = 140
LEG_TOP = HEM_Y + 1

# Half-widths of the torso alone; the arms are separate columns outside it.
DRESS_KEYS = [
    (70, 12.0),    # hidden behind the hair
    (78, 14.0),    # shoulders
    (90, 13.5),
    (104, 14.0),   # waist
    (116, 17.0),
    (124, 23.5),   # hem, widest
]

# Each leg is a column at CENTER +/- leg_offset with this half-width.
LEG_KEYS = [
    (125, 8.4),    # thigh
    (134, 8.0),
    (140, 8.6),    # boot cuff
    (160, 9.2),
    (170, 9.8),    # foot
    (176, 9.2),
]
LEG_OFFSET = 7.0

def _lerp(keys: list[tuple[int, float]], y: int) -> float:
    if y <= keys[0][0]:
        return keys[0][1]
    if y >= keys[-1][0]:
        return keys[-1][1]
    for (y0, w0), (y1, w1) in zip(keys, keys[1:]):
        if y0 <= y <= y1:
            t = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
            return w0 + (w1 - w0) * t
    return keys[-1][1]


def _row() -> list[str]:
    return ["."] * FRAME_W


def _put(row: list[str], x: int, char: str) -> None:
    if 0 <= x < len(row):
        row[x] = char


def _span(row: list[str], x0: float, x1: float, char: str) -> None:
    for x in range(max(0, int(round(x0))), min(len(row) - 1, int(round(x1))) + 1):
        row[x] = char


def dress_row(y: int, half: float) -> list[str]:
    """Light falls from the upper left: highlight, lit, mid, then deep shadow."""
    row = _row()
    xl = int(round(CENTER - half))
    xr = int(round(CENTER + half))
    width = xr - xl + 1
    if width < 5:
        _span(row, xl, xr, DRESS_MID)
        _put(row, xl, OUTLINE)
        _put(row, xr, OUTLINE)
        return row
    _span(row, xl, xr, DRESS_MID)
    _put(row, xl, OUTLINE)
    _put(row, xr, OUTLINE)
    _span(row, xl + 1, xl + max(1, width * 0.14), DRESS_HILITE)
    _span(row, xl + max(1, width * 0.14) + 1, xl + max(2, width * 0.36), DRESS_LIT)
    _span(row, xr - max(1, width * 0.24), xr - 1, DRESS_DARK)
    _span(row, xr - max(1, width * 0.10), xr - 1, DRESS_DEEP)
    return row


def leg_row(y: int) -> list[str]:
    """Two legs: bare skin above the boot line, knee-high boots below."""
    row = _row()
    half = _lerp(LEG_KEYS, y)
    boot = y >= BOOT_TOP
    for side in (-1, 1):
        centre = CENTER + side * LEG_OFFSET
        outer = centre + side * half
        inner = centre - side * half
        x0, x1 = sorted((int(round(outer)), int(round(inner))))
        if boot:
            _span(row, x0, x1, DRESS_DARK)
            _span(row, x0 + 1, x0 + max(1, (x1 - x0) * 0.3), DRESS_LIT)
            _put(row, x1 - 1, DRESS_DEEP)
            if y < BOOT_TOP + 3:            # cuff catches the light
                _span(row, x0, x1, DRESS_HILITE)
            _put(row, x0, OUTLINE)
            _put(row, x1, OUTLINE)
        else:
            _span(row, x0, x1, SKIN)
            _span(row, x0 + 1, x0 + 1, SKIN_LIGHT)
            _span(row, x1 - 1, x1, SKIN_SHADE)
            _put(row, x0, OUTLINE)
            _put(row, x1, OUTLINE)
    if boot and y >= 170:                   # sole
        _span(row, CENTER - LEG_OFFSET - 10.5, CENTER - 0.5, OUTLINE)
        _span(row, CENTER + 0.5, CENTER + LEG_OFFSET + 10.5, OUTLINE)
    return row


def build_standing_body() -> list[str]:
    """Neck, dress, legs and boots. Arms are separate parts laid on top."""
    grid = [["."] * FRAME_W for _ in range(STANDING_H)]
    for y in range(NECK_TOP, HEM_Y + 1):
        grid[y] = dress_row(y, _lerp(DRESS_KEYS, y))
    for y in range(LEG_TOP, STANDING_H):
        grid[y] = leg_row(y)

    rows = ["".join(r) for r in grid]

    # Waist seam and two folds, so the dress is not a flat blob.
    rows = _patch(rows, 103, CENTER - 13.5, DRESS_DEEP * 27)
    for dy in range(0, 18):
        y = 107 + dy
        indent = int(dy * 0.45)
        for dx in (-9.5 + indent, 9.5 - indent):
            rows = _patch(rows, y, int(CENTER + dx), DRESS_DEEP)
    return rows


def _patch(rows: list[str], y: int, x: float, text: str) -> list[str]:
    x = int(x)
    if not 0 <= y < len(rows):
        return rows
    row = list(rows[y])
    for i, ch in enumerate(text):
        if 0 <= x + i < len(row):
            row[x + i] = ch
    rows[y] = "".join(row)
    return rows


# --------------------------------------------------------------------- arms

ARM_KEYS = [
    (74, 24.0),    # shoulder: the arm's outer edge
    (84, 25.0),
    (100, 24.5),
    (114, 24.0),
    (122, 23.5),
]
ARM_WIDTH = 10.0
HAND_TOP = 112
HAND_BOTTOM = 126


def build_arm(side: int) -> list[str]:
    """A hanging arm with a mitten hand, in its own small frame.

    ``side`` is -1 for the character's left (viewer's left) and +1 for the right.
    The frame is FRAME_W wide and spans the same y range as the body, so parts can
    simply be blitted over the body at (0, 0).
    """
    rows = [["."] * FRAME_W for _ in range(STANDING_H)]
    for y in range(74, HAND_BOTTOM + 1):
        outer = _lerp(ARM_KEYS, y)
        inner = outer - ARM_WIDTH
        x0 = int(round(CENTER + side * inner))
        x1 = int(round(CENTER + side * outer))
        lo, hi = sorted((x0, x1))
        if y >= HAND_TOP:
            _span(rows[y], lo, hi, SKIN)
            _span(rows[y], lo, lo + 1, SKIN_LIGHT)
            _put(rows[y], hi - 1, SKIN_SHADE)
        else:
            # The arm reads lighter than the torso so the two silhouettes separate.
            _span(rows[y], lo, hi, DRESS_LIT)
            _span(rows[y], lo, lo + 1, DRESS_HILITE)
            _span(rows[y], hi - 1, hi, DRESS_MID)
        _put(rows[y], lo, OUTLINE)
        _put(rows[y], hi, OUTLINE)
    return ["".join(r) for r in rows]
