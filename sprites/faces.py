"""Face patches for the witch.

Every expression is a set of pixel-run replacements inside the reference head, so
the parts of the face that are not named stay pixel-identical to the reference and
expressions cannot drift in style.

Coordinates are absolute in the 83-wide frame and come from the reference analysis:
  left eye box  x27..33, right eye box x43..49, rows 43..46
  brows         row 40 (left x28..30, right x44..47)
  mouth         x34..41, rows 50..52
"""

from __future__ import annotations

SKIN = "8"          # #fcd8bf - dominant face colour, used to erase
LASH = "3"          # #1a1722
MOUTH_DARK = "d"    # #b06b5f
MOUTH_MID = "f"     # #d68b7a
MOUTH_EDGE = "b"    # #f8ceb6

EYE_LEFT_X = 27
EYE_RIGHT_X = 43
EYE_ROWS = (43, 44, 45, 46)


def _closed_eye(x: int) -> list[dict]:
    """A content ∩ closed eye inside the reference eye box."""
    S, L = SKIN, LASH
    return [
        {"y": 43, "x": x, "row": S * 2 + L * 3 + S * 2},
        {"y": 44, "x": x, "row": S + L + S * 3 + L + S},
        {"y": 45, "x": x, "row": S * 7},
        {"y": 46, "x": x, "row": S * 7},
    ]


def closed_eyes() -> list[dict]:
    return _closed_eye(EYE_LEFT_X) + _closed_eye(EYE_RIGHT_X)


def clear_mouth() -> list[dict]:
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


def mouth_flat() -> list[dict]:
    """A short level line - the 'thinking it over' mouth."""
    S = SKIN
    return clear_mouth() + [
        {"y": 52, "x": 36, "row": MOUTH_DARK * 4},
    ]


def mouth_wave() -> list[dict]:
    """A small zigzag - the uneasy mouth."""
    S = SKIN
    return clear_mouth() + [
        {"y": 51, "x": 36, "row": MOUTH_DARK + S + MOUTH_DARK},
        {"y": 52, "x": 37, "row": S + MOUTH_DARK + S},
    ]


def brows_angry() -> list[dict]:
    """Additive only: the inner ends of both brows drop one row."""
    return [
        {"y": 41, "x": 31, "row": LASH * 3},
        {"y": 41, "x": 44, "row": LASH * 2},
    ]


def brows_worried() -> list[dict]:
    """Additive only: the inner ends lift one row."""
    return [
        {"y": 39, "x": 31, "row": LASH},
        {"y": 39, "x": 42, "row": LASH},
    ]


# --------------------------------------------------------------- expressions

EXPRESSIONS: dict[str, list[dict]] = {
    "neutral": [],
    "pleased": closed_eyes() + mouth_small_smile(),
    "happy": closed_eyes() + mouth_open_happy(),
    "displeased": brows_angry() + mouth_frown(),
    "smirk": mouth_smirk(),
    "thinking": mouth_flat(),
    "worried": brows_worried() + mouth_wave(),
}
