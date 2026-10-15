# Proving a change

## The gates

```bash
./.venv/bin/python examples/demo.py           # reference-driven path
./.venv/bin/python examples/test_pixelate.py  # draft path
./.venv/bin/python sprites/build.py           # the patched pose set
```

Each is a plain script: it prints `[ok]`/`[PASS]` per check and exits non-zero on
failure. The check names say what they bind, so read the output rather than a list
here — it cannot go stale. `test_pixelate.py` is the gate for anything touching
`pixelkit/pixelate.py`; it is the file that turns the module's claims into
assertions, one per claim.

## What a finished sprite has to satisfy

1. **Palette conformance is exactly 1.0** — every opaque pixel equals a palette
   entry, checked against the PNG after a save/load round trip.
2. **Alpha is binary** — `[0, 255]` only. A third value means anti-aliasing came back.
3. **The block histogram has more than one size** — an all-`1×1` or all-`8×8` result
   means the quadtree collapsed or the sprite stopped being pixel art.
4. **Clearing the glow changes zero pixels inside the protected eye region.**

Draft against a frozen palette must also survive: the palette is kept verbatim and
only *extended*.

## Numbers are claims too

`README.md` quotes measured values, and every one of them is a property of the
current code and the current `in/` drafts. Changing a threshold, a `--cell`, a
palette cap or the median/mean choice moves them. When a number moves:

```bash
./.venv/bin/python sprites/redraw_drafts.py --colors 62 --palette sprites/palette.json
```

That rewrites `out/pixel/report.json`; update the figures quoted in `README.md`. The
test suite asserts behaviour, not numbers, so it will pass while the prose is quietly
wrong — that read-back is on you.

## Extending the tests

Add a `check("what it binds", condition, detail)` next to the claim it defends, in
the section for the step it belongs to. Prefer an assertion that would fail loudly on
a real regression over one that restates the implementation.
