# Conventions

## Language

Code prose — docstrings, comments, commit messages — is **English**. The Cyrillic in
this repo is data: the draft filenames in `in/` (`счастлива.webp`, `на_метле_влево.webp`)
and the pose names they map to. Quote those paths in shell commands.

## Style

Match the file you are editing; the house shape is:

- `from __future__ import annotations`, and `X | None` unions rather than `Optional`.
- Lines wrap near 100 columns; a handful reach ~110 where breaking hurts readability.
- Docstrings state *why* the step exists and what it costs, not what the next line does.
  The most valuable ones record the failure the current code was written to avoid —
  "deleting those instead would punch the backdrop through the middle of the hair".
- No formatter or linter is configured. Keep diffs local; do not reflow untouched code.

## Scripts

Executable scripts follow one shape:

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixelkit import render, verify  # noqa: E402
```

Two more rules that bite:

- **Output paths must resolve inside the repo.** Drivers call `.relative_to(ROOT)` on
  what they write, so `--out /tmp/...` fails. Use a directory under `out/`.
- **Drivers are idempotent and print a per-item line.** `sprites/build.py` rewrites the
  tracked `sprites/grids/*.txt`; it must reproduce them byte for byte, and does.

## Dependencies

Pillow and numpy, nothing else. The toolkit is deliberately pure Python so it runs
without a GPU, a display or a package manager beyond pip. A new dependency needs a
reason that survives the question "can this be twenty lines of numpy?".

## Tests

Plain scripts, no pytest: a `check(name, ok, detail)` helper accumulates results and
the script exits non-zero on failure. Keep them runnable with no arguments and no
network. See [verification](verification.md).
