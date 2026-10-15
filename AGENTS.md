# AGENTS.md

A sprite is data, not a picture: a character grid plus a palette, so every
requirement ("same witch, night palette, 2x") becomes a deterministic operation you
can verify pixel by pixel. This repo is the toolkit that enforces that — analysis,
redraw, render, verify — plus the drafts it was built against.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r pixelkit/requirements.txt
```

## Check your work

```bash
./.venv/bin/python examples/demo.py           # 9 assertions, reference-driven path
./.venv/bin/python examples/test_pixelate.py  # 25 assertions, draft path
./.venv/bin/python sprites/build.py           # 6 poses, one palette, grids reproduced
```

Each prints one line per check and exits non-zero on failure. Run them before you
call a change done.

## Where to look

| Need | Read |
|---|---|
| What the project is, module map, the rules | [README.md](README.md) |
| Pixel-art craft, the redraw workflow | [.agents/skills/](.agents/skills/) |
| How to prove a change is good | [verification](.agents/docs/verification.md) |
| Code style, script shape, dependencies | [conventions](.agents/docs/conventions.md) |
| Layout, ignored directories, host gotchas | [environment](.agents/docs/environment.md) |
