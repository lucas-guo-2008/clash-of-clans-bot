# Clash of Clans Attack Bot

A bot that farms loot in Clash of Clans: find bases, evaluate them, attack the good ones,
come home, repeat. Runs against Clash of Clans in BlueStacks Air on macOS, driven over ADB.

## What it does today

It **reads**. It can capture the emulator screen, recognize which UI buttons are on it,
read the loot panel into numbers, and tap a button it has located.

It does not yet navigate on its own. **There is no `main.py` and no "run the bot" command** —
the state machine that strings these pieces together is the next step. What exists is a set of
tools you drive by hand.

| Phase | Status |
|---|---|
| 0 — Plumbing: ADB, capture, recorder, template tools, one tap | done |
| 1 — Navigate: screen classifier + state machine | **next** |
| 2 — Read loot | done |
| 3 — Deploy troops | not started |
| 4 — Building detection (YOLO) | not started |
| 5 — Decision layer | not started |

Phases 1 and 2 were done out of order, so the bot can read a base but not yet get to one.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use `.venv/bin/python` rather than activating — a bare `pip` resolves to the pyenv global
install, not this venv.

Then start BlueStacks Air with Clash of Clans running, and connect:

```bash
adb connect 127.0.0.1:5555
adb devices          # expect 127.0.0.1:5555   device
```

Everything below runs from the repository root.

## Files you run

All four live in `tools/`. Nothing in `capture/`, `control/`, `vision/` or `policy/` is run
directly — those are imported.

| Command | What it does | Emulator? |
|---|---|---|
| `python -m tools.adb_ss` | Record frames. Prompts for a name prefix; `s` saves, `q` quits. Saves to `templates/initial_collection/<prefix>_<n>.png` and skips indices already on disk. | yes |
| `python -m tools.debug_match` | Live overlay boxing every button it recognizes, with match scores. `q` to quit. | yes |
| `python -m tools.tap_anchor <label>` | Locate a named button, tap it, and report what changed on screen. `--dry-run` locates without tapping. | yes |
| `python -m tools.extract_digits cut\|build` | Cut labelled digit glyphs from a frame, then vote them into `templates/digits/`. | no |
| `python -m tools.read_loot <frames...>` | Read the loot panel from saved PNGs. `--expect G E D` turns it into a pass/fail assertion. | no |

Known anchor labels: `attack`, `attack-button`, `find-match`, `end-battle`, `next-button`.

## Workflows

### Collect frames

Get to the screen you care about in BlueStacks, then:

```bash
python -m tools.adb_ss          # type a prefix, e.g. "scan", then press s on good frames
```

### Read loot from what you collected

```bash
python -m tools.read_loot templates/initial_collection/scan_*.png
```

Each frame prints `ok` or `REJECTED`, the three values, per-row minimum scores, and the reason
for any rejection. This never touches the emulator, so it's the loop to iterate in.

### Fix a rejected frame

A row reporting `low score` means the reader met a glyph its templates don't cover well. Read
the true values off the screenshot and teach it:

```bash
python -m tools.extract_digits cut <frame>.png --gold 504599 --elixir 452179 --dark 1299
python -m tools.extract_digits build
```

`cut` refuses a row whose glyph count doesn't match the digits you typed — that's the typo
check. `build` re-votes every template and **warns about any sample that disagrees with its own
template**; act on that warning rather than ignoring it. It catches glyphs accidentally cut from
a screen that renders the font at a different size or weight, which silently degrades a template.

### Assert a specific read

```bash
python -m tools.read_loot <frame>.png --expect 525962 1333432 29717
```

Exits non-zero on mismatch. Useful as a regression check after touching `vision/loot.py`.

### Check a navigation step by hand

```bash
python -m tools.tap_anchor attack --dry-run   # locate only
python -m tools.tap_anchor attack             # tap, then diff the anchors before/after
```

If the named anchor isn't on screen, nothing is tapped and the exit code is non-zero. That
refusal is deliberate: the bot does not tap screens it hasn't identified.

## Project layout

```
capture/   talking to the device. grab_frame() and nothing else.
vision/    pure functions: image in, findings out. No I/O, no ADB, no global state.
control/   actions on the device: tap, swipe, back.
policy/    what to do next. Thresholds now, the state machine later.
tools/     scripts a human runs. Never imported by the bot.
templates/ digits/ (tracked) plus digit_samples/ and initial_collection/ (gitignored bulk data)
```

The "vision is pure" rule is load-bearing: it is what lets the entire loot reader be developed
and regression-tested against saved PNGs with no emulator attached.

`feasibility testing/` is early throwaway exploration. Ignore it.

## Performance

Measured on this setup, 1920×1080:

| Stage | Time | Rate |
|---|---|---|
| `capture.grab_frame` | ~185 ms | 5.4 fps |
| `vision.anchors.find_anchors`, 5 templates, full frame | ~480 ms | 2.1 fps |
| `vision.loot.read_loot` | ~0.6 ms | negligible |

Base scanning needs 1–3 fps, so this is adequate — but anchor matching, not capture, is the
bottleneck. Restricting each template to the region it can actually appear in is the obvious
win when that starts to matter.

## Design notes

`CLAUDE.md` holds the reasoning: why template matching rather than a model, why the ROI is
binarized before matching, the font-size and template-provenance traps, and where the
thresholds came from. Read it before changing anything in `vision/`.
