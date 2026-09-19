# Clash of Clans Attack Bot

A bot that farms loot in Clash of Clans: find bases, evaluate them, attack the good ones,
come home, repeat. Runs against Clash of Clans in BlueStacks Air on macOS, driven over ADB.

## What it does today

It **navigates**. It captures the emulator screen, names which of four screens it is looking
at, walks home → army → attack menu → scout → Next → Next → End Battle → home on its own, and
reads the loot panel on every base it passes.

It does not yet attack. Loot is read and logged but never acted on, and no troops are deployed.
**There is still no `main.py`** — the loop lives in `tools/navigate.py`, which is a check you
run with an explicit gold budget, not a bot you leave running.

| Phase | Status |
|---|---|
| 0 — Plumbing: ADB, capture, recorder, template tools, one tap | done |
| 1 — Navigate: screen classifier + state machine | done |
| 2 — Read loot | done |
| 3 — Deploy troops | **next** |
| 4 — Building detection (YOLO) | not started |
| 5 — Decision layer | not started |

Phases 1 and 2 were done out of order: the loot reader came first, so the bot could read a base
before it could reach one.

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

All six live in `tools/`. Nothing in `capture/`, `control/`, `vision/` or `policy/` is run
directly — those are imported.

| Command | What it does | Emulator? |
|---|---|---|
| `python -m tools.adb_ss` | Record frames. Prompts for a name prefix; `s` saves, `q` quits. Saves to `templates/initial_collection/<prefix>_<n>.png` and skips indices already on disk. | yes |
| `python -m tools.debug_match` | Live overlay boxing every button it recognizes, with match scores. `q` to quit. | yes |
| `python -m tools.tap_anchor <label>` | Locate a named button, tap it, and report what changed on screen. `--dry-run` locates without tapping. | yes |
| `python -m tools.extract_digits cut\|build` | Cut labelled digit glyphs from a frame, then vote them into `templates/digits/`. | no |
| `python -m tools.read_loot <frames...>` | Read the loot panel from saved PNGs. `--expect G E D` turns it into a pass/fail assertion. | no |
| `python -m tools.classify_screens <frames...>` | Name the screen in saved PNGs and print every template's score. `--expect` turns it into a pass/fail assertion. | no |
| `python -m tools.navigate --max-nexts N` | Walk the whole navigation loop. `--max-nexts` is required. `--dry-run` classifies one screen without tapping. | yes |

Known anchor labels: `attack`, `attack-button`, `find-match`, `end-battle`, `next-button`.

## Screens

Five anchors name four screens. `vision/screens.py` holds the rule table; first rule whose
anchors are all present wins.

| Screen | Anchor | Where it leads |
|---|---|---|
| `home` | `attack` | tap → army |
| `army` | `attack-button` | tap → attack menu |
| `attack_menu` | `find-match` | tap → costs 900, finds a base |
| `scout` | `next-button`, `end-battle` | tap Next (900) or End Battle → home |
| `unknown` | none | wait, then back, then give up |

`unknown` covers two different things on purpose. The clouds between bases carry no buttons at
all, and neither does an unrecognized popup, so they are indistinguishable by anchor set. The
policy tells them apart by waiting: clouds clear on their own, a popup does not and earns the
back press. That is why there is no "searching" template — the screen it would match is both
brief and button-free.

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

### Run the navigation loop

```bash
python -m tools.navigate --max-nexts 2 --dry-run   # name one screen, tap nothing
python -m tools.navigate --max-nexts 2             # walk the loop: 2700 gold
```

`--max-nexts` is required and has no default, because every Next and the Find a Match that
starts the run cost 900 gold each; a run spends `900 × (1 + max-nexts)` and prints that before
it moves. `--dry-run` reports a single step, since without taps the screen never changes — to
check the classifier across the whole loop, drive the game by hand and run it once per screen.

Exit 0 means it got home; non-zero means it gave up, and any frame it could not name is saved
to `templates/unknown/` for you to look at.

### Check the classifier offline

```bash
python -m tools.classify_screens templates/initial_collection/adb_frame_*.png \
    --expect home attack_menu army scout
```

Prints every template's peak score on every frame, matched or not, then the gap between the
lowest score accepted and the highest rejected. `THRESHOLD` belongs inside that gap.

## Project layout

```
capture/   talking to the device. grab_frame() and nothing else.
vision/    pure functions: image in, findings out. No I/O, no ADB, no global state.
control/   actions on the device: tap, swipe, back.
policy/    what to do next. Thresholds now, the state machine later.
tools/     scripts a human runs. Never imported by the bot.
templates/ digits/ and buttons/ (tracked) plus digit_samples/, initial_collection/ and
           unknown/ (gitignored bulk data)
```

The "vision is pure" rule is load-bearing: it is what lets the entire loot reader be developed
and regression-tested against saved PNGs with no emulator attached.

`feasibility testing/` is early throwaway exploration. Ignore it.

## Performance

Measured on this setup, 1920×1080:

| Stage | Time | Rate |
|---|---|---|
| `capture.grab_frame` | ~185 ms | 5.4 fps |
| `vision.anchors.find_anchors`, 5 templates, full frame | ~465 ms | 2.2 fps |
| `vision.screens.classify` | ~0.01 ms | negligible |
| `vision.loot.read_loot` | ~0.6 ms | negligible |

One navigation step is a capture plus a match, about 0.65 s; naming the screen and reading the
loot are free next to that. Base scanning needs 1–3 fps, so this is adequate — but anchor
matching, not capture, is the bottleneck, and it grows linearly with each template added.
Restricting each template to the region it can actually appear in is the obvious win when that
starts to matter.

The scout screen's `Battle starts in: 23s` is the deadline that makes this worth watching: if
the bot dawdles there, the preview ends and it lands in a live battle it cannot yet name.

## Design notes

`CLAUDE.md` holds the reasoning: why template matching rather than a model, why the ROI is
binarized before matching, the font-size and template-provenance traps, and where the
thresholds came from. Read it before changing anything in `vision/`.
