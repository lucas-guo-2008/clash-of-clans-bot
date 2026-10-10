# Clash of Clans Attack Bot

A computer-vision bot that farms loot in Clash of Clans on its own. It captures the game screen
over ADB, recognises the interface, reads each opponent's loot, attacks the bases worth it with
the whole army, and records what it won — no game API and no injected code, just pixels in and
touches out. It runs against the real client in an Android emulator on macOS.

A run: home → army → attack menu → search → read the loot, **Next** until a base clears the
thresholds → find the deploy zone and drop the army around it → wait out the battle → read
"You got" off the result screen → Return Home. It stops back home after `ITERATIONS` battles.

> Automating the game is against Supercell's Terms of Service. This is a learning project, run on
> a throwaway account.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use `.venv/bin/python` rather than activating the venv — a bare `pip` resolves to the pyenv
global install.

Start BlueStacks with Clash of Clans open, **1920×1080, landscape**, then connect:

```bash
adb connect 127.0.0.1:5555
```

## Configure

- **`DECK`** in `main.py` — your army, slot by slot from the left, as `(kind, count)`:
  `troop` (tap the card, then `count` drops), `hero` (one drop; ability fires
  `HERO_ABILITY_DELAY` s later), `spell` (located, not cast yet), `skip`. If the deck on screen
  has a different number of cards, the bot stops rather than guess.
- **`ITERATIONS`** in `main.py` — battles per run (default 1). Troops are not retrained between
  battles.
- **Loot thresholds** in `policy/thresholds.py` — gold, elixir and dark elixir must all clear
  them (default 400k / 400k / 2k).

**Every search and every Next costs 900 gold**, so strict thresholds spend more before a base
qualifies.

## Run

```bash
.venv/bin/python main.py --dry-run          # name the screen it is looking at, tap nothing
.venv/bin/python main.py --show             # run, with a live window of what it sees and decides
.venv/bin/python main.py                    # run, console only
```

The console prints each change of screen or action, the loot read on each base, and what each
battle won. `--show` adds every template score, the deploy outline and drop points, the card
slots, and a crosshair on the next tap. Stop with Ctrl-C, or `q` in the window.

## Test

```bash
.venv/bin/python -m pytest -q
```

Runs in ~10 s with no emulator: screen recognition, loot and result reading, deploy points and
card slots over saved frames, plus the whole loop against a fake device and clock. Saved frames
are local data (not in git), so on a fresh clone the frame-based tests skip.

## Layout

```
main.py     the loop and its configuration
capture/    screen capture over ADB
vision/     pure image analysis: buttons, screens, loot, deploy zone, deck, overlay
policy/     pure decisions: what to do on each screen, attack thresholds
control/    taps
tools/      scripts a human runs (frame recorder, template cutter, offline overlays) — see --help
tests/      pytest suite
templates/  button and digit templates
```
