# Clash of Clans Attack Bot

A computer-vision bot that operates Clash of Clans on its own. It captures the game screen over
ADB, recognises the interface in front of it, reads the state of each prospective target, and
drives the client from its own decisions — no game API and no injected code, just pixels in and
touches out. It runs against the real client in an Android emulator on macOS.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use `.venv/bin/python` rather than activating the venv — a bare `pip` resolves to the pyenv
global install.

Start the emulator with Clash of Clans running, then connect:

```bash
adb connect 127.0.0.1:5555
```

Entry points live in `tools/`, each with its own `--help`.
