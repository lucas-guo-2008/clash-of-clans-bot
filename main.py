"""
Farm: home -> attack menu -> army -> scout -> Next until a base clears the thresholds ->
drop the army around it -> wait out the battle -> Return Home, ITERATIONS times.

  python main.py --dry-run
  python main.py
  python main.py --show      # live window: the screen at ~60 fps under everything the bot found and decided
  python main.py --no-save   # play without recording the run (no records or frames in templates/scout)

Stops back home after ITERATIONS battles. Prints the buildings detected on each judged base
(nothing decides on them yet) and logs every judged base to templates/scout/bases.jsonl, with
its frame. After each battle it prints what "You got" showed and a running summary of the run,
and logs the result to templates/scout/battles.jsonl, pointing at its base, with a battle frame
every few seconds. templates/scout/runs.jsonl holds each run's commit and settings.
"""

import argparse
import itertools
import json
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2

from capture.adb import connect, grab_frame
from capture.stream import Stream
from control.tap import tap_match, tap_point
from policy import thresholds
from policy.navigator import (ABORT, ARMS, DEPLOY, FULL, GATE_TIMEOUT, PATIENCE_SECONDS, STALL, STALL_SECONDS, STAR,
                              TAP, UNGATED, WAIT, Action, battle_stalled, decide, gate_open, star_won)
from vision.anchors import THRESHOLD, Match, peak_scores
from vision.buildings import MODEL_PATH, Building, detect_buildings, load_detector
from vision.buttons import load_button_templates
from vision.deploy import DeployPoint, card_spent, deck_slots, deploy_points, outline_lines, outline_mask
from vision.glyphs import load_digit_templates
from vision.loot import LootReading, read_loot
from vision.overlay import (CYAN, GREEN, GREY, MAGENTA, ORANGE, PINK, PURPLE, RED, WHITE, YELLOW, draw_anchors,
                            draw_buildings, draw_deploy, draw_tap, with_header)
from vision.result import load_result_templates, read_result
from vision.screens import Screen, classify

ROOT = Path(__file__).resolve().parent
RESULT_SETTLE_SECONDS = 1.0  # "You got" must read the same for this long before it counts
HOME_SETTLE_SECONDS = 2.0  # home must hold this long before Battle is tapped: a reward crate pops up over it
# ~0.5 s after home appears (live 2026-11-20), and tapping Battle first left the crate unopened
SCOUT_SETTLE_SECONDS = 0.6  # a readable loot panel must hold this long before the base is judged
SCOUT_UNREADABLE_SECONDS = 2.0  # an unreadable panel is skipped only this long after the base appeared
UNKNOWN_SAVE_SECONDS = 5.0  # an unrecognized stretch is saved only once it outlasts a cloud (live: under 1 s)
clock = time.monotonic  # every timing reads this, so tests can drive time
UNKNOWN_DIR = ROOT / "templates" / "unknown"
SCOUT_DIR = ROOT / "templates" / "scout"
BASES_LOG = "bases.jsonl"  # in SCOUT_DIR: one JSON record per judged base
BATTLES_LOG = "battles.jsonl"  # in SCOUT_DIR: one per settled result, pointing at its base
RUNS_LOG = "runs.jsonl"  # in SCOUT_DIR: one per start, the code and settings every record of that run came from
RECORDING = True  # False under --no-save: nothing is written to SCOUT_DIR (unknown frames are still kept)
BATTLE_FRAME_SECONDS = 5.0  # a battle frame is kept this often from the deploy (training data for what got destroyed)
WINDOW = "bot"
VIEW_FPS = 30  # the --show window's refresh cap; the video arrives at up to 60, findings at ~3.6

ITERATIONS = 40
rng = random.Random()  # draws each battle's arm; the tests seed their own

TOWN_HALL = 11
ARMY_BOOST = False
EVENT = False

# The army, slot by slot left to right, as (kind, count). Slots are found on screen; only
# what is in them is declared here, so reordering the army means editing this line.
#   troop: tap the card, then `count` drops cycling the deploy points
#   hero:  tap the card, one drop; the ability fires HERO_ABILITY_DELAY s later (card tapped again)
#   spell: located but not cast -- aiming waits for building detection (Phase 4/5)
DECK = (("troop", 24), ("skip", 1), ("hero", 1), ("spell", 11), ("skip", 1))
HERO_ABILITY_DELAY = 10.0
MIN_DEPLOY_POINTS = 12  # fewer means the outline was not found (not fully drawn yet, or a dark map skin)
OUTLINE_SECONDS = 5.0
CARD_CHECK_SECONDS = 3.0


def describe(matches) -> str:
    if not matches:
        return "(no anchors)"
    return " ".join(f"{m.label}:{m.score:.3f}" for m in sorted(matches.values(), key=lambda m: -m.score))


def describe_loot(reading: LootReading) -> str:
    if not reading.ok:
        reasons = ",".join(row.reason for row in reading.rows if row.reason) or "?"
        return f"loot=UNREADABLE({reasons})"
    return f"loot={reading.gold}/{reading.elixir}/{reading.dark_elixir}"


def describe_buildings(buildings: list[Building]) -> str:
    counts = Counter(b.name for b in buildings)
    return " ".join(f"{name}:{n}" for name, n in sorted(counts.items())) or "(none)"


def run_summary(looted: tuple[int, int, int], searches: int, minutes: float) -> str:
    return f"run: looted {looted[0]} gold, {looted[1]} elixir, {looted[2]} dark elixir across {searches} search{'es' if searches != 1 else ''} in {minutes:.1f} min"


def deploy_army(device, slots: list[Match], points: list[DeployPoint]) -> list[Match]:
    """Drop every troop and hero card left to right, cycling the deploy points.
    Returns the hero cards, for the ability tap later."""

    drops = itertools.cycle(points)
    heroes = []
    for slot, (kind, count) in zip(slots, DECK):
        if kind not in ("troop", "hero"):
            continue
        tap_match(device, slot)
        for _ in range(count if kind == "troop" else 1):
            tap_point(device, next(drops))
        if kind == "hero":
            heroes.append(slot)
    return heroes


def save_unknown(frame) -> Path:
    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    out = UNKNOWN_DIR / f"unknown_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    cv2.imwrite(str(out), frame)
    return out


def loot_fields(reading: LootReading, key: str) -> dict:
    """A reading as record fields: the numbers only if every row read, else why each row did not."""

    if reading.ok:
        return {key: {"gold": reading.gold, "elixir": reading.elixir, "dark_elixir": reading.dark_elixir},
                "unreadable": None}
    return {key: None, "unreadable": [row.reason for row in reading.rows]}


def append_record(log: str, record: dict) -> None:
    if not RECORDING:
        return
    SCOUT_DIR.mkdir(parents=True, exist_ok=True)
    with (SCOUT_DIR / log).open("a") as f:
        f.write(json.dumps(record) + "\n")


def save_png(frame, name: str) -> str:
    """Keep a frame in SCOUT_DIR, lossless: models train on what the bot sees live."""

    if not RECORDING:
        return name
    SCOUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SCOUT_DIR / name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return name


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_run() -> str:
    """Log this run's code and settings to runs.jsonl and return its id; every base and battle
    record carries it, so data from different rules or decks can be told apart later."""

    stamp = f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    status = git("status", "--porcelain", "--untracked-files=no")
    append_record(RUNS_LOG, {
        "id": stamp,
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),  # local settings, e.g. data-run thresholds
        "detector": MODEL_PATH.name,
        "thresholds": {"total": thresholds.MIN_TOTAL, "max_nexts": thresholds.MAX_NEXTS,
                       "max_unreadable": thresholds.MAX_UNREADABLE},
        "deck": DECK,
        "iterations": ITERATIONS,
        "hero_ability_delay": HERO_ABILITY_DELAY,
        "stall_seconds": STALL_SECONDS,
        "arms": ARMS,
        "town_hall": TOWN_HALL,
        "army_boost": ARMY_BOOST,
        "event": EVENT,
        "battle_frame_seconds": BATTLE_FRAME_SECONDS,
    })
    return stamp


def base_record(stamp: str, run_id: str, reading: LootReading, decision: str, buildings: list[Building],
                points: list[DeployPoint], frame_file: str, search_seconds: float | None, nexts: int = 0) -> dict:
    """One judged base as Phase 5 will read it back: enough to compute features for a base whose
    frame was not kept. A battle record points back at it by `id`."""

    return {
        "id": stamp,
        "run": run_id,
        "decision": decision,
        "frame": frame_file,
        # From the Attack / Next tap to this base being judged: clouds, panel settling, all of it.
        "search_seconds": None if search_seconds is None else round(search_seconds, 1),
        "nexts": nexts,  # bases skipped in a row before this one; an attack at MAX_NEXTS is the search cap's
        **loot_fields(reading, "loot"),
        # Detections depend on the model: a retrained one would count differently.
        "detector": MODEL_PATH.name,
        "buildings": [{**asdict(b), "conf": round(b.conf, 3)} for b in buildings],
        "deploy_points": [[p.x, p.y] for p in points],
    }


def save_base(frame, run_id: str, reading: LootReading, decision: str, buildings: list[Building],
              points: list[DeployPoint], search_seconds: float | None, nexts: int = 0) -> str:
    """Log a judged base to bases.jsonl with its frame, and return its id. Skipped bases keep their
    frame too: unlabeled detector data. `decision` is the loot rule's; an "attack" with no battle
    pointing at it was skipped for want of a deploy outline. (index.csv is the frozen Phase 4 log.)"""

    stamp = f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    frame_file = save_png(frame, f"scout_{stamp}.png")
    append_record(BASES_LOG, base_record(stamp, run_id, reading, decision, buildings, points, frame_file,
                                         search_seconds, nexts))
    return stamp


def save_battle(frame, run_id: str, result: LootReading, base_id: str | None, battle_seconds: float | None,
                trace: list[list], arm: str | None, ended_by: str | None, end_tap: list | None,
                battle_frames: list[list], final_frame: str | None, deployed: list[list], redeployed: list[list]) -> None:
    """Log a settled result to battles.jsonl with its frame, so a misread "You got" can be read
    again. `base` is the attacked base's id in bases.jsonl; None (and no duration) when this run
    did not deploy the army, e.g. it was started mid-battle. `trace` is the battle screen's
    Available Loot each time it changed, as [seconds since deploy, gold, elixir, dark elixir].
    `arm` is the stop rule drawn for this battle (None when this run did not deploy it).
    `ended_by` is the rule the bot ended the battle on, "star" (first star won) or "stall" (loot
    stopped coming); None when the game ended it, as it always does on the "full" arm. `ended_early` is the same fact as a bool, kept
    for the records written before there were two rules. `end_tap` is [seconds since deploy, gold,
    elixir, dark elixir] at the first end tap: the time, for the end lag, and the Available Loot last
    read, for what arrived after the tap. None when the bot did not end the battle.
    `battle_frames` are [seconds since deploy, file] every BATTLE_FRAME_SECONDS; `final_frame` is
    the last battle frame before the result: what the army left standing. `deployed` is the deploy
    points the army actually went down on (the base record's are from the frame it was judged on);
    `redeployed` is [seconds since deploy, x, y] for each troop dropped again after the game refused it."""

    stamp = f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    frame_file = save_png(frame, f"result_{stamp}.png")
    append_record(BATTLES_LOG, {
        "id": stamp,
        "run": run_id,
        "base": base_id,
        **loot_fields(result, "got"),
        # From the army going down to "You got" settling: includes the count-up and RESULT_SETTLE_SECONDS.
        "battle_seconds": None if battle_seconds is None else round(battle_seconds, 1),
        "loot_trace": trace,
        "ended_early": ended_by is not None,  # a rule that misjudged shows as loot cut short here
        "arm": arm,
        "ended_by": ended_by,
        "end_tap": end_tap,
        "deploy_points": deployed,
        "redeployed": redeployed,
        "battle_frames": battle_frames,
        "final_battle_frame": final_frame,
        "frame": frame_file,
    })


@dataclass(frozen=True)
class Findings:
    """What the bot found and decided in one step, for the --show window."""

    step: int
    screen: Screen
    action: Action
    peaks: dict[str, Match]
    reading: LootReading | None
    slots: list[Match]  # deck cards: on the scout screen, and in battle under --show
    spent: list[bool]  # per slot, in battle: the card is greyed out
    points: list[DeployPoint]  # in battle: the points the army went down on
    buildings: list[Building] | None  # in battle: the attacked base's, found on its scout frame
    tap: tuple[int, int] | None
    unknown_seconds: float
    battles: int
    lines: object  # the deploy outline (np.ndarray); in battle the scout frame's, so it trails a pan
    arm: str | None  # this battle's stop rule, once the army is down
    trace: list[list]  # this battle's Available Loot trace, a copy
    looted: tuple[int, int, int]  # the run so far
    minutes: float  # since the run started
    nexts: int = 0  # bases skipped in a row
    searches: int = 0  # Find a Match and Next taps this run
    since_drop: float | None = None  # seconds since the army went down, in battle


SCREEN_NAMES = {
    Screen.HOME: "Home village", Screen.ATTACK_MENU: "Attack menu", Screen.ARMY: "Army",
    Screen.SCOUT: "Scouting a base", Screen.BATTLE: "Battle", Screen.RESULT: "Battle result",
    Screen.END_CONFIRM: "Ending the battle", Screen.CRATE: "Reward crate", Screen.REWARD: "Reward",
    Screen.STAR_BONUS: "Star bonus", Screen.UNKNOWN: "Unrecognized screen",
}
STOP_RULES = {STAR: "at the first star", STALL: f"when loot stops for {STALL_SECONDS:.0f} s", FULL: "never (full battle)"}


def situation(f: Findings) -> list[tuple[str, tuple]]:
    """The header's second row: what this screen means, in words."""

    if f.screen is Screen.SCOUT and f.reading is not None:
        if not f.reading.ok:
            return [("loot panel unreadable, skipping soon", RED)]
        total = f.reading.gold + f.reading.elixir
        clears = total >= thresholds.MIN_TOTAL
        row = [(f"gold {f.reading.gold:,}", YELLOW), (f"elixir {f.reading.elixir:,}", PINK),
               (f"dark {f.reading.dark_elixir:,}", PURPLE),
               (f"gold+elixir {total / 1e6:.2f}M, {'clears' if clears else 'under'} the "
                f"{thresholds.MIN_TOTAL / 1e6:.1f}M bar", GREEN if clears else RED)]
        if f.nexts:
            row.append((f"skipped {f.nexts} in a row (attacks anyway after {thresholds.MAX_NEXTS})", GREY))
        return row
    if f.screen in (Screen.BATTLE, Screen.END_CONFIRM) and f.trace:
        shown = f.trace[0][1] + f.trace[0][2]
        taken = shown - f.trace[-1][1] - f.trace[-1][2]
        row = [(f"taken {taken / 1e6:.2f}M of {shown / 1e6:.2f}M", GREEN)]
        if f.since_drop is not None:
            row.append((f"{f.since_drop:.0f} s since the drop", WHITE))
        if f.arm is not None:
            row.append((f"ends {STOP_RULES[f.arm]}", CYAN))
        troops = [gone for gone, (kind, _) in zip(f.spent, DECK) if kind == "troop"]
        if troops:
            row.append((f"troop cards used up {sum(troops)}/{len(troops)}", GREY))
        return row
    if f.screen is Screen.UNKNOWN:
        return [(f"no known buttons for {f.unknown_seconds:.0f} s; gives up at {PATIENCE_SECONDS:.0f} s "
                 "(clouds between screens clear on their own)", RED)]
    return []


def legend(f: Findings) -> list[tuple[str, tuple]]:
    """The header's last row: what each colour on the picture is, with counts where they help."""

    row = [("buttons", GREEN)]
    if f.lines is not None or f.points:
        row += [("no-deploy line", MAGENTA), (f"{len(f.points)} drop points", CYAN)]
    if f.slots:
        row.append((f"{len(f.slots)} cards", GREEN))
    if f.buildings is not None:
        row += [(f"{len(f.buildings)} buildings:", WHITE), ("gold", YELLOW), ("elixir", PINK), ("dark", PURPLE),
                ("town hall", ORANGE), ("clan castle", WHITE)]
    if f.tap is not None:
        row.append(("next tap", RED))
    return row


def render(frame, f: Findings):
    """Everything the bot found and decided in one step, drawn over `frame` (the live video, or the
    bot's own frame)."""

    image = frame.copy()
    draw_buildings(image, f.buildings or [])
    spent = f.spent or [False] * len(f.slots)
    labels = [(slot, "spent", GREY) if gone else (slot, f"{kind} x{count}", GREEN)
              for slot, (kind, count), gone in zip(f.slots, DECK, spent)]
    draw_deploy(image, f.lines, f.points, labels)
    draw_anchors(image, f.peaks, THRESHOLD)
    if f.tap is not None:
        draw_tap(image, f.tap)

    doing = {DEPLOY: "attack: ", ABORT: "give up: "}.get(f.action.kind, "")
    total = f.looted[0] + f.looted[1]
    run = [(f"run {f.minutes:.0f} min", WHITE), (f"battles {f.battles} of {ITERATIONS}", WHITE),
           (f"looted {total / 1e6:.2f}M gold+elixir", GREEN)]
    if f.minutes >= 1:
        run.append((f"{total / 1e6 / (f.minutes / 60):.1f}M per hour", GREEN))
    run.append((f"{f.searches} searches", GREY))

    rows = [
        [(SCREEN_NAMES[f.screen].upper(), RED if f.screen is Screen.UNKNOWN else GREEN),
         (f"-> {doing}{f.action.why}", WHITE)],
        situation(f),
        run,
        legend(f),
    ]
    return with_header(image, rows, f.trace if f.screen in (Screen.BATTLE, Screen.END_CONFIRM) else None)


class Viewer:
    """The --show window. macOS draws windows only from the main thread, so under --show the bot runs on
    a worker thread and hands over each step's frame and findings; the window redraws the latest findings
    on every frame of the live video. Findings update at the bot's rate (~3.6 fps), so on a pan the boxes
    trail the picture by up to ~0.3 s."""

    def __init__(self) -> None:
        self.published = None  # (bot's frame, Findings) of the latest step
        self.stopped = threading.Event()  # q or Esc: the bot stops after its current step

    def publish(self, frame, findings: Findings) -> None:
        self.published = (frame, findings)  # one assignment: the window never sees half a step

    def run(self, bot: threading.Thread, hold: bool) -> None:
        """Show until the bot ends, or until a key once it has, with `hold` (a dry run)."""

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 600)  # points: on a Retina screen the full-size image shows 1:1 at this size
        stream = Stream()
        shown, shown_at = (None, None), 0.0
        try:
            while bot.is_alive() or hold:
                published, video = self.published, stream.latest
                due = time.monotonic() - shown_at >= 1 / VIEW_FPS
                if published is not None and due and (published is not shown[0] or video is not shown[1]):
                    frame, findings = published
                    if video is not None and video.shape != frame.shape:
                        video = cv2.resize(video, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                    # The full-size image, scaled to the window by macOS: OpenCV cannot read a macOS window's
                    # size (getWindowImageRect reports the last image's), and a smaller image went blurry.
                    cv2.imshow(WINDOW, render(frame if video is None else video, findings))
                    shown, shown_at = (published, stream.latest), time.monotonic()
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self.stopped.set()
                    return
                if hold and not bot.is_alive() and key != 255:
                    return
        finally:
            stream.close()
            cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="classify one screen and report, but do not tap")
    parser.add_argument("--show", action="store_true", help="show what the bot sees and decides, live")
    parser.add_argument("--no-save", action="store_true",
                        help="record nothing in templates/scout (runs, bases, battles and their frames)")
    args = parser.parse_args()
    global RECORDING
    RECORDING = not args.no_save

    buttons = load_button_templates()
    digits = load_digit_templates()
    result_digits = load_result_templates()
    detector = load_detector()
    device = connect()
    if not args.show:
        return farm(args, device, buttons, digits, result_digits, detector, None)

    view = Viewer()
    outcome = []

    def bot() -> None:
        try:
            outcome.append(farm(args, device, buttons, digits, result_digits, detector, view))
        except BaseException as error:  # handed to the main thread, which raises it
            outcome.append(error)

    worker = threading.Thread(target=bot, daemon=True)
    worker.start()
    view.run(worker, hold=args.dry_run)
    worker.join()
    if isinstance(outcome[0], BaseException):
        raise outcome[0]
    return outcome[0]


def farm(args, device, buttons, digits, result_digits, detector, view: Viewer | None) -> int:
    """The bot's loop, until it is done (0), gives up (1), or the window is closed (0)."""

    unknown_since = None
    unknown_saved = False  # this unrecognized stretch already has a frame in UNKNOWN_DIR
    battles = 0
    pending, pending_since = None, 0.0
    settled = None  # this result screen's reading, once it held for RESULT_SETTLE_SECONDS
    result_counted = False  # this battle's result is counted; the screen can come back through clouds after Return Home
    scout_values, scout_since, scout_seen = None, 0.0, None  # this base's latest loot reading, since when, first seen
    scout_saved = False  # this base is logged in BASES_LOG
    buildings = None  # this base's detections, found once when it is judged
    heroes: list[Match] = []  # deployed hero cards whose ability has not been used yet
    troops: list[Match] = []
    spare = iter(())
    check_at = 0.0
    redeployed: list[list] = []
    ability_at = 0.0
    attacked_id, deployed_at = None, None  # the base under attack (bases.jsonl id), and when its army went down
    base_id = None  # the base on screen, once judged; it becomes attacked_id only when the army goes down
    deployed = []  # the deploy points the army went down on
    trace: list[list] = []  # this battle's Available Loot readings, one per change (Phase 5f's stall window)
    arm = None 
    ended_by = None
    end_tap = None
    surrender_seen = False  # this battle's button has read Surrender: a troop landed, no star yet
    star_frames = 0  # battle frames in a row showing End Battle since then: the first star
    battle_frames: list[list] = []  # [seconds since deploy, file] kept every BATTLE_FRAME_SECONDS
    last_battle = None  # the latest battle frame since the deploy, kept as final_battle_frame with the result
    searched_at = None  # when Attack (army screen) or Next was last tapped
    run_id = None if args.dry_run else save_run()
    looted = (0, 0, 0)  # gold, elixir, dark elixir over every readable result
    searches = 0
    nexts = 0  # bases skipped in a row since the last deploy (the search cap, thresholds.MAX_NEXTS)
    unreadable = 0  # judged bases in a row whose loot panel could not be read (thresholds.MAX_UNREADABLE)
    started = clock()
    tapped_on, tapped_at = None, 0.0  # the screen of the last gated tap, and when
    last_screen, frames_on_screen = None, 0
    screen_since = 0.0  # when the screen on show appeared
    last_logged = None
    attacked_view = None  # --show: (outline, deploy points, buildings) of the base under attack

    for step in itertools.count(1):
        frame = grab_frame(device)
        peaks = peak_scores(frame, buttons)
        matches = {label: m for label, m in peaks.items() if m.score >= THRESHOLD}
        screen = classify(matches)
        reading = read_loot(frame, digits) if screen is Screen.SCOUT else None

        # Available Loot counts down as it is taken. Unreadable frames are skipped, not guessed.
        if screen is Screen.BATTLE and deployed_at is not None:
            since = clock() - deployed_at
            if since >= len(battle_frames) * BATTLE_FRAME_SECONDS:
                tag = attacked_id or f"{datetime.now():%Y%m%d_%H%M%S_%f}"
                battle_frames.append([round(since, 1), save_png(frame, f"battle_{tag}_{since:05.1f}.png")])
            surrender_seen = surrender_seen or "surrender" in matches
            left = read_loot(frame, digits)
            if left.ok:  # the end-of-battle banner dims the panel; the frame before it shows what was left standing
                last_battle = frame
            values = [left.gold, left.elixir, left.dark_elixir]
            if left.ok and (not trace or trace[-1][1:] != values):
                trace.append([round(clock() - deployed_at, 1), *values])

        on_star = screen is Screen.BATTLE and surrender_seen and "end-battle" in matches
        star_frames = star_frames + 1 if on_star else 0
        frames_on_screen = frames_on_screen + 1 if screen is last_screen else 1
        last_screen = screen

        now = clock()
        if frames_on_screen == 1:
            screen_since = now
        if screen is Screen.UNKNOWN:
            unknown_since = now if unknown_since is None else unknown_since
        else:
            unknown_since, unknown_saved = None, False
        unknown_seconds = now - unknown_since if unknown_since is not None else 0.0

        result = None
        if screen is Screen.RESULT:
            if settled is None:
                read = read_result(frame, result_digits)
                values = (read.gold, read.elixir, read.dark_elixir)
                if values != pending:
                    pending, pending_since = values, now
                elif now - pending_since >= RESULT_SETTLE_SECONDS:
                    settled = read  # Return Home is tapped on every settled result, counted or not
                    if not result_counted:
                        result, result_counted = read, True
                        battles += 1
                        if read.ok:
                            looted = (looted[0] + read.gold, looted[1] + read.elixir, looted[2] + read.dark_elixir)
        else:
            pending, settled = None, None
            if screen is not Screen.UNKNOWN:  # left the result screen for real, not just into clouds
                result_counted = False

        scout_settled = None
        if screen is Screen.SCOUT:
            values = (reading.gold, reading.elixir, reading.dark_elixir)
            if values != scout_values:
                scout_values, scout_since = values, now
            scout_seen = now if scout_seen is None else scout_seen
            if reading.ok and now - scout_since >= SCOUT_SETTLE_SECONDS:
                scout_settled = reading
            elif not reading.ok and now - scout_seen >= SCOUT_UNREADABLE_SECONDS:
                scout_settled = reading
        else:
            scout_values, scout_seen, scout_saved, buildings = None, None, False, None

        line = f"step {step:>3}  {screen.value:<12} {describe(matches)}"
        if reading is not None:
            line += f"  {describe_loot(reading)}"

        if screen is Screen.HOME and battles >= ITERATIONS:
            print(f"{line}\n\nhome after {battles} battle(s) -- done")
            print(run_summary(looted, searches, (clock() - started) / 60))
            return 0

        # Once the whole army is down and every ability used: end at the first star (the star arm), or,
        # Phase 5f, when the battle has stopped paying. The stall is the fallback for a battle with no star.
        # The full arm never ends a battle: the game does.
        end_with, end_rule = None, STALL
        if screen is Screen.BATTLE and deployed_at is not None and not heroes and not troops and arm != FULL:
            if arm == STAR and star_won(surrender_seen, star_frames):
                end_with, end_rule = "end-battle", STAR
            elif battle_stalled(trace, now - deployed_at):
                end_with = "end-battle" if "end-battle" in matches else "surrender"  # End Battle once a star is won
        if screen is Screen.RESULT and "claim-reward" in matches:
            end_with = "claim-reward"

        action = decide(screen, settled if screen is Screen.RESULT else scout_settled, unknown_seconds, end_with,
                        end_rule, nexts, unreadable)
        judged = scout_settled is not None and not scout_saved
        if judged:
            buildings = detect_buildings(frame, detector)
            decision = "attack" if action.kind == DEPLOY else "next"
            base_id = save_base(frame, run_id, scout_settled, decision, buildings, deploy_points(frame),
                                None if searched_at is None else now - searched_at, nexts)
            scout_saved = True
            unreadable = 0 if scout_settled.ok else unreadable + 1

        if tapped_on is not None and gate_open(screen, tapped_on, now - tapped_at, frames_on_screen):
            tapped_on = None
        if tapped_on is not None and action.kind in (TAP, DEPLOY) and action.anchor not in UNGATED:
            waited = f"{now - tapped_at:.1f}/{GATE_TIMEOUT:.0f} s"
            action = Action(WAIT, why=f"holding until the screen changes after the last tap ({waited})")

        slots, points = [], []
        if screen is Screen.SCOUT and (args.show or action.kind == DEPLOY):
            slots, points = deck_slots(frame), deploy_points(frame)
        if screen is Screen.HOME and action.kind == TAP and now - screen_since < HOME_SETTLE_SECONDS:
            action = Action(WAIT, why=f"home for {now - screen_since:.1f}/{HOME_SETTLE_SECONDS:.0f} s, in case a reward crate pops up")
        # The outline is not always fully drawn, and on a dark map skin not found at all. Drops inside the red
        # zone are refused, so wait for a better frame rather than guess, then give up on the base.
        if action.kind == DEPLOY and len(points) < MIN_DEPLOY_POINTS:
            if now - scout_seen < OUTLINE_SECONDS:
                action = Action(WAIT, why=f"deploy outline not found yet ({len(points)} points)")
            else:
                action = Action(TAP, "next-button", f"no deploy outline ({len(points)} points), skip the base")

        # Frames come ~5 per second: log when what the bot sees or means to do changes.
        target = f" {action.anchor}" if action.anchor else ""
        logged = (screen, action.kind, action.anchor)
        if logged != last_logged:
            print(f"{line}  -> {action.kind}{target}  ({action.why})")
            last_logged = logged
        if judged:
            print(f"        buildings {describe_buildings(buildings)}")

        if result is not None:
            got = f"{result.gold}/{result.elixir}/{result.dark_elixir}" if result.ok else "UNREADABLE (not counted)"
            print(f"        battle {battles} got {got}")
            print(f"        {run_summary(looted, searches, (clock() - started) / 60)}")
            final = None if last_battle is None else save_png(last_battle, f"battle_{attacked_id}_final.png")
            save_battle(frame, run_id, result, attacked_id, None if deployed_at is None else now - deployed_at, trace,
                        arm, ended_by, end_tap, battle_frames, final, deployed, redeployed)
            attacked_id, deployed_at, trace, ended_by, deployed, redeployed = None, None, [], None, [], []
            arm, end_tap = None, None
            surrender_seen, attacked_view = False, None
            battle_frames, last_battle = [], None

        # One frame per unrecognized stretch that outlasts a cloud, plus the one it gives up on
        # (usually a popup). Clouds between screens clear on their own and are not worth keeping.
        outlasted = unknown_seconds >= UNKNOWN_SAVE_SECONDS and not unknown_saved
        if outlasted or action.kind == ABORT:
            print(f"        saved {save_unknown(frame).relative_to(ROOT)}")
            unknown_saved = True

        if action.kind == DEPLOY:
            print(f"        {len(slots)}/{len(DECK)} deck cards, {len(points)} deploy points")

        tap = None
        if action.kind == TAP:
            tap = matches[action.anchor].centre
        elif action.kind == DEPLOY and slots and points:
            tap = slots[0].centre

        if view is not None:
            lines = outline_lines(outline_mask(frame)) if screen is Screen.SCOUT else None
            shown_points, shown_buildings, spent = points, buildings, []
            if action.kind == DEPLOY and points:
                attacked_view = (lines, points, buildings)  # drawn over the battle that follows
            if screen in (Screen.BATTLE, Screen.END_CONFIRM) and attacked_view is not None:
                lines, shown_points, _ = attacked_view
            if screen is Screen.BATTLE:  # detected again each step, so a destroyed building's box goes (~18 ms)
                shown_buildings = detect_buildings(frame, detector)
            if screen is Screen.BATTLE:
                slots = deck_slots(frame)
                spent = [kind == "troop" and card_spent(frame, slot) for slot, (kind, _) in zip(slots, DECK)]
            view.publish(frame, Findings(step, screen, action, peaks, reading, slots, spent, shown_points,
                                         shown_buildings, tap, unknown_seconds, battles, lines, arm, list(trace),
                                         looted, (clock() - started) / 60, nexts, searches,
                                         None if deployed_at is None else clock() - deployed_at))

        if args.dry_run:
            if tap is not None:
                print(f"        would tap {action.anchor or 'card'} at {tap}")
            return 0

        if action.kind == ABORT:
            print(f"\ngiving up: {action.why}", file=sys.stderr)
            return 1

        if action.kind == TAP:
            tap_match(device, matches[action.anchor])
            if action.anchor not in UNGATED:
                tapped_on, tapped_at = screen, clock()
            if action.anchor in ("find-match", "next-button"):
                searches += 1
            if action.anchor == "next-button":
                nexts += 1
            if action.anchor in ("attack-button", "next-button"):  # the clouds start here
                searched_at = clock()
            if action.anchor in ("end-battle", "surrender"):
                if ended_by is None:
                    ended_by = end_rule
                    end_tap = [round(clock() - deployed_at, 1), *(trace[-1][1:] if trace else [None] * 3)]
        elif action.kind == DEPLOY:
            if len(slots) != len(DECK):
                print(f"\ndeck shows {len(slots)} cards but DECK lists {len(DECK)} -- update DECK in main.py",
                      file=sys.stderr)
                return 1
            else:
                deployed_at, trace, battle_frames, last_battle = clock(), [], [], None
                nexts, unreadable = 0, 0
                surrender_seen, ended_by, end_tap, arm = False, None, None, rng.choice(ARMS)
                attacked_id, deployed = base_id, [[p.x, p.y] for p in points]
                heroes = deploy_army(device, slots, points)
                troops = [slot for slot, (kind, _) in zip(slots, DECK) if kind == "troop"]
                spare = itertools.islice(itertools.cycle(points), len(points) // 2, len(points) // 2 + len(points))
                ability_at, check_at = clock() + HERO_ABILITY_DELAY, clock() + CARD_CHECK_SECONDS
                tapped_on, tapped_at = screen, clock()

        if screen is Screen.BATTLE and heroes and clock() >= ability_at:
            for hero in heroes:
                tap_match(device, hero)
            print(f"        hero abilities: tapped {len(heroes)} hero card(s)")
            heroes = []
        if screen is Screen.BATTLE and troops and clock() >= check_at:
            troops = [slot for slot in troops if not card_spent(frame, slot)]
            point = next(spare, None) if troops else None
            if point is None:
                troops = []
            else:
                tap_match(device, troops[0])
                tap_point(device, point)
                redeployed.append([round(clock() - deployed_at, 1), point.x, point.y])
                print(f"        {troops[0].label} not spent: one more drop at ({point.x}, {point.y})")
                check_at = clock() + CARD_CHECK_SECONDS
        if screen is Screen.RESULT:
            heroes, troops = [], []

        if view is not None and view.stopped.is_set():
            print("\nstopped from the window")
            return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped")
        sys.exit(0)
