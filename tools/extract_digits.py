"""Cut digit templates out of saved attack-search frames.

Two steps:

  cut    segment a frame's loot rows and save each glyph under the digit it is.
         Pass the values you read off the screen and labelling is automatic --
         the glyph order is the digit order.

  build  majority-vote the collected samples into templates/digits/0-9.png.

Typical use:

  python -m tools.extract_digits cut templates/initial_collection/adb_frame_4.png --gold 525962 --elixir 1333432 --dark 29717
  python -m tools.extract_digits build
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.glyphs import DIGITS, majority_vote, save_template  # noqa: E402
from vision.loot import (  # noqa: E402
    LOOT_X,
    ROW_BANDS,
    ROW_NAMES,
    binarize,
    canonicalize,
    segment_glyphs,
)

SAMPLES_DIR = ROOT / "templates" / "digit_samples"


def save_samples(glyphs, expected, frame_stem, tag, samples_dir: Path) -> int:
    for index, (glyph, digit) in enumerate(zip(glyphs, expected)):
        out_dir = samples_dir / digit
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{frame_stem}_{tag}{index}.png"), canonicalize(glyph) * 255)
    return len(glyphs)


def cut(frame_path: Path, values: dict[str, str | None], samples_dir: Path) -> int:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise SystemExit(f"could not read {frame_path}")

    x0, x1 = LOOT_X
    saved = 0
    for name, (y0, y1) in zip(ROW_NAMES, ROW_BANDS):
        expected = values.get(name)
        if expected is None:
            continue

        glyphs = segment_glyphs(binarize(frame[y0:y1, x0:x1]))
        if len(glyphs) != len(expected):
            print(
                f"  {name}: segmented {len(glyphs)} glyphs but '{expected}' has "
                f"{len(expected)} digits -- skipped. Check the value, or the row band.",
                file=sys.stderr,
            )
            continue

        saved += save_samples(glyphs, expected, frame_path.stem, name, samples_dir)
        print(f"  {name}: {len(glyphs)} glyphs -> {expected}")

    return saved


def cut_band(frame_path: Path, value: str, band: tuple[int, int, int, int], samples_dir: Path) -> int:
    """Cut one arbitrary row of digits."""

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise SystemExit(f"could not read {frame_path}")

    y0, y1, x0, x1 = band
    glyphs = segment_glyphs(binarize(frame[y0:y1, x0:x1]))
    if len(glyphs) != len(value):
        raise SystemExit(
            f"segmented {len(glyphs)} glyphs (h={[g.h for g in glyphs]}) but '{value}' has "
            f"{len(value)} digits -- adjust the band"
        )
    if not all(22 <= g.h <= 25 for g in glyphs):
        raise SystemExit(
            f"glyph heights {[g.h for g in glyphs]} are outside the loot panel's 22-24 px. "
            "A different font size would need rescaling, which is what makes it unusable."
        )
    saved = save_samples(glyphs, value, frame_path.stem, "band", samples_dir)
    print(f"  band {band}: {len(glyphs)} glyphs -> {value}")
    return saved


CONSISTENT = 0.95


def build(samples_dir: Path) -> None:
    missing, suspect = [], []
    for digit in DIGITS:
        paths = sorted((samples_dir / digit).glob("*.png")) if (samples_dir / digit).is_dir() else []
        if not paths:
            missing.append(digit)
            continue
        samples = [(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127).astype("uint8") for p in paths]
        template = majority_vote(samples)

        agreements = [float((sample == template).mean()) for sample in samples]
        odd = [(a, p.stem) for a, p in zip(agreements, paths) if a < CONSISTENT]
        suspect.extend((digit, a, stem) for a, stem in odd)

        path = save_template(template, digit)
        note = f"  worst sample {min(agreements):.3f}" if len(samples) > 1 else ""
        print(f"  {digit}: {len(samples)} sample(s) -> {path.relative_to(ROOT)}{note}")

    if suspect:
        print("\nSamples disagreeing with their own template -- likely cut from a screen", file=sys.stderr)
        print("that renders this font differently. Remove them and rebuild:", file=sys.stderr)
        for digit, agreement, stem in suspect:
            print(f"   {digit}: {agreement:.3f}  {samples_dir.name}/{digit}/{stem}.png", file=sys.stderr)

    if missing:
        print(
            f"\nStill missing: {''.join(missing)}. Capture scan-screen frames whose loot "
            "values contain them and re-run cut.",
            file=sys.stderr,
        )
    else:
        print("\nAll ten digits present.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cut_parser = sub.add_parser("cut", help="segment a frame and save labelled glyph samples")
    cut_parser.add_argument("frame", type=Path)
    cut_parser.add_argument("--gold", help="gold value as shown, digits only")
    cut_parser.add_argument("--elixir", help="elixir value as shown, digits only")
    cut_parser.add_argument("--dark", help="dark elixir value as shown, digits only")
    cut_parser.add_argument(
        "--value", help="digits of one arbitrary row, used with --band"
    )
    cut_parser.add_argument(
        "--band",
        nargs=4,
        type=int,
        metavar=("Y0", "Y1", "X0", "X1"),
        help="cut one row from this box instead of the standard loot rows",
    )

    sub.add_parser("build", help="majority-vote samples into templates/digits")

    args = parser.parse_args()

    if args.command == "cut":
        print(f"{args.frame}:")
        if bool(args.band) != bool(args.value):
            raise SystemExit("--band and --value go together")
        if args.band:
            saved = cut_band(args.frame, args.value, tuple(args.band), SAMPLES_DIR)
        else:
            values = {"gold": args.gold, "elixir": args.elixir, "dark_elixir": args.dark}
            if not any(values.values()):
                raise SystemExit("pass at least one of --gold/--elixir/--dark, or --value with --band")
            saved = cut(args.frame, values, SAMPLES_DIR)
        print(f"saved {saved} glyph sample(s) to {SAMPLES_DIR.relative_to(ROOT)}")
    else:
        build(SAMPLES_DIR)


if __name__ == "__main__":
    main()
