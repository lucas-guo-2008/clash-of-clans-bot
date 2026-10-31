"""Turn a Roboflow YOLO export into a training set on the bot's own PNGs.

Roboflow renames every image `<stem>_png.rf.<hash>.jpg` and re-encodes it to JPEG. The labels
are normalized, so they pair with the original 1920x1080 PNG in templates/scout unchanged, and
training sees the pixels the bot will see live.

  - <out>/{train,valid,test}/images/<stem>.png  hard links to templates/scout
  - <out>/{train,valid,test}/labels/<stem>.txt  copied from the export
  - <out>/data.yaml                             the export's class order, paths into <out>

It also relinks datasets/scout: batch1 = every labeled frame, pool = every other frame not in
excluded.csv. Both folders hold only hard links, so emptying them loses nothing.

Typical use:

  python -m tools.build_dataset datasets/coc-buildings-v1 datasets/buildings_v1
"""

import argparse
import ast
import csv
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SCOUT_DIR = ROOT / "templates" / "scout"
SCOUT_SETS = ROOT / "datasets" / "scout"
SPLITS = ("train", "valid", "test")
FRAME_SIZE = (1080, 1920)


def original_stem(export_name: str) -> str:
    return export_name.split("_png.rf.")[0]


def read_names(data_yaml: Path) -> list[str]:
    # Roboflow writes `names: [...]` on one line; no PyYAML dependency for one key.
    for line in data_yaml.read_text().splitlines():
        if line.startswith("names:"):
            return ast.literal_eval(line.split(":", 1)[1].strip())
    raise ValueError(f"no names in {data_yaml}")


def build(export: Path, out: Path) -> dict[str, list[str]]:
    if out.exists():
        sys.exit(f"{out} exists; remove it first")
    stems: dict[str, list[str]] = {}
    for split in SPLITS:
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        stems[split] = []
        for label in sorted((export / split / "labels").glob("*.txt")):
            stem = original_stem(label.stem)
            png = SCOUT_DIR / f"{stem}.png"
            frame = cv2.imread(str(png))
            if frame is None or frame.shape[:2] != FRAME_SIZE:
                sys.exit(f"{label.name}: no 1920x1080 original at {png}")
            (out / split / "images" / png.name).hardlink_to(png)
            shutil.copyfile(label, out / split / "labels" / f"{stem}.txt")
            stems[split].append(stem)

    names = read_names(export / "data.yaml")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(names)}\nnames: {names}\n"
    )
    return stems


def relink_scout(labeled: set[str]) -> tuple[int, int]:
    with open(SCOUT_SETS / "excluded.csv") as f:
        excluded = {Path(row["file"]).stem for row in csv.DictReader(f)}
    if labeled & excluded:
        sys.exit(f"labeled frames listed in excluded.csv: {sorted(labeled & excluded)}")
    batch1, pool = SCOUT_SETS / "batch1", SCOUT_SETS / "pool"
    for folder in (batch1, pool):
        for png in folder.glob("*.png"):
            if not (SCOUT_DIR / png.name).samefile(png):
                sys.exit(f"{png} is not a link into {SCOUT_DIR}; not deleting it")
            png.unlink()
        folder.mkdir(exist_ok=True)
    counts = [0, 0]
    for png in sorted(SCOUT_DIR.glob("*.png")):
        if png.stem in excluded:
            continue
        is_labeled = png.stem in labeled
        (batch1 if is_labeled else pool).joinpath(png.name).hardlink_to(png)
        counts[0 if is_labeled else 1] += 1
    return counts[0], counts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export", type=Path, help="unzipped Roboflow YOLO export")
    parser.add_argument("out", type=Path, help="training set to create")
    args = parser.parse_args()

    stems = build(args.export, args.out)
    for split in SPLITS:
        print(f"{split}: {len(stems[split])}")
    labeled = {s for split in SPLITS for s in stems[split]}
    n_batch1, n_pool = relink_scout(labeled)
    print(f"batch1: {n_batch1}  pool: {n_pool}")


if __name__ == "__main__":
    main()
