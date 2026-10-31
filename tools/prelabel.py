"""Pre-label unlabeled frames with a trained detector, for review in Roboflow.

Writes a YOLO folder Roboflow imports with its boxes as annotations:

  <out>/images/<stem>.png   hard links to the frames
  <out>/labels/<stem>.txt   one `cls cx cy w h` line per box, normalized
  <out>/data.yaml           the model's class order (= the export's)

Boxes come from vision.buildings.detect_buildings, so they are exactly what the bot would see:
its CONF (low on purpose -- deleting a wrong box is faster than drawing a missed one), and only
the most confident Town Hall, Clan Castle and Dark Elixir Storage.

Typical use:

  python -m tools.prelabel models/buildings_v2.pt datasets/scout/pool datasets/prelabel_v2
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from vision.buildings import FRAME_SIZE, Building, detect_buildings


def label_lines(buildings: list[Building], names: dict[int, str]) -> list[str]:
    """One `cls cx cy w h` line per box, normalized to the frame."""

    h, w = FRAME_SIZE
    index = {name: cls for cls, name in names.items()}
    return [f"{index[b.name]} {(b.x0 + b.x1) / 2 / w:.6f} {(b.y0 + b.y1) / 2 / h:.6f} "
            f"{(b.x1 - b.x0) / w:.6f} {(b.y1 - b.y0) / h:.6f}" for b in buildings]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="weights, e.g. models/buildings_v2.pt")
    parser.add_argument("frames", type=Path, help="folder of PNG frames")
    parser.add_argument("out", type=Path, help="folder to create")
    args = parser.parse_args()

    if args.out.exists():
        sys.exit(f"{args.out} exists; remove it first")
    frames = sorted(args.frames.glob("*.png"))
    (args.out / "images").mkdir(parents=True)
    (args.out / "labels").mkdir()

    model = YOLO(args.model)
    names = model.names
    n_boxes = 0
    for frame in frames:
        lines = label_lines(detect_buildings(cv2.imread(str(frame)), model), names)
        (args.out / "images" / frame.name).hardlink_to(frame)
        (args.out / "labels" / f"{frame.stem}.txt").write_text("".join(f"{l}\n" for l in lines))
        n_boxes += len(lines)

    (args.out / "data.yaml").write_text(
        f"nc: {len(names)}\nnames: {[names[i] for i in range(len(names))]}\n")
    print(f"{len(frames)} frames, {n_boxes} boxes -> {args.out}")


if __name__ == "__main__":
    main()
