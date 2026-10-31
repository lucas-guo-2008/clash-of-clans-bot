"""Pre-label unlabeled frames with a trained detector, for review in Roboflow.

Writes a YOLO folder Roboflow imports with its boxes as annotations:

  <out>/images/<stem>.png   hard links to the frames
  <out>/labels/<stem>.txt   one `cls cx cy w h` line per box, normalized
  <out>/data.yaml           the model's class order (= the export's)

CONF is low on purpose: deleting a wrong box is faster than drawing a missed one. A base has
one Town Hall, Clan Castle and Dark Elixir Storage, so only the most confident box of each is
kept -- the detector has no NMS and sometimes doubles them.

Typical use:

  python -m tools.prelabel models/buildings_v1.pt datasets/scout/pool datasets/prelabel_v1
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

IMGSZ = 1280
DEVICE = "mps"
CONF = 0.25
SINGLETONS = {"town_hall", "clan_castle", "dark_elixir_storage"}


def label_lines(result, names: dict[int, str]) -> list[str]:
    boxes = sorted(zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                       result.boxes.xywhn.tolist()), key=lambda b: -b[1])
    lines, seen = [], set()
    for cls, _, (cx, cy, w, h) in boxes:
        cls = int(cls)
        if names[cls] in SINGLETONS:
            if cls in seen:
                continue
            seen.add(cls)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="weights, e.g. models/buildings_v1.pt")
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
    results = model.predict([str(f) for f in frames], imgsz=IMGSZ, device=DEVICE, conf=CONF,
                            stream=True, verbose=False)
    for frame, result in zip(frames, results):
        lines = label_lines(result, names)
        (args.out / "images" / frame.name).hardlink_to(frame)
        (args.out / "labels" / f"{frame.stem}.txt").write_text("".join(f"{l}\n" for l in lines))
        n_boxes += len(lines)

    (args.out / "data.yaml").write_text(
        f"nc: {len(names)}\nnames: {[names[i] for i in range(len(names))]}\n")
    print(f"{len(frames)} frames, {n_boxes} boxes -> {args.out}")


if __name__ == "__main__":
    main()
