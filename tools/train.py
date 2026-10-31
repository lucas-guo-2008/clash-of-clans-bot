"""Train the building detector on a set made by tools/build_dataset.py.

Starts from COCO-pretrained nano weights (downloaded once to runs/pretrained/), writes the run
to runs/<name>/ (curves, confusion matrix, weights), then copies the best checkpoint -- best
mAP on valid -- to models/buildings_<name>.pt.

Typical use:

  python -m tools.train datasets/buildings_v1 v1
"""

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
PRETRAINED = ROOT / "runs" / "pretrained" / "yolo26n.pt"
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"

TRAIN_ARGS: dict = {
    "imgsz": 1280,
    "epochs": 100,
    "batch": 8,
    "device": "mps",
    "fliplr": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path, help="folder holding data.yaml")
    parser.add_argument("name", help="run name, e.g. v1")
    args = parser.parse_args()

    target = MODELS_DIR / f"buildings_{args.name}.pt"
    if target.exists():
        sys.exit(f"{target} exists; pick another name")
    PRETRAINED.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(PRETRAINED)
    model.train(
        data=str((args.dataset / "data.yaml").resolve()),
        project=str(RUNS_DIR),
        name=args.name,
        **TRAIN_ARGS,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copyfile(best, target)
    print(f"{best} -> {target}")


if __name__ == "__main__":
    main()
