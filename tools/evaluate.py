"""Evaluate a building detector on one split of a set made by tools/build_dataset.py.

Three views, all written under runs/eval/<model>_<split>/:

  1. mAP view -- Ultralytics val at its default conf (0.001): per-class precision, recall,
     mAP50, mAP50-95. This is the number to compare models by.
  2. Deploy view -- val again at CONF (0.25), the threshold the bot would use: confusion matrix
     (conf25/confusion_matrix.png) and one GT / FP / TP / FN grid per frame
     (conf25/visualizations/). Val at 0.001 fills the matrix's background column with junk.
  3. Count view -- the Phase 5 metric: per frame, how many of each building were detected at
     CONF vs. how many are labeled.

Typical use:

  python -m tools.evaluate models/buildings_v1.pt datasets/buildings_v1 test
"""

import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "runs" / "eval"
IMGSZ = 1280
DEVICE = "mps"
CONF = 0.25


def map_view(model: YOLO, data: str, split: str, name: str) -> None:
    metrics = model.val(data=data, split=split, imgsz=IMGSZ, device=DEVICE, plots=False,
                        project=str(EVAL_DIR), name=name, exist_ok=True, verbose=False)
    box = metrics.box
    print(f"\n{'class':<20} {'P':>6} {'R':>6} {'mAP50':>6} {'50-95':>6}")
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        print(f"{model.names[c]:<20} {p:6.3f} {r:6.3f} {ap50:6.3f} {ap:6.3f}")
    p, r, ap50, ap = box.mean_results()
    print(f"{'all':<20} {p:6.3f} {r:6.3f} {ap50:6.3f} {ap:6.3f}")


def deploy_view(model: YOLO, data: str, split: str, name: str) -> None:
    model.val(data=data, split=split, imgsz=IMGSZ, device=DEVICE, conf=CONF, plots=True,
              visualize=True, project=str(EVAL_DIR), name=f"{name}/conf25", exist_ok=True,
              verbose=False)


def read_counts(label: Path) -> Counter:
    return Counter(int(line.split()[0]) for line in label.read_text().splitlines() if line.strip())


def summarize_counts(true: list[int], pred: list[int]) -> dict[str, float]:
    """One class's counts over every frame (true[i], pred[i] = frame i) -> named summary numbers."""
    errors = [p - t for t, p in zip(true, pred)]
    return {
        "exact": sum(e == 0 for e in errors) / len(errors),
        "mae": sum(abs(e) for e in errors) / len(errors),
        "bias": sum(errors) / len(errors),
    }


def count_view(model: YOLO, dataset: Path, split: str) -> None:
    images = sorted((dataset / split / "images").glob("*.png"))
    true = [read_counts(dataset / split / "labels" / f"{img.stem}.txt") for img in images]
    pred = [Counter(int(c) for c in r.boxes.cls.tolist())
            for r in model.predict([str(i) for i in images], imgsz=IMGSZ, device=DEVICE,
                                   conf=CONF, verbose=False)]
    print(f"\ncounts at conf {CONF}, {len(images)} frames")
    for c, cname in model.names.items():
        t = [counts[c] for counts in true]
        p = [counts[c] for counts in pred]
        summary = "  ".join(f"{k} {v:.2f}" for k, v in summarize_counts(t, p).items())
        print(f"{cname:<20} true {sum(t):4d}  pred {sum(p):4d}  {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="weights, e.g. models/buildings_v1.pt")
    parser.add_argument("dataset", type=Path, help="folder holding data.yaml")
    parser.add_argument("split", choices=("train", "valid", "test"))
    args = parser.parse_args()

    model = YOLO(args.model)
    data = str((args.dataset / "data.yaml").resolve())
    name = f"{args.model.stem}_{args.split}"
    split = "val" if args.split == "valid" else args.split
    map_view(model, data, split, name)
    deploy_view(model, data, split, name)
    count_view(model, args.dataset, args.split)
    print(f"\nplots: {EVAL_DIR / name}")


if __name__ == "__main__":
    main()
