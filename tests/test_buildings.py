"""The building detector, run over v2's test split: frames it never trained on, labeled by hand."""

from collections import Counter
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from conftest import ROOT
from vision.buildings import MODEL_PATH, SINGLETONS, detect_buildings, load_detector

HELD_OUT = ROOT / "datasets" / "buildings_v2" / "test"
STEMS = sorted(p.stem for p in (HELD_OUT / "images").glob("*.png"))


@pytest.fixture(scope="session")
def detector():
    if not MODEL_PATH.exists():
        pytest.skip(f"{MODEL_PATH.name} is not on this machine")
    return load_detector()


def labeled_counts(stem: str, names: dict[int, str]) -> Counter:
    lines = (HELD_OUT / "labels" / f"{stem}.txt").read_text().splitlines()
    return Counter(names[int(line.split()[0])] for line in lines if line.strip())


def within_tolerance(true: Counter, pred: Counter) -> bool:
    """One frame's labeled vs detected count per class -> whether the detector passes.

    Singletons exact; any other class off by at most one; at most two off in all, so one
    misclassification (one class +1, another -1) passes and a second does not.
    """
    errors = {name: abs(pred[name] - true[name]) for name in true.keys() | pred.keys()}
    return (all(errors.get(name, 0) == 0 for name in SINGLETONS)
            and max(errors.values(), default=0) <= 1
            and sum(errors.values()) <= 2)


@pytest.mark.parametrize("stem", STEMS)
def test_counts_match_labels(detector, stem):
    frame = cv2.imread(str(HELD_OUT / "images" / f"{stem}.png"))
    pred = Counter(b.name for b in detect_buildings(frame, detector))
    true = labeled_counts(stem, detector.names)
    assert within_tolerance(true, pred), f"labeled {dict(true)}, detected {dict(pred)}"
    assert all(pred[name] <= 1 for name in SINGLETONS)


class FakeDetector:
    """Stands in for the YOLO model: predict returns the given boxes as Ultralytics does."""

    names = {0: "clan_castle", 1: "gold_mine"}

    def __init__(self, boxes):
        cls, conf, xyxy = zip(*boxes)
        self.boxes = SimpleNamespace(cls=np.array(cls), conf=np.array(conf), xyxy=np.array(xyxy))

    def predict(self, frame, **_):
        return [SimpleNamespace(boxes=self.boxes)]


def test_doubled_singleton_keeps_only_the_most_confident():
    model = FakeDetector([(0, 0.4, (0, 0, 9, 9)), (1, 0.8, (5, 5, 9, 9)), (0, 0.9, (10, 10, 20, 20)),
                          (1, 0.7, (30, 30, 40, 40))])
    found = detect_buildings(np.zeros((1080, 1920, 3), np.uint8), model)
    assert [(b.name, b.conf) for b in found] == [("clan_castle", 0.9), ("gold_mine", 0.8), ("gold_mine", 0.7)]


def test_resized_frame_is_refused(frame, detector):
    with pytest.raises(ValueError):
        detect_buildings(cv2.resize(frame(24), (960, 540)), detector)
