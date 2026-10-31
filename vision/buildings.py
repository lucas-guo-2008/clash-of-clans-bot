"""Finding loot buildings on the scout screen with the Phase 4 YOLO detector."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "buildings_v2.pt"
FRAME_SIZE = (1080, 1920)  # (h, w)

IMGSZ = 1280
DEVICE = "mps"
CONF = 0.25
SINGLETONS = {"town_hall", "clan_castle", "dark_elixir_storage"}


@dataclass(frozen=True)
class Building:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float


def load_detector(path: Path = MODEL_PATH) -> YOLO:
    if not path.exists():
        raise FileNotFoundError(f"no detector weights at {path}; train them with tools/train.py")
    model = YOLO(path)
    detect_buildings(np.zeros((*FRAME_SIZE, 3), np.uint8), model)
    return model


def detect_buildings(frame: np.ndarray, model: YOLO) -> list[Building]:
    """Every loot building on a BGR scout frame, most confident first."""

    if frame.shape[:2] != FRAME_SIZE:
        raise ValueError(
            f"expected a {FRAME_SIZE[1]}x{FRAME_SIZE[0]} frame, got {frame.shape[1]}x{frame.shape[0]}; "
            "the detector was trained on device pixels"
        )

    boxes = model.predict(frame, imgsz=IMGSZ, device=DEVICE, conf=CONF, verbose=False)[0].boxes
    found = sorted(zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()), key=lambda b: -b[1])
    buildings, seen = [], set()
    for cls, conf, (x0, y0, x1, y1) in found:
        name = model.names[int(cls)]
        if name in SINGLETONS:
            if name in seen:
                continue
            seen.add(name)
        buildings.append(Building(name, round(x0), round(y0), round(x1), round(y1), conf))
    return buildings
