"""Waste detector abstraction.

Virtual camera frames go into WasteDetector.detect(). Swap MockDetector for
your trained model without changing navigation or the arm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    extra: dict | None = None


class WasteDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """image: RGB HxWx3 uint8 -> list of detections."""


# Distinct HSV windows for the colored primitive waste objects.
_HSV_RANGES = {
    "plastic_bottle": [((35, 80, 60), (85, 255, 255))],
    "can": [((12, 90, 80), (32, 255, 255))],
    "plastic_bag": [((135, 70, 70), (175, 255, 255))],
    "cardboard": [((8, 80, 60), (22, 255, 220))],
    "food_container": [((95, 80, 50), (130, 255, 255))],
}


class MockColorDetector(WasteDetector):
    """Placeholder detector that runs on virtual-camera RGB (not ground-truth IDs)."""

    def __init__(self, confidence_threshold: float = 0.5, force_low_confidence: bool = False):
        self.threshold = confidence_threshold
        self.force_low_confidence = force_low_confidence

    def detect(self, image: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, w = image.shape[:2]
        found: list[Detection] = []
        for name, ranges in _HSV_RANGES.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            for lo, hi in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
            mask = cv2.medianBlur(mask, 5)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 80:
                    continue
                if area > 0.08 * h * w:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                cy = y + bh / 2.0
                if cy < 0.28 * h:
                    continue
                if bw > 0.42 * w or bh > 0.55 * h:
                    continue
                aspect = bw / max(bh, 1)
                if aspect > 3.2 or aspect < 0.18:
                    continue
                fill = area / max(bw * bh, 1)
                if fill < 0.28:
                    continue
                size_score = min(1.0, area / 2500.0)
                conf = float(np.clip(0.45 + 0.4 * fill + 0.2 * size_score, 0.0, 0.99))
                if self.force_low_confidence:
                    conf = min(conf, 0.35)
                found.append(Detection(name, conf, (x, y, x + bw, y + bh)))
        found = _nms(found, 0.45)
        found.sort(key=lambda d: d.confidence, reverse=True)
        return found


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(dets, key=lambda d: d.confidence, reverse=True):
        if all(_iou(det, k) < iou_thr for k in kept):
            kept.append(det)
    return kept


class OracleDetector(WasteDetector):
    """Debug-only: project known waste bodies into the image (not for the demo pipeline)."""

    def __init__(self, client: int, waste_items, camera, threshold: float = 0.5):
        self.client = client
        self.waste_items = waste_items
        self.camera = camera
        self.threshold = threshold

    def detect(self, image: np.ndarray) -> list[Detection]:
        # Uses segmentation mask object ids when available.
        return MockColorDetector(self.threshold).detect(image)


class ExistingModelDetector(WasteDetector):
    """Loads the team's model.

    Supported options in config:
      - detector.backend: existing
      - detector.model_path: path to .pt (Ultralytics YOLO) OR
      - detector.existing_model_module: 'package.module:ClassName' with a detect(image) method
    """

    def __init__(self, cfg: dict):
        self.threshold = float(cfg.get("confidence_threshold", 0.5))
        self.class_names = list(cfg.get("class_names", []))
        self._impl = self._load(cfg)

    def _load(self, cfg: dict) -> Any:
        module_spec = cfg.get("existing_model_module")
        if module_spec:
            mod_name, _, cls_name = module_spec.partition(":")
            mod = import_module(mod_name)
            obj = getattr(mod, cls_name)() if cls_name else mod
            if not hasattr(obj, "detect"):
                raise TypeError(f"{module_spec} must provide detect(image)")
            return obj
        path = Path(cfg.get("model_path", "models/waste_detector.pt"))
        if not path.exists():
            raise FileNotFoundError(
                f"Existing model not found at {path}. Keep backend: mock until you copy weights here, "
                "or set detector.existing_model_module."
            )
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install ultralytics to use a YOLO .pt waste model") from exc
        return YOLO(str(path))

    def detect(self, image: np.ndarray) -> list[Detection]:
        if hasattr(self._impl, "detect") and self._impl.__class__.__name__ != "YOLO":
            raw = self._impl.detect(image)
            return [self._coerce(x) for x in raw]
        # Ultralytics YOLO
        bgr = image[:, :, ::-1]
        results = self._impl.predict(bgr, verbose=False)
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = r.names.get(cls_id, str(cls_id))
                if self.class_names and name not in self.class_names:
                    # map numeric class ids if model uses the same order
                    if 0 <= cls_id < len(self.class_names):
                        name = self.class_names[cls_id]
                out.append(Detection(name, conf, tuple(int(v) for v in xyxy)))
        return out

    def _coerce(self, item: Any) -> Detection:
        if isinstance(item, Detection):
            return item
        if isinstance(item, dict):
            return Detection(
                class_name=item.get("class_name") or item.get("label") or "waste",
                confidence=float(item.get("confidence", item.get("score", 0))),
                bbox=tuple(item["bbox"]),
            )
        raise TypeError(f"Unsupported detection type: {type(item)}")


def build_detector(cfg: dict, *, force_low_confidence: bool = False) -> WasteDetector:
    backend = (cfg.get("backend") or "mock").lower()
    thr = float(cfg.get("confidence_threshold", 0.5))
    if backend == "mock":
        return MockColorDetector(thr, force_low_confidence=force_low_confidence)
    if backend == "gemini":
        from waste_robot.gemini_detector import GeminiWasteDetector

        return GeminiWasteDetector(cfg, force_low_confidence=force_low_confidence)
    if backend == "existing":
        return ExistingModelDetector(cfg)
    if backend == "oracle":
        return MockColorDetector(thr)
    raise ValueError(f"Unknown detector backend: {backend}")
