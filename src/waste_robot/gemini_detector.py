"""Gemini vision detector on virtual-camera snapshots (not a webcam).

Pipeline:
  virtual RGB frame
      -> cheap mock blob trigger (an 'object' appeared)
      -> JPEG snapshot of THAT camera frame
      -> Gemini: trash or not, with 0-1000 boxes
      -> only confirmed trash detections drive navigation/pickup
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

import cv2
import numpy as np

from waste_robot.detector import Detection, MockColorDetector, WasteDetector

GEMINI_PROMPT = """
You are a trash-detection system for a robot arm. Look at this image and
find ALL litter/trash items visible (e.g. plastic bottles, cans, wrappers,
paper, plastic bags, cups, cardboard, food containers). Ignore anything that
is not trash (rocks, plants, benches, buildings, grass, the robot itself).
List every piece of trash you can see, even small or partially visible ones.

Respond with ONLY valid JSON, no other text, no markdown code fences, in
exactly this format:
{"objects": [{"label": "plastic bottle", "box_2d": [ymin, xmin, ymax, xmax]}]}

box_2d values must be integers from 0 to 1000, representing the bounding
box position as a fraction of image height/width (not pixels).
If there is no trash visible, respond with {"objects": []}
"""

_LABEL_MAP = {
    "plastic bottle": "plastic_bottle",
    "bottle": "plastic_bottle",
    "water bottle": "plastic_bottle",
    "can": "can",
    "soda can": "can",
    "tin can": "can",
    "aluminium can": "can",
    "aluminum can": "can",
    "plastic bag": "plastic_bag",
    "bag": "plastic_bag",
    "wrapper": "plastic_bag",
    "cardboard": "cardboard",
    "paper": "cardboard",
    "box": "cardboard",
    "food container": "food_container",
    "container": "food_container",
    "cup": "food_container",
    "styrofoam": "food_container",
}

_NOT_TRASH = {
    "rock",
    "stone",
    "plant",
    "tree",
    "grass",
    "bench",
    "building",
    "robot",
    "wood",
    "flower",
    "person",
    "ground",
    "path",
    "bin",
}


def normalize_label(label: str) -> str | None:
    raw = (label or "waste").strip().lower()
    if raw in _NOT_TRASH:
        return None
    if raw in _LABEL_MAP:
        return _LABEL_MAP[raw]
    for key, mapped in _LABEL_MAP.items():
        if key in raw:
            return mapped
    slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_") or "waste"
    if slug in _NOT_TRASH:
        return None
    return slug


def parse_gemini_objects(text: str, width: int, height: int) -> list[Detection]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    data = json.loads(cleaned)
    objects = data.get("objects", []) if isinstance(data, dict) else []
    out: list[Detection] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        class_name = normalize_label(str(obj.get("label", "waste")))
        if class_name is None:
            continue
        box = obj.get("box_2d")
        if not box or len(box) != 4:
            continue
        ymin, xmin, ymax, xmax = [float(v) for v in box]
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        x1 = int(np.clip(xmin / 1000.0 * width, 0, width - 1))
        y1 = int(np.clip(ymin / 1000.0 * height, 0, height - 1))
        x2 = int(np.clip(xmax / 1000.0 * width, 1, width))
        y2 = int(np.clip(ymax / 1000.0 * height, 1, height))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        out.append(
            Detection(
                class_name,
                0.92,
                (x1, y1, x2, y2),
                extra={"source": "gemini", "label": obj.get("label")},
            )
        )
    return out


class GeminiWasteDetector(WasteDetector):
    def __init__(self, cfg: dict, *, force_low_confidence: bool = False, client: Any | None = None):
        self.threshold = float(cfg.get("confidence_threshold", 0.5))
        self.force_low_confidence = force_low_confidence
        self.scan_interval = float(cfg.get("gemini_scan_interval", 4.0))
        self.jpeg_quality = int(cfg.get("gemini_jpeg_quality", 80))
        self.model_name = str(cfg.get("gemini_model", "gemini-3.6-flash"))
        self._fallback_models = [
            self.model_name,
            "gemini-3.6-flash",
            "gemini-1.5-flash",
            "gemini-flash-latest",
        ]
        # keep unique order
        seen: set[str] = set()
        models: list[str] = []
        for name in self._fallback_models:
            if name not in seen:
                seen.add(name)
                models.append(name)
        self._fallback_models = models
        self.trigger = MockColorDetector(self.threshold, force_low_confidence=force_low_confidence)
        self.last_status = "Gemini: idle"
        self._lock = threading.Lock()
        self._last_dets: list[Detection] = []
        self._last_t = 0.0
        self._pending = False
        self._client = client
        if force_low_confidence:
            return
        if self._client is None:
            self._client = self._make_client()

    def _make_client(self) -> Any:
        from waste_robot.config import _load_dotenv

        _load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in a .env file at the project root:\n"
                "  GEMINI_API_KEY=your-key-here\n"
                "or set it in PowerShell:  $env:GEMINI_API_KEY = \"your-key-here\""
            )
        from google import genai

        return genai.Client(api_key=api_key)

    def invalidate(self) -> None:
        with self._lock:
            self._last_dets = []
            self._last_t = 0.0

    def detect(self, image: np.ndarray) -> list[Detection]:
        if self.force_low_confidence:
            return self.trigger.detect(image)

        now = time.time()
        with self._lock:
            cached = list(self._last_dets)
            pending = self._pending
            last_t = self._last_t

        due = last_t <= 0.0 or (now - last_t) >= self.scan_interval
        if due and not pending:
            snapshot = np.ascontiguousarray(image.copy())
            with self._lock:
                self._pending = True
                self.last_status = "Gemini: virtual-camera snapshot sent"
            worker = threading.Thread(target=self._query, args=(snapshot,), daemon=True)
            worker.start()
            return cached

        if pending:
            self.last_status = "Gemini: virtual-camera snapshot sent"
        elif not cached:
            self.last_status = "Gemini: not trash (ignored)"
        else:
            self.last_status = f"Gemini: trash confirmed ({len(cached)})"
        return cached

    def _query(self, rgb: np.ndarray) -> None:
        dets: list[Detection] = []
        status = "Gemini: not trash (ignored)"
        try:
            dets = self._call_gemini(rgb)
            status = (
                f"Gemini: trash confirmed ({len(dets)})" if dets else "Gemini: not trash (ignored)"
            )
            print(status)
            for d in dets:
                print(f"  gemini box={d.bbox} class={d.class_name}")
        except Exception as exc:
            status = f"Gemini call failed: {exc}"
            print(status)
            dets = []
        with self._lock:
            self._last_dets = dets
            self._last_t = time.time()
            self._pending = False
            self.last_status = status

    def _call_gemini(self, rgb: np.ndarray) -> list[Detection]:
        from google.genai import types

        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return []
        last_error: Exception | None = None
        config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        for model_name in self._fallback_models:
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=[
                        GEMINI_PROMPT,
                        types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"),
                    ],
                    config=config,
                )
                self.model_name = model_name
                text = (response.text or "").strip()
                return parse_gemini_objects(text, w, h)
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                if "not found" in msg or "not supported" in msg or "invalid" in msg:
                    continue
                raise
        raise last_error or RuntimeError("Gemini call failed")
