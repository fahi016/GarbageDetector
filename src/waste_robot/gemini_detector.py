"""Gemini vision detector on virtual-camera snapshots (not a webcam).

Two-tier pipeline (this is the point of the module, not a detail):
  1. cheap local trigger (`MockColorDetector`, already used standalone as the
     'mock' backend) runs on every patrol frame and answers one question:
     "does anything object-shaped and non-background-colored sit in view?"
  2. only when that trigger fires (and a cooldown has elapsed) do we JPEG-
     encode the frame and actually call the Gemini API to classify it and
     get a bounding box. This is what keeps API usage sane instead of
     calling Gemini every simulation frame.

`confirm()` / `poll()` expose that second tier explicitly so the mission
state machine can sit in CAPTURING_IMAGE / GEMINI_ANALYSIS states with
proper logging instead of it happening invisibly inside `detect()`.
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

# Prompt: tells Gemini exactly what this image is, what to ignore, and the
# exact structured JSON shape to answer in (see spec section 3).
GEMINI_PROMPT = """
You are the vision system on an autonomous garbage-collection robot working
in a simulated outdoor plaza. Your ONLY job is to find litter/trash that the
robot's arm should pick up in THIS camera frame.

Rules:
- Only report actual trash/litter: plastic bottles, cans, plastic bags,
  wrappers, cups, cardboard/paper, food containers, and similar debris.
- IGNORE the robot itself (chassis, arm, gripper, basket), the ground,
  grass, paths, walls, buildings, trees, benches, planters, rocks, and
  litter bins -- none of those are garbage to collect.
- Only report an item if it looks reachable/pickable by a small ground
  robot arm (on or near the ground nearby, not already inside a bin, not
  far away in the background).
- Respond with ONLY valid JSON. No markdown fences, no commentary.

If you see trash, respond in exactly this shape (one entry per item,
bounding_box values are fractions 0.0-1.0 of image width/height):
{"garbage_detected": true, "objects": [
  {"garbage_type": "plastic_bottle", "confidence": 0.92,
   "bounding_box": {"x": 0.45, "y": 0.52, "width": 0.12, "height": 0.25},
   "description": "Plastic bottle lying on the ground"}
]}

If there is no trash visible, respond with exactly:
{"garbage_detected": false, "garbage_type": null, "confidence": 0,
 "bounding_box": null, "description": "No garbage detected"}
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
    "rock", "stone", "plant", "tree", "grass", "bench", "building", "robot",
    "wood", "flower", "person", "ground", "path", "bin", "arm", "gripper",
    "basket", "none",
}


def normalize_label(label: str | None) -> str | None:
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


def _clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    ix1 = int(np.clip(x1, 0, width - 1))
    iy1 = int(np.clip(y1, 0, height - 1))
    ix2 = int(np.clip(x2, 1, width))
    iy2 = int(np.clip(y2, 1, height))
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return None
    return ix1, iy1, ix2, iy2


def _bbox_from_fractional(box: dict, width: int, height: int) -> tuple[int, int, int, int] | None:
    """{"x","y","width","height"} as fractions 0..1 of the image."""
    try:
        x = float(box.get("x", box.get("left", 0.0)))
        y = float(box.get("y", box.get("top", 0.0)))
        w = float(box.get("width", box.get("w", 0.0)))
        h = float(box.get("height", box.get("h", 0.0)))
    except (TypeError, ValueError):
        return None
    return _clip_box(x * width, y * height, (x + w) * width, (y + h) * height, width, height)


def _bbox_from_box2d(box: list, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Legacy [ymin, xmin, ymax, xmax] on a 0-1000 scale (kept for robustness)."""
    if len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    return _clip_box(xmin / 1000.0 * width, ymin / 1000.0 * height, xmax / 1000.0 * width, ymax / 1000.0 * height, width, height)


def _one_detection(obj: Any, width: int, height: int) -> Detection | None:
    if not isinstance(obj, dict):
        return None
    if obj.get("garbage_detected") is False:
        return None
    label = obj.get("garbage_type") or obj.get("label") or obj.get("type")
    class_name = normalize_label(label if isinstance(label, str) else None)
    if class_name is None:
        return None
    try:
        confidence = float(obj.get("confidence", 0.75) or 0.75)
    except (TypeError, ValueError):
        confidence = 0.75
    confidence = float(np.clip(confidence, 0.0, 1.0))
    box = obj.get("bounding_box", obj.get("bbox", obj.get("box_2d")))
    bbox = None
    if isinstance(box, dict):
        bbox = _bbox_from_fractional(box, width, height)
    elif isinstance(box, (list, tuple)):
        bbox = _bbox_from_box2d(list(box), width, height)
    if bbox is None:
        return None
    return Detection(
        class_name,
        confidence,
        bbox,
        extra={"source": "gemini", "label": label, "description": obj.get("description", "")},
    )


def parse_gemini_response(text: str, width: int, height: int) -> list[Detection]:
    """Robust parse of Gemini's JSON: handles the list-of-objects shape, a
    bare single-object shape, markdown fences, and minor key variations."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    if data.get("garbage_detected") is False:
        return []
    objects = data.get("objects")
    if objects is None:
        # Bare single-object schema (no "objects" wrapper).
        objects = [data] if any(k in data for k in ("garbage_type", "bounding_box", "bbox")) else []
    if isinstance(objects, dict):
        objects = [objects]
    if not isinstance(objects, list):
        return []
    out: list[Detection] = []
    for obj in objects:
        det = _one_detection(obj, width, height)
        if det is not None:
            out.append(det)
    return out


class GeminiWasteDetector(WasteDetector):
    def __init__(self, cfg: dict, *, force_low_confidence: bool = False, client: Any | None = None):
        self.threshold = float(cfg.get("confidence_threshold", 0.5))
        self.force_low_confidence = force_low_confidence
        self.scan_interval = float(cfg.get("gemini_scan_interval", 4.0))
        self.jpeg_quality = int(cfg.get("gemini_jpeg_quality", 80))
        self.model_name = str(cfg.get("gemini_model", "gemini-3.6-flash"))
        self._fallback_models = [self.model_name, "gemini-3.6-flash", "gemini-1.5-flash", "gemini-flash-latest"]
        seen: set[str] = set()
        models: list[str] = []
        for name in self._fallback_models:
            if name not in seen:
                seen.add(name)
                models.append(name)
        self._fallback_models = models
        # Tier 1: cheap local "something's there" trigger (also the whole
        # implementation of the 'mock' backend) -- gates tier 2 below.
        self.trigger = MockColorDetector(self.threshold, force_low_confidence=force_low_confidence)
        self.last_status = "Gemini: idle"
        self._lock = threading.Lock()
        self._last_dets: list[Detection] = []
        self._last_t = 0.0
        self._pending = False
        self._done_since_confirm = False
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

    # -- Tier 1: cheap, synchronous, safe to call every patrol frame --------

    def quick_scan(self, image: np.ndarray) -> list[Detection]:
        """Cheap local 'potential garbage' check -- does NOT call Gemini."""
        if self.force_low_confidence:
            return self.trigger.detect(image)
        return self.trigger.detect(image)

    # -- Tier 2: expensive, async, explicitly triggered ----------------------

    def confirm(self, image: np.ndarray) -> bool:
        """Start an async Gemini request if not already pending/cooling down.

        Returns True if a request was actually started (caller should move to
        GEMINI_ANALYSIS and poll()), False if skipped (still in cooldown or a
        request is already in flight).
        """
        if self.force_low_confidence:
            return False
        now = time.time()
        with self._lock:
            if self._pending:
                return False
            if self._last_t > 0.0 and (now - self._last_t) < self.scan_interval:
                return False
            self._pending = True
            self._done_since_confirm = False
        snapshot = np.ascontiguousarray(image.copy())
        print("[CAPTURING_IMAGE] Snapshot captured from virtual camera")
        print(f"[GEMINI_ANALYSIS] Gemini request started (model={self.model_name})")
        threading.Thread(target=self._query, args=(snapshot,), daemon=True).start()
        return True

    def poll(self) -> tuple[bool, list[Detection]]:
        """Non-blocking: (done, detections). done=False while still pending."""
        with self._lock:
            if self._pending:
                return False, []
            return True, list(self._last_dets)

    def _query(self, rgb: np.ndarray) -> None:
        dets: list[Detection] = []
        status = "Gemini: not trash (ignored)"
        try:
            dets = self._call_gemini(rgb)
            print("[GEMINI_ANALYSIS] Gemini response received")
            if dets:
                status = f"Gemini: trash confirmed ({len(dets)})"
                for d in dets:
                    print(f"  garbage_detected=true type={d.class_name} confidence={d.confidence:.2f} bbox={d.bbox}")
            else:
                status = "Gemini: garbage_detected=false"
                print("  garbage_detected=false -- no garbage in frame")
        except Exception as exc:
            status = f"Gemini call failed: {exc}"
            print(f"[GEMINI_ANALYSIS] {status}")
            dets = []
        with self._lock:
            self._last_dets = dets
            self._last_t = time.time()
            self._pending = False
            self.last_status = status

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Back-compat synchronous-looking API (dashboard/status display,
        and the `oracle`/tests path). Internally still just tier 1 + cached
        tier-2 results -- the mission loop drives confirm()/poll() directly."""
        if self.force_low_confidence:
            return self.trigger.detect(image)
        with self._lock:
            cached = list(self._last_dets)
            pending = self._pending
        if pending:
            self.last_status = "Gemini: analyzing snapshot"
        elif not cached:
            self.last_status = "Gemini: not trash (ignored)"
        else:
            self.last_status = f"Gemini: trash confirmed ({len(cached)})"
        return cached

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
                    contents=[GEMINI_PROMPT, types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg")],
                    config=config,
                )
                self.model_name = model_name
                text = (response.text or "").strip()
                return parse_gemini_response(text, w, h)
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                if "not found" in msg or "not supported" in msg or "invalid" in msg:
                    continue
                raise
        raise last_error or RuntimeError("Gemini call failed")
