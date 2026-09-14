"""Unit tests that do not require a display."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from waste_robot.bus import Bus
from waste_robot.detector import MockColorDetector
from waste_robot.gemini_detector import parse_gemini_response
from waste_robot.state_machine import RobotState


def test_mock_detector_finds_green_blob():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[80:140, 120:170] = (40, 200, 70)  # plastic bottle green
    # keep blob small so background-size rejection does not fire
    dets = MockColorDetector(0.3).detect(img)
    assert any(d.class_name == "plastic_bottle" for d in dets), dets


def test_low_confidence_filtered_by_threshold():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[80:140, 120:170] = (40, 200, 70)
    dets = MockColorDetector(0.5, force_low_confidence=True).detect(img)
    assert all(d.confidence < 0.5 for d in dets)


def test_bus_pubsub():
    bus = Bus()
    got = []
    bus.subscribe("detections", got.append)
    bus.publish("detections", ["x"])
    assert got == [["x"]]


def test_state_names():
    assert RobotState.PATROLLING.value == "PATROLLING"
    assert RobotState.GRABBING.value == "GRABBING"
    assert RobotState.GEMINI_ANALYSIS.value == "GEMINI_ANALYSIS"


def test_gemini_json_fractional_bbox():
    text = (
        '{"garbage_detected": true, "objects": [{"garbage_type": "plastic bottle", '
        '"confidence": 0.9, "bounding_box": {"x": 0.3125, "y": 0.4167, '
        '"width": 0.3125, "height": 0.4167}, "description": "a bottle"}]}'
    )
    dets = parse_gemini_response(text, 320, 240)
    assert len(dets) == 1
    assert dets[0].class_name == "plastic_bottle"
    x1, y1, x2, y2 = dets[0].bbox
    assert (x1, y1, x2, y2) == (100, 100, 200, 200)


def test_gemini_no_garbage_detected():
    text = '{"garbage_detected": false, "garbage_type": null, "confidence": 0, "bounding_box": null, "description": "No garbage detected"}'
    assert parse_gemini_response(text, 320, 240) == []


def test_gemini_ignores_non_trash_labels():
    text = '{"garbage_detected": true, "objects": [{"garbage_type": "rock", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}}]}'
    assert parse_gemini_response(text, 320, 240) == []


def test_gemini_legacy_box2d_still_parses():
    text = '{"objects": [{"label": "can", "box_2d": [200, 100, 400, 300]}]}'
    dets = parse_gemini_response(text, 320, 240)
    assert len(dets) == 1
    assert dets[0].class_name == "can"


if __name__ == "__main__":
    test_mock_detector_finds_green_blob()
    test_low_confidence_filtered_by_threshold()
    test_bus_pubsub()
    test_state_names()
    test_gemini_json_fractional_bbox()
    test_gemini_no_garbage_detected()
    test_gemini_ignores_non_trash_labels()
    test_gemini_legacy_box2d_still_parses()
    print("All unit tests passed.")
