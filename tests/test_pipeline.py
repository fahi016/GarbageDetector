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
from waste_robot.gemini_detector import parse_gemini_objects
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
    assert RobotState.SEARCHING.value == "SEARCHING"
    assert RobotState.GRASPING.value == "GRASPING"


def test_gemini_json_boxes_to_pixels():
    text = '{"objects": [{"label": "plastic bottle", "box_2d": [200, 100, 400, 300]}]}'
    dets = parse_gemini_objects(text, 320, 240)
    assert len(dets) == 1
    assert dets[0].class_name == "plastic_bottle"
    x1, y1, x2, y2 = dets[0].bbox
    assert (x1, y1, x2, y2) == (32, 48, 96, 96)


def test_gemini_ignores_non_trash_labels():
    text = '{"objects": [{"label": "rock", "box_2d": [100, 100, 200, 200]}]}'
    assert parse_gemini_objects(text, 320, 240) == []


if __name__ == "__main__":
    test_mock_detector_finds_green_blob()
    test_low_confidence_filtered_by_threshold()
    test_bus_pubsub()
    test_state_names()
    test_gemini_json_boxes_to_pixels()
    test_gemini_ignores_non_trash_labels()
    print("All unit tests passed.")
