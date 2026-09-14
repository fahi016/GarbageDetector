"""Original primitive waste objects (bottles, cans, bags, cardboard, containers)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WASTE_SPECS = {
    "plastic_bottle": {"rgba": (0.05, 0.95, 0.35, 1.0), "kind": "cylinder", "radius": 0.032, "height": 0.18, "mass": 0.05},
    "can": {"rgba": (0.92, 0.70, 0.08, 1.0), "kind": "cylinder", "radius": 0.033, "height": 0.11, "mass": 0.04},
    "plastic_bag": {"rgba": (0.88, 0.18, 0.72, 1.0), "kind": "box", "size": (0.12, 0.09, 0.04), "mass": 0.02},
    "cardboard": {"rgba": (0.72, 0.50, 0.22, 1.0), "kind": "box", "size": (0.16, 0.12, 0.03), "mass": 0.06},
    "food_container": {"rgba": (0.12, 0.42, 0.85, 1.0), "kind": "box", "size": (0.12, 0.09, 0.06), "mass": 0.05},
}


@dataclass
class WasteItem:
    name: str
    waste_type: str
    collected: bool = False
    body_id: int = -1


def default_layout(scenario: str, seed: int = 7) -> list[tuple[str, tuple[float, float], float]]:
    """Return (type, xy, yaw) layouts for test scenarios."""
    rng = np.random.default_rng(seed)
    types = list(WASTE_SPECS.keys())
    if scenario == "single":
        return [("plastic_bottle", (1.35, 0.05), 0.2)]
    if scenario == "obstacle":
        return [("can", (3.6, 2.6), 0.4)]
    if scenario == "types":
        return [
            ("plastic_bottle", (1.4, 0.3), 0.1),
            ("can", (1.6, -1.1), 0.5),
            ("plastic_bag", (0.4, 1.5), 1.0),
            ("cardboard", (-1.3, 1.2), 0.2),
        ]
    if scenario == "false_detection":
        return [("plastic_bottle", (1.5, 0.0), 0.0)]
    if scenario == "failed_pickup":
        return [("plastic_bag", (1.3, 0.15), 0.8)]
    spots = [
        (1.45, 0.10),
        (2.20, -1.40),
        (-1.50, 1.60),
        (0.80, 2.10),
        (-2.10, -1.30),
        (2.80, 1.50),
        (-0.60, -2.00),
        (1.10, -2.40),
        (3.10, -0.40),
        (-1.00, 0.90),
    ]
    n = 10
    out = []
    for i, (x, y) in enumerate(spots[:n]):
        out.append((types[i % len(types)], (x, y), float(rng.uniform(0, 3.14))))
    return out


DECOY_SPECS = {
    "rock": {"kind": "sphere", "radius": 0.08, "rgba": (0.50, 0.50, 0.52, 1.0)},
    "plant": {"kind": "cylinder", "radius": 0.045, "height": 0.16, "rgba": (0.08, 0.62, 0.22, 1.0)},
    "wood_block": {"kind": "box", "size": (0.10, 0.08, 0.05), "rgba": (0.55, 0.38, 0.16, 1.0)},
}


def decoy_layout(scenario: str) -> list[tuple[str, tuple[float, float], float]]:
    """Non-trash objects Gemini should refuse to pick."""
    if scenario not in ("multi",):
        return []
    return [
        ("rock", (0.55, -0.85), 0.1),
        ("plant", (1.85, 0.55), 0.0),
        ("wood_block", (-0.9, -0.7), 0.4),
    ]
