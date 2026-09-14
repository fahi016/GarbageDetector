"""Realistic multi-part waste objects (bottles, cans, bags, cardboard, containers).

Each waste type is built from several primitive MuJoCo geoms (a "part list")
instead of one bare cuboid/cylinder, so it actually reads as that object from
the virtual camera. Every geom still doubles as real collision geometry, so
the gripper interacts with genuine per-part shapes for free.

Per-instance variety (size, rotation, position, and accent colors) is driven
by a seeded RNG so scenarios stay reproducible. The *dominant* ("base") color
of each type is kept inside that type's existing HSV detection window in
`waste_robot/detector.py` so the color-based mock detector keeps working
unmodified; only the small accent parts (caps, labels, rims, lids, tape) get
freely randomized colors for visual realism.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np


# hue_range is normalized 0..1 (matches colorsys), chosen to land inside the
# corresponding OpenCV 0..180 HSV window in detector.py:_HSV_RANGES so the
# mock color detector keeps firing on the "base" parts of each type.
# Each part: kind, size (cylinder: r,hh | box: hx,hy,hz | sphere: r),
# pos (x,y,z from ground), euler (optional, radians), role (base/accent1/
# accent2), mass.
WASTE_SPECS = {
    "plastic_bottle": {
        "hue_range": (0.22, 0.44),
        "parts": [
            {"role": "base", "kind": "cylinder", "size": (0.032, 0.065), "pos": (0, 0, 0.085), "mass": 0.030},
            {"role": "base", "kind": "cylinder", "size": (0.024, 0.014), "pos": (0, 0, 0.164), "mass": 0.006},
            {"role": "accent2", "kind": "cylinder", "size": (0.033, 0.020), "pos": (0, 0, 0.095), "mass": 0.004},
            {"role": "accent1", "kind": "cylinder", "size": (0.011, 0.016), "pos": (0, 0, 0.194), "mass": 0.004},
            {"role": "accent1", "kind": "cylinder", "size": (0.014, 0.010), "pos": (0, 0, 0.220), "mass": 0.003},
        ],
    },
    "can": {
        "hue_range": (0.075, 0.165),
        "parts": [
            {"role": "base", "kind": "cylinder", "size": (0.033, 0.050), "pos": (0, 0, 0.070), "mass": 0.028},
            {"role": "accent2", "kind": "cylinder", "size": (0.034, 0.022), "pos": (0, 0, 0.070), "mass": 0.006},
            {"role": "accent1", "kind": "cylinder", "size": (0.036, 0.006), "pos": (0, 0, 0.126), "mass": 0.004},
            {"role": "accent1", "kind": "box", "size": (0.006, 0.010, 0.004), "pos": (0, 0, 0.136), "mass": 0.001},
        ],
    },
    "plastic_bag": {
        "hue_range": (0.76, 0.96),
        "parts": [
            {"role": "base", "kind": "box", "size": (0.045, 0.035, 0.020), "pos": (0.00, 0.00, 0.040), "mass": 0.008},
            {"role": "base", "kind": "box", "size": (0.030, 0.040, 0.018), "pos": (0.028, -0.016, 0.048), "euler": (0, 0, 0.6), "mass": 0.006},
            {"role": "base", "kind": "box", "size": (0.035, 0.022, 0.016), "pos": (-0.022, 0.020, 0.036), "euler": (0, 0, -0.4), "mass": 0.004},
            {"role": "base", "kind": "box", "size": (0.020, 0.028, 0.014), "pos": (0.010, 0.032, 0.034), "euler": (0.3, 0, 1.0), "mass": 0.002},
        ],
    },
    "cardboard": {
        "hue_range": (0.03, 0.12),
        "parts": [
            {"role": "base", "kind": "box", "size": (0.075, 0.055, 0.018), "pos": (0, 0, 0.038), "mass": 0.042},
            {"role": "base", "kind": "box", "size": (0.075, 0.030, 0.006), "pos": (0, 0.075, 0.075), "euler": (0.95, 0, 0), "mass": 0.006},
            {"role": "base", "kind": "box", "size": (0.075, 0.030, 0.006), "pos": (0, -0.075, 0.075), "euler": (-0.95, 0, 0), "mass": 0.006},
            {"role": "accent2", "kind": "box", "size": (0.010, 0.058, 0.003), "pos": (0, 0, 0.058), "mass": 0.003},
        ],
    },
    "food_container": {
        "hue_range": (0.52, 0.70),
        "parts": [
            {"role": "base", "kind": "box", "size": (0.060, 0.045, 0.025), "pos": (0, 0, 0.045), "mass": 0.035},
            {"role": "accent1", "kind": "box", "size": (0.062, 0.047, 0.006), "pos": (0, 0.038, 0.078), "euler": (0.5, 0, 0), "mass": 0.010},
            {"role": "accent2", "kind": "box", "size": (0.025, 0.018, 0.004), "pos": (0, 0, 0.074), "mass": 0.005},
        ],
    },
}

# Accent palette shared across types: a few plausible cap/label/lid/tape
# colors, deliberately outside every WASTE_SPECS hue window so they read as
# distinct trim rather than blending into the base part.
ACCENT_PALETTE = [
    (0.95, 0.95, 0.95, 1.0),   # white
    (0.08, 0.08, 0.09, 1.0),   # black
    (0.85, 0.10, 0.10, 1.0),   # red
    (0.75, 0.76, 0.78, 1.0),   # silver
    (0.10, 0.20, 0.65, 1.0),   # deep blue
    (0.95, 0.80, 0.15, 1.0),   # yellow
]


def base_color(rng: np.random.Generator, waste_type: str) -> tuple[float, float, float, float]:
    lo, hi = WASTE_SPECS[waste_type]["hue_range"]
    hue = float(rng.uniform(lo, hi))
    sat = float(rng.uniform(0.55, 0.95))
    val = float(rng.uniform(0.55, 0.95))
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (r, g, b, 1.0)


def accent_color(rng: np.random.Generator) -> tuple[float, float, float, float]:
    idx = int(rng.integers(0, len(ACCENT_PALETTE)))
    return ACCENT_PALETTE[idx]


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
