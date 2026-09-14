"""2D detection -> camera frame -> world coordinates using virtual depth."""

from __future__ import annotations

import numpy as np

from waste_robot.camera import CameraFrame
from waste_robot.detector import Detection


def bbox_center(det: Detection) -> tuple[int, int]:
    x1, y1, x2, y2 = det.bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def median_depth(depth: np.ndarray, u: int, v: int, win: int = 4) -> float:
    h, w = depth.shape
    x0, x1 = max(0, u - win), min(w, u + win + 1)
    y0, y1 = max(0, v - win), min(h, v + win + 1)
    patch = depth[y0:y1, x0:x1]
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def detection_to_world(frame: CameraFrame, det: Detection) -> dict | None:
    """Deproject bbox center with linearized depth, then invert the view matrix."""
    u, v = bbox_center(det)
    if not (0 <= u < frame.width and 0 <= v < frame.height):
        return None
    z = median_depth(frame.depth, u, v)
    if not np.isfinite(z) or z < frame.near or z > frame.far * 0.98:
        return None
    x = (u - frame.width / 2.0) * z / ((frame.width / 2.0) / np.tan(np.deg2rad(frame.fov_deg) / 2.0))
    y = (v - frame.height / 2.0) * z / ((frame.height / 2.0) / np.tan(np.deg2rad(frame.fov_deg) / 2.0))
    # OpenGL camera: x right, y up, z backward (points have negative z in eye space).
    # PyBullet linearized depth is positive distance along camera forward, matching eye -Z.
    eye = np.array([x, -y, -z, 1.0], dtype=np.float64)
    try:
        inv_view = np.linalg.inv(frame.view)
    except np.linalg.LinAlgError:
        return None
    world_h = inv_view @ eye
    world = world_h[:3] / world_h[3]
    return {
        "class_name": det.class_name,
        "confidence": det.confidence,
        "bbox": det.bbox,
        "pixel": (u, v),
        "camera_xyz": (float(x), float(y), float(z)),
        "world_xyz": (float(world[0]), float(world[1]), float(world[2])),
    }


def raycast_fallback(engine, cam_pos, cam_target, max_dist=8.0) -> np.ndarray | None:
    direction = np.asarray(cam_target, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    n = np.linalg.norm(direction)
    if n < 1e-9:
        return None
    hits = engine.lidar(8, max_dist)
    d = float(np.min(hits))
    if d >= max_dist * 0.99:
        return None
    return np.asarray(cam_pos, dtype=np.float64) + direction / n * d


def pixel_world_direction(frame: CameraFrame, u: int, v: int) -> np.ndarray:
    """World-space unit ray direction from the camera through pixel (u, v)."""
    fx = (frame.width / 2.0) / np.tan(np.deg2rad(frame.fov_deg) / 2.0)
    fy = fx
    ex = (u - frame.width / 2.0) / fx
    ey = (v - frame.height / 2.0) / fy
    eye_dir = np.array([ex, -ey, -1.0, 0.0], dtype=np.float64)  # w=0: direction, not a point
    inv_view = np.linalg.inv(frame.view)
    world_dir = (inv_view @ eye_dir)[:3]
    n = np.linalg.norm(world_dir)
    return world_dir / n if n > 1e-9 else world_dir


def raycast_detection_target(engine, frame: CameraFrame, det: Detection, max_dist: float | None = None) -> dict | None:
    """Cast a single simulator ray through the detection bbox center.

    This is the "simulator-native mechanism" localization step: instead of
    trusting a ground-truth object position, fire a ray along the camera's
    real viewing direction for that pixel and see what it actually hits.
    Accepts only `waste_*` bodies (real trash geometry); a `decoy_*` hit or a
    miss means "reject" so callers can fall back to the depth estimate or
    drop the detection.
    """
    u, v = bbox_center(det)
    if not (0 <= u < frame.width and 0 <= v < frame.height):
        return None
    direction = pixel_world_direction(frame, u, v)
    reach = float(max_dist if max_dist is not None else frame.far)
    body_id, dist = engine.raycast(frame.camera_pos, direction, max_dist=reach)
    if body_id is None or dist <= 0:
        return None
    name = str(engine.model.body(body_id).name)
    if not name.startswith("waste_"):
        return None
    world_xyz = engine.data.xpos[body_id].copy()
    return {
        "body_id": body_id,
        "body_name": name,
        "distance": dist,
        "world_xyz": (float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])),
    }
