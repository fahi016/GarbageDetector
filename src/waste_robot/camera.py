"""Virtual RGB-D camera mounted on the robot (not a physical webcam)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waste_robot.engine import Engine, look_at_view


@dataclass
class CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray
    view: np.ndarray
    projection: np.ndarray
    width: int
    height: int
    fov_deg: float
    near: float
    far: float
    camera_pos: np.ndarray
    camera_target: np.ndarray
    camera_up: np.ndarray


class VirtualCamera:
    def __init__(self, engine: Engine, cfg: dict):
        self.engine = engine
        self.width = int(cfg["width"])
        self.height = int(cfg["height"])
        self.fov = float(cfg["fov_deg"])
        self.near = float(cfg["near"])
        self.far = float(cfg["far"])
        self.fx = (self.width / 2.0) / np.tan(np.deg2rad(self.fov) / 2.0)
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def capture(self) -> CameraFrame:
        pos, x_axis, y_axis, z_axis = self.engine.camera_frame_axes("robot_cam")
        target = pos - z_axis
        view = look_at_view(pos, target, y_axis)
        rendered = self.engine.render_rgb_depth("robot_cam")
        if rendered is None:
            rgb, depth = self._fallback(pos, view)
        else:
            rgb, depth = rendered
            depth = np.where(np.isfinite(depth), depth, self.far)
            depth = np.clip(depth, self.near, self.far)
        seg = np.zeros((self.height, self.width), dtype=np.int32)
        proj = np.eye(4)
        return CameraFrame(
            rgb=rgb,
            depth=depth,
            segmentation=seg,
            view=view,
            projection=proj,
            width=self.width,
            height=self.height,
            fov_deg=self.fov,
            near=self.near,
            far=self.far,
            camera_pos=pos,
            camera_target=target,
            camera_up=y_axis,
        )

    def _fallback(self, cam_pos: np.ndarray, view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Software view if GPU/offscreen GL is unavailable."""
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rgb[:, :] = (70, 120, 70)
        depth = np.full((self.height, self.width), self.far, dtype=np.float32)
        inv_view = np.linalg.inv(view)
        # Project waste bodies as colored disks.
        for bid in range(1, self.engine.model.nbody):
            name = self.engine.model.body(bid).name
            if not str(name).startswith("waste_"):
                continue
            world = np.append(self.engine.data.xpos[bid], 1.0)
            eye = view @ world
            if eye[2] >= -self.near:
                continue
            z = -eye[2]
            u = int(self.cx + self.fx * eye[0] / z)
            v = int(self.cy - self.fy * eye[1] / z)
            if not (0 <= u < self.width and 0 <= v < self.height):
                continue
            rgba = None
            for g in range(self.engine.model.ngeom):
                if int(self.engine.model.geom_bodyid[g]) == bid:
                    rgba = self.engine.model.geom_rgba[g]
                    break
            color = (40, 200, 70) if rgba is None else tuple(int(255 * c) for c in rgba[:3])
            rr = max(4, int(18 * 1.2 / max(z, 0.3)))
            yy, xx = np.ogrid[-rr : rr + 1, -rr : rr + 1]
            mask = xx * xx + yy * yy <= rr * rr
            for dy, dx in zip(*np.where(mask)):
                py, px = v + dy - rr, u + dx - rr
                if 0 <= py < self.height and 0 <= px < self.width:
                    rgb[py, px] = color
                    depth[py, px] = min(depth[py, px], z)
        _ = inv_view
        _ = cam_pos
        return rgb, depth
