"""4-wheel skid-steer / differential-drive base (planar x, y, yaw)."""

from __future__ import annotations

import numpy as np

from waste_robot.engine import Engine


class MobileBase:
    def __init__(self, engine: Engine, cfg: dict):
        self.engine = engine
        self.radius = float(cfg["wheel_radius"])
        self.track = float(cfg["track_width"])
        self.vmax = float(cfg["max_linear_speed"])
        self.wmax = float(cfg["max_angular_speed"])
        self._last_pose = self.pose()
        self.distance_travelled = 0.0
        self.wheels = {"wheel_fl_joint": 0, "wheel_fr_joint": 1, "wheel_rl_joint": 2, "wheel_rr_joint": 3}

    def pose(self) -> tuple[np.ndarray, float]:
        return self.engine.base_pose()

    def set_cmd_vel(self, v: float, w: float) -> None:
        v = float(np.clip(v, -self.vmax, self.vmax))
        w = float(np.clip(w, -self.wmax, self.wmax))
        self.engine.set_base_cmd(v, w)

    def stop(self) -> None:
        self.set_cmd_vel(0.0, 0.0)

    def update_odometry(self) -> None:
        pos, yaw = self.pose()
        self.distance_travelled += float(np.linalg.norm(pos[:2] - self._last_pose[0][:2]))
        self._last_pose = (pos, yaw)

    def lidar(self, n_rays: int, max_range: float) -> np.ndarray:
        return self.engine.lidar(n_rays, max_range)
