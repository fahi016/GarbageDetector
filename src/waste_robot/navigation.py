"""Go-to-goal navigation with A* when blocked and lidar obstacle avoidance."""

from __future__ import annotations

import math

import numpy as np

from simulation.worlds.park_plaza import Obstacle, astar_path
from waste_robot.mobile_base import MobileBase


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Navigator:
    def __init__(self, base: MobileBase, obstacles: list[Obstacle], cfg: dict):
        self.base = base
        self.obstacles = obstacles
        self.approach = float(cfg["approach_distance"])
        self.xy_tol = float(cfg["goal_xy_tolerance"])
        self.yaw_tol = math.radians(float(cfg["goal_yaw_tolerance_deg"]))
        self.inflate = float(cfg["obstacle_inflate"])
        self.path: list[tuple[float, float]] = []
        self.i = 0

    def approach_pose(self, waste_xy: tuple[float, float], robot_xy: tuple[float, float]) -> tuple[float, float, float]:
        dx = waste_xy[0] - robot_xy[0]
        dy = waste_xy[1] - robot_xy[1]
        yaw = math.atan2(dy, dx)
        ax = waste_xy[0] - math.cos(yaw) * self.approach
        ay = waste_xy[1] - math.sin(yaw) * self.approach
        return ax, ay, yaw

    def plan(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> None:
        self.path = astar_path(self.obstacles, start_xy, goal_xy, inflate=self.inflate)
        self.i = 0

    def step(self, goal_xy: tuple[float, float], goal_yaw: float | None, lidar: np.ndarray) -> tuple[float, float, bool]:
        """Returns (v, w, arrived)."""
        pos, yaw = self.base.pose()
        if not self.path:
            self.plan((pos[0], pos[1]), goal_xy)
        # advance along path
        while self.i < len(self.path) - 1:
            px, py = self.path[self.i]
            if math.hypot(px - pos[0], py - pos[1]) < 0.28:
                self.i += 1
            else:
                break
        tx, ty = self.path[min(self.i, len(self.path) - 1)]
        # final goal uses true goal
        if self.i >= len(self.path) - 1:
            tx, ty = goal_xy

        dist = math.hypot(goal_xy[0] - pos[0], goal_xy[1] - pos[1])
        heading = math.atan2(ty - pos[1], tx - pos[0])
        err_h = wrap_angle(heading - yaw)

        # lidar reactive avoidance (front sector)
        n = len(lidar)
        front = np.concatenate([lidar[n // 2 - n // 8 : n // 2 + n // 8 + 1]])
        min_front = float(np.min(front)) if front.size else 4.0
        avoid_w = 0.0
        if min_front < 0.55:
            left = float(np.mean(lidar[: n // 2]))
            right = float(np.mean(lidar[n // 2 :]))
            avoid_w = 1.1 if left > right else -1.1

        arrived_xy = dist < self.xy_tol
        if arrived_xy:
            if goal_yaw is None:
                self.base.stop()
                return 0.0, 0.0, True
            err_y = wrap_angle(goal_yaw - yaw)
            if abs(err_y) < self.yaw_tol:
                self.base.stop()
                return 0.0, 0.0, True
            w = np.clip(2.2 * err_y, -self.base.wmax, self.base.wmax)
            return 0.0, float(w), False

        v = self.base.vmax * (0.35 + 0.65 * max(0.0, math.cos(err_h)))
        if abs(err_h) > 0.7:
            v *= 0.15
        if min_front < 0.7:
            v *= 0.25
        w = np.clip(2.0 * err_h + avoid_w, -self.base.wmax, self.base.wmax)
        return float(v), float(w), False

    def search_cmd(self, lidar: np.ndarray, t: float) -> tuple[float, float]:
        n = len(lidar)
        min_front = float(np.min(lidar[n // 2 - 4 : n // 2 + 5]))
        if min_front < 0.6:
            return 0.0, 0.8
        # slow cruise + gentle scan
        return 0.22, 0.35 * math.sin(0.4 * t)
