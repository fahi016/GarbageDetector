"""Load the combined mobile base + 6-DOF arm from the compiled MuJoCo model."""

from __future__ import annotations

import numpy as np

from waste_robot.engine import Engine


class RobotModel:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.arm_joints = [engine.joint_id(f"joint_{k}") for k in range(1, 7)]
        self.finger_left = engine.joint_id("finger_left_joint")
        self.finger_right = engine.joint_id("finger_right_joint")
        self.fingers = {"finger_left_joint": self.finger_left, "finger_right_joint": self.finger_right}
        self.ee_site = engine.ee_site
        self.camera_name = "robot_cam"
        self.wheel_joints = {}

    @property
    def robot_id(self) -> int:
        return self.engine.robot_body

    def basket_world_pose(self):
        pos, mat = self.engine.site_pose(self.engine.basket_site)
        return pos, mat
