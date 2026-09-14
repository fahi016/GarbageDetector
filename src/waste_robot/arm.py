"""6-DOF arm IK, gripper, and simplified grasp constraint."""

from __future__ import annotations

import numpy as np
import mujoco

from waste_robot.engine import Engine


class ArmController:
    def __init__(self, engine: Engine, robot, cfg: dict):
        self.engine = engine
        self.model = engine.model
        self.data = engine.data
        self.joints = robot.arm_joints
        self.fingers = robot.fingers
        self.ee_site = robot.ee_site
        self.cfg = cfg
        self.home = list(cfg["home_joints"])
        self.held_id: int | None = None
        self.lower = [float(self.model.jnt_range[j][0]) for j in self.joints]
        self.upper = [float(self.model.jnt_range[j][1]) for j in self.joints]
        self.act = [engine.actuator_id(f"act_joint_{k}") for k in range(1, 7)]
        self.act_fl = engine.actuator_id("act_finger_left_joint")
        self.act_fr = engine.actuator_id("act_finger_right_joint")
        self._hold_position(self.home)
        self.open_gripper()
        engine.forward()

    def _hold_position(self, q: list[float]) -> None:
        for j, qi, act in zip(self.joints, q, self.act):
            adr = int(self.model.jnt_qposadr[j])
            lo, hi = self.model.jnt_range[j]
            qi = float(np.clip(qi, lo, hi))
            self.data.qpos[adr] = qi
            self.data.ctrl[act] = qi

    def current_q(self) -> list[float]:
        return [float(self.data.qpos[int(self.model.jnt_qposadr[j])]) for j in self.joints]

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos, mat = self.engine.site_pose(self.ee_site)
        return pos, mat

    def _ik(self, xyz: np.ndarray) -> list[float] | None:
        target = np.asarray(xyz, dtype=np.float64)
        for _ in range(int(self.cfg["ik_max_iterations"])):
            mujoco.mj_forward(self.model, self.data)
            pos = self.data.site_xpos[self.ee_site]
            err = target - pos
            if float(np.linalg.norm(err)) < 0.02:
                return self.current_q()
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site)
            cols = [int(self.model.jnt_dofadr[j]) for j in self.joints]
            jmat = jacp[:, cols]
            lam = 1e-3
            try:
                dq = jmat.T @ np.linalg.solve(jmat @ jmat.T + lam * np.eye(3), 0.6 * err)
            except np.linalg.LinAlgError:
                return None
            dq = np.clip(dq, -0.2, 0.2)
            q = self.current_q()
            q = [float(np.clip(qi + dqi, lo, hi)) for qi, dqi, lo, hi in zip(q, dq, self.lower, self.upper)]
            self._hold_position(q)
        mujoco.mj_forward(self.model, self.data)
        if float(np.linalg.norm(target - self.data.site_xpos[self.ee_site])) < 0.07:
            return self.current_q()
        return None

    def move_ee(self, xyz, orn=None, timeout: float | None = None) -> bool:
        q_goal = self._ik(np.asarray(xyz, dtype=np.float64))
        if q_goal is None:
            return False
        start = np.array(self.current_q())
        goal = np.array(q_goal)
        steps = 48
        for k in range(steps):
            a = (k + 1) / steps
            self._hold_position((start * (1 - a) + goal * a).tolist())
            for _ in range(6):
                self.engine.step()
            pos, _ = self.ee_pose()
            if np.linalg.norm(pos - np.asarray(xyz)) < 0.04 and k > steps // 3:
                return True
        pos, _ = self.ee_pose()
        return bool(np.linalg.norm(pos - np.asarray(xyz)) < 0.12)

    def go_home(self, steps: int = 40) -> None:
        start = np.array(self.current_q())
        goal = np.array(self.home)
        for k in range(steps):
            a = (k + 1) / steps
            self._hold_position((start * (1 - a) + goal * a).tolist())
            self.engine.step()

    def open_gripper(self) -> None:
        gap = float(self.cfg["gripper_open"])
        self._set_fingers(gap)

    def close_gripper(self) -> None:
        gap = float(self.cfg["gripper_closed"])
        self._set_fingers(gap)

    def _set_fingers(self, opening: float) -> None:
        pos = float(np.clip(opening, 0.0, 0.045))
        self.data.ctrl[self.act_fl] = pos
        self.data.ctrl[self.act_fr] = pos
        self.data.qpos[int(self.model.jnt_qposadr[self.fingers["finger_left_joint"]])] = pos
        self.data.qpos[int(self.model.jnt_qposadr[self.fingers["finger_right_joint"]])] = pos

    def try_attach(self, waste_body_ids: list[int]) -> int | None:
        if self.engine.held_body is not None:
            return self.held_id
        ee, _ = self.ee_pose()
        best = None
        best_d = 1e9
        for bid in waste_body_ids:
            pos = self.data.xpos[bid]
            d = float(np.linalg.norm(pos - ee))
            if d < best_d:
                best_d = d
                best = bid
        if best is None or best_d > float(self.cfg["grasp_attach_distance"]):
            return None
        self.engine.grasp(best)
        self.held_id = best
        return best

    def release(self) -> None:
        self.engine.release()
        self.held_id = None
        self.open_gripper()
