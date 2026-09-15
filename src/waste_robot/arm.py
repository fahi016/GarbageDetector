"""6-DOF arm IK, gripper, and simplified grasp constraint."""

from __future__ import annotations

import numpy as np
import mujoco

from waste_robot.engine import Engine


class ArmController:
    def __init__(self, engine: Engine, robot, cfg: dict, bridge=None):
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
        # Optional ArmSerialBridge: when set, every joint/gripper update
        # below is also streamed to the real ESP32-driven arm.
        self.bridge = bridge
        self._gripper_opening = float(cfg["gripper_open"])
        self._hold_position(self.home)
        self.open_gripper()
        engine.forward()

    def _hold_position(self, q: list[float]) -> None:
        applied = []
        for j, qi, act in zip(self.joints, q, self.act):
            adr = int(self.model.jnt_qposadr[j])
            lo, hi = self.model.jnt_range[j]
            qi = float(np.clip(qi, lo, hi))
            self.data.qpos[adr] = qi
            self.data.ctrl[act] = qi
            applied.append(qi)
        if self.bridge is not None:
            self.bridge.send(applied, self._gripper_opening)

    def current_q(self) -> list[float]:
        return [float(self.data.qpos[int(self.model.jnt_qposadr[j])]) for j in self.joints]

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos, mat = self.engine.site_pose(self.ee_site)
        return pos, mat

    def _ik(self, xyz: np.ndarray, joint_mask: list[int] | None = None) -> list[float] | None:
        """Jacobian IK. `joint_mask` restricts which of the 6 joints may move
        (others stay frozen at their current value) — used for the wrist-only
        fine-alignment pass. Returns the full 6-length joint solution.

        IMPORTANT: this only *evaluates* candidate poses (via mj_forward,
        kinematics-only) while searching -- it always restores the arm to
        its pre-call pose before returning, so it never mutates live sim
        state. Callers animate from the real current pose to the returned
        solution; without this the arm would silently teleport to the goal
        during the search itself, and any later interpolation would be a
        no-op ("start" and "goal" already equal) -- exactly the invisible/
        instant-snap motion this whole staged-motion system exists to avoid.
        """
        mask = list(joint_mask) if joint_mask is not None else list(range(len(self.joints)))
        target = np.asarray(xyz, dtype=np.float64)
        original_q = self.current_q()
        solved: list[float] | None = None
        for _ in range(int(self.cfg["ik_max_iterations"])):
            mujoco.mj_forward(self.model, self.data)
            pos = self.data.site_xpos[self.ee_site]
            err = target - pos
            if float(np.linalg.norm(err)) < 0.02:
                solved = self.current_q()
                break
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site)
            cols_all = [int(self.model.jnt_dofadr[j]) for j in self.joints]
            cols = [cols_all[i] for i in mask]
            jmat = jacp[:, cols]
            lam = 1e-3
            try:
                dq = jmat.T @ np.linalg.solve(jmat @ jmat.T + lam * np.eye(3), 0.6 * err)
            except np.linalg.LinAlgError:
                solved = None
                break
            dq = np.clip(dq, -0.2, 0.2)
            q = self.current_q()
            for k, idx in enumerate(mask):
                q[idx] = float(np.clip(q[idx] + dq[k], self.lower[idx], self.upper[idx]))
            self._hold_position(q)
        else:
            mujoco.mj_forward(self.model, self.data)
            if float(np.linalg.norm(target - self.data.site_xpos[self.ee_site])) < 0.07:
                solved = self.current_q()
        self._hold_position(original_q)
        mujoco.mj_forward(self.model, self.data)
        return solved

    def move_joints_staged(
        self,
        target_q: list[float],
        groups: list[list[int]],
        steps_per_group: int = 28,
        substeps: int = 4,
    ) -> None:
        """Move joints group-by-group so each group's motion is visually distinct.

        Joints not in the active group hold their pre-group value while the
        active group's joints interpolate smoothly to `target_q`. This is the
        core of the "clearly 6-DOF, not one rigid blob" arm motion: the base
        rotates, *then* the shoulder drops, *then* the elbow bends, etc.
        """
        current = np.array(self.current_q())
        goal = np.array(target_q)
        for group in groups:
            group_start = current.copy()
            for k in range(steps_per_group):
                a = (k + 1) / steps_per_group
                q = current.copy()
                for idx in group:
                    q[idx] = group_start[idx] * (1 - a) + goal[idx] * a
                self._hold_position(q.tolist())
                for _ in range(substeps):
                    self.engine.step()
            for idx in group:
                current[idx] = goal[idx]

    # -- Named 6-DOF pickup phases (each moves a distinct joint group) -------

    def _base_bearing_q(self, target_xyz) -> float:
        """Desired joint_1 angle to point the arm's waist at target_xyz.

        A single revolute joint can't satisfy a full 3D position IK target
        (that needs 3+ DOF), so aiming the base is a plain bearing
        calculation, not an IK solve: angle from the (fixed-to-chassis) arm
        base to the target, expressed relative to the chassis heading.
        """
        base_pos, _ = self.engine.body_pose("arm_base")
        _, chassis_yaw = self.engine.base_pose()
        dx = float(target_xyz[0]) - base_pos[0]
        dy = float(target_xyz[1]) - base_pos[1]
        desired = float(np.arctan2(dy, dx)) - chassis_yaw
        desired = (desired + np.pi) % (2 * np.pi) - np.pi
        return float(np.clip(desired, self.lower[0], self.upper[0]))

    def stage_align_base(self, target_xyz) -> bool:
        """Phase 1 (scan/align): rotate only the waist/base joint toward the target."""
        q_goal = self.current_q()
        q_goal[0] = self._base_bearing_q(target_xyz)
        self.move_joints_staged(q_goal, groups=[[0]], steps_per_group=34)
        return True

    def stage_reach(self, pregrasp_xyz, grasp_xyz) -> bool:
        """Phase 2 (reach): shoulder -> elbow -> wrist orientation -> descend onto the target.

        Two full-IK solves, not one: first a safe hover pose above the
        target (shoulder/elbow/forearm do the big motion), then the actual
        grasp point (wrist pitch does the final descent). A single wrist-
        only IK pass can't cover that descent -- of the 3 wrist joints, two
        are roll axes through the end-effector site and barely move it, so
        that step is reserved for small last-centimeter fine alignment.
        """
        q_hover = self._ik(pregrasp_xyz)
        if q_hover is None:
            return False
        self.move_joints_staged(q_hover, groups=[[1], [2], [3, 4]], steps_per_group=24)
        q_final = self._ik(grasp_xyz)
        if q_final is None:
            return False
        self.move_joints_staged(q_final, groups=[[5], [1, 2]], steps_per_group=22)
        return True

    def stage_fine_align(self, target_xyz) -> bool:
        """Phase 3 (fine alignment): small wrist-only correction onto the grasp point."""
        q_goal = self._ik(target_xyz, joint_mask=[3, 4, 5])
        if q_goal is None:
            return False
        self.move_joints_staged(q_goal, groups=[[3], [4], [5]], steps_per_group=16)
        return True

    def stage_lift(self, lift_xyz) -> bool:
        """Phase 5 (lift): wrist up, retract elbow, raise shoulder."""
        q_goal = self._ik(lift_xyz)
        if q_goal is None:
            return False
        self.move_joints_staged(q_goal, groups=[[3, 4, 5], [2], [1]], steps_per_group=24)
        return True

    def stage_deposit(self, basket_xyz) -> bool:
        """Phase 6 (deposit): rotate base toward the bin, then swing the arm over it."""
        q_base = self.current_q()
        q_base[0] = self._base_bearing_q(basket_xyz)
        self.move_joints_staged(q_base, groups=[[0]], steps_per_group=28)
        q_goal = self._ik(basket_xyz)
        if q_goal is None:
            return False
        self.move_joints_staged(q_goal, groups=[[1, 2], [3, 4], [5]], steps_per_group=24)
        return True

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
        self._gripper_opening = pos
        if self.bridge is not None:
            self.bridge.send(self.current_q(), pos)

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
