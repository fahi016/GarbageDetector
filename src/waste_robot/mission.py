"""Autonomous waste-collection mission: perception -> nav -> manipulation loop."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from simulation.assets.waste import WasteItem
from waste_robot.arm import ArmController
from waste_robot.bus import Bus
from waste_robot.camera import VirtualCamera
from waste_robot.coordinate_estimator import detection_to_world
from waste_robot.dashboard import Dashboard
from waste_robot.detector import Detection, WasteDetector
from waste_robot.engine import Engine
from waste_robot.metrics import MissionMetrics
from waste_robot.mobile_base import MobileBase
from waste_robot.navigation import Navigator
from waste_robot.state_machine import RobotState


@dataclass
class Target:
    class_name: str
    confidence: float
    world_xyz: np.ndarray
    bbox: tuple[int, int, int, int]


class Mission:
    def __init__(
        self,
        engine: Engine,
        robot,
        base: MobileBase,
        arm: ArmController,
        camera: VirtualCamera,
        detector: WasteDetector,
        navigator: Navigator,
        waste: list[WasteItem],
        cfg: dict,
        bus: Bus,
        dashboard: Dashboard,
        scenario: str = "multi",
        headless: bool = False,
    ):
        self.engine = engine
        self.robot = robot
        self.base = base
        self.arm = arm
        self.camera = camera
        self.detector = detector
        self.nav = navigator
        self.waste = waste
        self.cfg = cfg
        self.bus = bus
        self.dashboard = dashboard
        self.scenario = scenario
        self.headless = headless
        self.state = RobotState.SEARCHING
        self.target: Target | None = None
        self.metrics = MissionMetrics()
        self.battery = 100.0
        self.running = True
        self.threshold = float(cfg["detector"]["confidence_threshold"])
        if scenario == "false_detection":
            self.threshold = 0.50
        self.collect_limit = int(cfg["simulator"].get("collect_limit") or 0)
        self.max_seconds = float(cfg["simulator"]["max_mission_seconds"])
        self._nav_t0 = None
        self._pick_t0 = None
        self._grasp_tries = 0
        self.frame = None
        self.last_dets: list[Detection] = []
        self.arm_status = "READY"
        self.debug = ""
        self._last_col_t = 0.0
        self._nav_goal = (0.0, 0.0, 0.0)
        self.decoys: list[WasteItem] = []

    def stop(self) -> None:
        self.running = False
        self.state = RobotState.STOPPED
        self.base.stop()

    def active_waste_ids(self) -> list[int]:
        return [w.body_id for w in self.waste if not w.collected]

    def remaining(self) -> int:
        return sum(1 for w in self.waste if not w.collected)

    def _count_collisions(self) -> None:
        now = time.time()
        if now - self._last_col_t < 1.0:
            return
        if self.engine.contacts_with_static() > 0:
            self.metrics.collisions += 1
            self._last_col_t = now

    def _mark_collected(self, body_id: int) -> None:
        for w in self.waste:
            if w.body_id == body_id and not w.collected:
                w.collected = True
                self.metrics.waste_collected += 1
                self.engine.hide_body(body_id)

    def _near_basket(self, body_id: int) -> bool:
        bpos, _ = self.robot.basket_world_pose()
        wpos = self.engine.data.xpos[body_id]
        dx, dy, dz = wpos[0] - bpos[0], wpos[1] - bpos[1], wpos[2] - bpos[2]
        return abs(dx) < 0.18 and abs(dy) < 0.20 and -0.08 < dz < 0.35

    def _best_detection(self, dets: list[Detection]) -> Detection | None:
        ok = [d for d in dets if d.confidence >= self.threshold]
        return ok[0] if ok else None

    def _perceive(self) -> None:
        t0 = time.perf_counter()
        self.frame = self.camera.capture()
        querying = self.state == RobotState.SEARCHING
        if querying:
            self.last_dets = self.detector.detect(self.frame.rgb)
        dt = time.perf_counter() - t0
        self.metrics.detection_frames += 1
        self.metrics.detection_seconds += dt
        self.bus.publish("camera/image_raw", self.frame.rgb)
        self.bus.publish("camera/depth", self.frame.depth)
        self.bus.publish("detections", self.last_dets)
        for d in self.last_dets:
            self.metrics.confidences.append(d.confidence)

    def _set_target_from_det(self, det: Detection) -> bool:
        info = detection_to_world(self.frame, det)
        if info is None:
            return False
        xyz = np.array(info["world_xyz"], dtype=np.float64)
        if xyz[2] < 0 or xyz[2] > 1.5:
            xyz[2] = 0.08
        nearest = None
        nearest_d = 1e9
        for w in self.waste:
            if w.collected:
                continue
            pos = self.engine.data.xpos[w.body_id]
            d = float(np.hypot(pos[0] - xyz[0], pos[1] - xyz[1]))
            if d < nearest_d:
                nearest_d = d
                nearest = pos.copy()
        if nearest is None or nearest_d >= 1.25:
            self.metrics.gemini_rejected += 1
            self.debug = "Gemini box is not near remaining trash — ignored"
            return False
        xyz = nearest
        xyz[2] = max(0.06, float(xyz[2]))
        self.target = Target(det.class_name, det.confidence, xyz, det.bbox)
        self.metrics.waste_detected += 1
        if det.extra and det.extra.get("source") == "gemini":
            self.metrics.gemini_confirmed += 1
        self.bus.publish("waste_pose", info)
        self.debug = (
            f"Gemini trash: {det.class_name}  conf={det.confidence:.2f}  "
            f"X={xyz[0]:.2f} Y={xyz[1]:.2f} Z={xyz[2]:.2f}"
        )
        return True

    def _grasp_xyz(self, pregrasp: bool) -> np.ndarray:
        xyz = self.target.world_xyz.copy()
        best = None
        best_d = 1e9
        pos, _ = self.base.pose()
        for w in self.waste:
            if w.collected:
                continue
            p = self.engine.data.xpos[w.body_id]
            d = float(np.hypot(p[0] - pos[0], p[1] - pos[1]))
            if d < best_d:
                best_d = d
                best = p.copy()
        if best is not None and best_d < 1.4:
            xyz = best
        if pregrasp:
            xyz[2] = max(float(xyz[2]) + float(self.cfg["arm"]["pregrasp_height"]), 0.20)
        else:
            xyz[2] = max(0.08, float(xyz[2]) + 0.02)
        return xyz

    def _distance_to_target(self) -> float:
        if self.target is None:
            return 0.0
        pos, _ = self.base.pose()
        return float(np.hypot(self.target.world_xyz[0] - pos[0], self.target.world_xyz[1] - pos[1]))

    def _maybe_complete(self) -> bool:
        if self.remaining() == 0:
            self.state = RobotState.COMPLETE
            self.base.stop()
            self.running = False
            return True
        if self.collect_limit and self.metrics.waste_collected >= self.collect_limit:
            self.state = RobotState.COMPLETE
            self.base.stop()
            self.running = False
            return True
        return False

    def tick(self, sim_time: float) -> None:
        if self.dashboard.consume_quit():
            self.stop()
            return
        self.base.update_odometry()
        self.metrics.distance_travelled = self.base.distance_travelled
        self.metrics.mission_time = time.time() - self.metrics.start_time
        self.battery = max(5.0, 100.0 - self.metrics.mission_time * 0.08)
        if self.metrics.mission_time > self.max_seconds:
            self.stop()
            return
        self._count_collisions()

        cam_period = 1.0 / float(self.cfg["simulator"]["camera_render_hz"])
        if not hasattr(self, "_last_cam"):
            self._last_cam = 0.0
        do_vision = (sim_time - self._last_cam) >= cam_period or self.frame is None
        status = getattr(self.detector, "last_status", "")
        if do_vision:
            self._last_cam = sim_time
            self._perceive()
            status = getattr(self.detector, "last_status", "")
            if status and self.state == RobotState.SEARCHING:
                self.debug = status

        if self.state == RobotState.SEARCHING:
            self.arm_status = "READY"
            lidar = self.base.lidar(self.cfg["robot"]["lidar_rays"], self.cfg["robot"]["lidar_range"])
            v, w = self.nav.search_cmd(lidar, sim_time)
            self.base.set_cmd_vel(v, w)
            det = None
            for cand in self.last_dets:
                if cand.confidence >= self.threshold and self._set_target_from_det(cand):
                    det = cand
                    break
            if det:
                self.base.stop()
                self.state = RobotState.WASTE_DETECTED
            elif status:
                self.debug = status

        elif self.state == RobotState.WASTE_DETECTED:
            pos, _ = self.base.pose()
            ax, ay, yaw = self.nav.approach_pose(
                (self.target.world_xyz[0], self.target.world_xyz[1]), (pos[0], pos[1])
            )
            self.nav.plan((pos[0], pos[1]), (ax, ay))
            self._nav_goal = (ax, ay, yaw)
            self._nav_t0 = time.time()
            self.state = RobotState.NAVIGATING

        elif self.state == RobotState.NAVIGATING:
            lidar = self.base.lidar(self.cfg["robot"]["lidar_rays"], self.cfg["robot"]["lidar_range"])
            ax, ay, yaw = self._nav_goal
            v, w, arrived = self.nav.step((ax, ay), None, lidar)
            self.base.set_cmd_vel(v, w)
            if arrived:
                if self._nav_t0:
                    self.metrics.navigation_time += time.time() - self._nav_t0
                self.state = RobotState.POSITIONING

        elif self.state == RobotState.POSITIONING:
            pos, cyaw = self.base.pose()
            desired = math.atan2(self.target.world_xyz[1] - pos[1], self.target.world_xyz[0] - pos[0])
            err = (desired - cyaw + math.pi) % (2 * math.pi) - math.pi
            if abs(err) < math.radians(10):
                self.base.stop()
                self.state = RobotState.ARM_APPROACHING
                self._pick_t0 = time.time()
                self.arm_status = "APPROACHING"
            else:
                self.base.set_cmd_vel(0.0, 1.4 * np.sign(err) + 1.8 * err)

        elif self.state == RobotState.ARM_APPROACHING:
            self.base.stop()
            tgt = self._grasp_xyz(pregrasp=True)
            ok = self.arm.move_ee(tgt.tolist())
            self.arm.open_gripper()
            self.state = RobotState.GRASPING if ok else RobotState.RETURNING_ARM
            self.arm_status = "GRASPING" if ok else "IK_FAIL"

        elif self.state == RobotState.GRASPING:
            tgt = self._grasp_xyz(pregrasp=False)
            self.arm.move_ee(tgt.tolist(), timeout=3.0)
            for _ in range(24):
                self.engine.step()
            self.arm.close_gripper()
            for _ in range(18):
                self.engine.step()
            if self.scenario == "failed_pickup" and self._grasp_tries == 0:
                attached = None
            else:
                attached = self.arm.try_attach(self.active_waste_ids())
            if attached is None:
                self.metrics.failed_grasps += 1
                self._grasp_tries += 1
                if self._grasp_tries >= 3:
                    self._grasp_tries = 0
                    self.target = None
                    self.state = RobotState.RETURNING_ARM
                else:
                    self.state = RobotState.ARM_APPROACHING
            else:
                self._grasp_tries = 0
                self.state = RobotState.LIFTING
                self.arm_status = "LIFTING"

        elif self.state == RobotState.LIFTING:
            pos, _ = self.arm.ee_pose()
            lift = pos.copy()
            lift[2] = float(self.cfg["arm"]["lift_height"])
            self.arm.move_ee(lift.tolist(), timeout=3.0)
            self.state = RobotState.MOVING_TO_BASKET
            self.arm_status = "TO_BASKET"

        elif self.state == RobotState.MOVING_TO_BASKET:
            bpos, _ = self.robot.basket_world_pose()
            dest = [bpos[0], bpos[1], bpos[2] + 0.28]
            self.arm.move_ee(dest, timeout=4.0)
            self.state = RobotState.RELEASING

        elif self.state == RobotState.RELEASING:
            held = self.arm.held_id
            self.arm.release()
            for _ in range(40):
                self.engine.step()
            if held is not None:
                if self._near_basket(held) or np.linalg.norm(
                    self.engine.data.xpos[held][:2] - self.robot.basket_world_pose()[0][:2]
                ) < 0.38:
                    self._mark_collected(held)
            if self._pick_t0:
                self.metrics.pickup_times.append(time.time() - self._pick_t0)
            self.target = None
            self.state = RobotState.RETURNING_ARM
            self.arm_status = "RETURNING"

        elif self.state == RobotState.RETURNING_ARM:
            self.arm.go_home()
            self.arm_status = "READY"
            invalidate = getattr(self.detector, "invalidate", None)
            if callable(invalidate):
                invalidate()
            if not self._maybe_complete():
                self.state = RobotState.SEARCHING

        self.bus.publish("robot_state", self.state.value)
        self.bus.publish("cmd_vel", (0, 0))

        dist = self._distance_to_target()
        tgt_name = self.target.class_name if self.target else "none"
        if do_vision and self.frame is not None:
            world = None if self.headless else self.engine.render_overview()
            self.dashboard.show(
                self.frame.rgb,
                self.last_dets,
                self.threshold,
                self.state,
                tgt_name,
                dist,
                self.arm_status,
                self.metrics.waste_collected,
                self.battery,
                extra=self.debug,
                world_rgb=world,
            )
