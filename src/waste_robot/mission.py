"""Autonomous garbage-collection mission state machine.

Patrol -> see garbage -> capture image -> Gemini detection -> locate target ->
approach -> 6-DOF arm movement -> grab -> lift -> deposit -> resume patrol.

Every state transition is logged so the pipeline is legible in the terminal
even without the GUI windows open.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from simulation.assets.waste import WasteItem
from waste_robot.arm import ArmController
from waste_robot.bus import Bus
from waste_robot.camera import VirtualCamera
from waste_robot.coordinate_estimator import bbox_center, detection_to_world, pixel_world_direction
from waste_robot.dashboard import Dashboard
from waste_robot.detector import Detection, WasteDetector
from waste_robot.engine import Engine
from waste_robot.metrics import MissionMetrics
from waste_robot.mobile_base import MobileBase
from waste_robot.navigation import Navigator
from waste_robot.state_machine import RobotState

# Loop of open-plaza waypoints the robot patrols while searching for waste.
# The first leg heads straight out from the start position (covers the
# close-in single-item test scenarios); the rest sweeps the wider plaza for
# the multi-item demo. Navigator/astar_path already steers around obstacles
# and reactively avoids anything lidar sees, so these just need to be
# reasonable open-area targets, not hand-tuned exact clearances.
PATROL_WAYPOINTS = [
    (1.8, 0.0),
    (2.6, 2.6),
    (0.0, 3.2),
    (-2.6, 2.4),
    (-3.2, -0.5),
    (-1.0, -3.0),
    (2.0, -2.6),
    (0.0, 0.0),
]


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
        self.state = RobotState.IDLE
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
        self.patrol_points = list(PATROL_WAYPOINTS)
        self.patrol_idx = 0
        self._approach_phase = "drive"
        self._potential_det: Detection | None = None

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
        """Highest-confidence candidate above threshold that also looks like
        plausible ground litter, using the depth camera as a sanity check:

        - near-ground height (rejects tall background scenery -- trees, bin
          lids, building facades -- that happens to share a waste type's
          color, i.e. "does it look reachable/pickable" from the spec).
        - small, roughly-constant depth across its own bbox (rejects a
          sprawling background patch, e.g. a strip of grass at the color
          threshold's edge stretching off into the distance -- a real
          waste item is a small compact object at one depth).
        """
        if self.frame is None:
            return None
        ranked = sorted((d for d in dets if d.confidence >= self.threshold), key=lambda d: d.confidence, reverse=True)
        for d in ranked:
            x1, y1, x2, y2 = d.bbox
            depth_patch = self.frame.depth[y1:y2, x1:x2]
            finite = depth_patch[np.isfinite(depth_patch)]
            if finite.size == 0 or float(finite.max() - finite.min()) > 0.5:
                continue
            info = detection_to_world(self.frame, d)
            if info is not None and -0.10 <= info["world_xyz"][2] <= 0.40:
                return d
        return None

    def _localize_target(self, det: Detection) -> tuple[np.ndarray | None, str]:
        """Camera pixel -> simulator raycast -> world XYZ (spec section 6).

        Casts a single ray from the camera through the detection's bbox
        center. A hit on real waste geometry gives a precise, simulator-
        verified position; a hit on a decoy is an explicit reject; anything
        else falls back to the depth-based estimate rather than trusting a
        known object's ground-truth coordinates.
        """
        u, v = bbox_center(det)
        if 0 <= u < self.frame.width and 0 <= v < self.frame.height:
            direction = pixel_world_direction(self.frame, u, v)
            body_id, dist = self.engine.raycast(self.frame.camera_pos, direction, max_dist=self.frame.far)
            if body_id is not None and dist > 0:
                name = str(self.engine.model.body(body_id).name)
                if name.startswith("waste_"):
                    return self.engine.data.xpos[body_id].copy(), f"raycast hit {name}"
                if name.startswith("decoy_"):
                    return None, f"raycast hit {name} (not trash)"
        info = detection_to_world(self.frame, det)
        if info is None:
            return None, "no valid depth at target pixel"
        return np.array(info["world_xyz"], dtype=np.float64), "depth estimate (raycast inconclusive)"

    def _grasp_xyz(self, pregrasp: bool) -> np.ndarray:
        xyz = self.target.world_xyz.copy()
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

    def _start_approach(self, xyz: np.ndarray) -> None:
        pos, _ = self.base.pose()
        ax, ay, yaw = self.nav.approach_pose((xyz[0], xyz[1]), (pos[0], pos[1]))
        self.nav.plan((pos[0], pos[1]), (ax, ay))
        self._nav_goal = (ax, ay, yaw)
        self._nav_t0 = time.time()
        self._approach_phase = "drive"
        self.state = RobotState.APPROACHING_TARGET
        print(f"[APPROACHING_TARGET] driving to approach point ({ax:.2f}, {ay:.2f})")

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
        if do_vision:
            self._last_cam = sim_time
            self.frame = self.camera.capture()
            self.bus.publish("camera/image_raw", self.frame.rgb)
            self.bus.publish("camera/depth", self.frame.depth)
            self.metrics.detection_frames += 1

        if self.state == RobotState.IDLE:
            print("[IDLE] Mission starting -- beginning patrol")
            self.state = RobotState.PATROLLING

        elif self.state == RobotState.PATROLLING:
            self.arm_status = "READY"
            lidar = self.base.lidar(self.cfg["robot"]["lidar_rays"], self.cfg["robot"]["lidar_range"])
            goal = self.patrol_points[self.patrol_idx]
            v, w, arrived = self.nav.step(goal, None, lidar)
            self.base.set_cmd_vel(v, w)
            if arrived:
                self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_points)
                self.nav.path = []
            if do_vision and self.frame is not None:
                t0 = time.perf_counter()
                candidates = self.detector.quick_scan(self.frame.rgb)
                self.metrics.detection_seconds += time.perf_counter() - t0
                self.last_dets = candidates
                for d in candidates:
                    self.metrics.confidences.append(d.confidence)
                best = self._best_detection(candidates)
                if best is not None:
                    self._potential_det = best
                    self.base.stop()
                    self.debug = f"Potential garbage: {best.class_name} conf={best.confidence:.2f}"
                    print(f"[POTENTIAL_GARBAGE_FOUND] local trigger -- {self.debug} bbox={best.bbox}")
                    self.state = RobotState.POTENTIAL_GARBAGE_FOUND

        elif self.state == RobotState.POTENTIAL_GARBAGE_FOUND:
            self.base.stop()
            self.state = RobotState.CAPTURING_IMAGE

        elif self.state == RobotState.CAPTURING_IMAGE:
            self.base.stop()
            if self.frame is None:
                self.frame = self.camera.capture()
            started = self.detector.confirm(self.frame.rgb)
            if not started:
                print("[CAPTURING_IMAGE] confirmation skipped (cooldown/in-flight) -- resuming patrol")
                self.debug = "Gemini on cooldown -- resuming patrol"
                self._potential_det = None
                self.state = RobotState.PATROLLING
            else:
                self.state = RobotState.GEMINI_ANALYSIS

        elif self.state == RobotState.GEMINI_ANALYSIS:
            self.base.stop()
            done, dets = self.detector.poll()
            if not done:
                self.debug = "Waiting for Gemini analysis..."
            else:
                self.last_dets = dets
                confirmed = self._best_detection(dets)
                if confirmed is None:
                    self.metrics.gemini_rejected += 1
                    self.debug = "Gemini: no garbage confirmed -- resuming patrol"
                    print("[GEMINI_ANALYSIS] garbage_detected=false or below threshold -- resuming patrol")
                    self._potential_det = None
                    self.state = RobotState.PATROLLING
                else:
                    xyz, reason = self._localize_target(confirmed)
                    if xyz is None:
                        self.metrics.gemini_rejected += 1
                        self.debug = f"Target rejected ({reason}) -- resuming patrol"
                        print(f"[GEMINI_ANALYSIS] target rejected: {reason} -- resuming patrol")
                        self._potential_det = None
                        self.state = RobotState.PATROLLING
                    else:
                        xyz[2] = float(np.clip(xyz[2], 0.05, 1.2))
                        self.target = Target(confirmed.class_name, confirmed.confidence, xyz, confirmed.bbox)
                        self.metrics.waste_detected += 1
                        if confirmed.extra and confirmed.extra.get("source") == "gemini":
                            self.metrics.gemini_confirmed += 1
                        self.debug = (
                            f"Target: {confirmed.class_name} conf={confirmed.confidence:.2f} "
                            f"@ ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})"
                        )
                        print(f"[GEMINI_ANALYSIS] garbage_detected=true type={confirmed.class_name} "
                              f"confidence={confirmed.confidence:.2f} target={xyz.round(2).tolist()} ({reason})")
                        self._potential_det = None
                        self._start_approach(xyz)

        elif self.state == RobotState.APPROACHING_TARGET:
            lidar = self.base.lidar(self.cfg["robot"]["lidar_rays"], self.cfg["robot"]["lidar_range"])
            ax, ay, _ = self._nav_goal
            if self._approach_phase == "drive":
                v, w, arrived = self.nav.step((ax, ay), None, lidar)
                self.base.set_cmd_vel(v, w)
                if arrived:
                    if self._nav_t0:
                        self.metrics.navigation_time += time.time() - self._nav_t0
                    self._approach_phase = "rotate"
            else:
                pos, cyaw = self.base.pose()
                desired = math.atan2(self.target.world_xyz[1] - pos[1], self.target.world_xyz[0] - pos[0])
                err = (desired - cyaw + math.pi) % (2 * math.pi) - math.pi
                if abs(err) < math.radians(10):
                    self.base.stop()
                    self._pick_t0 = time.time()
                    self.arm_status = "ALIGNING"
                    print("[ARM_POSITIONING] base stopped -- beginning 6-DOF pickup sequence")
                    self.state = RobotState.ARM_POSITIONING
                else:
                    self.base.set_cmd_vel(0.0, 1.4 * np.sign(err) + 1.8 * err)

        elif self.state == RobotState.ARM_POSITIONING:
            self.base.stop()
            print("[ARM_POSITIONING] phase 1/3 -- rotate base toward target, open gripper")
            self.arm.open_gripper()
            aligned = self.arm.stage_align_base(self.target.world_xyz.tolist())
            pregrasp = self._grasp_xyz(pregrasp=True)
            grasp_xyz = self._grasp_xyz(pregrasp=False)
            print("[ARM_POSITIONING] phase 2/3 -- reach (shoulder -> elbow -> wrist -> descend)")
            reached = self.arm.stage_reach(pregrasp.tolist(), grasp_xyz.tolist())
            print("[ARM_POSITIONING] phase 3/3 -- fine wrist alignment onto garbage")
            self.arm.stage_fine_align(grasp_xyz.tolist())
            if aligned and reached:
                self.arm_status = "POSITIONED"
                self.state = RobotState.GRABBING
            else:
                print("[ARM_POSITIONING] arm could not reach target (IK failure) -- aborting, resuming patrol")
                self.arm_status = "IK_FAIL"
                self.target = None
                self.state = RobotState.ARM_RESET

        elif self.state == RobotState.GRABBING:
            print("[GRABBING] phase 4 -- closing gripper")
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
                print(f"[GRABBING] gripper failed to grab the object (attempt {self._grasp_tries})")
                if self._grasp_tries >= 3:
                    self._grasp_tries = 0
                    self.target = None
                    self.state = RobotState.ARM_RESET
                else:
                    self.state = RobotState.ARM_POSITIONING
            else:
                self._grasp_tries = 0
                print("[GRABBING] garbage grasped successfully")
                self.arm_status = "LIFTING"
                self.state = RobotState.LIFTING

        elif self.state == RobotState.LIFTING:
            pos, _ = self.arm.ee_pose()
            lift_xyz = pos.copy()
            lift_xyz[2] = float(self.cfg["arm"]["lift_height"])
            print("[LIFTING] phase 5 -- wrist up -> elbow retract -> shoulder raise")
            self.arm.stage_lift(lift_xyz.tolist())
            self.arm_status = "TO_BIN"
            self.state = RobotState.MOVING_TO_BIN

        elif self.state == RobotState.MOVING_TO_BIN:
            bpos, _ = self.robot.basket_world_pose()
            hover = [bpos[0], bpos[1], bpos[2] + 0.28]
            print("[MOVING_TO_BIN] phase 6 -- rotate base to bin, swing arm over it, lower")
            self.arm.stage_deposit(hover)
            lower = [bpos[0], bpos[1], bpos[2] + 0.12]
            self.arm.stage_fine_align(lower)
            self.state = RobotState.RELEASING

        elif self.state == RobotState.RELEASING:
            held = self.arm.held_id
            print("[RELEASING] phase 7 -- opening gripper, releasing garbage into bin")
            self.arm.release()
            for _ in range(40):
                self.engine.step()
            if held is not None:
                bpos, _ = self.robot.basket_world_pose()
                near = self._near_basket(held) or np.linalg.norm(
                    self.engine.data.xpos[held][:2] - bpos[:2]
                ) < 0.38
                if near:
                    self._mark_collected(held)
                    print(f"[RELEASING] garbage deposited in bin (total collected={self.metrics.waste_collected})")
                else:
                    print("[RELEASING] warning: item landed outside the bin")
            if self._pick_t0:
                self.metrics.pickup_times.append(time.time() - self._pick_t0)
            self.target = None
            self.arm_status = "RESETTING"
            self.state = RobotState.ARM_RESET

        elif self.state == RobotState.ARM_RESET:
            print("[ARM_RESET] phase 8 -- returning arm to home position")
            self.arm.go_home()
            self.arm_status = "READY"
            invalidate = getattr(self.detector, "invalidate", None)
            if callable(invalidate):
                invalidate()
            if not self._maybe_complete():
                print("[PATROLLING] resuming patrol")
                self.state = RobotState.PATROLLING

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
