"""MuJoCo physics engine: model, step, cameras, lidar, kinematic base."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import mujoco


def look_at_view(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenGL-style 4x4 view matrix (world -> camera)."""
    f = target - eye
    n = np.linalg.norm(f)
    f = f / (n if n > 1e-9 else 1.0)
    z = -f
    x = np.cross(up, z)
    xn = np.linalg.norm(x)
    if xn < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / xn
    y = np.cross(z, x)
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = np.stack([x, y, z], axis=0)
    view[:3, 3] = -view[:3, :3] @ eye
    return view


class Engine:
    def __init__(self, xml: str, *, gui: bool, cam_width: int, cam_height: int, overview: bool):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.gui = gui
        self.dt = float(self.model.opt.timestep)
        self.held_body: int | None = None
        self._held_geom_contype: dict[int, int] = {}
        self._base_pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # x, y, yaw
        self._cmd = np.array([0.0, 0.0], dtype=np.float64)  # v, w
        self.jx = self.model.joint("base_x").id
        self.jy = self.model.joint("base_y").id
        self.jyaw = self.model.joint("base_yaw").id
        self.q_x = int(self.model.jnt_qposadr[self.jx])
        self.q_y = int(self.model.jnt_qposadr[self.jy])
        self.q_yaw = int(self.model.jnt_qposadr[self.jyaw])
        self.dof_x = int(self.model.jnt_dofadr[self.jx])
        self.robot_body = int(self.model.body("robot").id)
        self._disable_robot_contacts()
        self.ee_site = int(self.model.site("ee_site").id)
        self.basket_site = int(self.model.site("basket_site").id)
        self.lidar_site = int(self.model.site("lidar_site").id)
        self.cam_id = int(self.model.camera("robot_cam").id)
        self.overview_id = int(self.model.camera("overview").id)
        self._renderer = None
        self._overview = None
        self._viewer = None
        if gui or True:
            try:
                self._renderer = mujoco.Renderer(self.model, cam_height, cam_width)
            except Exception:
                self._renderer = None
        if overview and gui:
            try:
                self._overview = mujoco.Renderer(self.model, 360, 640)
            except Exception:
                self._overview = None
        if gui:
            try:
                from mujoco import viewer

                self._viewer = viewer.launch_passive(self.model, self.data)
            except Exception:
                self._viewer = None

    def _disable_robot_contacts(self) -> None:
        """Kinematic base/arm should not fling waste; lidar still uses geom rays."""
        for g in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[g])
            if self._is_robot_body(bid):
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0

    def joint_id(self, name: str) -> int:
        return int(self.model.joint(name).id)

    def actuator_id(self, name: str) -> int:
        return int(self.model.actuator(name).id)

    def body_id(self, name: str) -> int:
        return int(self.model.body(name).id)

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self.body_id(name)
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    def site_pose(self, site_id: int) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site_xpos[site_id].copy(), self.data.site_xmat[site_id].copy()

    def set_base_cmd(self, v: float, w: float) -> None:
        self._cmd[0] = v
        self._cmd[1] = w

    def base_pose(self) -> tuple[np.ndarray, float]:
        x = float(self.data.qpos[self.q_x])
        y = float(self.data.qpos[self.q_y])
        yaw = float(self.data.qpos[self.q_yaw])
        z = float(self.data.xpos[self.robot_body][2])
        return np.array([x, y, z], dtype=np.float64), yaw

    def apply_base_kinematics(self) -> None:
        yaw = float(self.data.qpos[self.q_yaw])
        v, w = self._cmd
        self.data.qpos[self.q_x] += v * math_cos(yaw) * self.dt
        self.data.qpos[self.q_y] += v * math_sin(yaw) * self.dt
        self.data.qpos[self.q_yaw] = yaw + w * self.dt
        self.data.qvel[self.dof_x] = v * math_cos(yaw)
        self.data.qvel[self.dof_x + 1] = v * math_sin(yaw)
        self.data.qvel[self.dof_x + 2] = w

    def _attach_held(self) -> None:
        if self.held_body is None:
            return
        pos, mat = self.site_pose(self.ee_site)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        jnt = int(self.model.body_jntadr[self.held_body])
        adr = int(self.model.jnt_qposadr[jnt])
        dof = int(self.model.jnt_dofadr[jnt])
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat
        self.data.qvel[dof : dof + 6] = 0.0

    def grasp(self, body_id: int) -> None:
        self.release()
        self.held_body = body_id
        for g in range(self.model.ngeom):
            if int(self.model.geom_bodyid[g]) == body_id:
                self._held_geom_contype[g] = int(self.model.geom_contype[g])
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0

    def release(self) -> None:
        for g, ct in self._held_geom_contype.items():
            self.model.geom_contype[g] = ct
            self.model.geom_conaffinity[g] = 1
        self._held_geom_contype.clear()
        self.held_body = None

    def hide_body(self, body_id: int) -> None:
        jnt = int(self.model.body_jntadr[body_id])
        adr = int(self.model.jnt_qposadr[jnt])
        self.data.qpos[adr : adr + 3] = np.array([0.0, 0.0, -1.0])
        for g in range(self.model.ngeom):
            if int(self.model.geom_bodyid[g]) == body_id:
                self.model.geom_rgba[g, 3] = 0.0
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0

    def step(self) -> None:
        self.apply_base_kinematics()
        self._attach_held()
        mujoco.mj_step(self.model, self.data)
        self._attach_held()
        if self._viewer is not None:
            try:
                if self._viewer.is_running():
                    self._viewer.sync()
                else:
                    self._viewer = None
            except Exception:
                self._viewer = None

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def render_rgb_depth(self, camera: str = "robot_cam") -> tuple[np.ndarray, np.ndarray] | None:
        if self._renderer is None:
            return None
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera)
        rgb = self._renderer.render()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera)
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()
        return rgb.copy(), depth.copy()

    def render_overview(self) -> np.ndarray | None:
        if self._overview is None:
            return None
        self._overview.update_scene(self.data, camera="overview")
        return self._overview.render().copy()

    def camera_frame_axes(self, cam_name: str = "robot_cam") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cid = int(self.model.camera(cam_name).id)
        pos = self.data.cam_xpos[cid].copy()
        r = self.data.cam_xmat[cid].reshape(3, 3)
        x_axis, y_axis, z_axis = r[:, 0], r[:, 1], r[:, 2]
        return pos, x_axis, y_axis, z_axis

    def lidar(self, n_rays: int, max_range: float) -> np.ndarray:
        pos = self.data.site_xpos[self.lidar_site].copy()
        yaw = float(self.data.qpos[self.q_yaw])
        vec = np.zeros(3 * n_rays, dtype=np.float64)
        angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
        for i, a in enumerate(angles):
            ang = yaw + a
            vec[3 * i : 3 * i + 3] = (np.cos(ang) * max_range, np.sin(ang) * max_range, 0.0)
        geomid = np.full(n_rays, -1, dtype=np.int32)
        dist = np.full(n_rays, -1.0, dtype=np.float64)
        geomgroup = np.ones(6, dtype=np.uint8)
        mujoco.mj_multiRay(
            self.model,
            self.data,
            pos,
            vec,
            geomgroup,
            True,
            self.robot_body,
            geomid,
            dist,
            None,
            n_rays,
            max_range,
        )
        hits = np.full(n_rays, max_range, dtype=np.float64)
        for i in range(n_rays):
            if geomid[i] >= 0 and dist[i] >= 0:
                hits[i] = float(min(dist[i], max_range))
        return hits

    def raycast(self, pos: np.ndarray, direction: np.ndarray, max_dist: float = 10.0) -> tuple[int | None, float]:
        """Single ray (mj_ray) from pos along direction; returns (hit_body_id, distance)."""
        n = np.linalg.norm(direction)
        if n < 1e-9:
            return None, -1.0
        vec = (np.asarray(direction, dtype=np.float64) / n) * max_dist
        geomid = np.full(1, -1, dtype=np.int32)
        geomgroup = np.ones(6, dtype=np.uint8)
        dist = mujoco.mj_ray(
            self.model,
            self.data,
            np.asarray(pos, dtype=np.float64),
            vec,
            geomgroup,
            True,
            self.robot_body,
            geomid,
        )
        if dist < 0 or geomid[0] < 0:
            return None, -1.0
        # mj_ray returns x where hit = pos + x*vec, and |vec| == max_dist here,
        # so the real-world distance in meters is x * max_dist.
        body_id = int(self.model.geom_bodyid[geomid[0]])
        return body_id, float(dist * max_dist)

    def contacts_with_static(self) -> int:
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            r1 = self._is_robot_body(b1)
            r2 = self._is_robot_body(b2)
            if r1 ^ r2:
                other = b2 if r1 else b1
                if other != 0 and not str(self.model.body(other).name).startswith("waste_"):
                    n += 1
        return n

    def _is_robot_body(self, bid: int) -> bool:
        while bid > 0:
            if bid == self.robot_body:
                return True
            bid = int(self.model.body_parentid[bid])
        return False

    def close(self) -> None:
        self.release()
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._overview is not None:
            self._overview.close()
            self._overview = None


def math_cos(a: float) -> float:
    return float(np.cos(a))


def math_sin(a: float) -> float:
    return float(np.sin(a))


@dataclass
class SimContext:
    engine: Engine
    mission: object
    robot: object
