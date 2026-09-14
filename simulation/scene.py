"""Build the MuJoCo plaza + robot + waste as one MJCF document."""

from __future__ import annotations

import math

import numpy as np

from simulation.assets.waste import WASTE_SPECS, accent_color, base_color
from simulation.worlds.park_plaza import plaza_obstacles


def _rgba(c) -> str:
    return " ".join(f"{float(v):.4f}" for v in c)


def _cam_xyaxes(pitch_deg: float) -> str:
    pitch = math.radians(pitch_deg)
    forward = (math.cos(pitch), 0.0, math.sin(pitch))
    x_axis = (0.0, -1.0, 0.0)
    z_axis = (-forward[0], -forward[1], -forward[2])
    y_axis = (
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    )
    n = math.sqrt(sum(v * v for v in y_axis)) or 1.0
    y_axis = tuple(v / n for v in y_axis)
    return f"{x_axis[0]:.5f} {x_axis[1]:.5f} {x_axis[2]:.5f} {y_axis[0]:.5f} {y_axis[1]:.5f} {y_axis[2]:.5f}"


def _robot_xml(start_xy, start_yaw: float, pitch_deg: float, fovy: float) -> str:
    x, y = start_xy
    xyaxes = _cam_xyaxes(pitch_deg)
    return f"""
    <body name="robot" pos="{x:.4f} {y:.4f} 0.12" euler="0 0 {start_yaw:.4f}">
      <joint name="base_x" type="slide" axis="1 0 0" damping="80"/>
      <joint name="base_y" type="slide" axis="0 1 0" damping="80"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" damping="40"/>
      <inertial pos="0 0 0.02" mass="28" diaginertia="0.55 0.85 1.05"/>
      <!-- Chassis/basket/arm are deliberately low-saturation: every waste
           HSV bucket in detector.py requires S above ~0.28, so a desaturated
           robot body can never trigger the color-based waste detectors on
           itself even when it enters its own camera frame. -->
      <geom name="chassis" type="box" size="0.39 0.25 0.08" pos="0 0 0.05" rgba="0.58 0.59 0.62 1" friction="1.2 0.1 0.01"/>
      <geom name="mast" type="cylinder" size="0.025 0.16" pos="0.20 0 0.28" rgba="0.55 0.57 0.60 1"/>
      <geom name="cam_box" type="box" size="0.04 0.05 0.03" pos="0.22 0 0.44" rgba="0.25 0.25 0.28 1"/>
      <geom name="wheel_fl" type="cylinder" size="0.08 0.022" pos="0.24 0.21 -0.04" euler="1.5708 0 0" rgba="0.08 0.08 0.08 1" contype="0" conaffinity="0"/>
      <geom name="wheel_fr" type="cylinder" size="0.08 0.022" pos="0.24 -0.21 -0.04" euler="1.5708 0 0" rgba="0.08 0.08 0.08 1" contype="0" conaffinity="0"/>
      <geom name="wheel_rl" type="cylinder" size="0.08 0.022" pos="-0.24 0.21 -0.04" euler="1.5708 0 0" rgba="0.08 0.08 0.08 1" contype="0" conaffinity="0"/>
      <geom name="wheel_rr" type="cylinder" size="0.08 0.022" pos="-0.24 -0.21 -0.04" euler="1.5708 0 0" rgba="0.08 0.08 0.08 1" contype="0" conaffinity="0"/>
      <geom name="basket_floor" type="box" size="0.15 0.18 0.01" pos="-0.28 0 0.14" rgba="0.30 0.32 0.34 1"/>
      <geom name="basket_l" type="box" size="0.01 0.18 0.11" pos="-0.42 0 0.24" rgba="0.30 0.32 0.34 1"/>
      <geom name="basket_r" type="box" size="0.01 0.18 0.11" pos="-0.14 0 0.24" rgba="0.30 0.32 0.34 1"/>
      <geom name="basket_f" type="box" size="0.15 0.01 0.11" pos="-0.28 0.17 0.24" rgba="0.30 0.32 0.34 1"/>
      <geom name="basket_b" type="box" size="0.15 0.01 0.11" pos="-0.28 -0.17 0.24" rgba="0.30 0.32 0.34 1"/>
      <site name="basket_site" pos="-0.28 0 0.20" size="0.03"/>
      <site name="lidar_site" pos="0 0 0.12" size="0.02"/>
      <camera name="robot_cam" pos="0.24 0 0.46" xyaxes="{xyaxes}" fovy="{fovy:.2f}"/>

      <body name="arm_base" pos="0.12 0 0.16">
        <geom type="cylinder" size="0.055 0.03" rgba="0.34 0.36 0.40 1"/>
        <body name="arm_link1" pos="0 0 0.04">
          <joint name="joint_1" type="hinge" axis="0 0 1" range="-2.8 2.8" damping="2" armature="0.02"/>
          <geom type="cylinder" size="0.045 0.06" pos="0 0 0.06" rgba="0.34 0.36 0.40 1"/>
          <body name="arm_link2" pos="0 0 0.12">
            <joint name="joint_2" type="hinge" axis="0 1 0" range="-1.7 1.7" damping="2" armature="0.02"/>
            <geom type="box" size="0.03 0.03 0.14" pos="0 0 0.14" rgba="0.55 0.57 0.60 1"/>
            <body name="arm_link3" pos="0 0 0.28">
              <joint name="joint_3" type="hinge" axis="0 1 0" range="-2.4 2.4" damping="2" armature="0.015"/>
              <geom type="box" size="0.025 0.025 0.12" pos="0 0 0.12" rgba="0.34 0.36 0.40 1"/>
              <body name="arm_link4" pos="0 0 0.24">
                <joint name="joint_4" type="hinge" axis="0 0 1" range="-2.8 2.8" damping="1" armature="0.008"/>
                <geom type="cylinder" size="0.03 0.04" pos="0 0 0.04" rgba="0.55 0.57 0.60 1"/>
                <body name="arm_link5" pos="0 0 0.08">
                  <joint name="joint_5" type="hinge" axis="0 1 0" range="-1.8 1.8" damping="1" armature="0.006"/>
                  <geom type="box" size="0.022 0.022 0.035" pos="0 0 0.035" rgba="0.34 0.36 0.40 1"/>
                  <body name="arm_link6" pos="0 0 0.07">
                    <joint name="joint_6" type="hinge" axis="0 0 1" range="-2.8 2.8" damping="1" armature="0.004"/>
                    <geom type="cylinder" size="0.025 0.03" pos="0 0 0.03" rgba="0.55 0.57 0.60 1"/>
                    <body name="gripper_palm" pos="0 0 0.06">
                      <geom type="box" size="0.03 0.04 0.02" pos="0 0 0.02" rgba="0.08 0.08 0.08 1"/>
                      <site name="ee_site" pos="0 0 0.09" size="0.015" rgba="1 0.2 0.1 1"/>
                      <body name="finger_left" pos="0 0.03 0.03">
                        <joint name="finger_left_joint" type="slide" axis="0 1 0" range="0 0.045" damping="5"/>
                        <geom type="box" size="0.006 0.009 0.04" pos="0 0 0.04" rgba="0.08 0.08 0.08 1"/>
                      </body>
                      <body name="finger_right" pos="0 -0.03 0.03">
                        <joint name="finger_right_joint" type="slide" axis="0 -1 0" range="0 0.045" damping="5"/>
                        <geom type="box" size="0.006 0.009 0.04" pos="0 0 0.04" rgba="0.08 0.08 0.08 1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    """


def _decoy_xml(decoys) -> str:
    from simulation.assets.waste import DECOY_SPECS

    chunks = []
    for i, (dtype, xy, yaw) in enumerate(decoys):
        spec = DECOY_SPECS[dtype]
        name = f"decoy_{i}"
        rgba = _rgba(spec["rgba"])
        if spec["kind"] == "cylinder":
            h = spec["height"]
            z = h / 2 + 0.02
            geom = f'<geom name="{name}_g" type="cylinder" size="{spec["radius"]} {h/2:.4f}" rgba="{rgba}" mass="0.08"/>'
        elif spec["kind"] == "sphere":
            r = spec["radius"]
            z = r + 0.02
            geom = f'<geom name="{name}_g" type="sphere" size="{r}" rgba="{rgba}" mass="0.12"/>'
        else:
            sx, sy, sz = spec["size"]
            z = sz / 2 + 0.02
            geom = f'<geom name="{name}_g" type="box" size="{sx/2:.4f} {sy/2:.4f} {sz/2:.4f}" rgba="{rgba}" mass="0.10"/>'
        chunks.append(
            f'<body name="{name}" pos="{xy[0]:.4f} {xy[1]:.4f} {z:.4f}" euler="0 0 {yaw:.4f}">'
            f"<freejoint/>{geom}</body>"
        )
    return "\n".join(chunks)


def _part_geom_xml(geom_name: str, part: dict, scale: float, rgba: tuple) -> str:
    """One collision+visual geom for a compound waste part, scaled uniformly."""
    kind = part["kind"]
    px, py, pz = (v * scale for v in part["pos"])
    euler = part.get("euler")
    euler_attr = f' euler="{euler[0]:.4f} {euler[1]:.4f} {euler[2]:.4f}"' if euler else ""
    color = _rgba(rgba)
    mass = float(part["mass"]) * (scale ** 3)
    if kind == "cylinder":
        r, hh = (v * scale for v in part["size"])
        return (
            f'<geom name="{geom_name}" type="cylinder" size="{r:.4f} {hh:.4f}" '
            f'pos="{px:.4f} {py:.4f} {pz:.4f}"{euler_attr} rgba="{color}" mass="{mass:.4f}" '
            f'friction="1 0.05 0.01"/>'
        )
    if kind == "sphere":
        r = part["size"][0] * scale
        return (
            f'<geom name="{geom_name}" type="sphere" size="{r:.4f}" '
            f'pos="{px:.4f} {py:.4f} {pz:.4f}"{euler_attr} rgba="{color}" mass="{mass:.4f}" '
            f'friction="1 0.05 0.01"/>'
        )
    hx, hy, hz = (v * scale for v in part["size"])
    return (
        f'<geom name="{geom_name}" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" '
        f'pos="{px:.4f} {py:.4f} {pz:.4f}"{euler_attr} rgba="{color}" mass="{mass:.4f}" '
        f'friction="1 0.05 0.01"/>'
    )


def _waste_xml(layout, seed: int = 7) -> str:
    """Emit one compound multi-geom body per waste item (realistic trash, not a bare cuboid)."""
    rng = np.random.default_rng(seed * 1000 + 1)
    chunks = []
    for i, (wtype, xy, yaw) in enumerate(layout):
        spec = WASTE_SPECS[wtype]
        name = f"waste_{i}"
        scale = float(rng.uniform(0.90, 1.15))
        dx, dy = rng.uniform(-0.05, 0.05, size=2)
        role_colors: dict[str, tuple] = {"base": base_color(rng, wtype)}
        geoms = []
        for pi, part in enumerate(spec["parts"]):
            role = part["role"]
            if role not in role_colors:
                role_colors[role] = accent_color(rng)
            geoms.append(_part_geom_xml(f"{name}_p{pi}", part, scale, role_colors[role]))
        chunks.append(
            f'<body name="{name}" pos="{xy[0] + dx:.4f} {xy[1] + dy:.4f} 0.001" euler="0 0 {yaw:.4f}">'
            f'<freejoint/>{"".join(geoms)}</body>'
        )
    return "\n".join(chunks)


def _static_world_xml(size: float) -> str:
    half = size / 2.0
    parts = [
        f'<geom name="grass" type="plane" size="{half+1:.2f} {half+1:.2f} 0.1" rgba="0.42 0.46 0.32 1"/>',
        '<geom name="plaza" type="box" size="3.75 3.75 0.01" pos="0 0 0.005" rgba="0.55 0.55 0.52 1" contype="0" conaffinity="0"/>',
        f'<geom name="path_x" type="box" size="{half:.2f} 0.80 0.008" pos="0 0 0.006" rgba="0.62 0.60 0.55 1" contype="0" conaffinity="0"/>',
        f'<geom name="path_y" type="box" size="0.80 {half:.2f} 0.008" pos="0 0 0.006" rgba="0.62 0.60 0.55 1" contype="0" conaffinity="0"/>',
        f'<geom name="wall_n" type="box" size="{half:.2f} 0.06 0.45" pos="0 {half:.2f} 0.45" rgba="0.72 0.70 0.66 1"/>',
        f'<geom name="wall_s" type="box" size="{half:.2f} 0.06 0.45" pos="0 {-half:.2f} 0.45" rgba="0.72 0.70 0.66 1"/>',
        f'<geom name="wall_e" type="box" size="0.06 {half:.2f} 0.45" pos="{half:.2f} 0 0.45" rgba="0.72 0.70 0.66 1"/>',
        f'<geom name="wall_w" type="box" size="0.06 {half:.2f} 0.45" pos="{-half:.2f} 0 0.45" rgba="0.72 0.70 0.66 1"/>',
        '<geom name="bldg1" type="box" size="1.20 1.10 1.40" pos="6.2 6.0 1.4" rgba="0.64 0.62 0.60 1"/>',
        '<geom name="bldg2" type="box" size="1.30 1.00 1.10" pos="-6.0 6.1 1.1" rgba="0.60 0.58 0.56 1"/>',
        '<geom name="bldg3" type="box" size="1.10 1.20 1.60" pos="6.4 -6.0 1.6" rgba="0.58 0.59 0.61 1"/>',
        '<geom name="bldg4" type="box" size="1.25 1.05 1.00" pos="-6.3 -5.8 1.0" rgba="0.62 0.60 0.58 1"/>',
    ]
    trees = [(-3.2, 3.4), (3.5, 3.2), (-3.6, -3.3), (3.4, -3.5), (-2.0, 6.5), (2.2, -6.6)]
    for i, (x, y) in enumerate(trees):
        parts.append(f'<geom name="trunk_{i}" type="cylinder" size="0.12 0.45" pos="{x} {y} 0.45" rgba="0.35 0.34 0.33 1"/>')
        parts.append(f'<geom name="canopy_{i}" type="sphere" size="0.55" pos="{x} {y} 1.15" rgba="0.35 0.36 0.34 1"/>')
    parts.append('<geom name="bench1" type="box" size="0.55 0.18 0.22" pos="-1.8 2.4 0.22" rgba="0.40 0.39 0.38 1"/>')
    parts.append('<geom name="bench2" type="box" size="0.18 0.55 0.22" pos="1.9 -2.3 0.22" rgba="0.40 0.39 0.38 1"/>')
    for i, (x, y) in enumerate([(4.2, 1.2), (-4.3, -1.0), (1.3, 4.4)]):
        parts.append(f'<geom name="bin_{i}" type="cylinder" size="0.18 0.35" pos="{x} {y} 0.35" rgba="0.15 0.15 0.16 1"/>')
        parts.append(f'<geom name="lid_{i}" type="cylinder" size="0.19 0.03" pos="{x} {y} 0.72" rgba="0.22 0.23 0.22 1"/>')
    parts.append('<geom name="planter1" type="box" size="0.225 0.225 0.18" pos="2.6 0.9 0.18" rgba="0.44 0.43 0.42 1"/>')
    parts.append('<geom name="planter2" type="box" size="0.225 0.225 0.18" pos="-2.7 -0.8 0.18" rgba="0.44 0.43 0.42 1"/>')
    parts.append('<camera name="overview" pos="9.5 -9.5 8.2" xyaxes="0.707 0.707 0 -0.40 0.40 0.82" fovy="48"/>')
    parts.append('<light pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.75" specular="0.3 0.3 0.3"/>')
    parts.append('<light pos="6 -4 7" dir="-0.3 0.2 -1" diffuse="0.35 0.35 0.32"/>')
    _ = plaza_obstacles  # occupancy list stays in park_plaza.py
    return "\n".join(parts)


def build_scene_xml(
    *,
    size: float,
    layout,
    start_xy=(0.0, 0.0),
    start_yaw: float = 0.0,
    pitch_deg: float = -28.0,
    fovy: float = 70.0,
    timestep: float = 0.0041667,
    decoys=(),
    seed: int = 7,
) -> str:
    robot = _robot_xml(start_xy, start_yaw, pitch_deg, fovy)
    waste = _waste_xml(layout, seed=seed)
    decoy = _decoy_xml(decoys)
    world = _static_world_xml(size)
    acts = "\n".join(
        f'<position name="act_{n}" joint="{n}" kp="90" dampratio="1.1" inheritrange="1"/>'
        for n in [f"joint_{k}" for k in range(1, 7)] + ["finger_left_joint", "finger_right_joint"]
    )
    return f"""
<mujoco model="waste_plaza">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep:.7f}" gravity="0 0 -9.81" integrator="Euler" noslip_iterations="2"/>
  <visual>
    <headlight ambient="0.45 0.45 0.42" diffuse="0.55 0.55 0.5"/>
    <global azimuth="135" elevation="-25"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.42 0.62 0.88" rgb2="0.90 0.94 0.98" width="256" height="256"/>
  </asset>
  <worldbody>
    {world}
    {robot}
    {waste}
    {decoy}
  </worldbody>
  <actuator>
    {acts}
  </actuator>
</mujoco>
"""
