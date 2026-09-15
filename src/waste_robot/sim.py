"""Simulation bootstrap: physics, world, robot, sensors, mission."""

from __future__ import annotations

from waste_robot.arm import ArmController
from waste_robot.bus import Bus
from waste_robot.camera import VirtualCamera
from waste_robot.dashboard import Dashboard
from waste_robot.detector import build_detector
from waste_robot.engine import Engine, SimContext
from waste_robot.mobile_base import MobileBase
from waste_robot.mission import Mission
from waste_robot.navigation import Navigator
from waste_robot.robot import RobotModel
from simulation.assets.waste import WasteItem, decoy_layout, default_layout
from simulation.scene import build_scene_xml
from simulation.worlds.park_plaza import plaza_obstacles


def build_sim(cfg: dict, scenario: str = "multi", gui: bool | None = None, headless: bool = False) -> SimContext:
    if gui is None:
        gui = bool(cfg["simulator"]["gui"]) and not headless
    start = tuple(cfg["robot"]["start_xy"])
    yaw = float(cfg["robot"]["start_yaw_deg"]) * 3.14159265 / 180.0
    layout = default_layout(scenario, seed=int(cfg["world"]["seed"]))
    decoys_spec = decoy_layout(scenario)
    xml = build_scene_xml(
        size=float(cfg["world"]["size"]),
        layout=layout,
        start_xy=start,
        start_yaw=yaw,
        pitch_deg=float(cfg["camera"]["pitch_deg"]),
        fovy=float(cfg["camera"]["fov_deg"]),
        timestep=float(cfg["simulator"]["timestep"]),
        decoys=decoys_spec,
        seed=int(cfg["world"]["seed"]),
    )
    engine = Engine(
        xml,
        gui=gui,
        cam_width=int(cfg["camera"]["width"]),
        cam_height=int(cfg["camera"]["height"]),
        overview=gui,
    )
    robot = RobotModel(engine)
    waste = []
    for i, (wtype, _xy, _yaw) in enumerate(layout):
        name = f"waste_{i}"
        item = WasteItem(name=name, waste_type=wtype)
        item.body_id = engine.body_id(name)
        waste.append(item)

    decoys = []
    for i, (dtype, _xy, _yaw) in enumerate(decoys_spec):
        name = f"decoy_{i}"
        item = WasteItem(name=name, waste_type=dtype)
        item.body_id = engine.body_id(name)
        decoys.append(item)

    base = MobileBase(engine, cfg["robot"])
    bridge = None
    serial_port = cfg["arm"].get("serial_port")
    if serial_port:
        from waste_robot.arm_serial_bridge import ArmSerialBridge

        bridge = ArmSerialBridge(
            serial_port,
            baud=int(cfg["arm"].get("serial_baud", 115200)),
            max_hz=float(cfg["arm"].get("serial_max_hz", 50)),
        )
    arm = ArmController(engine, robot, cfg["arm"], bridge=bridge)
    camera = VirtualCamera(engine, cfg["camera"])
    force_low = scenario == "false_detection"
    detector = build_detector(cfg["detector"], force_low_confidence=force_low)
    nav = Navigator(base, plaza_obstacles(), cfg["robot"])
    bus = Bus()
    dash = Dashboard(enabled=not headless)
    mission = Mission(
        engine, robot, base, arm, camera, detector, nav, waste, cfg, bus, dash, scenario=scenario, headless=headless
    )
    mission.decoys = decoys
    for _ in range(48):
        engine.step()
    arm.go_home(20)
    return SimContext(engine=engine, mission=mission, robot=robot)


def step_physics(engine: Engine) -> None:
    engine.step()
