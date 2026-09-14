#!/usr/bin/env python3
"""Headless scenario checks used for development and CI-style demos."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from waste_robot.config import load_config
from waste_robot.sim import build_sim, step_physics
from waste_robot.state_machine import RobotState


def run_scenario(name: str, seconds: float, collect_limit: int = 1) -> dict:
    cfg = load_config()
    cfg["detector"]["backend"] = "mock"
    cfg["simulator"]["max_mission_seconds"] = seconds
    cfg["simulator"]["collect_limit"] = collect_limit
    cfg["simulator"]["realtime"] = False
    ctx = build_sim(cfg, scenario=name, gui=False, headless=True)
    mission = ctx.mission
    dt = float(cfg["simulator"]["timestep"])
    control_every = max(1, int(round((1.0 / 30.0) / dt)))
    step = 0
    t = 0.0
    t_end = time.time() + seconds
    try:
        while mission.running and time.time() < t_end:
            step_physics(ctx.engine)
            if step % control_every == 0:
                mission.tick(t)
            step += 1
            t += dt
    finally:
        ctx.engine.close()
    return {
        "scenario": name,
        "state": mission.state.value,
        "collected": mission.metrics.waste_collected,
        "detected": mission.metrics.waste_detected,
        "failed_grasps": mission.metrics.failed_grasps,
        "report": mission.metrics.report_text(),
    }


def main() -> int:
    tests = [
        ("single", 25.0, 1),
        ("false_detection", 8.0, 1),
        ("failed_pickup", 30.0, 1),
        ("types", 55.0, 2),
        ("obstacle", 40.0, 1),
    ]
    failed = 0
    for name, sec, lim in tests:
        print(f"\n=== TEST {name} ({sec}s) ===")
        result = run_scenario(name, sec, lim)
        print(result["report"])
        if name == "false_detection":
            if result["collected"] > 0:
                print("FAIL: false_detection collected waste")
                failed += 1
            else:
                print("PASS: ignored low-confidence detections / no collection")
        elif name == "failed_pickup":
            if result["failed_grasps"] < 1:
                print("FAIL: expected at least one failed grasp")
                failed += 1
            elif result["collected"] < 1:
                print("FAIL: robot did not recover after a missed grasp")
                failed += 1
            else:
                print("PASS: missed grasp then recovered")
        else:
            print("Info: collected=", result["collected"], "detected=", result["detected"], "state=", result["state"])
            if result["collected"] < 1:
                print(f"FAIL: {name} did not collect waste")
                failed += 1
            else:
                print(f"PASS: {name} collected {result['collected']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
