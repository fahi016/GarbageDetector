#!/usr/bin/env python3
"""Launch the smart waste-collection robot simulation.

Examples:
  python scripts/run_simulation.py
  python scripts/run_simulation.py --scenario single
  python scripts/run_simulation.py --scenario obstacle --headless --max-seconds 40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from waste_robot.config import load_config
from waste_robot.sim import build_sim, step_physics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart waste collection robot simulation")
    p.add_argument(
        "--scenario",
        default="multi",
        choices=["single", "multi", "obstacle", "types", "false_detection", "failed_pickup"],
    )
    p.add_argument("--config", default=str(ROOT / "config" / "simulation.yaml"))
    p.add_argument("--headless", action="store_true")
    p.add_argument("--gui", action="store_true", help="Force GUI even if yaml says otherwise")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--collect-limit", type=int, default=None)
    p.add_argument("--detector", choices=["mock", "existing", "oracle", "gemini"], default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_seconds is not None:
        cfg["simulator"]["max_mission_seconds"] = args.max_seconds
    if args.collect_limit is not None:
        cfg["simulator"]["collect_limit"] = args.collect_limit
    if args.detector:
        cfg["detector"]["backend"] = args.detector

    print("Smart Waste Collection Robot — simulation")
    print(f"Scenario: {args.scenario}")
    print("Keys:  S = stop mission,  Q = quit  (camera window)")
    print("Windows: VIRTUAL CAMERA | AI DETECTION, PLAZA OVERVIEW, and the 3D MuJoCo viewer.")
    ctx = build_sim(cfg, scenario=args.scenario, gui=True if args.gui else None, headless=args.headless)
    mission = ctx.mission
    dt = float(cfg["simulator"]["timestep"])
    control_every = max(1, int(round((1.0 / float(cfg["simulator"]["control_hz"])) / dt)))
    step = 0
    t = 0.0
    try:
        while mission.running:
            step_physics(ctx.engine)
            if step % control_every == 0:
                mission.tick(t)
            step += 1
            t += dt
            if cfg["simulator"].get("realtime") and not args.headless:
                time.sleep(dt * 0.15)
    except KeyboardInterrupt:
        mission.stop()
    finally:
        mission.metrics.mission_time = time.time() - mission.metrics.start_time
        mission.metrics.distance_travelled = mission.base.distance_travelled
        report = mission.metrics.report_text()
        print(report)
        out = ROOT / cfg["metrics"]["output_dir"] / f"mission_{args.scenario}.json"
        mission.metrics.save(out)
        print(f"Saved metrics to {out}")
        mission.dashboard.close()
        ctx.engine.close()
        if getattr(mission.arm, "bridge", None) is not None:
            mission.arm.bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
