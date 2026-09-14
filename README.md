# Smart Waste Collection Robot — End-to-End Simulation

Autonomous loop:

**virtual camera → your waste model (or the built-in mock) → 3D location → drive → 6-DOF arm → grasp → basket → search again**

This is a robotics architecture, not a canned animation. Every stage talks to the next through a small in-process bus that uses ROS 2 topic names, so later you can swap PyBullet for Gazebo and the simulated robot for the real one.

---

## 1. Platform choice (research summary)

| Platform | CV / camera | Arm | Mobile nav | ROS 2 | Python | Physics | Laptop | Demo ease | Sim-to-real |
|---|---|---|---|---|---|---|---|---|---|
| **ROS 2 + Gazebo Harmonic** | Excellent plugins | MoveIt 2 | Nav2 | Native | Yes | Strong | Medium, **Linux** | Medium | **Best** |
| NVIDIA Isaac Sim | Best photoreal RGB | Excellent | Good | Isaac ROS | Yes | Excellent | **Needs RTX + 32 GB** | Hard | Excellent vision |
| Webots | Good | Many built-in arms | Good | `webots_ros2` (WSL on Windows) | Yes | Good | Easy | Easy | Good |
| CoppeliaSim | Good | Excellent IK | OK | Via ROS | Yes | Good | Easy | Medium | Medium |
| **PyBullet** | RGB + depth + seg API | IK + constraints | You implement | Manual bridge | **Best** | Good enough | **Easiest** | Easy | Architecture-level |
| Unity / Unreal | Beautiful | Game IK | Game AI | Poor robotics | C#/Blueprints | Game physics | Heavy | Pretty ≠ robotics | Weak |

### Recommendation for *this* capstone, on a Windows student laptop

**Now:** **MuJoCo + Python**, with ROS 2-shaped modules.  
**Next (physical robot):** same nodes on **Ubuntu + ROS 2 + Gazebo or the real drivers**. Optionally **Webots** if the team stays on Windows and installs WSL2.

**Why not ROS 2 + Gazebo as the first runnable demo?** Official ROS 2 on Windows is not a comfortable student path; Gazebo + Nav2 + MoveIt 2 is an Ubuntu stack. Starting there usually means a week of install issues and no demo.

**Why not Isaac Sim?** Photoreal cameras are ideal for your detector, but a typical laptop GPU will not run it. NVIDIA’s own guidance is roughly RTX-class GPU, ~32 GB RAM, tens of GB of disk.

**Why MuJoCo (not PyBullet) on this laptop?**

- PyPI `pybullet` has no Windows wheel for Python 3.12 and needs MSVC to compile
- `pip install mujoco` installs a binary wheel (already verified on this machine)
- Virtual RGB-D camera is MuJoCo’s offscreen renderer — **no webcam**
- 6-DOF Jacobian IK + kinematic grasp attach
- OpenGL 3D viewer plus an OpenCV **VIRTUAL CAMERA | AI DETECTION** window

**6-DOF for the car?** A wheeled ground robot should **not** fly with 6-DOF. This project uses **planar mobility (x, y, yaw)** with **4-wheel skid-steer / differential drive**. The **arm** is the 6-DOF system.

---

## 2. Architecture

```
Virtual environment (MuJoCo plaza)
        ↓
Virtual camera (RGB-D on the mast)
        ↓
Cheap local color trigger  ← "something's there" (also the whole 'mock' backend)
        ↓
Snapshot → Gemini vision (garbage_detected / type / confidence / bbox)
        ↓
Coordinate estimator (bbox → camera raycast → world XYZ, depth as fallback)
        ↓
Navigator (A* on static map + lidar avoidance)
        ↓
Mobile base (cmd_vel → wheel velocities)
        ↓
6-DOF arm (staged per-joint IK: align → reach → descend → fine-align → lift → deposit)
        ↓
Onboard basket (physics drop)
```

State machine: `IDLE → PATROLLING → POTENTIAL_GARBAGE_FOUND → CAPTURING_IMAGE → GEMINI_ANALYSIS → APPROACHING_TARGET → ARM_POSITIONING → GRABBING → LIFTING → MOVING_TO_BIN → RELEASING → ARM_RESET → PATROLLING`

The robot patrols a waypoint loop rather than sitting still; a cheap color-blob check gates when an (API-billed) Gemini vision call actually fires, rather than calling it every frame. See `waste_robot/gemini_detector.py` for the prompt and `waste_robot/arm.py` for the staged 6-DOF motion.

---

## 3. Hardware requirements (MuJoCo demo)

| | Minimum | Comfortable demo |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 6+ cores |
| GPU | none (Tiny renderer) | any OpenGL GPU |
| Disk | 1 GB | 2 GB |
| OS | Windows 10/11, Ubuntu 22.04+, macOS | same |
| Python | 3.10–3.12 | 3.12 (this repo was started on 3.12) |

Isaac Sim comparison: plan for an NVIDIA RTX GPU, 16–32 GB RAM, 40+ GB disk. Use it later for photoreal data if the lab has a workstation.

---

## 4. Install and run

In PowerShell, from this folder:

```powershell
cd "C:\Users\Mohammed Faheem P\OneDrive\Desktop\CapstoneProject"
python -m pip install -r requirements.txt
python tests\test_pipeline.py
python scripts\run_simulation.py --scenario multi
```

What you should see:

1. A 3D plaza (grass, paths, buildings, trees, bins, scattered waste)
2. An orange 4-wheel robot with a blue 6-DOF arm and a rear basket
3. A second window **VIRTUAL CAMERA | AI DETECTION** with bounding boxes
4. Status text: state, target, distance, arm, collected count, battery
5. The robot turns/drives, stops near waste, moves the arm, grips, drops into the basket
6. A mission report printed in the terminal when it finishes or you press **S** / **Q**

### Scenarios

```powershell
python scripts\run_simulation.py --scenario single
python scripts\run_simulation.py --scenario multi
python scripts\run_simulation.py --scenario obstacle
python scripts\run_simulation.py --scenario types
python scripts\run_simulation.py --scenario false_detection
python scripts\run_simulation.py --scenario failed_pickup
```

Headless (no windows, for testing):

```powershell
python scripts\run_simulation.py --scenario single --headless --max-seconds 30 --collect-limit 1
python scripts\run_tests.py
```

### Common errors

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: mujoco` | `python -m pip install -r requirements.txt` |
| Black camera / crash in DIRECT | N/A; `--headless` still uses MuJoCo’s offscreen renderer |
| OpenCV window does not appear | GPU/display issue; still OK with `--headless` |
| Robot does not see waste | Confirm waste is in front; lower `detector.confidence_threshold` |
| Arm misses | IK timeout; check the camera depth target in the HUD extra line |
| `existing` backend FileNotFound | Copy weights to `models/waste_detector.pt` or stay on `mock` |
| `GEMINI_API_KEY is not set` | Put the key in `.env` as `GEMINI_API_KEY=...` then rerun |

---

## 5. Plug in your existing AI model

Do **not** replace the pipeline. Implement `detect(image)` and switch config.

**Option A — Gemini vision (default)**

A cheap local color trigger flags "something's there," then a virtual-camera
snapshot goes to Gemini, which returns structured JSON: `garbage_detected`,
`garbage_type`, `confidence`, a fractional `bounding_box`, and a
`description` (see `waste_robot/gemini_detector.py:GEMINI_PROMPT`). Requires
`GEMINI_API_KEY` in `.env`.

```yaml
detector:
  backend: gemini
  gemini_model: gemini-3.6-flash
```

**Option B — Ultralytics YOLO `.pt`**

1. Copy weights to `models/waste_detector.pt`
2. `pip install ultralytics`
3. In `config/simulation.yaml`:

```yaml
detector:
  backend: existing
  model_path: models/waste_detector.pt
```

**Option B — your Python class**

```yaml
detector:
  backend: existing
  existing_model_module: waste_robot.existing_model_example:YourWasteModel
```

See `src/waste_robot/existing_model_example.py`. Input is an **RGB numpy frame from the simulator**, never a physical webcam.

Screen-share demo: share the OpenCV **VIRTUAL CAMERA** window. That window is the virtual camera feed with boxes, which is the correct academic demonstration of “screen input” without using a real camera.

---

## 6. Project layout

```
CapstoneProject/
├── config/simulation.yaml
├── simulation/robots/waste_collector.urdf
├── simulation/worlds/park_plaza.py
├── simulation/assets/waste.py
├── src/waste_robot/          # detector, coords, nav, arm, FSM, dashboard
├── scripts/run_simulation.py
├── tests/test_pipeline.py
├── ros2_ws/src/README.md     # how to promote Bus topics to ROS 2
├── models/                   # drop your .pt here
└── docs/ASSETS.md
```

---

## 7. What is simplified in v1 (and how to upgrade)

| v1 | Later |
|---|---|
| Color mock detector | Your CNN/YOLO weights |
| A* on a known obstacle map + lidar reflex | Nav2 + costmaps + AMCL |
| MuJoCo Jacobian IK, staged per-joint interpolation | MoveIt 2 |
| Grasp = proximity + fixed constraint | Contact-rich grasp / suction |
| Procedural compound-geom trash (no mesh assets) | Photoreal Isaac/Gazebo meshes |
| In-process `Bus` | `rclpy` publishers |

These are replacement boundaries, not a rewrite.

---

## 8. Metrics

Every run writes `outputs/mission_<scenario>.json` and prints:

```
========== MISSION REPORT ==========
Waste detected / collected, pickup success, distance, time, collisions, detection FPS
```

---

## 9. License

Project code: MIT (`LICENSE`). Third-party runtime libraries: see `docs/ASSETS.md`.
