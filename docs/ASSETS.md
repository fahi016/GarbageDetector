# Open-source assets and licenses

This simulation **does not download third-party meshes**. The plaza, robot,
arm, basket, and waste are original primitive geometry (boxes/cylinders/spheres)
so the academic project has a clear license (MIT, see `/LICENSE`).

## Software used at runtime

| Name | URL | License | Why | Install | Simulator |
|---|---|---|---|---|---|
| MuJoCo | https://github.com/google-deepmind/mujoco | Apache-2.0 | Physics, MJCF robot, virtual RGB-D camera, 3D viewer | `pip install mujoco` | This project |
| NumPy | https://numpy.org | BSD | Geometry and image arrays | `pip install numpy` | any |
| OpenCV | https://opencv.org | Apache-2.0 | Detection overlay + dashboard | `pip install opencv-python` | any |
| PyYAML | https://pyyaml.org | MIT | Config files | `pip install PyYAML` | any |
| Python 3.10–3.12 | https://python.org | PSF | Language | installer | any |

## Evaluated but not bundled (legal to use later)

| Name | URL | License | Notes |
|---|---|---|---|
| Webots | https://github.com/cyberbotics/webots | Apache-2.0 | Excellent on Windows; ROS 2 still wants WSL |
| Gazebo Harmonic | https://gazebosim.org | Apache-2.0 | Best ROS 2 pairing; use on Ubuntu |
| Nav2 | https://github.com/ros-navigation/navigation2 | Apache-2.0 | Full autonomy stack |
| MoveIt 2 | https://github.com/moveit/moveit2 | BSD | Arm planning |
| Universal Robots UR5 description | https://github.com/UniversalRobots/Universal_Robots_ROS2_Description | BSD-3-Clause | Real 6-DOF arm URDF when you add ROS 2 |
| PyBullet | https://github.com/bulletphysics/bullet3 | zlib | Alternative engine; no Python 3.12 Windows pip wheel | source/conda | Evaluated |
| ROS 2 | https://docs.ros.org | Apache-2.0 | Physical-robot middleware |
| Ultralytics YOLO | https://github.com/ultralytics/ultralytics | AGPL-3.0 (watch license) | Optional wrapper for `detector.backend: existing` |

Do **not** scrape 3D marketplace models without a license file. For photoreal
waste meshes later, prefer:

- YCB object set (https://www.ycbbenchmarks.com) — research use, check their terms
- Google Scanned Objects (Apache-2.0) if you need scanned bottles/cans
