# ROS 2 upgrade path (not required to run the current demo)

The running simulation uses an in-process `Bus` with ROS 2 *topic names*.
When you move to Ubuntu + ROS 2 Humble/Jazzy, map those names to real messages:

| Internal topic | ROS 2 type | Notes |
|---|---|---|
| `camera/image_raw` | `sensor_msgs/Image` | Virtual camera RGB, later a real camera |
| `camera/depth` | `sensor_msgs/Image` | Linearized depth |
| `detections` | custom or `vision_msgs/Detection2DArray` | Output of *your* model |
| `waste_pose` | `geometry_msgs/PoseStamped` | After deprojection + TF |
| `cmd_vel` | `geometry_msgs/Twist` | Differential-drive command |
| `joint_states` | `sensor_msgs/JointState` | Arm + wheels |
| `robot_state` | `std_msgs/String` | FSM state |

Suggested packages to add later (do **not** install these on Windows for the first demo):

- ROS 2 Humble or Jazzy on Ubuntu 22.04/24.04
- Gazebo Harmonic (`ros_gz`) **or** Webots (`webots_ros2`)
- Nav2 for the mobile base
- MoveIt 2 for the arm
- `tf2_ros` for camera → base → map

Recommended physical-robot launch split:

```
ros2 launch waste_bringup sim.launch.py      # MuJoCo/Gazebo
ros2 launch waste_bringup robot.launch.py    # real drivers
```

Both launches should start the same `waste_detector`, `coordinate_estimator`,
and state-machine nodes. Only the camera and actuator drivers change.
