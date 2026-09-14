"""In-process pub/sub used as a ROS 2 topic analog.

Later, replace Bus.publish with rclpy publishers using the same topic names.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


TopicCallback = Callable[[Any], None]


class Bus:
    TOPICS = {
        "camera/image_raw": "sensor_msgs/Image",
        "camera/depth": "sensor_msgs/Image",
        "detections": "waste_msgs/DetectionArray",
        "waste_pose": "geometry_msgs/PoseStamped",
        "cmd_vel": "geometry_msgs/Twist",
        "joint_states": "sensor_msgs/JointState",
        "robot_state": "std_msgs/String",
        "metrics": "waste_msgs/MissionMetrics",
    }

    def __init__(self) -> None:
        self._subs: dict[str, list[TopicCallback]] = defaultdict(list)
        self.latest: dict[str, Any] = {}

    def publish(self, topic: str, msg: Any) -> None:
        self.latest[topic] = msg
        for cb in self._subs.get(topic, []):
            cb(msg)

    def subscribe(self, topic: str, callback: TopicCallback) -> None:
        self._subs[topic].append(callback)
