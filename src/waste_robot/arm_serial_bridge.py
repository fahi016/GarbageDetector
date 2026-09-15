"""Streams ArmController's joint trajectory to a real ESP32-driven 6-DOF
arm over Serial, in real time.

Wire-format and per-servo limits mirror firmware/esp32_arm_controller --
keep the two in sync if either changes. The ESP32 does not smooth or
re-plan anything it receives; it clamps to its own limits and writes each
frame to the servos immediately, so the *rate* at which frames are sent
here is effectively the timing resolution of the real-hardware motion.
"""

from __future__ import annotations

import math
import time

import serial

# Degrees added to a sim joint angle (radians, 0 = home) to get the servo
# angle in degrees, for servos 1-5 ("zero position for servo 1-5 is 90 deg").
SERVO_ZERO_DEG = 90.0

# Per-servo physical limits -- order: base, shoulder, elbow, wrist,
# wristRot, gripper. Sent angles are clamped to these before transmission
# as a first line of defense; the ESP32 clamps independently as the second.
SERVO_MIN = (2, 0, 0, 45, 0, 90)
SERVO_MAX = (179, 180, 180, 145, 180, 180)

# Gripper finger opening (meters, from ArmController._set_fingers) at the
# two servo extremes: 90 deg is open, 180 deg is fully closed.
GRIPPER_OPEN_M = 0.04
GRIPPER_CLOSED_M = 0.0


def joints_to_servo_degrees(q_rad: list[float], gripper_opening_m: float) -> list[int]:
    """Convert 5 sim joint angles (radians) + gripper opening (meters) into
    6 clamped, integer servo angles (degrees) in wire order."""
    degrees = [SERVO_ZERO_DEG + math.degrees(q) for q in q_rad[:5]]
    span = GRIPPER_OPEN_M - GRIPPER_CLOSED_M
    frac_open = 0.0 if span == 0 else (gripper_opening_m - GRIPPER_CLOSED_M) / span
    frac_open = min(1.0, max(0.0, frac_open))
    degrees.append(180.0 - frac_open * 90.0)
    return [int(round(min(SERVO_MAX[i], max(SERVO_MIN[i], degrees[i])))) for i in range(6)]


class ArmSerialBridge:
    """Opens the ESP32 serial link and forwards arm poses as `<a,b,c,d,e,f>`
    frames, throttled to `max_hz` so the sim's high-rate interpolation
    doesn't flood the UART faster than the servos can usefully track."""

    def __init__(self, port: str, baud: int = 115200, max_hz: float = 50.0) -> None:
        self._ser = serial.Serial(port, baud, timeout=0)
        self._min_interval = 1.0 / max_hz
        self._last_send = 0.0
        time.sleep(2.0)  # let the ESP32 finish its boot reset after the port opens

    def send(self, q_rad: list[float], gripper_opening_m: float) -> None:
        now = time.monotonic()
        if now - self._last_send < self._min_interval:
            return
        self._last_send = now
        degs = joints_to_servo_degrees(q_rad, gripper_opening_m)
        frame = "<" + ",".join(str(d) for d in degs) + ">\n"
        self._ser.write(frame.encode("ascii"))

    def close(self) -> None:
        self._ser.close()
