"""OpenCV dashboard: virtual camera + detections + robot status."""

from __future__ import annotations

import numpy as np

from waste_robot.detector import Detection
from waste_robot.state_machine import RobotState


def draw_detections(rgb: np.ndarray, detections: list[Detection], threshold: float) -> np.ndarray:
    import cv2

    vis = rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = (40, 220, 40) if det.confidence >= threshold else (40, 40, 200)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.0%}"
        cv2.putText(vis, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


def compose_hud(
    camera_bgr: np.ndarray,
    state: RobotState,
    target: str,
    distance: float,
    arm: str,
    collected: int,
    battery: float,
    extra: str = "",
) -> np.ndarray:
    import cv2

    h, w = camera_bgr.shape[:2]
    panel_h = 140
    canvas = np.zeros((h + panel_h, max(w, 520), 3), dtype=np.uint8)
    canvas[:h, :w] = camera_bgr
    cv2.putText(canvas, "VIRTUAL CAMERA  |  AI DETECTION", (12, h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    lines = [
        f"Robot Status: {state.value}",
        f"Target: {target}    Distance: {distance:.2f} m",
        f"Arm Status: {arm}    Waste Collected: {collected}",
        f"Battery: {battery:.0f}%",
        extra,
    ]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (12, h + 46 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 255), 1)
    return canvas


class Dashboard:
    def __init__(self, enabled: bool = True, window: str = "VIRTUAL CAMERA | AI DETECTION"):
        self.enabled = enabled
        self.window = window
        self.world_window = "PLAZA OVERVIEW"
        self._ready = False
        self._quit = False

    def consume_quit(self) -> bool:
        flag = self._quit
        self._quit = False
        return flag

    def show(
        self,
        image_rgb: np.ndarray,
        detections,
        threshold,
        state,
        target,
        distance,
        arm,
        collected,
        battery,
        extra="",
        world_rgb=None,
    ):
        if not self.enabled:
            return
        import cv2

        vis = draw_detections(image_rgb, detections, threshold)
        bgr = vis[:, :, ::-1]
        hud = compose_hud(bgr, state, target, distance, arm, collected, battery, extra)
        cv2.imshow(self.window, hud)
        if world_rgb is not None:
            cv2.imshow(self.world_window, world_rgb[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), ord("s"), ord("S"), 27):
            self._quit = True
        self._ready = True

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
