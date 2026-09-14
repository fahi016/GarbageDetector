from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MissionMetrics:
    waste_detected: int = 0
    waste_collected: int = 0
    failed_grasps: int = 0
    collisions: int = 0
    distance_travelled: float = 0.0
    mission_time: float = 0.0
    navigation_time: float = 0.0
    pickup_times: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    detection_frames: int = 0
    detection_seconds: float = 0.0
    gemini_confirmed: int = 0
    gemini_rejected: int = 0
    start_time: float = field(default_factory=time.time)

    def detection_fps(self) -> float:
        if self.detection_seconds <= 0:
            return 0.0
        return self.detection_frames / self.detection_seconds

    def pickup_success_rate(self) -> float:
        attempts = self.waste_collected + self.failed_grasps
        if attempts == 0:
            return 0.0
        return 100.0 * self.waste_collected / attempts

    def report_text(self) -> str:
        avg_pick = sum(self.pickup_times) / len(self.pickup_times) if self.pickup_times else 0.0
        mean_conf = sum(self.confidences) / len(self.confidences) if self.confidences else 0.0
        mins = int(self.mission_time // 60)
        secs = int(self.mission_time % 60)
        return (
            "========== MISSION REPORT ==========\n"
            f"Waste detected:       {self.waste_detected}\n"
            f"Waste collected:      {self.waste_collected}\n"
            f"Pickup success rate:  {self.pickup_success_rate():.0f}%\n"
            f"Failed grasps:        {self.failed_grasps}\n"
            f"Mean confidence:      {mean_conf:.2f}\n"
            f"Avg pickup time:      {avg_pick:.1f} s\n"
            f"Distance travelled:   {self.distance_travelled:.1f} m\n"
            f"Navigation time:      {self.navigation_time:.1f} s\n"
            f"Mission time:         {mins}m {secs:02d}s\n"
            f"Collisions:           {self.collisions}\n"
            f"Gemini confirmed:     {self.gemini_confirmed}\n"
            f"Gemini rejected:      {self.gemini_rejected}\n"
            f"Detection FPS:        {self.detection_fps():.1f} FPS\n"
            "====================================\n"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
