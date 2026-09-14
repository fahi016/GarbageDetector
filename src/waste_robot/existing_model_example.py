"""Drop-in example for connecting a real waste detector.

1. Copy your weights to models/waste_detector.pt  (YOLO)
   OR implement detect() below and set in config:

     detector:
       backend: existing
       existing_model_module: waste_robot.existing_model_example:YourWasteModel

2. Restart the simulation. No other files need to change.
"""

from __future__ import annotations

import numpy as np


class YourWasteModel:
    def __init__(self):
        # Load torch/keras/onnx weights here.
        pass

    def detect(self, image: np.ndarray) -> list[dict]:
        """Must accept an RGB uint8 frame from the virtual camera.

        Return a list of dicts:
          {"class_name": str, "confidence": float, "bbox": (x1, y1, x2, y2)}
        """
        raise NotImplementedError("Plug your model into detect().")
