"""Person detector: RF-DETR Small (Apache-2.0 code and COCO weights) behind one call."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COCO_PERSON = 1  # RF-DETR uses the 1-based COCO ids


@dataclass(frozen=True)
class Detection:
    box_uv: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    conf: float

    @property
    def foot_uv(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box_uv
        return (x1 + x2) / 2.0, y2


class PersonDetector:
    def __init__(self, threshold: float = 0.5) -> None:
        from rfdetr import RFDETRSmall

        self.threshold = threshold
        self.model = RFDETRSmall()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        det = self.model.predict(rgb, threshold=self.threshold)
        out: list[Detection] = []
        for box, cid, conf in zip(det.xyxy, det.class_id, det.confidence, strict=True):
            if int(cid) == COCO_PERSON:
                out.append(Detection(tuple(float(v) for v in box), float(conf)))  # type: ignore[arg-type]
        return out
