"""Perception backend contract and shared detection result format."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from PIL import Image, ImageDraw, ImageFont


@dataclass
class BBox:
    """A 2D object bounding box with optional 3D pose information."""

    label: str
    xyxy: Tuple[float, float, float, float]
    confidence: Optional[float] = None
    center_3d: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center_xy(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def contains_xy(self, xy: Tuple[float, float]) -> bool:
        x, y = xy
        x1, y1, x2, y2 = self.xyxy
        return x1 <= x <= x2 and y1 <= y <= y2

    def to_object_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [int(round(v)) for v in self.xyxy],
            "label": self.label,
            "score": float(self.confidence) if self.confidence is not None else 1.0,
            "center_point": [int(round(v)) for v in self.center_xy],
            "mask": None,
            **self.metadata,
        }


@dataclass
class DetectionResult:
    """Unified detection result format compatible with ObjectDetector output."""

    success: bool
    objects: List[Dict[str, Any]]
    image_shape: Tuple[int, int]
    raw_output: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None

    def get_object_bbox(self, offset: Sequence[int] = (0, 0, 0, 0)) -> List[int]:
        if not self.objects:
            return []
        bbox = list(max(self.objects, key=lambda obj: obj["score"])["bbox"])
        bbox[0] -= offset[0]
        bbox[1] -= offset[1]
        bbox[2] += offset[2]
        bbox[3] += offset[3]
        return [int(v) for v in bbox]

    def get_object_center(self) -> Optional[List[int]]:
        if not self.objects:
            return None
        best_obj = max(self.objects, key=lambda obj: obj["score"])
        return best_obj.get("center_point")

    def set_random_obj_point(self, margin_ratio: float = 0.8) -> None:
        for obj in self.objects:
            bbox = obj["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            radius = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 * margin_ratio
            r = radius * math.sqrt(random.random())
            theta = 2 * math.pi * random.random()
            obj["center_point"] = [
                int(center_x + r * math.cos(theta)),
                int(center_y + r * math.sin(theta)),
            ]

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    @property
    def get_all_labels(self) -> List[str]:
        return [obj["label"] for obj in self.objects]

    @property
    def output_to_json(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "objects": [
                {
                    "bounding_box": obj["bbox"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "center_point": obj.get("center_point"),
                }
                for obj in self.objects
            ],
            "image_shape": self.image_shape,
        }
        if self.summary:
            result["summary"] = self.summary
        return result

    def to_bboxes(self) -> Dict[str, List[BBox]]:
        grouped: Dict[str, List[BBox]] = {}
        for obj in self.objects:
            bbox = obj["bbox"]
            grouped.setdefault(obj["label"], []).append(
                BBox(
                    label=obj["label"],
                    xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=float(obj.get("score", 1.0)),
                    metadata={"center_point": obj.get("center_point"), "mask": obj.get("mask")},
                )
            )
        return grouped


@runtime_checkable
class PerceptionBackend(Protocol):
    """Detect named objects in an RGB image."""

    def detect(self, image: Image.Image, object_names: Sequence[str]) -> DetectionResult:
        """Return a unified DetectionResult."""


class BasePerceptionBackend:
    """Shared output helpers for concrete perception backends."""

    def __init__(self, output_dir: str = "output/perception", session_name: Optional[str] = None) -> None:
        timestamp = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(self.session_output_dir, exist_ok=True)

    def save_detection(
        self,
        image: Image.Image,
        detection_result: DetectionResult,
        save_dir: Optional[str] = None,
    ) -> str:
        if not save_dir:
            clean_name = "_".join(detection_result.get_all_labels).replace(" ", "_")
            clean_name = clean_name or "no_detection"
            save_dir = os.path.join(self.session_output_dir, f"{clean_name}_result")
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "detection_result.json"), "w") as f:
            json.dump(detection_result.output_to_json, f, indent=2)

        self._save_visualization(
            image,
            [obj["bbox"] for obj in detection_result.objects],
            [obj["label"] for obj in detection_result.objects],
            [obj["score"] for obj in detection_result.objects],
            os.path.join(save_dir, "visualization.png"),
            "labeled",
            center_points=[obj.get("center_point") for obj in detection_result.objects],
        )

        image.copy().save(os.path.join(save_dir, "original_img.png"))
        return save_dir

    def _save_visualization(
        self,
        image: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        scores: Sequence[float],
        output_path: str,
        viz_type: str = "detection",
        center_points: Optional[List[Optional[Sequence[float]]]] = None,
    ) -> None:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial", 20)
        except OSError:
            font = ImageFont.load_default()

        color = (255, 0, 0) if viz_type in {"detection", "labeled"} else (0, 255, 0)
        show_index = viz_type == "detection"
        show_label = viz_type == "labeled"

        if center_points is None:
            center_points = [None] * len(boxes)
        elif len(center_points) < len(boxes):
            center_points.extend([None] * (len(boxes) - len(center_points)))

        for idx, (box, label, score, center_point) in enumerate(zip(boxes, labels, scores, center_points)):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            if show_index:
                draw.text((x1, y1 - 20), f"[{idx}] {score:.2f}", fill=color)
            elif show_label:
                text_bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                draw.rectangle([x1, y1 - th - 4, x1 + tw, y1], fill=color)
                draw.text((x1, y1 - th - 4), label, fill=(255, 255, 255), font=font)

            if center_point:
                cx, cy = int(round(center_point[0])), int(round(center_point[1]))
                radius = 5
                draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    fill=(0, 255, 0),
                    outline=(0, 0, 0),
                )

        img.save(output_path)


class EmptyPerceptionBackend(BasePerceptionBackend):
    """Detector stub for dry-run execution and tests."""

    def detect(self, image: Image.Image, object_names: Sequence[str]) -> DetectionResult:
        width, height = image.size
        return DetectionResult(
            success=False,
            objects=[],
            image_shape=(height, width),
            summary={"method": "empty", "requested_objects": list(object_names)},
        )
