"""YOLO-World v2 perception backend."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence

import yaml
from PIL import Image

from agenticlab_human.perception.backend.perception_backend import (
    BasePerceptionBackend,
    DetectionResult,
)


class YoloWorldDetector(BasePerceptionBackend):
    """Ultralytics detector that supports YOLO-World and regular YOLO models."""

    def __init__(
        self,
        model_path: str = "yolov8s-worldv2.pt",
        model_type: str = "auto",
        confidence: float = 0.05,
        image_size: int = 640,
        output_dir: str = "output/perception/yolo_world",
        session_name: Optional[str] = None,
    ) -> None:
        super().__init__(output_dir=output_dir, session_name=session_name)
        self.model_path = model_path
        self.model_type = model_type
        self.confidence = confidence
        self.image_size = image_size
        self._model = None
        self._is_world_model = self._resolve_is_world_model()

    def detect(self, image: Image.Image, object_names: Sequence[str]) -> DetectionResult:
        object_names = [name for name in object_names if name]
        width, height = image.size
        if not object_names:
            return DetectionResult(
                success=False,
                objects=[],
                image_shape=(height, width),
                summary={"method": "yolo_world", "reason": "no object names provided"},
            )

        model = self._load_model()
        if self._uses_text_classes:
            model.set_classes(list(object_names))
        else:
            unavailable = self._get_unavailable_requested_classes(object_names)
            if unavailable:
                return DetectionResult(
                    success=False,
                    objects=[],
                    image_shape=(height, width),
                    summary={
                        "method": "yolo_world",
                        "model_path": self.model_path,
                        "model_type": self._resolved_model_type,
                        "classes": list(object_names),
                        "unavailable_requested_classes": unavailable,
                        "available_model_classes": self._get_model_names(),
                        "reason": (
                            "Regular YOLO models are closed-set detectors. "
                            "Requested classes must exist in model.names."
                        ),
                    },
                )
        results = model.predict(image, imgsz=self.image_size, conf=self.confidence, verbose=False)
        objects = self._results_to_objects(
            results,
            requested_classes=object_names if not self._is_world_model else None,
        )
        return DetectionResult(
            success=bool(objects),
            objects=objects,
            image_shape=(height, width),
            raw_output={"num_result_batches": len(results)},
            summary={
                "method": "yolo_world",
                "model_path": self.model_path,
                "model_type": self._resolved_model_type,
                "classes": list(object_names),
                "confidence": self.confidence,
                "image_size": self.image_size,
                "available_model_classes": self._get_model_names(),
            },
        )

    def _load_model(self):
        if self._model is None:
            try:
                if self._resolved_model_type == "world":
                    from ultralytics import YOLOWorld

                    self._model = YOLOWorld(self.model_path)
                elif self._resolved_model_type == "yoloe":
                    from ultralytics import YOLOE

                    self._model = YOLOE(self.model_path)
                else:
                    from ultralytics import YOLO

                    self._model = YOLO(self.model_path)
            except ImportError as exc:
                raise ImportError(
                    "YoloWorldDetector requires the optional 'ultralytics' package "
                    "and its torch dependencies."
                ) from exc
        return self._model

    def _resolve_is_world_model(self) -> bool:
        return self._resolved_model_type in {"world", "yoloe"}

    @property
    def _uses_text_classes(self) -> bool:
        return self._resolved_model_type in {"world", "yoloe"}

    @property
    def _resolved_model_type(self) -> str:
        if self.model_type not in {"auto", "world", "regular", "yoloe"}:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        if self.model_type != "auto":
            return self.model_type
        lowered = self.model_path.lower()
        if "yoloe" in lowered:
            return "yoloe"
        if "world" in lowered:
            return "world"
        return "regular"

    def _get_model_names(self) -> Dict[int, str]:
        if not self._model:
            return {}
        names = getattr(self._model, "names", {}) or {}
        return {int(k): str(v) for k, v in names.items()}

    def _get_unavailable_requested_classes(self, requested_classes: Sequence[str]) -> List[str]:
        available = {name.lower() for name in self._get_model_names().values()}
        return [name for name in requested_classes if name.lower() not in available]

    def _results_to_objects(
        self,
        results: Sequence[Any],
        requested_classes: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []
        requested = {name.lower() for name in requested_classes or []}
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()
            for box, score, class_id in zip(xyxy, scores, class_ids):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                label = names.get(int(class_id), str(int(class_id)))
                if requested and label.lower() not in requested:
                    continue
                objects.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "label": label,
                        "score": float(score),
                        "center_point": [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))],
                        "mask": None,
                    }
                )
        objects.sort(key=lambda obj: obj["score"], reverse=True)
        return objects


def _load_detector_from_config(config_path: Optional[str]) -> YoloWorldDetector:
    if not config_path:
        return YoloWorldDetector()
    cfg = yaml.safe_load(open(config_path)) or {}
    detector_cfg = cfg.get("YoloWorldDetector", cfg.get("ObjDetector", cfg))
    return YoloWorldDetector(
        model_path=detector_cfg.get("model_path", "yolov8s-worldv2.pt"),
        model_type=detector_cfg.get("model_type", "auto"),
        confidence=float(detector_cfg.get("confidence", detector_cfg.get("conf", 0.05))),
        image_size=int(detector_cfg.get("image_size", detector_cfg.get("imgsz", 640))),
        output_dir=detector_cfg.get("output_dir", "output/perception/yolo_world"),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Ultralytics detection and save ObjectDetector-format output.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--classes", nargs="+", required=True, help="Object names/classes to request or filter.")
    parser.add_argument("--config", default="configs/perception/obj_detector_config.yaml", help="Detector config path.")
    parser.add_argument("--model-path", default=None, help="Override model path, e.g. yolo26n.pt.")
    parser.add_argument(
        "--model-type",
        choices=["auto", "world", "regular", "yoloe"],
        default=None,
        help="Override model type.",
    )
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override inference image size.")
    parser.add_argument("--save-dir", default=None, help="Optional output directory.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    image = Image.open(args.image).convert("RGB")
    detector = _load_detector_from_config(args.config)
    if args.model_path:
        detector.model_path = args.model_path
        detector._is_world_model = detector._resolve_is_world_model()
        detector._model = None
    if args.model_type:
        detector.model_type = args.model_type
        detector._is_world_model = detector._resolve_is_world_model()
        detector._model = None
    if args.conf is not None:
        detector.confidence = args.conf
    if args.imgsz is not None:
        detector.image_size = args.imgsz
    try:
        result = detector.detect(image, args.classes)
    except ImportError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 2
    save_dir = detector.save_detection(image, result, args.save_dir)
    print(json.dumps({"success": result.success, "num_objects": result.num_objects, "save_dir": save_dir}, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
