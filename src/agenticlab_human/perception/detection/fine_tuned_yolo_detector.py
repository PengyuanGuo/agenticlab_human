"""Closed-set detector backed by a fine-tuned Ultralytics YOLO checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from agenticlab_human.perception.backend.perception_backend import (
    BasePerceptionBackend,
    DetectionResult,
)


class FineTunedYoloDetector(BasePerceptionBackend):
    """Run fixed-class detection with an explicitly configured checkpoint."""

    def __init__(
        self,
        model_path: str,
        *,
        confidence: float = 0.25,
        image_size: int = 960,
        output_dir: str = "output/perception/fine_tuned_yolo",
        session_name: str | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir, session_name=session_name)
        self.model_path = str(model_path)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self._model = model
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

    def detect(
        self,
        image: Image.Image,
        object_names: Sequence[str],
    ) -> DetectionResult:
        requested_names = [str(name) for name in object_names if name]
        width, height = image.size
        if not requested_names:
            return DetectionResult(
                success=False,
                objects=[],
                image_shape=(height, width),
                summary={"method": "fine_tuned_yolo", "reason": "no classes requested"},
            )

        model = self._load_model()
        model_names = _normalize_model_names(getattr(model, "names", {}) or {})
        class_ids = _requested_class_ids(requested_names, model_names)
        unavailable = [
            name
            for name in requested_names
            if name.lower() not in {label.lower() for label in model_names.values()}
        ]
        if unavailable:
            return DetectionResult(
                success=False,
                objects=[],
                image_shape=(height, width),
                summary={
                    "method": "fine_tuned_yolo",
                    "model_path": self.model_path,
                    "requested_classes": requested_names,
                    "unavailable_requested_classes": unavailable,
                    "available_model_classes": model_names,
                    "reason": "requested classes are not present in the checkpoint",
                },
            )

        results = model.predict(
            image,
            classes=class_ids,
            imgsz=self.image_size,
            conf=self.confidence,
            verbose=False,
        )
        objects = _results_to_objects(results, requested_names)
        return DetectionResult(
            success=bool(objects),
            objects=objects,
            image_shape=(height, width),
            raw_output={"num_result_batches": len(results)},
            summary={
                "method": "fine_tuned_yolo",
                "model_path": self.model_path,
                "requested_classes": requested_names,
                "confidence": self.confidence,
                "image_size": self.image_size,
                "available_model_classes": model_names,
            },
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        checkpoint = Path(self.model_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"fine-tuned YOLO checkpoint does not exist: {checkpoint}"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "FineTunedYoloDetector requires the optional ultralytics package"
            ) from exc
        self._model = YOLO(str(checkpoint))
        return self._model


def _normalize_model_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(index): str(label) for index, label in names.items()}
    return {index: str(label) for index, label in enumerate(names)}


def _requested_class_ids(
    requested_names: Sequence[str],
    model_names: dict[int, str],
) -> list[int]:
    requested = {name.lower() for name in requested_names}
    return [
        class_id
        for class_id, label in model_names.items()
        if label.lower() in requested
    ]


def _results_to_objects(
    results: Sequence[Any],
    requested_names: Sequence[str],
) -> list[dict[str, Any]]:
    requested = {name.lower() for name in requested_names}
    objects: list[dict[str, Any]] = []
    for result in results:
        names = _normalize_model_names(getattr(result, "names", {}) or {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy_values = boxes.xyxy.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()
        for xyxy, score, class_id in zip(
            xyxy_values,
            scores,
            class_ids,
            strict=True,
        ):
            label = names.get(int(class_id), str(int(class_id)))
            if label.lower() not in requested:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
            objects.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "score": float(score),
                    "center_point": [
                        int(round((x1 + x2) / 2.0)),
                        int(round((y1 + y2) / 2.0)),
                    ],
                    "mask": None,
                }
            )
    objects.sort(key=lambda item: item["score"], reverse=True)
    return objects
