import json
import os

import pytest
from PIL import Image

from agenticlab_human.core.action_sequence import Action, ActionSequence
from agenticlab_human.execution.execution_context import ExecutionContext
from agenticlab_human.perception.backend.perception_backend import (
    BasePerceptionBackend,
    DetectionResult,
)
from agenticlab_human.perception.detection.yolo_world_detector import YoloWorldDetector


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class FakeBoxes:
    xyxy = FakeTensor([[10.2, 20.3, 110.7, 120.8]])
    conf = FakeTensor([0.91])
    cls = FakeTensor([0])


class FakeYoloResult:
    names = {0: "green cube"}
    boxes = FakeBoxes()


class FakeDetectionBackend:
    def detect(self, image, object_names):
        return DetectionResult(
            success=True,
            objects=[
                {
                    "bbox": [10, 20, 110, 120],
                    "label": object_names[0],
                    "score": 0.9,
                    "center_point": [60, 70],
                    "mask": None,
                }
            ],
            image_shape=(image.size[1], image.size[0]),
        )


class FakeSceneProvider:
    def capture_rgbd(self):
        return Image.new("RGB", (160, 120), color="white"), None


def test_detection_result_save_matches_object_detector_format(tmp_path):
    image = Image.new("RGB", (160, 120), color="white")
    result = DetectionResult(
        success=True,
        objects=[
            {
                "bbox": [10, 20, 110, 100],
                "label": "green cube",
                "score": 0.92,
                "center_point": [60, 60],
                "mask": None,
            }
        ],
        image_shape=(120, 160),
        summary={"method": "test"},
    )
    backend = BasePerceptionBackend(output_dir=str(tmp_path), session_name="session")

    save_dir = backend.save_detection(image, result)

    assert os.path.exists(os.path.join(save_dir, "detection_result.json"))
    assert os.path.exists(os.path.join(save_dir, "visualization.png"))
    assert os.path.exists(os.path.join(save_dir, "original_img.png"))
    saved = json.loads(open(os.path.join(save_dir, "detection_result.json")).read())
    assert saved["objects"][0]["bounding_box"] == [10, 20, 110, 100]
    assert saved["objects"][0]["label"] == "green cube"
    assert saved["objects"][0]["center_point"] == [60, 60]


def test_yolo_world_result_conversion_aligns_with_detection_result_format(tmp_path):
    detector = YoloWorldDetector(output_dir=str(tmp_path))

    objects = detector._results_to_objects([FakeYoloResult()])

    assert objects == [
        {
            "bbox": [10, 20, 111, 121],
            "label": "green cube",
            "score": 0.91,
            "center_point": [60, 70],
            "mask": None,
        }
    ]


def test_regular_yolo_result_conversion_filters_requested_classes(tmp_path):
    detector = YoloWorldDetector(model_path="yolo26n.pt", model_type="regular", output_dir=str(tmp_path))

    objects = detector._results_to_objects([FakeYoloResult()], requested_classes=["person"])

    assert objects == []


def test_auto_model_type_recognizes_yoloe(tmp_path):
    detector = YoloWorldDetector(model_path="yoloe-26x-seg.pt", output_dir=str(tmp_path))

    assert detector._resolved_model_type == "yoloe"
    assert detector._uses_text_classes is True


def test_execution_context_accepts_detection_result_output():
    context = ExecutionContext(scene_provider=FakeSceneProvider(), detector=FakeDetectionBackend())
    action_sequence = ActionSequence(
        task="test",
        task_description="test",
        actions=[Action(id=1, name="pick", args={"object": "green cube"})],
    )

    report = context.prepare_for_sequence(action_sequence)

    assert report.prepared is True
    assert context.get_bbox("green cube").xyxy == (10.0, 20.0, 110.0, 120.0)
    assert context.get_bbox("green cube").confidence == 0.9


@pytest.mark.skipif(
    os.environ.get("AGENTICLAB_RUN_YOLO_WORLD_SMOKE") != "1",
    reason="Set AGENTICLAB_RUN_YOLO_WORLD_SMOKE=1 in a local YOLO-World environment.",
)
def test_yolo_world_detector_real_smoke(tmp_path):
    pytest.importorskip("ultralytics")
    image = Image.open("data/data_for_test/task_parser/04_stack1_color.png").convert("RGB")
    detector = YoloWorldDetector(output_dir=str(tmp_path))

    result = detector.detect(image, ["orange cube", "yellow cube", "green cube", "blue cube", "pink plate"])
    save_dir = detector.save_detection(image, result)

    assert os.path.exists(os.path.join(save_dir, "detection_result.json"))
