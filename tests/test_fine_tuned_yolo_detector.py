from types import SimpleNamespace

from PIL import Image

from agenticlab_human.perception.detection.fine_tuned_yolo_detector import (
    FineTunedYoloDetector,
)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    names = {0: "number_block_1", 1: "yellow_bin"}

    def __init__(self):
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(
                names=self.names,
                boxes=SimpleNamespace(
                    xyxy=FakeTensor([[10.0, 20.0, 30.0, 40.0]]),
                    conf=FakeTensor([0.9]),
                    cls=FakeTensor([1.0]),
                ),
            )
        ]


def test_fine_tuned_yolo_filters_with_checkpoint_class_ids(tmp_path):
    model = FakeModel()
    detector = FineTunedYoloDetector(
        "unused.pt",
        output_dir=str(tmp_path),
        model=model,
    )

    result = detector.detect(
        Image.new("RGB", (64, 48)),
        ["yellow_bin"],
    )

    assert result.success is True
    assert result.objects[0]["label"] == "yellow_bin"
    assert model.calls[0]["classes"] == [1]


def test_fine_tuned_yolo_reports_unavailable_class_without_inference(tmp_path):
    model = FakeModel()
    detector = FineTunedYoloDetector(
        "unused.pt",
        output_dir=str(tmp_path),
        model=model,
    )

    result = detector.detect(
        Image.new("RGB", (64, 48)),
        ["unknown_object"],
    )

    assert result.success is False
    assert result.summary["unavailable_requested_classes"] == ["unknown_object"]
    assert model.calls == []
