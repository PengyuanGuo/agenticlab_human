import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from agenticlab_human.perception.backend.perception_backend import BBox
from agenticlab_human.perception.grasping.backend import MockGraspInferenceBackend
from agenticlab_human.perception.grasping.client import GraspNetHTTPClient
from agenticlab_human.perception.grasping.http_backend import GraspNetHTTPBackend
from agenticlab_human.perception.grasping.server import create_app


def test_grasp_http_backend_plans_for_saved_object_bbox(tmp_path):
    rgb_path = tmp_path / "frame_rgb.png"
    depth_path = tmp_path / "frame_depth_mm.npy"
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(rgb_path)
    np.save(depth_path, np.full((48, 64), 800.0, dtype=np.float32))

    with TestClient(create_app(MockGraspInferenceBackend())) as transport:
        client = GraspNetHTTPClient(
            "http://testserver",
            timeout_s=None,
            session=transport,
        )
        backend = GraspNetHTTPBackend(
            "http://testserver",
            client=client,
        )
        backend.initialize()
        candidates = backend.plan_for_object(
            rgb_path=rgb_path,
            depth_path=depth_path,
            bbox=BBox(
                label="number_block_1",
                xyxy=(16.0, 12.0, 48.0, 36.0),
            ),
            object_name="number_block_1",
        )
        backend.shutdown()

    assert len(candidates) == 1
    assert candidates[0].object_name == "number_block_1"
    assert candidates[0].metadata["frame"] == "camera"
