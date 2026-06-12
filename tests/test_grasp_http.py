import json
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from agenticlab_human.perception.grasping.backend import MockGraspInferenceBackend
from agenticlab_human.perception.grasping.client import GraspNetHTTPClient
from agenticlab_human.perception.grasping import server
from agenticlab_human.perception.grasping.server import create_app


def _write_capture(tmp_path):
    rgb_path = tmp_path / "orbbec-test_rgb.png"
    depth_path = tmp_path / "orbbec-test_depth_mm.npy"

    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[12:36, 16:48] = [200, 40, 20]
    depth_mm = np.full((48, 64), 800.0, dtype=np.float32)
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth_mm)
    return rgb_path


def test_grasp_http_reads_local_rgbd_files_and_returns_camera_pose(tmp_path):
    rgb_path = _write_capture(tmp_path)
    app = create_app(MockGraspInferenceBackend())

    with TestClient(app) as transport:
        client = GraspNetHTTPClient(
            "http://testserver",
            timeout_s=None,
            session=transport,
        )
        health = client.health()
        response = client.predict_from_files(
            rgb_path=rgb_path,
            bbox_xyxy=[16, 12, 48, 36],
            object_label="test_block",
            max_grasps=5,
        )

    assert health.status == "ok"
    assert health.backend == "mock"
    assert response.success is True
    assert response.pose_frame == "camera"
    assert response.object_label == "test_block"
    assert response.num_grasps == 1
    assert response.input_summary["workspace_points"] == 52 * 44

    grasp = response.grasps[0]
    assert grasp.object_label == "test_block"
    assert grasp.width == 0.05
    assert grasp.pose_4x4[3] == [0.0, 0.0, 0.0, 1.0]

    candidates = client.to_grasp_candidates(response)
    assert len(candidates) == 1
    assert candidates[0].object_name == "test_block"
    assert candidates[0].metadata["frame"] == "camera"
    assert candidates[0].metadata["width"] == 0.05


def test_grasp_http_rejects_missing_shared_file(tmp_path):
    rgb_path = _write_capture(tmp_path)
    depth_path = tmp_path / "orbbec-test_depth_mm.npy"
    depth_path.unlink()
    app = create_app(MockGraspInferenceBackend())

    with TestClient(app) as transport:
        response = transport.post(
            "/v1/grasp/predict",
            json={
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "object_label": "test_block",
            },
        )

    assert response.status_code == 400
    assert "depth_path does not exist" in response.json()["detail"]


def test_grasp_http_launches_visualizer_after_success(tmp_path, monkeypatch):
    rgb_path = _write_capture(tmp_path)
    camera_config_path = tmp_path / "camera.yaml"
    camera_config_path.write_text(
        """
mock:
  intrinsic_matrix:
  - [100.0, 0.0, 32.0]
  - [0.0, 100.0, 24.0]
  - [0.0, 0.0, 1.0]
  factor_depth: 1000.0
  width: 64
  height: 48
""".strip()
    )
    backend = MockGraspInferenceBackend()
    backend.camera_config_path = str(camera_config_path)
    backend.camera_name = "mock"
    commands = []

    def fake_popen(command, **kwargs):
        commands.append((command, kwargs))
        return object()

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    app = create_app(
        backend,
        visualize_seconds=10,
        visualization_dir=tmp_path / "grasp_viz",
    )

    with TestClient(app) as transport:
        response = transport.post(
            "/v1/grasp/predict",
            json={
                "rgb_path": str(rgb_path),
                "depth_path": str(tmp_path / "orbbec-test_depth_mm.npy"),
                "bbox_xyxy": [16, 12, 48, 36],
                "object_label": "test_block",
            },
        )

    assert response.status_code == 200
    assert len(commands) == 1
    command, options = commands[0]
    assert command[1:4] == [
        "-m",
        "agenticlab_human.perception.grasping.visualizer",
        "--manifest",
    ]
    assert options["cwd"] == server.REPO_ROOT

    manifest_path = command[4]
    manifest = json.loads(open(manifest_path).read())
    assert manifest["visualize_seconds"] == 10
    assert manifest["camera_name"] == "mock"
    assert manifest["bbox_xyxy"] == [16.0, 12.0, 48.0, 36.0]
    assert len(manifest["grasps"]) == 1
