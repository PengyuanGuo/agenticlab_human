import json
from types import SimpleNamespace

import numpy as np

from agenticlab_human.execution.action_backend import ActionResult
from agenticlab_human.execution.pipeline import (
    ExecutionRuntime,
    capture_scene_from_x5_server,
    execute_pick,
    execute_place,
)
from agenticlab_human.execution.robot.x5.contracts import (
    CameraIntrinsics,
    RGBDFrame,
)
from agenticlab_human.perception.backend.grasp_backend import GraspCandidate
from agenticlab_human.perception.backend.perception_backend import DetectionResult


class FakeX5Client:
    def __init__(self):
        self.capture_count = 0
        self.closed = False

    def health(self):
        ready = SimpleNamespace(ready=True, detail="ready")
        return SimpleNamespace(camera=ready, robot=ready)

    def capture_rgbd(self):
        self.capture_count += 1
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        depth_mm = np.full((48, 64), 1000.0, dtype=np.float32)
        return RGBDFrame(
            rgb=rgb,
            depth_mm=depth_mm,
            intrinsics=CameraIntrinsics(
                fx=100.0,
                fy=100.0,
                cx=32.0,
                cy=24.0,
                width=64,
                height=48,
            ),
            timestamp_ns=1_000_000_000 + self.capture_count,
            frame_id=f"frame-{self.capture_count:02d}",
        )

    def close(self):
        self.closed = True


class FakeDetector:
    def detect(self, image, object_names):
        label = object_names[0]
        return DetectionResult(
            success=True,
            objects=[
                {
                    "bbox": [20, 12, 44, 36],
                    "label": label,
                    "score": 0.9,
                    "center_point": [32, 24],
                    "mask": None,
                }
            ],
            image_shape=(48, 64),
        )


class FakeGraspBackend:
    def __init__(self):
        self.initialized = False

    def initialize(self):
        self.initialized = True

    def shutdown(self):
        self.initialized = False

    def plan_for_object(self, *, bbox, object_name, **kwargs):
        assert self.initialized is True
        return [
            GraspCandidate(
                pose=np.eye(4),
                score=0.9,
                object_name=object_name,
                metadata={"frame": "camera"},
            )
        ]


class FakeActionBackend:
    def __init__(self):
        self.T_world_camera = np.eye(4)
        self.calls = []

    def initialize(self):
        self.calls.append(("initialize",))

    def shutdown(self, move_home=False):
        self.calls.append(("shutdown", move_home))

    def pick(self, object_name, **kwargs):
        self.calls.append(("pick", object_name, kwargs))
        return ActionResult(success=True, action_name="pick")

    def place(self, object_name, target_name, **kwargs):
        self.calls.append(("place", object_name, target_name, kwargs))
        return ActionResult(success=True, action_name="place")

    def move_home(self):
        return ActionResult(success=True, action_name="move-home")

    def get_eef_pose(self):
        return None


def _runtime(tmp_path):
    return ExecutionRuntime(
        x5_client=FakeX5Client(),
        detector=FakeDetector(),
        grasp_backend=FakeGraspBackend(),
        action_backend=FakeActionBackend(),
        run_dir=tmp_path / "run",
        place_depth_patch_px=9,
        place_offset_world_x_m=0.1,
        execute=False,
    )


def test_capture_scene_wraps_existing_rgbd_and_saved_paths(tmp_path):
    client = FakeX5Client()

    snapshot = capture_scene_from_x5_server(client, tmp_path)

    assert snapshot.frame.frame_id == "frame-01"
    assert snapshot.rgb_path.exists()
    assert snapshot.depth_path.exists()
    assert snapshot.metadata_path.exists()


def test_execute_pick_then_place_reuses_runtime_and_recaptures_scene(tmp_path):
    runtime = _runtime(tmp_path)

    with runtime:
        pick_result = execute_pick(runtime, "number_block_1")
        assert pick_result.success is True
        assert runtime.held_object == "number_block_1"

        place_result = execute_place(
            runtime,
            "number_block_1",
            "yellow_bin",
        )

    assert place_result.success is True
    assert runtime.held_object is None
    assert runtime.x5_client.capture_count == 2
    place_call = next(
        call for call in runtime.action_backend.calls if call[0] == "place"
    )
    target_pose = place_call[3]["target_pose"]
    target_data = json.loads(
        (runtime.run_dir / "place" / "target_pose.json").read_text()
    )
    np.testing.assert_allclose(target_pose, target_data["p_world_place"])
    assert target_data["place_offset_world_x_m"] == 0.1


def test_execute_place_rejects_object_not_held_by_runtime(tmp_path):
    runtime = _runtime(tmp_path)

    with runtime:
        result = execute_place(runtime, "number_block_1", "yellow_bin")

    assert result.success is False
    assert "held_object" in result.error
    assert runtime.x5_client.capture_count == 0
