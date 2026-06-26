import json
from types import SimpleNamespace

import numpy as np

from agenticlab_human.core.action_sequence import Action, ActionSequence
from agenticlab_human.execution.action_backend import ActionResult
from agenticlab_human.execution.pipeline import (
    ExecutionRuntime,
    PickGripVerification,
    PlaceSupportArmMotion,
    capture_scene_from_x5_server,
    execute_action_sequence,
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
    def __init__(self, *, grip_statuses=None, init_gripper_results=None):
        self.capture_count = 0
        self.closed = False
        self.calls = []
        self.joints_rad = np.radians(
            [74.0, 32.0, 0.0, 99.0, -100.0, 77.0, 46.0]
        ).tolist()
        self.response_count = 0
        self.grip_statuses = list(grip_statuses or [2])
        self.grip_status_reads = 0
        self.init_gripper_results = list(init_gripper_results or [])
        self.init_gripper_reads = 0

    def health(self):
        ready = SimpleNamespace(ready=True, detail="ready")
        return SimpleNamespace(camera=ready, robot=ready)

    def capture_rgbd(self):
        self.capture_count += 1
        self.calls.append(("capture_rgbd",))
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

    def get_state(self, arm):
        self.calls.append(("get_state", arm))
        grip_status = self.grip_statuses[
            min(self.grip_status_reads, len(self.grip_statuses) - 1)
        ]
        if arm == "left":
            self.grip_status_reads += 1
        arm_state = SimpleNamespace(joints_rad=self.joints_rad.copy())
        gripper_state = SimpleNamespace(grip_status=grip_status)
        return SimpleNamespace(
            success=True,
            error=None,
            state_after=SimpleNamespace(
                arms={arm: arm_state},
                gripper=gripper_state,
                grippers={arm: gripper_state},
            ),
        )

    def move_joints(
        self,
        arm,
        joints_rad,
        *,
        torso_joints_deg,
        speed_ratio,
        wait,
    ):
        self.response_count += 1
        self.calls.append(
            (
                "move_joints",
                arm,
                list(joints_rad),
                list(torso_joints_deg),
                speed_ratio,
                wait,
            )
        )
        self.joints_rad = list(joints_rad)
        return SimpleNamespace(
            success=True,
            error=None,
            request_id=f"pipeline-{self.response_count}",
            duration_ms=1.0,
        )

    def open_gripper(self, *, arm, wait):
        self.response_count += 1
        self.calls.append(("open_gripper", arm, wait))
        return SimpleNamespace(
            success=True,
            error=None,
            request_id=f"pipeline-{self.response_count}",
            duration_ms=1.0,
        )

    def init_gripper(self, *, arm):
        self.response_count += 1
        self.calls.append(("init_gripper", arm))
        configured_result = True
        if self.init_gripper_reads < len(self.init_gripper_results):
            configured_result = self.init_gripper_results[self.init_gripper_reads]
        self.init_gripper_reads += 1
        if isinstance(configured_result, str):
            success = False
            error = configured_result
        else:
            success = bool(configured_result)
            error = None if success else "init gripper failed"
        return SimpleNamespace(
            success=success,
            error=error,
            request_id=f"pipeline-{self.response_count}",
            duration_ms=1.0,
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


def _runtime(
    tmp_path,
    *,
    grip_statuses=None,
    pick_max_retries=1,
    init_gripper_results=None,
):
    return ExecutionRuntime(
        x5_client=FakeX5Client(
            grip_statuses=grip_statuses,
            init_gripper_results=init_gripper_results,
        ),
        detector=FakeDetector(),
        grasp_backend=FakeGraspBackend(),
        action_backend=FakeActionBackend(),
        run_dir=tmp_path / "run",
        place_depth_patch_px=9,
        place_offset_world_x_m=0.1,
        place_support_motion=PlaceSupportArmMotion(
            arm="right",
            home_joints_deg=[74.0, 32.0, 0.0, 99.0, -100.0, 77.0, 46.0],
            home_torso_deg=[0.0],
            place_torso_deg=[-8.0],
            speed_ratio=0.5,
            max_joint_delta_deg=150.0,
        ),
        pick_grip_verification=PickGripVerification(
            arm="left",
            expected_grip_status=2,
            max_retries=pick_max_retries,
        ),
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
    move_calls = [
        call for call in runtime.x5_client.calls if call[0] == "move_joints"
    ]
    assert [call[3] for call in move_calls] == [[-8.0], [0.0]]
    assert (
        place_result.metadata["right_arm_place_pre_steps"][-1]["torso_joints_deg"]
        == [-8.0]
    )
    assert (
        place_result.metadata["right_arm_home_steps"][-1]["torso_joints_deg"]
        == [0.0]
    )
    place_call = next(
        call for call in runtime.action_backend.calls if call[0] == "place"
    )
    target_pose = place_call[3]["target_pose"]
    target_data = json.loads(
        (runtime.run_dir / "action_002_place" / "target_pose.json").read_text()
    )
    np.testing.assert_allclose(target_pose, target_data["p_world_place"])
    assert target_data["place_offset_world_x_m"] == 0.1
    assert (
        runtime.run_dir / "action_001_pick" / "grasp_candidates.json"
    ).exists()
    assert (
        runtime.run_dir / "action_002_place" / "target_pose.json"
    ).exists()


def test_execute_pick_stays_single_attempt_when_gripper_is_empty(tmp_path):
    runtime = _runtime(tmp_path, grip_statuses=[1, 2], pick_max_retries=1)

    with runtime:
        result = execute_pick(runtime, "number_block_1")

    assert result.success is True
    assert runtime.held_object == "number_block_1"
    assert runtime.x5_client.capture_count == 1
    pick_calls = [
        call for call in runtime.action_backend.calls if call[0] == "pick"
    ]
    assert len(pick_calls) == 1
    assert ("open_gripper", "left", True) not in runtime.x5_client.calls
    assert ("init_gripper", "left") not in runtime.x5_client.calls
    assert "pick_attempts" not in result.metadata
    assert "gripper_verification" not in result.metadata
    assert (
        runtime.run_dir / "action_001_pick" / "grasp_candidates.json"
    ).exists()


def test_action_sequence_retries_pick_when_gripper_is_empty(tmp_path):
    runtime = _runtime(tmp_path, grip_statuses=[1, 2], pick_max_retries=1)
    sequence = ActionSequence(
        task="retry-pick",
        task_description="retry one pick",
        actions=[
            Action(
                id=1,
                name="pick",
                args={"object": "number_block_1"},
            )
        ],
    )

    with runtime:
        report = execute_action_sequence(runtime, sequence)

    result = report.results[0]
    assert report.success is True
    assert result.success is True
    assert runtime.held_object == "number_block_1"
    assert runtime.x5_client.capture_count == 2
    pick_calls = [
        call for call in runtime.action_backend.calls if call[0] == "pick"
    ]
    assert len(pick_calls) == 2
    assert ("init_gripper", "left") in runtime.x5_client.calls
    assert ("open_gripper", "left", True) not in runtime.x5_client.calls
    attempts = result.metadata["pick_attempts"]
    assert [attempt["grip_status"] for attempt in attempts] == [1, 2]
    assert attempts[0]["verified"] is False
    assert len(attempts[1]["retry_reset_steps"]) == 1
    assert attempts[1]["retry_reset_steps"][0]["previous_grip_status"] == 1
    assert attempts[1]["verified"] is True
    assert result.metadata["action_id"] == 1
    assert (
        runtime.run_dir / "action_001_pick" / "grasp_candidates.json"
    ).exists()
    assert (
        runtime.run_dir / "action_002_pick" / "grasp_candidates.json"
    ).exists()


def test_action_sequence_drop_state_initializes_gripper_before_pick_retry(tmp_path):
    runtime = _runtime(
        tmp_path,
        grip_statuses=[3, 2],
        pick_max_retries=1,
    )
    sequence = ActionSequence(
        task="drop-retry-pick",
        task_description="reset gripper after drop before retry",
        actions=[
            Action(
                id=1,
                name="pick",
                args={"object": "number_block_1"},
            )
        ],
    )

    with runtime:
        report = execute_action_sequence(runtime, sequence)

    result = report.results[0]
    assert report.success is True
    assert result.success is True
    init_calls = [
        call for call in runtime.x5_client.calls if call[0] == "init_gripper"
    ]
    assert init_calls == [("init_gripper", "left")]
    attempts = result.metadata["pick_attempts"]
    assert [attempt["grip_status"] for attempt in attempts] == [3, 2]
    retry_reset_steps = attempts[1]["retry_reset_steps"]
    assert [step["success"] for step in retry_reset_steps] == [True]
    assert retry_reset_steps[0]["step"] == "init_gripper_before_pick_retry"
    assert retry_reset_steps[0]["previous_grip_status"] == 3


def test_action_sequence_stops_when_gripper_init_before_pick_retry_fails(tmp_path):
    runtime = _runtime(
        tmp_path,
        grip_statuses=[3, 2],
        pick_max_retries=1,
        init_gripper_results=["init gripper failed"],
    )
    sequence = ActionSequence(
        task="drop-reset-fail",
        task_description="reset gripper fails before retry",
        actions=[
            Action(
                id=1,
                name="pick",
                args={"object": "number_block_1"},
            )
        ],
    )

    with runtime:
        report = execute_action_sequence(runtime, sequence)

    result = report.results[0]
    assert report.success is False
    assert result.success is False
    assert result.error == "init gripper failed"
    assert runtime.x5_client.capture_count == 1
    pick_calls = [
        call for call in runtime.action_backend.calls if call[0] == "pick"
    ]
    assert len(pick_calls) == 1
    assert result.metadata["pick_attempts"][0]["retryable"] is True
    retry_reset_steps = result.metadata["retry_reset_steps"]
    assert len(retry_reset_steps) == 1
    assert retry_reset_steps[0]["success"] is False
    assert retry_reset_steps[0]["previous_grip_status"] == 3


def test_action_sequence_fails_when_gripper_retry_still_empty(tmp_path):
    runtime = _runtime(tmp_path, grip_statuses=[1, 1], pick_max_retries=1)
    sequence = ActionSequence(
        task="retry-pick-fail",
        task_description="retry one pick and fail",
        actions=[
            Action(
                id=1,
                name="pick",
                args={"object": "number_block_1"},
            )
        ],
    )

    with runtime:
        report = execute_action_sequence(runtime, sequence)

    result = report.results[0]
    assert report.success is False
    assert result.success is False
    assert "grip_status=1" in result.error
    assert runtime.held_object is None
    assert runtime.x5_client.capture_count == 2
    assert result.metadata["pick_attempts"][-1]["verified"] is False
    assert result.metadata["gripper_verification"]["grip_status"] == 1


def test_execute_action_sequence_runs_multiple_pick_place_pairs(tmp_path):
    runtime = _runtime(tmp_path)
    sequence = ActionSequence(
        task="two-blocks",
        task_description="place two blocks",
        actions=[
            Action(
                id=1,
                name="pick",
                args={"object": "number_block_1", "from": "cloth"},
                pddl_str="(pick number_block_1 cloth)",
            ),
            Action(
                id=2,
                name="place",
                args={"object": "number_block_1", "target": "yellow_bin"},
                pddl_str="(place number_block_1 yellow_bin)",
            ),
            Action(
                id=3,
                name="pick",
                args={"object": "number_block_2", "from": "cloth"},
                pddl_str="(pick number_block_2 cloth)",
            ),
            Action(
                id=4,
                name="place",
                args={"object": "number_block_2", "target": "red_bin"},
                pddl_str="(place number_block_2 red_bin)",
            ),
        ],
        goal_conditions=["(on-top-of number_block_2 red_bin)"],
    )

    with runtime:
        report = execute_action_sequence(runtime, sequence)

    assert report.success is True
    assert report.task == "two-blocks"
    assert report.total_actions == 4
    assert runtime.x5_client.capture_count == 4
    assert [
        result.metadata["action_id"]
        for result in report.results
    ] == [1, 2, 3, 4]
    assert runtime.held_object is None
    action_calls = [
        call
        for call in runtime.action_backend.calls
        if call[0] in {"pick", "place"}
    ]
    assert [call[0] for call in action_calls] == [
        "pick",
        "place",
        "pick",
        "place",
    ]
    assert action_calls[0][1] == "number_block_1"
    assert action_calls[0][2]["from_name"] == "cloth"
    assert action_calls[1][1:3] == ("number_block_1", "yellow_bin")
    assert action_calls[2][1] == "number_block_2"
    assert action_calls[3][1:3] == ("number_block_2", "red_bin")
    assert (
        runtime.run_dir / "action_001_pick" / "grasp_candidates.json"
    ).exists()
    assert (
        runtime.run_dir / "action_002_place" / "target_pose.json"
    ).exists()
    assert (
        runtime.run_dir / "action_003_pick" / "grasp_candidates.json"
    ).exists()
    assert (
        runtime.run_dir / "action_004_place" / "target_pose.json"
    ).exists()


def test_execute_place_rejects_object_not_held_by_runtime(tmp_path):
    runtime = _runtime(tmp_path)

    with runtime:
        result = execute_place(runtime, "number_block_1", "yellow_bin")

    assert result.success is False
    assert "held_object" in result.error
    assert runtime.x5_client.capture_count == 0
