from pathlib import Path

from agenticlab_human.core.action_sequence import Action, ActionSequence
from agenticlab_human.execution.action import ActionExecutor
from agenticlab_human.execution.action_backend import DryRunActionBackend
from agenticlab_human.execution.robot.flexiv.flexiv_backend import FlexivActionBackend
from agenticlab_human.execution.execution_context import ExecutionContext
from agenticlab_human.perception.backend.grasp_backend import GraspCandidate
from agenticlab_human.perception.backend.perception_backend import BBox


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SESSION = REPO_ROOT / "output/task_parser/20260527_112644"


class FakeSceneProvider:
    def capture_rgbd(self):
        return "rgb", "depth"


class FakeDetector:
    def detect(self, rgb, object_names):
        return {
            name: [BBox(label=name, xyxy=(0.0, 0.0, 20.0, 20.0), confidence=0.9)]
            for name in object_names
        }


class FakeGraspPlanner:
    def plan_scene(self, rgb, depth):
        return [
            GraspCandidate(pose="green-grasp", score=0.8, image_xy=(10.0, 10.0)),
            GraspCandidate(pose="outside-grasp", score=0.7, image_xy=(100.0, 100.0)),
        ]

    def plan_for_object(self, rgb, depth, bbox):
        return [GraspCandidate(pose=f"refreshed-{bbox.label}", score=1.0, image_xy=bbox.center_xy)]


def test_execution_context_prepares_bbox_and_grasp_cache():
    action_sequence = ActionSequence(
        task="test",
        task_description="test",
        actions=[
            Action(id=1, name="pick", args={"object": "green-cube-1", "from": "table-1"}),
            Action(id=2, name="place-on-object", args={"object": "green-cube-1", "target": "box-1"}),
        ],
    )
    context = ExecutionContext(
        scene_provider=FakeSceneProvider(),
        detector=FakeDetector(),
        grasp_planner=FakeGraspPlanner(),
    )

    report = context.prepare_for_sequence(action_sequence)

    assert report.prepared is True
    assert report.bbox_counts["green-cube-1"] == 1
    assert report.grasp_counts["green-cube-1"] == 1
    assert context.get_best_grasp("green-cube-1").pose == "green-grasp"


def test_action_executor_dry_run_executes_historical_sequence():
    action_sequence = ActionSequence.load(str(HISTORICAL_SESSION))
    executor = ActionExecutor(backend=DryRunActionBackend())

    prepare_report = executor.prepare(action_sequence)
    execution_report = executor.execute_sequence(action_sequence)

    assert prepare_report.prepared is False
    assert execution_report.success is True
    assert execution_report.total_actions == 12
    assert len(execution_report.results) == 12
    assert execution_report.results[0].metadata["grasp_count"] == 0


def test_flexiv_backend_plans_pick_from_camera_frame_grasp_without_execution():
    action_sequence = ActionSequence.load(str(REPO_ROOT / "data/data_for_test/task_parser/execution/action_sequence.json"))
    context = ExecutionContext()
    T_cam_grasp = [
        [-0.15578863, 0.90573424, 0.39417678, 0.04381273],
        [-0.33291125, -0.423846, 0.84233284, -0.07053141],
        [0.93, 0.0, 0.3675595, 0.536],
        [0.0, 0.0, 0.0, 1.0],
    ]
    context.grasps["number-block-3-1"] = [
        GraspCandidate(
            pose=T_cam_grasp,
            score=1.0,
            object_name="number-block-3-1",
            metadata={"width": 0.08437179774045944},
        )
    ]
    backend = FlexivActionBackend(
        robot_config_path=str(REPO_ROOT / "configs/robot/flexiv_config.yaml"),
        camera_config_path=str(REPO_ROOT / "configs/perception/camera_config.yaml"),
        camera_name="FemtoBolt",
        execute=False,
    )
    executor = ActionExecutor(backend=backend, context=context)

    report = executor.execute_sequence(action_sequence)

    assert report.success is True
    metadata = report.results[0].metadata
    assert metadata["execute"] is False
    assert metadata["camera_name"] == "FemtoBolt"
    assert len(metadata["grasp_pose6d"]) == 6
    assert len(metadata["approach_pose6d"]) == 6
    assert metadata["grasp_width"] == 0.08437179774045944
