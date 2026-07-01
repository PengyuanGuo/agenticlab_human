from pathlib import Path

from agenticlab_human.core.action_sequence import Action, ActionSequence
from agenticlab_human.execution.action_backend import ActionResult, ExecutionReport
from agenticlab_human.execution.pipeline_loop import (
    PourExecutionDefaults,
    execute_action_sequence_loop,
    execute_action_sequence_loop_plan,
)


class FakeRuntime:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True


def _sequence() -> ActionSequence:
    return ActionSequence(
        task="pick-place-pour",
        task_description="pick, place, then pour",
        actions=[
            Action(1, "pick", {"object": "number_block_1"}),
            Action(2, "place", {"object": "number_block_1", "target": "red_bin"}),
            Action(3, "pour", {"arm": "right", "control_gripper": "true"}),
        ],
    )


def test_loop_repeats_pick_place_pour_and_offsets_action_ids(tmp_path):
    runtime = FakeRuntime(tmp_path / "run")
    pick_place_actions = []
    pour_calls = []

    def fake_pick_place_executor(runtime, sequence):
        action = sequence.actions[0]
        pick_place_actions.append(action)
        return ExecutionReport(
            success=True,
            task=sequence.task,
            total_actions=1,
            results=[
                ActionResult(
                    success=True,
                    action_name=action.name,
                    metadata={"action_id": action.id},
                )
            ],
        )

    def fake_pour_executor(**kwargs):
        pour_calls.append(kwargs)
        return ExecutionReport(
            success=True,
            task="right-arm-pour",
            total_actions=11,
            results=[ActionResult(success=True, action_name="pour")],
        )

    report = execute_action_sequence_loop(
        runtime,
        _sequence(),
        max_loops=3,
        pour_defaults=PourExecutionDefaults(
            config_path="configs/robot/x5_config.yaml",
            server_url="http://x5",
            arm="right",
            speed_ratio=None,
            control_gripper=False,
        ),
        pick_place_executor=fake_pick_place_executor,
        pour_executor=fake_pour_executor,
    )

    assert report.success is True
    assert report.total_actions == 9
    assert [result.metadata["action_id"] for result in report.results] == list(
        range(1, 10)
    )
    assert [result.metadata["loop_index"] for result in report.results] == [
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
    ]
    assert [action.id for action in pick_place_actions] == [1, 2, 4, 5, 7, 8]
    assert len(pour_calls) == 3
    assert all(call["arm"] == "right" for call in pour_calls)
    assert all(call["server_url"] == "http://x5" for call in pour_calls)
    assert all(call["control_gripper"] is True for call in pour_calls)
    assert report.metadata["completed_loops"] == 3


def test_loop_stops_on_failed_pour(tmp_path):
    runtime = FakeRuntime(tmp_path / "run")
    pour_calls = 0

    def fake_pick_place_executor(runtime, sequence):
        action = sequence.actions[0]
        return ExecutionReport(
            success=True,
            task=sequence.task,
            total_actions=1,
            results=[
                ActionResult(
                    success=True,
                    action_name=action.name,
                    metadata={"action_id": action.id},
                )
            ],
        )

    def fake_pour_executor(**kwargs):
        nonlocal pour_calls
        pour_calls += 1
        return ExecutionReport(
            success=False,
            task="right-arm-pour",
            total_actions=11,
            error="pour failed",
        )

    report = execute_action_sequence_loop(
        runtime,
        _sequence(),
        max_loops=3,
        pour_defaults=PourExecutionDefaults(
            config_path="configs/robot/x5_config.yaml",
            server_url="http://x5",
            arm="right",
            speed_ratio=None,
            control_gripper=False,
        ),
        pick_place_executor=fake_pick_place_executor,
        pour_executor=fake_pour_executor,
    )

    assert report.success is False
    assert report.failed_action_id == 3
    assert report.failed_action_name == "pour"
    assert report.error == "pour failed"
    assert len(report.results) == 3
    assert pour_calls == 1
    assert report.metadata["completed_loops"] == 0


def test_loop_plan_uses_pipeline_config_defaults_and_writes_report(tmp_path):
    plan_path = tmp_path / "action_sequence.json"
    config_path = tmp_path / "x5_pipeline.yaml"
    run_dir = tmp_path / "run"
    _sequence().save(str(plan_path))
    config_path.write_text(
        """
pipeline:
  robot_config: configs/robot/custom_x5.yaml
  x5_server_url: http://configured-x5
""".lstrip()
    )
    runtime = FakeRuntime(run_dir)
    pour_calls = []

    def fake_runtime_factory(*, config_path):
        return runtime

    def fake_pick_place_executor(runtime, sequence):
        action = sequence.actions[0]
        return ExecutionReport(
            success=True,
            task=sequence.task,
            total_actions=1,
            results=[
                ActionResult(
                    success=True,
                    action_name=action.name,
                    metadata={"action_id": action.id},
                )
            ],
        )

    def fake_pour_executor(**kwargs):
        pour_calls.append(kwargs)
        return ExecutionReport(
            success=True,
            task="right-arm-pour",
            total_actions=11,
            results=[ActionResult(success=True, action_name="pour")],
        )

    report = execute_action_sequence_loop_plan(
        plan_path=str(plan_path),
        config_path=str(config_path),
        max_loops=2,
        runtime_factory=fake_runtime_factory,
        pick_place_executor=fake_pick_place_executor,
        pour_executor=fake_pour_executor,
    )

    assert report.success is True
    assert runtime.entered is True
    assert runtime.exited is True
    assert [call["config_path"] for call in pour_calls] == [
        "configs/robot/custom_x5.yaml",
        "configs/robot/custom_x5.yaml",
    ]
    assert [call["server_url"] for call in pour_calls] == [
        "http://configured-x5",
        "http://configured-x5",
    ]
    assert (run_dir / "action_sequence.json").exists()
    assert (run_dir / "execution_report.json").exists()
