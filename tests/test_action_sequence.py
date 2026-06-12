from pathlib import Path

from agenticlab_human.core.action_sequence import ActionSequence


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PLAN = REPO_ROOT / "output/task_parser/20260527_112644/task_plan.json"


def test_historical_task_plan_maps_pddl_args_to_action_sequence_contract():
    action_sequence = ActionSequence.load(str(HISTORICAL_PLAN))

    assert len(action_sequence.actions) == 12

    first_action = action_sequence.actions[0]
    assert first_action.name == "pick"
    assert first_action.args["object"] == "green-cube-1"
    assert first_action.args["from"] == "orange-cube-1"
    assert "arg1" not in first_action.args

    last_action = action_sequence.actions[-1]
    assert last_action.name == "place"
    assert last_action.args["object"] == "blue-cube-1"
    assert last_action.args["target"] == "green-cube-1"
    assert "arg1" not in last_action.args
    assert last_action.pddl_str == "(place blue-cube-1 green-cube-1)"

    assert action_sequence.goal_conditions is not None
    assert len(action_sequence.goal_conditions) == 4


def test_place_uses_object_target_args_without_domain_content():
    action_sequence = ActionSequence.from_pddl_strings(
        ["(pick block-1 table-1)", "(place block-1 target-1)"]
    )

    assert action_sequence.actions[0].args == {
        "object": "block-1",
        "from": "table-1",
    }
    assert action_sequence.actions[1].name == "place"
    assert action_sequence.actions[1].args == {
        "object": "block-1",
        "target": "target-1",
    }
