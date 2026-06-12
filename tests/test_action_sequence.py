from pathlib import Path

import pytest

from agenticlab_human.core.action_sequence import ActionSequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_SEQUENCE_PATH = (
    REPO_ROOT / "output/task_parser/20260601_162529/action_sequence.json"
)


def test_load_reads_action_sequence_json_contract():
    action_sequence = ActionSequence.load(str(ACTION_SEQUENCE_PATH))

    assert len(action_sequence.actions) == 2

    first_action = action_sequence.actions[0]
    assert first_action.name == "pick"
    assert first_action.args["object"] == "number-block-3"
    assert first_action.args["from"] == "beige-cloth"
    assert "arg1" not in first_action.args

    last_action = action_sequence.actions[-1]
    assert last_action.name == "place"
    assert last_action.args["object"] == "number-block-3"
    assert last_action.args["target"] == "yellow-box"
    assert "arg1" not in last_action.args
    assert last_action.pddl_str == "(place number-block-3 yellow-box)"

    assert action_sequence.goal_conditions is not None
    assert len(action_sequence.goal_conditions) == 1


def test_load_rejects_session_directory_and_task_plan():
    session_dir = ACTION_SEQUENCE_PATH.parent

    with pytest.raises(ValueError, match="path to action_sequence.json"):
        ActionSequence.load(str(session_dir))
    with pytest.raises(ValueError, match="path to action_sequence.json"):
        ActionSequence.load(str(session_dir / "task_plan.json"))


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
