"""ActionSequence – the decoupled contract between Planner and Executor.

The Planner builds this contract from TaskPlan and writes action_sequence.json.
The Executor uses ActionSequence.load(path), which accepts only that JSON file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agenticlab_human.planning.task_parser import TaskPlan

# ---------------------------------------------------------------------------
# Parameter-name semantics
# ---------------------------------------------------------------------------
# Maps PDDL variable names (after stripping '?') to human-readable arg keys
# used in the ActionSequence JSON contract.
_PARAM_RENAME: Dict[str, str] = {
    "obj":       "object",
    "underobj":  "from",
    "target":    "target",
    "surface":   "surface",
    "location":  "location",
    "loc":       "location",
    "container": "container",
    "from":      "from",
    "to":        "to",
}


def _rename_param(name: str) -> str:
    return _PARAM_RENAME.get(name, name)


def _default_param_keys(action_name: str, count: int) -> List[str]:
    defaults = {
        "pick": ["object", "from"],
        "place": ["object", "target"],
    }
    keys = defaults.get(action_name, [])
    return [
        keys[index] if index < len(keys) else f"arg{index}"
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# PDDL helpers
# ---------------------------------------------------------------------------

def _extract_action_params(domain_content: str) -> Dict[str, List[str]]:
    """Return {action_name: [semantic_arg_keys]} parsed from PDDL domain text."""
    param_map: Dict[str, List[str]] = {}
    pattern = r':action\s+(\S+).*?:parameters\s*\(([^)]*)\)'
    for m in re.finditer(pattern, domain_content, re.DOTALL | re.IGNORECASE):
        action_name = m.group(1).strip()
        var_names = re.findall(r'\?(\w+)', m.group(2))
        param_map[action_name] = [_rename_param(v) for v in var_names]
    return param_map


def _parse_pddl_string(pddl_str: str):
    """Parse '(action-name arg1 arg2 ...)' → (name, [arg1, arg2, ...])."""
    cleaned = pddl_str.strip().lstrip('(').rstrip(')')
    parts = cleaned.split()
    if not parts:
        return "", []
    return parts[0], parts[1:]


def _canonical_action(name: str, args: Dict[str, str]) -> tuple[str, Dict[str, str]]:
    """Normalize the public place contract to object/target arguments."""

    if name == "place":
        normalized_args = dict(args)
        if "target" not in normalized_args:
            for key in ("surface", "location", "container", "to"):
                value = normalized_args.pop(key, None)
                if value:
                    normalized_args["target"] = value
                    break
        return "place", normalized_args
    return name, args


def _make_task_slug(description: str) -> str:
    slug = description.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')
    return slug[:60]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """A single robot action with semantic argument names."""
    id: int
    name: str
    args: Dict[str, str]
    pddl_str: Optional[str] = None  # original PDDL string, kept for traceability

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id, "name": self.name, "args": self.args}
        if self.pddl_str:
            d["pddl_str"] = self.pddl_str
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Action":
        name, args = _canonical_action(d["name"], d["args"])
        return cls(
            id=d["id"],
            name=name,
            args=args,
            pddl_str=d.get("pddl_str"),
        )


@dataclass
class ActionSequence:
    """Ordered list of Actions that forms the contract between Planner and Executor."""
    task: str
    task_description: str
    actions: List[Action]
    goal_conditions: Optional[List[str]] = field(default=None)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_task_plan(cls, task_plan: "TaskPlan") -> "ActionSequence":
        """Build an ActionSequence from a TaskPlan produced by TaskParser."""
        return cls.from_pddl_strings(
            pddl_strings=task_plan.action_sequence,
            task_description=task_plan.task_description,
            domain_content=task_plan.updated_domain or "",
            goal_conditions=task_plan.goal_conditions,
        )

    @classmethod
    def from_pddl_strings(
        cls,
        pddl_strings: List[str],
        task: str = "",
        task_description: str = "",
        domain_content: str = "",
        goal_conditions: Optional[List[str]] = None,
    ) -> "ActionSequence":
        """Build directly from PDDL-style strings without a TaskPlan object."""
        param_map = _extract_action_params(domain_content) if domain_content else {}
        actions: List[Action] = []
        for i, pddl_str in enumerate(pddl_strings, start=1):
            raw_name, raw_args = _parse_pddl_string(pddl_str)
            keys = param_map.get(
                raw_name,
                _default_param_keys(raw_name, len(raw_args)),
            )
            args = {
                (keys[j] if j < len(keys) else f"arg{j}"): v
                for j, v in enumerate(raw_args)
            }
            name, args = _canonical_action(raw_name, args)
            actions.append(Action(id=i, name=name, args=args, pddl_str=pddl_str))
        return cls(
            task=task or _make_task_slug(task_description),
            task_description=task_description,
            actions=actions,
            goal_conditions=goal_conditions,
        )

    @classmethod
    def load(cls, path: str) -> "ActionSequence":
        """Load the executor contract from an action_sequence.json file."""
        p = Path(path)
        if p.name != "action_sequence.json":
            raise ValueError(
                "ActionSequence.load() requires a path to action_sequence.json, "
                f"got: {path}"
            )
        if not p.is_file():
            raise FileNotFoundError(path)
        data = json.loads(p.read_text())
        if not isinstance(data, dict) or "actions" not in data:
            raise ValueError(
                f"Invalid action_sequence.json contract in {path}: "
                "expected a JSON object containing 'actions'"
            )
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "task": self.task,
            "task_description": self.task_description,
            "actions": [a.to_dict() for a in self.actions],
        }
        if self.goal_conditions is not None:
            d["goal_conditions"] = self.goal_conditions
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActionSequence":
        return cls(
            task=d.get("task", ""),
            task_description=d.get("task_description", ""),
            actions=[Action.from_dict(a) for a in d.get("actions", [])],
            goal_conditions=d.get("goal_conditions"),
        )

    def save(self, path: str) -> None:
        """Persist to a JSON file."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def __repr__(self) -> str:
        return (
            f"ActionSequence(task={self.task!r}, "
            f"actions={len(self.actions)}, "
            f"goal_conditions={len(self.goal_conditions or [])})"
        )
