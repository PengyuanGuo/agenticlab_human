"""ActionSequence – the decoupled contract between Planner and Executor.

Converts PDDL-style action strings produced by TaskParser into a
typed, JSON-serializable representation:

    {
      "task": "stack-cubes-on-pink-plate",
      "task_description": "Stack the cubes ...",
      "actions": [
        {"id": 1, "name": "pick",            "args": {"object": "green-cube-1", "from": "orange-cube-1"}},
        {"id": 2, "name": "place-on-object", "args": {"object": "green-cube-1", "target": "yellow-cube-1"}},
        ...
      ],
      "goal_conditions": ["(on-top-of orange-cube-1 pink-plate-1)", ...]
    }

Source-of-truth hierarchy
--------------------------
  task_plan.json        Planner full output; primary source of truth.
  action_sequence.json  Optional executor-friendly cache; can be regenerated.
  action_sequence.txt   Human-readable PDDL strings; used as last-resort fallback.

Use the unified entry point ActionSequence.load(path) – it accepts a session
directory, a task_plan.json, an action_sequence.json, or an action_sequence.txt
and does the right thing automatically.
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
        return cls(
            id=d["id"],
            name=d["name"],
            args=d["args"],
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
        domain = task_plan.updated_domain or ""
        param_map = _extract_action_params(domain) if domain else {}

        actions: List[Action] = []
        for i, pddl_str in enumerate(task_plan.action_sequence, start=1):
            name, raw_args = _parse_pddl_string(pddl_str)
            keys = param_map.get(name, [])
            args = {
                (keys[j] if j < len(keys) else f"arg{j}"): v
                for j, v in enumerate(raw_args)
            }
            actions.append(Action(id=i, name=name, args=args, pddl_str=pddl_str))

        return cls(
            task=_make_task_slug(task_plan.task_description),
            task_description=task_plan.task_description,
            actions=actions,
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
            name, raw_args = _parse_pddl_string(pddl_str)
            keys = param_map.get(name, [])
            args = {
                (keys[j] if j < len(keys) else f"arg{j}"): v
                for j, v in enumerate(raw_args)
            }
            actions.append(Action(id=i, name=name, args=args, pddl_str=pddl_str))
        return cls(
            task=task or _make_task_slug(task_description),
            task_description=task_description,
            actions=actions,
            goal_conditions=goal_conditions,
        )

    @classmethod
    def from_task_plan_json(cls, path: str) -> "ActionSequence":
        """Build from a task_plan.json file – the Planner source of truth.

        Uses action_sequence + updated_domain for arg-name resolution, plus
        goal_conditions and task_description if present.
        """
        data = json.loads(Path(path).read_text())
        return cls.from_pddl_strings(
            pddl_strings=data.get("action_sequence", []),
            task_description=data.get("task_description", ""),
            domain_content=data.get("updated_domain", ""),
            goal_conditions=data.get("goal_conditions"),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "ActionSequence":
        """Deserialize from an action_sequence.json file written by save()."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def load_from_dir(cls, session_dir: str) -> "ActionSequence":
        """Load from a persisted session directory.

        Priority order:
          1. action_sequence.json  (pre-built executor artifact)
          2. task_plan.json        (Planner source of truth – preferred fallback)
          3. action_sequence.txt + domain.pddl  (last resort)
        """
        session_path = Path(session_dir)

        action_seq_json = session_path / "action_sequence.json"
        if action_seq_json.exists():
            return cls.from_json_file(str(action_seq_json))

        task_plan_json = session_path / "task_plan.json"
        if task_plan_json.exists():
            return cls.from_task_plan_json(str(task_plan_json))

        txt_file = session_path / "action_sequence.txt"
        if not txt_file.exists():
            raise FileNotFoundError(
                f"No loadable ActionSequence found in {session_dir}. "
                "Expected action_sequence.json, task_plan.json, or action_sequence.txt."
            )
        domain_file = session_path / "domain.pddl"
        pddl_strings = [
            line.strip()
            for line in txt_file.read_text().splitlines()
            if line.strip() and not line.startswith(';')
        ]
        return cls.from_pddl_strings(
            pddl_strings,
            domain_content=domain_file.read_text() if domain_file.exists() else "",
        )

    @classmethod
    def load(cls, path: str) -> "ActionSequence":
        """Unified entry point – accepts any of:

          * A session directory  → load_from_dir()
          * A task_plan.json     → from_task_plan_json()
          * An action_sequence.json (has "actions" key) → from_json_file()
          * An action_sequence.txt  → from_pddl_strings()
        """
        p = Path(path)
        if p.is_dir():
            return cls.load_from_dir(path)
        if p.suffix == ".json":
            data = json.loads(p.read_text())
            if "actions" in data:
                # action_sequence.json: already in contract format
                return cls.from_dict(data)
            if "action_sequence" in data:
                # task_plan.json: Planner source of truth
                return cls.from_task_plan_json(path)
            raise ValueError(
                f"Unrecognized JSON schema in {path}. "
                "Expected 'actions' (action_sequence.json) or 'action_sequence' (task_plan.json)."
            )
        if p.suffix == ".txt":
            pddl_strings = [
                line.strip()
                for line in p.read_text().splitlines()
                if line.strip() and not line.startswith(';')
            ]
            return cls.from_pddl_strings(pddl_strings)
        raise ValueError(f"Cannot load ActionSequence from: {path}")

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
