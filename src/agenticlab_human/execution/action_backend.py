"""Robot backend contract for semantic action execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from agenticlab_human.perception.backend.grasp_backend import GraspCandidate
from agenticlab_human.perception.backend.perception_backend import BBox


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass
class ActionResult:
    """Result returned by one semantic action."""

    success: bool
    action_name: str
    message: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action_name": self.action_name,
            "message": self.message,
            "error": self.error,
            "metadata": _jsonable(self.metadata),
        }


@dataclass
class ExecutionReport:
    """Structured report for a whole ActionSequence execution."""

    success: bool
    task: str
    total_actions: int
    results: List[ActionResult] = field(default_factory=list)
    prepared: bool = False
    failed_action_id: Optional[int] = None
    failed_action_name: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "task": self.task,
            "total_actions": self.total_actions,
            "results": [result.to_dict() for result in self.results],
            "prepared": self.prepared,
            "failed_action_id": self.failed_action_id,
            "failed_action_name": self.failed_action_name,
            "error": self.error,
            "metadata": _jsonable(self.metadata),
        }


@runtime_checkable
class ActionBackend(Protocol):
    """Robot-specific backend consumed by ActionExecutor."""

    def initialize(self) -> None:
        """Connect or initialize resources before execution."""

    def shutdown(self, move_home: bool = False) -> None:
        """Release resources after execution."""

    def pick(
        self,
        object_name: str,
        from_name: Optional[str] = None,
        grasp_candidates: Optional[Sequence[GraspCandidate]] = None,
        object_bbox: Optional[BBox] = None,
        object_pose: Any = None,
    ) -> ActionResult:
        """Pick an object using cached grasp affordances."""

    def place_on_object(
        self,
        object_name: str,
        target_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        """Place an object on another object."""

    def place_on_surface(
        self,
        object_name: str,
        surface_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        """Place an object on a named surface."""

    def place_in_container(
        self,
        object_name: str,
        container_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        """Place an object into a named container."""

    def move_home(self) -> ActionResult:
        """Move to a safe home pose."""

    def get_eef_pose(self) -> Any:
        """Return current end-effector pose."""


class DryRunActionBackend:
    """Backend that records semantic calls without moving hardware."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def initialize(self) -> None:
        self.calls.append({"name": "initialize"})

    def shutdown(self, move_home: bool = False) -> None:
        self.calls.append({"name": "shutdown", "move_home": move_home})

    def pick(
        self,
        object_name: str,
        from_name: Optional[str] = None,
        grasp_candidates: Optional[Sequence[GraspCandidate]] = None,
        object_bbox: Optional[BBox] = None,
        object_pose: Any = None,
    ) -> ActionResult:
        self.calls.append({"name": "pick", "object": object_name, "from": from_name})
        return ActionResult(
            success=True,
            action_name="pick",
            message=f"Dry-run pick {object_name}.",
            metadata={
                "object": object_name,
                "from": from_name,
                "grasp_count": len(grasp_candidates or []),
                "has_bbox": object_bbox is not None,
                "has_pose": object_pose is not None,
            },
        )

    def place_on_object(
        self,
        object_name: str,
        target_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        self.calls.append({"name": "place_on_object", "object": object_name, "target": target_name})
        return ActionResult(
            success=True,
            action_name="place-on-object",
            message=f"Dry-run place {object_name} on {target_name}.",
            metadata={
                "object": object_name,
                "target": target_name,
                "has_target_bbox": target_bbox is not None,
                "has_target_pose": target_pose is not None,
            },
        )

    def place_on_surface(
        self,
        object_name: str,
        surface_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        self.calls.append({"name": "place_on_surface", "object": object_name, "surface": surface_name})
        return ActionResult(
            success=True,
            action_name="place-on-surface",
            message=f"Dry-run place {object_name} on {surface_name}.",
            metadata={
                "object": object_name,
                "surface": surface_name,
                "has_target_bbox": target_bbox is not None,
                "has_target_pose": target_pose is not None,
            },
        )

    def place_in_container(
        self,
        object_name: str,
        container_name: str,
        target_bbox: Optional[BBox] = None,
        target_pose: Any = None,
    ) -> ActionResult:
        self.calls.append({"name": "place_in_container", "object": object_name, "container": container_name})
        return ActionResult(
            success=True,
            action_name="place-in-container",
            message=f"Dry-run place {object_name} in {container_name}.",
            metadata={
                "object": object_name,
                "container": container_name,
                "has_target_bbox": target_bbox is not None,
                "has_target_pose": target_pose is not None,
            },
        )

    def move_home(self) -> ActionResult:
        self.calls.append({"name": "move_home"})
        return ActionResult(success=True, action_name="move-home", message="Dry-run move home.")

    def get_eef_pose(self) -> Any:
        return None
