"""Grasp planning abstractions used by the execution layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from agenticlab_human.perception.backend.perception_backend import BBox


@dataclass
class GraspCandidate:
    """A grasp pose candidate for an object."""

    pose: Any
    score: Optional[float] = None
    image_xy: Optional[Tuple[float, float]] = None
    object_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GraspBackend(Protocol):
    """Plan grasp candidates for one detected object."""

    def plan_for_object(self, rgb: Any, depth: Any, bbox: BBox) -> List[GraspCandidate]:
        """Plan grasp candidates inside the object's bounding box."""


class EmptyGraspBackend:
    """Grasp planner stub for dry-run execution and tests."""

    def plan_for_object(self, rgb: Any, depth: Any, bbox: BBox) -> List[GraspCandidate]:
        return []
