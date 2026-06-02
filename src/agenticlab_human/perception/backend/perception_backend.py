"""Perception abstractions used by the execution layer.

Concrete detectors such as YOLOWorld or GroundingDINO should implement this
interface outside action.py, so the executor can be tested without model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable


@dataclass
class BBox:
    """A 2D object bounding box with optional 3D pose information."""

    label: str
    xyxy: Tuple[float, float, float, float]
    confidence: Optional[float] = None
    center_3d: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center_xy(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def contains_xy(self, xy: Tuple[float, float]) -> bool:
        x, y = xy
        x1, y1, x2, y2 = self.xyxy
        return x1 <= x <= x2 and y1 <= y <= y2


@runtime_checkable
class PerceptionBackend(Protocol):
    """Detect named objects in an RGB image."""

    def detect(self, rgb: Any, object_names: Sequence[str]) -> Dict[str, List[BBox]]:
        """Return object-name keyed bbox candidates."""


class EmptyPerceptionBackend:
    """Detector stub for dry-run execution and tests."""

    def detect(self, rgb: Any, object_names: Sequence[str]) -> Dict[str, List[BBox]]:
        return {name: [] for name in object_names}
