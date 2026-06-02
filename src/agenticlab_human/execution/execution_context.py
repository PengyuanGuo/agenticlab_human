"""Execution-time scene and affordance cache.

The context is the bridge between semantic actions and perception/grasp
backends. It keeps action.py free of YOLO, AnyGrasp, camera, and model imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple, runtime_checkable

from agenticlab_human.core.action_sequence import ActionSequence
from agenticlab_human.execution.action_utils import (
    assign_grasps_to_objects,
    extract_interested_objects,
    extract_pick_objects,
)
from agenticlab_human.perception.backend.grasp_backend import GraspBackend, GraspCandidate
from agenticlab_human.perception.backend.perception_backend import BBox, PerceptionBackend


@runtime_checkable
class SceneProvider(Protocol):
    """Provider that captures a synchronized RGB-D scene."""

    def capture_rgbd(self) -> Tuple[Any, Any]:
        """Return (rgb, depth)."""


@dataclass
class PrepareReport:
    """Summary of the executor preparation step."""

    prepared: bool
    interested_objects: List[str] = field(default_factory=list)
    pick_objects: List[str] = field(default_factory=list)
    bbox_counts: Dict[str, int] = field(default_factory=dict)
    grasp_counts: Dict[str, int] = field(default_factory=dict)
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prepared": self.prepared,
            "interested_objects": self.interested_objects,
            "pick_objects": self.pick_objects,
            "bbox_counts": self.bbox_counts,
            "grasp_counts": self.grasp_counts,
            "message": self.message,
            "error": self.error,
        }


class ExecutionContext:
    """Cache bboxes, grasp candidates, object poses, and stale state."""

    def __init__(
        self,
        scene_provider: Optional[SceneProvider] = None,
        detector: Optional[PerceptionBackend] = None,
        grasp_planner: Optional[GraspBackend] = None,
    ) -> None:
        self.scene_provider = scene_provider
        self.detector = detector
        self.grasp_planner = grasp_planner
        self.rgb: Any = None
        self.depth: Any = None
        self.bboxes: Dict[str, List[BBox]] = {}
        self.grasps: Dict[str, List[GraspCandidate]] = {}
        self.object_states: Dict[str, Dict[str, Any]] = {}
        self.stale_objects: Set[str] = set()
        self.last_prepare_report: Optional[PrepareReport] = None

    def prepare_for_sequence(self, action_sequence: ActionSequence) -> PrepareReport:
        """Capture once, detect all referenced objects, and cache scene grasps."""

        interested_objects = extract_interested_objects(action_sequence)
        pick_objects = extract_pick_objects(action_sequence)

        if not self.scene_provider:
            report = PrepareReport(
                prepared=False,
                interested_objects=interested_objects,
                pick_objects=pick_objects,
                message="No scene provider configured; preparation skipped.",
            )
            self.last_prepare_report = report
            return report

        if not self.detector:
            report = PrepareReport(
                prepared=False,
                interested_objects=interested_objects,
                pick_objects=pick_objects,
                message="No perception backend configured; preparation skipped.",
            )
            self.last_prepare_report = report
            return report

        self.rgb, self.depth = self.scene_provider.capture_rgbd()
        self.bboxes = self.detector.detect(self.rgb, interested_objects)
        self.object_states = {
            name: {"stale": False, "last_location": None}
            for name in interested_objects
        }
        self.stale_objects.clear()

        if self.grasp_planner:
            all_grasps = self.grasp_planner.plan_scene(self.rgb, self.depth)
            self.grasps = assign_grasps_to_objects(
                all_grasps=all_grasps,
                bboxes=self.bboxes,
                object_names=pick_objects,
            )
        else:
            self.grasps = {name: [] for name in pick_objects}

        report = PrepareReport(
            prepared=True,
            interested_objects=interested_objects,
            pick_objects=pick_objects,
            bbox_counts={name: len(self.bboxes.get(name, [])) for name in interested_objects},
            grasp_counts={name: len(self.grasps.get(name, [])) for name in pick_objects},
            message="Preparation completed.",
        )
        self.last_prepare_report = report
        return report

    def get_bboxes(self, object_name: str) -> List[BBox]:
        return self.bboxes.get(object_name, [])

    def get_bbox(self, object_name: str) -> Optional[BBox]:
        candidates = self.get_bboxes(object_name)
        if not candidates:
            return None
        return max(candidates, key=lambda b: b.confidence if b.confidence is not None else 0.0)

    def get_grasps(self, object_name: str) -> List[GraspCandidate]:
        return self.grasps.get(object_name, [])

    def get_best_grasp(self, object_name: str) -> Optional[GraspCandidate]:
        candidates = self.get_grasps(object_name)
        return candidates[0] if candidates else None

    def get_object_pose(self, object_name: str) -> Any:
        bbox = self.get_bbox(object_name)
        if bbox and bbox.center_3d is not None:
            return bbox.center_3d
        return self.object_states.get(object_name, {}).get("pose")

    def mark_stale(self, object_name: str) -> None:
        self.stale_objects.add(object_name)
        self.object_states.setdefault(object_name, {})["stale"] = True

    def mark_object_location(self, object_name: str, location_name: Optional[str]) -> None:
        state = self.object_states.setdefault(object_name, {})
        state["last_location"] = location_name
        state["stale"] = True
        self.stale_objects.add(object_name)

    def is_stale(self, object_name: str) -> bool:
        return object_name in self.stale_objects

    def refresh_object(self, object_name: str) -> bool:
        """Refresh bbox and object-level grasps when backends support it."""

        if not self.scene_provider or not self.detector:
            return False
        self.rgb, self.depth = self.scene_provider.capture_rgbd()
        refreshed = self.detector.detect(self.rgb, [object_name])
        self.bboxes[object_name] = refreshed.get(object_name, [])

        bbox = self.get_bbox(object_name)
        if self.grasp_planner and bbox:
            self.grasps[object_name] = self.grasp_planner.plan_for_object(
                self.rgb,
                self.depth,
                bbox,
            )

        self.stale_objects.discard(object_name)
        self.object_states.setdefault(object_name, {})["stale"] = False
        return True

    def describe_cache(self) -> Dict[str, Any]:
        return {
            "prepared": bool(self.last_prepare_report and self.last_prepare_report.prepared),
            "bbox_counts": {name: len(items) for name, items in self.bboxes.items()},
            "grasp_counts": {name: len(items) for name, items in self.grasps.items()},
            "stale_objects": sorted(self.stale_objects),
        }
