"""Small pure helpers for semantic action execution."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

from agenticlab_human.core.action_sequence import ActionSequence
from agenticlab_human.perception.backend.grasp_backend import GraspCandidate
from agenticlab_human.perception.backend.perception_backend import BBox


_OBJECT_ARG_KEYS = {
    "object",
    "from",
    "target",
    "surface",
    "location",
    "container",
    "to",
}


def extract_interested_objects(action_sequence: ActionSequence) -> List[str]:
    """Return all object-like names referenced by an ActionSequence."""

    names: Set[str] = set()
    for action in action_sequence.actions:
        for key, value in action.args.items():
            if key in _OBJECT_ARG_KEYS and value:
                names.add(value)
    return sorted(names)


def extract_pick_objects(action_sequence: ActionSequence) -> List[str]:
    """Return objects that need grasp candidates."""

    names: Set[str] = set()
    for action in action_sequence.actions:
        if action.name == "pick" and action.args.get("object"):
            names.add(action.args["object"])
    return sorted(names)


def assign_grasps_to_objects(
    all_grasps: Sequence[GraspCandidate],
    bboxes: Dict[str, Sequence[BBox]],
    object_names: Iterable[str],
) -> Dict[str, List[GraspCandidate]]:
    """Assign full-scene grasp candidates to objects by image projection.

    A concrete AnyGrasp integration can attach image coordinates to each grasp
    candidate after projecting the grasp center into the RGB frame. Candidates
    without image coordinates are ignored here and can still be handled by a
    backend-specific fallback.
    """

    assigned: Dict[str, List[GraspCandidate]] = {name: [] for name in object_names}
    for grasp in all_grasps:
        if grasp.object_name:
            assigned.setdefault(grasp.object_name, []).append(grasp)
            continue
        if grasp.image_xy is None:
            continue
        for object_name in assigned:
            if any(bbox.contains_xy(grasp.image_xy) for bbox in bboxes.get(object_name, [])):
                assigned[object_name].append(grasp)
                break

    for candidates in assigned.values():
        candidates.sort(key=lambda g: g.score if g.score is not None else 0.0, reverse=True)
    return assigned
