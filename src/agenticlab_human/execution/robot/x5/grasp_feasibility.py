"""X5 grasp reachability filtering using server-side SDK IK checks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from agenticlab_human.execution.robot.x5.x5_remote_backend import (
    build_world_tcp_pick_poses,
)
from agenticlab_human.perception.grasping.grasp_backend import GraspCandidate


class X5IKClient(Protocol):
    """Small subset of X5HTTPClient needed for check-only IK."""

    def get_state(self, arm: str = "all") -> Any: ...

    def check_ik_point(
        self,
        arm: str,
        tcp_pose_xyz_rotvec: list[float],
        *,
        inverse_type: int = 0,
        seed_joints_rad: list[float] | None = None,
        request_id: str | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class GraspFeasibilityConfig:
    joint_limits_deg: Sequence[Sequence[float]]
    soft_limit_margin_deg: float = 3.0
    movel_sample_step_m: float = 0.01
    inverse_type: int = 0
    max_candidates: int | None = None


@dataclass(frozen=True)
class GraspFeasibilityEvaluation:
    candidate_index: int
    candidate_score: float | None
    feasible: bool
    reason: str | None = None
    approach_pose_xyz_rotvec: list[float] | None = None
    grasp_pose_xyz_rotvec: list[float] | None = None
    ik_joints_deg: list[list[float]] = field(default_factory=list)
    min_soft_margin_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "candidate_score": self.candidate_score,
            "feasible": self.feasible,
            "reason": self.reason,
            "approach_pose_xyz_rotvec": self.approach_pose_xyz_rotvec,
            "grasp_pose_xyz_rotvec": self.grasp_pose_xyz_rotvec,
            "ik_joints_deg": self.ik_joints_deg,
            "min_soft_margin_deg": self.min_soft_margin_deg,
        }


@dataclass(frozen=True)
class GraspFeasibilitySelection:
    feasible_candidates: list[GraspCandidate]
    evaluations: list[GraspFeasibilityEvaluation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible_count": len(self.feasible_candidates),
            "evaluations": [
                evaluation.to_dict()
                for evaluation in self.evaluations
            ],
        }


def select_feasible_grasps(
    *,
    candidates: Sequence[GraspCandidate],
    ik_client: X5IKClient,
    arm: str,
    T_world_camera: Any,
    approach_distance_m: float,
    config: GraspFeasibilityConfig,
    seed_joints_rad: Sequence[float] | None = None,
) -> GraspFeasibilitySelection:
    """Return IK-feasible grasps sorted by the original grasp score."""

    limits = _normalize_soft_limits(
        config.joint_limits_deg,
        config.soft_limit_margin_deg,
    )
    initial_seed = (
        _finite_vector(seed_joints_rad, 7, "seed_joints_rad").tolist()
        if seed_joints_rad is not None
        else _current_joints_rad(ik_client, arm)
    )
    indexed_candidates = list(enumerate(candidates))
    if config.max_candidates is not None:
        indexed_candidates = indexed_candidates[: int(config.max_candidates)]

    feasible_candidates: list[GraspCandidate] = []
    evaluations: list[GraspFeasibilityEvaluation] = []
    for index, candidate in indexed_candidates:
        candidate_result = _evaluate_candidate(
            candidate,
            candidate_index=index,
            ik_client=ik_client,
            arm=arm,
            T_world_camera=T_world_camera,
            approach_distance_m=approach_distance_m,
            config=config,
            soft_limits_deg=limits,
            seed_joints_rad=initial_seed,
        )
        evaluations.append(candidate_result.evaluation)
        if candidate_result.evaluation.feasible:
            feasible_candidates.append(
                _candidate_with_feasibility(candidate, candidate_result)
            )

    feasible_candidates.sort(
        key=lambda candidate: (
            candidate.score if candidate.score is not None else float("-inf")
        ),
        reverse=True,
    )
    return GraspFeasibilitySelection(
        feasible_candidates=feasible_candidates,
        evaluations=evaluations,
    )


@dataclass(frozen=True)
class _CandidateEvaluationResult:
    evaluation: GraspFeasibilityEvaluation
    plan: dict[str, np.ndarray]


def _evaluate_candidate(
    candidate: GraspCandidate,
    *,
    candidate_index: int,
    ik_client: X5IKClient,
    arm: str,
    T_world_camera: Any,
    approach_distance_m: float,
    config: GraspFeasibilityConfig,
    soft_limits_deg: np.ndarray,
    seed_joints_rad: list[float],
) -> _CandidateEvaluationResult:
    plan = build_world_tcp_pick_poses(
        T_world_camera,
        candidate.pose,
        approach_distance_m=approach_distance_m,
    )
    approach_pose = _pose_list(plan["approach_pose_xyz_rotvec"])
    grasp_pose = _pose_list(plan["grasp_pose_xyz_rotvec"])
    ik_joints_deg: list[list[float]] = []
    min_soft_margin: float | None = None
    last_seed = list(seed_joints_rad)

    try:
        approach_ik = _check_pose_ik(
            ik_client,
            arm,
            approach_pose,
            inverse_type=config.inverse_type,
            seed_joints_rad=last_seed,
            soft_limits_deg=soft_limits_deg,
        )
        ik_joints_deg.append(approach_ik.joints_deg)
        min_soft_margin = _min_optional_margin(
            min_soft_margin,
            approach_ik.min_soft_margin_deg,
        )
        last_seed = approach_ik.joints_rad

        for sample_pose in _sample_movel_poses(
            approach_pose,
            grasp_pose,
            step_m=config.movel_sample_step_m,
        ):
            sample_ik = _check_pose_ik(
                ik_client,
                arm,
                sample_pose,
                inverse_type=config.inverse_type,
                seed_joints_rad=last_seed,
                soft_limits_deg=soft_limits_deg,
            )
            ik_joints_deg.append(sample_ik.joints_deg)
            min_soft_margin = _min_optional_margin(
                min_soft_margin,
                sample_ik.min_soft_margin_deg,
            )
            last_seed = sample_ik.joints_rad

        evaluation = GraspFeasibilityEvaluation(
            candidate_index=candidate_index,
            candidate_score=candidate.score,
            feasible=True,
            approach_pose_xyz_rotvec=approach_pose,
            grasp_pose_xyz_rotvec=grasp_pose,
            ik_joints_deg=ik_joints_deg,
            min_soft_margin_deg=min_soft_margin,
        )
    except Exception as exc:
        evaluation = GraspFeasibilityEvaluation(
            candidate_index=candidate_index,
            candidate_score=candidate.score,
            feasible=False,
            reason=str(exc),
            approach_pose_xyz_rotvec=approach_pose,
            grasp_pose_xyz_rotvec=grasp_pose,
            ik_joints_deg=ik_joints_deg,
            min_soft_margin_deg=min_soft_margin,
        )
    return _CandidateEvaluationResult(evaluation=evaluation, plan=plan)


@dataclass(frozen=True)
class _IKCheck:
    joints_rad: list[float]
    joints_deg: list[float]
    min_soft_margin_deg: float


def _check_pose_ik(
    ik_client: X5IKClient,
    arm: str,
    pose_xyz_rotvec: list[float],
    *,
    inverse_type: int,
    seed_joints_rad: list[float],
    soft_limits_deg: np.ndarray,
) -> _IKCheck:
    response = ik_client.check_ik_point(
        arm,
        pose_xyz_rotvec,
        inverse_type=inverse_type,
        seed_joints_rad=seed_joints_rad,
    )
    if not bool(getattr(response, "success", False)):
        raise RuntimeError(getattr(response, "error", None) or "IK check failed")

    metadata = getattr(response, "metadata", {}) or {}
    joints_rad = metadata.get("ik_joints_rad")
    joints_deg = metadata.get("ik_joints_deg")
    if joints_deg is None and joints_rad is not None:
        joints_deg = np.degrees(np.asarray(joints_rad, dtype=float)).tolist()
    if joints_rad is None and joints_deg is not None:
        joints_rad = np.radians(np.asarray(joints_deg, dtype=float)).tolist()
    if joints_rad is None or joints_deg is None:
        raise RuntimeError("IK check response is missing joint values")

    joints_deg = _finite_vector(joints_deg, 7, "ik_joints_deg").tolist()
    joints_rad = _finite_vector(joints_rad, 7, "ik_joints_rad").tolist()
    min_margin = _check_soft_limits(joints_deg, soft_limits_deg)
    return _IKCheck(
        joints_rad=joints_rad,
        joints_deg=joints_deg,
        min_soft_margin_deg=min_margin,
    )


def _sample_movel_poses(
    approach_pose: Sequence[float],
    grasp_pose: Sequence[float],
    *,
    step_m: float,
) -> list[list[float]]:
    if step_m <= 0.0:
        raise ValueError("movel_sample_step_m must be positive")
    start = _finite_vector(approach_pose, 6, "approach_pose")
    end = _finite_vector(grasp_pose, 6, "grasp_pose")
    distance_m = float(np.linalg.norm(end[:3] - start[:3]))
    step_count = max(1, int(math.ceil(distance_m / float(step_m))))
    return [
        (start + (end - start) * (index / step_count)).tolist()
        for index in range(1, step_count + 1)
    ]


def _candidate_with_feasibility(
    candidate: GraspCandidate,
    result: _CandidateEvaluationResult,
) -> GraspCandidate:
    metadata = dict(candidate.metadata)
    metadata["x5_feasibility"] = result.evaluation.to_dict()
    return GraspCandidate(
        pose=candidate.pose,
        score=candidate.score,
        image_xy=candidate.image_xy,
        object_name=candidate.object_name,
        metadata=metadata,
    )


def _current_joints_rad(ik_client: X5IKClient, arm: str) -> list[float]:
    response = ik_client.get_state(arm)
    if not bool(getattr(response, "success", False)):
        raise RuntimeError(getattr(response, "error", None) or "get_state failed")
    state_after = getattr(response, "state_after")
    joints_rad = state_after.arms[arm].joints_rad
    return _finite_vector(joints_rad, 7, "current_joints_rad").tolist()


def _normalize_soft_limits(
    joint_limits_deg: Sequence[Sequence[float]],
    margin_deg: float,
) -> np.ndarray:
    limits = np.asarray(joint_limits_deg, dtype=float)
    if limits.shape != (7, 2) or not np.all(np.isfinite(limits)):
        raise ValueError("joint_limits_deg must contain 7 finite [lower, upper] pairs")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("joint_limits_deg lower bounds must be less than upper bounds")
    margin = float(margin_deg)
    if margin < 0.0 or not math.isfinite(margin):
        raise ValueError("soft_limit_margin_deg must be finite and non-negative")
    soft_limits = limits.copy()
    soft_limits[:, 0] += margin
    soft_limits[:, 1] -= margin
    if np.any(soft_limits[:, 0] >= soft_limits[:, 1]):
        raise ValueError("soft_limit_margin_deg collapses at least one joint range")
    return soft_limits


def _check_soft_limits(
    joints_deg: Sequence[float],
    soft_limits_deg: np.ndarray,
) -> float:
    joints = _finite_vector(joints_deg, 7, "joints_deg")
    lower = soft_limits_deg[:, 0]
    upper = soft_limits_deg[:, 1]
    violations = np.flatnonzero((joints < lower) | (joints > upper))
    if len(violations):
        index = int(violations[0])
        raise RuntimeError(
            f"joint {index + 1} IK {joints[index]:.3f} deg is outside "
            f"soft limit [{lower[index]:.3f}, {upper[index]:.3f}] deg"
        )
    return float(np.min(np.minimum(joints - lower, upper - joints)))


def _min_optional_margin(
    current: float | None,
    candidate: float,
) -> float:
    return candidate if current is None else min(current, candidate)


def _pose_list(value: Any) -> list[float]:
    return _finite_vector(value, 6, "pose_xyz_rotvec").tolist()


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values")
    return vector
