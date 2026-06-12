"""Data contracts for the no-planning execution pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticlab_human.execution.robot.x5.contracts import RGBDFrame


@dataclass(frozen=True)
class SceneSnapshot:
    """One aligned RGB-D frame and the files saved for downstream services."""

    frame: RGBDFrame
    rgb_path: Path
    depth_path: Path
    metadata_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame.frame_id,
            "timestamp_ns": self.frame.timestamp_ns,
            "rgb_path": str(self.rgb_path),
            "depth_path": str(self.depth_path),
            "metadata_path": str(self.metadata_path),
            "intrinsics": self.frame.intrinsics.model_dump(),
        }
