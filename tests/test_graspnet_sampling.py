import pytest
import torch


pytest.importorskip("MinkowskiEngine")
pytest.importorskip("pointnet2._ext")

from graspness_unofficial.models.graspnet import (  # noqa: E402
    NoGraspablePointsError,
    _graspable_sample_indices,
)


def test_graspable_sample_indices_repeat_small_point_set():
    indices = _graspable_sample_indices(3, 8, torch.device("cpu"))

    assert indices.dtype == torch.int32
    assert indices.tolist() == [[0, 1, 2, 0, 1, 2, 0, 1]]


def test_graspable_sample_indices_reject_empty_point_set():
    with pytest.raises(NoGraspablePointsError, match="no graspable seed points"):
        _graspable_sample_indices(0, 8, torch.device("cpu"))


def test_graspable_sample_indices_uses_fps_for_large_point_set():
    assert _graspable_sample_indices(9, 8, torch.device("cpu")) is None
