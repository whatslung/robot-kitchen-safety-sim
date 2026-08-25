import numpy as np
import pytest

from backend.multiview import CalibrationError, CameraCalibration


def test_homography_restores_unseen_floor_point():
    calibration = CameraCalibration.from_points(
        image=[[0.1, 0.2], [0.9, 0.2], [0.8, 0.9], [0.2, 0.9]],
        world=[[-4, -3], [4, -3], [3, 5], [-3, 5]],
        valid_world_polygon=[[-4, -3], [4, -3], [3, 5], [-3, 5]],
    )

    assert np.allclose(calibration.project((0.5, 0.55)), (0.0, 1.0), atol=1e-6)
    assert calibration.reprojection_rms < 1e-8


def test_calibration_rejects_collinear_points():
    with pytest.raises(CalibrationError):
        CameraCalibration.from_points(
            image=[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]],
            world=[[0, 0], [1, 1], [2, 2], [3, 3]],
            valid_world_polygon=[[-1, -1], [4, -1], [4, 4], [-1, 4]],
        )


def test_projection_outside_valid_floor_polygon_is_ignored():
    calibration = CameraCalibration.from_points(
        image=[[0, 0], [1, 0], [1, 1], [0, 1]],
        world=[[0, 0], [1, 0], [1, 1], [0, 1]],
        valid_world_polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
    )

    assert calibration.project((1.2, 0.5)) is None
