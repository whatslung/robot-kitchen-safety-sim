import numpy as np
import pytest

from backend.multiview import CalibrationError, CameraCalibration, MultiViewFusion


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


def test_calibration_rejects_degenerate_valid_polygon():
    with pytest.raises(CalibrationError):
        CameraCalibration.from_points(
            image=[[0, 0], [1, 0], [1, 1], [0, 1]],
            world=[[0, 0], [1, 0], [1, 1], [0, 1]],
            valid_world_polygon=[[0, 0], [1, 0], [2, 0]],
        )


def test_calibration_accepts_valid_camera_projection_with_negative_svd_scale():
    image = [
        [-0.17218452, 1.13428753], [1.68595407, 1.83888089], [6.29462695, 3.58645793],
        [0.32296442, 0.30013605], [1.16884945, 0.44316661], [2.37089659, 0.64642056],
        [0.48646005, 0.02470352], [1.03596224, 0.08449146], [1.72032998, 0.15895328],
    ]
    world = [
        [-4.5, -3.8], [0, -3.8], [4.5, -3.8],
        [-4.5, 0], [0, 0], [4.5, 0],
        [-4.5, 3.8], [0, 3.8], [4.5, 3.8],
    ]

    calibration = CameraCalibration.from_points(
        image=image,
        world=world,
        valid_world_polygon=[[-5.75, -5.75], [5.75, -5.75], [5.75, 5.75], [-5.75, 5.75]],
    )

    assert calibration.reprojection_rms < 1e-6


def test_projection_outside_valid_floor_polygon_is_ignored():
    calibration = CameraCalibration.from_points(
        image=[[0, 0], [1, 0], [1, 1], [0, 1]],
        world=[[0, 0], [1, 0], [1, 1], [0, 1]],
        valid_world_polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
    )

    assert calibration.project((1.2, 0.5)) is None


def _identity_calibration():
    return CameraCalibration.from_points(
        image=[[0, 0], [1, 0], [1, 1], [0, 1]],
        world=[[0, 0], [1, 0], [1, 1], [0, 1]],
        valid_world_polygon=[[-5, -5], [5, -5], [5, 5], [-5, 5]],
    )


def _fusion(*camera_ids, gate=0.2):
    fusion = MultiViewFusion(gate=gate, fusion_window_ms=250, coast_ms=750, remove_ms=1500)
    for camera_id in camera_ids:
        fusion.calibrate(camera_id, _identity_calibration())
    return fusion


def _person(local_id, cx, cy=0.2, conf=0.9):
    return {
        "label": "person",
        "id": local_id,
        "conf": conf,
        "cx": cx,
        "cy": cy,
        "w": 0.1,
        "h": 0.2,
    }


def test_two_cameras_merge_same_person_into_one_global_id():
    fusion = _fusion("mvNW", "mvCenter")

    first = fusion.update("mvNW", [_person(2, 0.40)], 1000)
    second = fusion.update("mvCenter", [_person(7, 0.41)], 1080)

    assert first[0]["global_id"] == second[0]["global_id"]
    assert second[0]["world"] == pytest.approx({"x": 0.41, "z": 0.3})
    snapshot = fusion.snapshot(1080)
    assert len(snapshot) == 1
    assert snapshot[0]["sources"] == ["mvCenter", "mvNW"]


def test_same_numeric_local_id_in_different_cameras_is_not_a_global_key():
    fusion = _fusion("mvNW", "mvCenter", gate=0.15)

    near_left = fusion.update("mvNW", [_person(3, 0.2)], 1000)
    far_right = fusion.update("mvCenter", [_person(3, 0.8)], 1050)

    assert near_left[0]["global_id"] != far_right[0]["global_id"]
    assert len(fusion.snapshot(1050)) == 2


def test_measurement_outside_gate_starts_new_global_track():
    fusion = _fusion("mvNW", "mvCenter", gate=0.2)

    first = fusion.update("mvNW", [_person(1, 0.1)], 1000)
    distant = fusion.update("mvCenter", [_person(8, 0.7)], 1100)

    assert first[0]["global_id"] != distant[0]["global_id"]


def test_unassigned_local_ids_are_ephemeral_and_do_not_collapse_people():
    fusion = _fusion("mvNW", gate=0.2)

    first = fusion.update("mvNW", [_person(-1, 0.2), _person(-1, 0.8)], 1000)
    second = fusion.update("mvNW", [_person(-1, 0.22), _person(-1, 0.78)], 1100)

    assert [box["global_id"] for box in first] == [1, 2]
    assert [box["global_id"] for box in second] == [1, 2]
    assert ("mvNW", -1) not in fusion._local_bindings


def test_cross_camera_unbound_measurement_outside_fusion_window_starts_new_track():
    fusion = _fusion("mvNW", "mvCenter", gate=0.2)

    first = fusion.update("mvNW", [_person(1, 0.4)], 1000)
    late = fusion.update("mvCenter", [_person(7, 0.41)], 1400)

    assert first[0]["global_id"] != late[0]["global_id"]


def test_older_timestamp_does_not_rewind_global_state():
    fusion = _fusion("mvNW")
    fusion.update("mvNW", [_person(1, 0.2)], 1000)
    fusion.update("mvNW", [_person(1, 0.4)], 1200)
    before = fusion.snapshot(1200)

    stale_response = fusion.update("mvNW", [_person(1, 0.05)], 1100)

    assert "global_id" not in stale_response[0]
    assert fusion.snapshot(1200) == before


def test_reset_clears_tracks_but_preserves_calibration():
    fusion = _fusion("mvNW")
    fusion.update("mvNW", [_person(1, 0.2)], 1000)

    fusion.reset_tracks()

    assert fusion.tracks == {}
    assert sorted(fusion.calibrations) == ["mvNW"]
