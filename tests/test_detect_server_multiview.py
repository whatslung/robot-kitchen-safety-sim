import os

import pytest
from fastapi.testclient import TestClient

os.environ["DETECT_DISABLE_MODEL"] = "1"

from backend import detect_server as server  # noqa: E402
from backend.multiview import MultiViewFusion  # noqa: E402


CALIBRATION = {
    "camera": "mvNW",
    "points": [
        {"image": [0, 0], "world": [0, 0]},
        {"image": [1, 0], "world": [1, 0]},
        {"image": [1, 1], "world": [1, 1]},
        {"image": [0, 1], "world": [0, 1]},
    ],
    "valid_world_polygon": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
}


@pytest.fixture(autouse=True)
def isolated_server_state(monkeypatch):
    server.CAMS.clear()
    server.FUSION = MultiViewFusion(gate=0.8, fusion_window_ms=250, coast_ms=750, remove_ms=1500)
    monkeypatch.setattr(
        server,
        "run_detect",
        lambda _image: [{"label": "person", "conf": 0.9, "cx": 0.4, "cy": 0.2, "w": 0.1, "h": 0.2}],
    )
    monkeypatch.setattr(
        server,
        "track_and_measure",
        lambda detections, _w, _h, _camera, _t: [
            {**detection, "id": index + 10, "vx": 0.0, "vy": 0.0}
            for index, detection in enumerate(detections)
        ],
    )


def test_calibrated_detect_returns_world_global_id_and_tracks():
    with TestClient(server.app) as client:
        calibration = client.post("/calibrate", json=CALIBRATION)
        response = client.post(
            "/detect",
            json={"camera": "mvNW", "image": "unused", "t": 1000, "seq": 14},
        )

    assert calibration.status_code == 200
    assert calibration.json()["camera"] == "mvNW"
    assert "reprojection_rms" in calibration.json()
    assert response.status_code == 200
    payload = response.json()
    assert payload["seq"] == 14
    assert payload["boxes"][0]["global_id"] == 1
    assert payload["boxes"][0]["world"] == pytest.approx({"x": 0.4, "z": 0.3})
    assert len(payload["global_tracks"]) == 1


def test_uncalibrated_detect_preserves_legacy_box_contract():
    with TestClient(server.app) as client:
        response = client.post(
            "/detect",
            json={"camera": "legacy", "image": "unused", "t": 1000, "seq": 2},
        )

    assert response.status_code == 200
    box = response.json()["boxes"][0]
    assert "world" not in box
    assert "global_id" not in box
    assert response.json()["global_tracks"] == []


def test_track_reset_keeps_calibration_visible_in_health():
    with TestClient(server.app) as client:
        assert client.post("/calibrate", json=CALIBRATION).status_code == 200
        client.post("/detect", json={"camera": "mvNW", "image": "unused", "t": 1000})
        reset = client.post("/tracks/reset")
        health = client.get("/health")

    assert reset.status_code == 200
    assert reset.json() == {"ok": True, "calibrated_cameras": ["mvNW"]}
    assert health.json()["calibrated_cameras"] == ["mvNW"]
    assert health.json()["global_track_count"] == 0


def test_health_reports_integer_global_track_count_and_camera_ages():
    with TestClient(server.app) as client:
        assert client.post("/calibrate", json=CALIBRATION).status_code == 200
        client.post("/detect", json={"camera": "mvNW", "image": "unused", "t": 1000})
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["global_track_count"], int)
    assert isinstance(payload["camera_update_age_ms"]["mvNW"], int)


def test_malformed_calibration_has_stable_422_error():
    with TestClient(server.app) as client:
        response = client.post("/calibrate", json={"camera": "mvNW", "points": []})

    assert response.status_code == 422
    assert response.json() == {"error": "invalid calibration"}
