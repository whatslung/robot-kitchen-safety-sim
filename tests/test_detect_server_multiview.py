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


class EndpointPredictor:
    def __init__(self):
        self.batch_calls = 0

    @staticmethod
    def _modes(history):
        x, z = history[-1]
        return [
            {
                "path": [(x + 0.1 * step, z + offset) for step in range(1, 13)],
                "w": probability,
                "sigma": [0.2] * 12,
            }
            for probability, offset in [(0.7, 0.0), (0.2, 0.3), (0.1, -0.3)]
        ]

    def predict_batch(self, histories):
        self.batch_calls += 1
        return [self._modes(history) for history in histories]

    def predict_modes(self, history):
        return self._modes(history)


def _api_history(offset=0.0):
    return [
        {"t": 1000 + index * 400, "x": offset + 0.1 * index, "z": 0.0}
        for index in range(8)
    ]


def test_predict_batches_global_tracks_once(monkeypatch):
    predictor = EndpointPredictor()
    monkeypatch.setattr(server, "_get_predictor", lambda: predictor)

    with TestClient(server.app) as client:
        response = client.post(
            "/predict",
            json={
                "tracks": [
                    {"id": 9, "history": _api_history(1.0), "age_ms": 10, "stale": False},
                    {"id": 2, "history": _api_history(), "age_ms": 20, "stale": False},
                ],
                "robot": {"x": -1.1, "z": 0.795, "stop_radius": 3.1, "slow_radius": 3.9},
            },
        )

    assert response.status_code == 200
    assert predictor.batch_calls == 1
    assert [track["id"] for track in response.json()["tracks"]] == [2, 9]
    assert all(track["source"] == "lstm" for track in response.json()["tracks"])


def test_predict_keeps_legacy_single_history_response(monkeypatch):
    predictor = EndpointPredictor()
    monkeypatch.setattr(server, "_get_predictor", lambda: predictor)
    history = [[0.2 * index, 0.0] for index in range(8)]

    with TestClient(server.app) as client:
        response = client.post("/predict", json={"hist": history})

    assert response.status_code == 200
    assert "modes" in response.json()
    assert len(response.json()["modes"]) == 3


def test_predictor_can_be_disabled_without_weight_download(monkeypatch):
    import huggingface_hub

    monkeypatch.setenv("PREDICT_DISABLE_MODEL", "1")
    monkeypatch.setattr(server, "_PREDICTOR", None)
    monkeypatch.setattr(server, "_PREDICTOR_ERR", None)
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("weight download attempted")),
    )

    assert server._get_predictor() is None
    assert server._PREDICTOR_ERR == "disabled by PREDICT_DISABLE_MODEL"
