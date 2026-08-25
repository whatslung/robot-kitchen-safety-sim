import math

from backend.prediction_contract import predict_global_tracks, risk_entry


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict_batch(self, histories):
        self.calls.append(histories)
        outputs = []
        for history in histories:
            x, z = history[-1]
            outputs.append(
                [
                    {
                        "path": [(x + 0.1 * step, z + offset) for step in range(1, 13)],
                        "w": probability,
                        "sigma": [0.1 + 0.01 * step for step in range(12)],
                    }
                    for probability, offset in [(0.6, 0.0), (0.3, 0.4), (0.1, -0.4)]
                ]
            )
        return outputs


def _long_history(start_x=0.0):
    return [
        {"t": 1000 + index * 400, "x": start_x + index * 0.1, "z": 0.0}
        for index in range(8)
    ]


def test_risk_entry_returns_first_time_per_radius():
    mode = {"prob": 1.0, "path": [[0.4, 3.5, 0.0], [0.8, 2.8, 0.0], [1.2, 1.9, 0.0]]}

    risk = risk_entry([mode], center=(0, 0), stop_radius=2.0, slow_radius=3.1)

    assert risk == {"stop_entry_s": 1.2, "slow_entry_s": 0.8}


def test_batch_prediction_orders_ids_and_formats_three_lstm_modes():
    predictor = FakePredictor()
    tracks = [
        {"id": 9, "history": _long_history(1.0), "age_ms": 10, "stale": False},
        {"id": 2, "history": _long_history(0.0), "age_ms": 20, "stale": False},
    ]

    results = predict_global_tracks(
        tracks,
        predictor=predictor,
        robot={"x": 0.0, "z": 0.0, "stop_radius": 1.0, "slow_radius": 2.0},
    )

    assert [result["id"] for result in results] == [2, 9]
    assert len(predictor.calls) == 1
    assert len(predictor.calls[0]) == 2
    for result in results:
        assert result["source"] == "lstm"
        assert len(result["modes"]) == 3
        probabilities = [mode["prob"] for mode in result["modes"]]
        assert probabilities == sorted(probabilities, reverse=True)
        assert all(len(row) == 3 for mode in result["modes"] for row in mode["path"])
        assert result["modes"][0]["path"][0][0] == 0.4


def test_two_observations_use_finite_constant_velocity_fallback():
    results = predict_global_tracks(
        [
            {
                "id": 4,
                "history": [
                    {"t": 1000, "x": 0.0, "z": 0.0},
                    {"t": 1400, "x": 0.2, "z": 0.1},
                ],
                "age_ms": 30,
                "stale": False,
            }
        ],
        predictor=FakePredictor(),
        robot={"x": -1.1, "z": 0.795, "stop_radius": 3.1, "slow_radius": 3.9},
    )

    assert results[0]["source"] == "kalman"
    assert len(results[0]["modes"]) == 1

    def assert_finite(value):
        if isinstance(value, dict):
            for child in value.values():
                assert_finite(child)
        elif isinstance(value, list):
            for child in value:
                assert_finite(child)
        elif isinstance(value, (int, float)):
            assert math.isfinite(value)

    assert_finite(results)


def test_fallback_prefers_filtered_global_kalman_state():
    results = predict_global_tracks(
        [
            {
                "id": 5,
                "x": 1.0,
                "z": 2.0,
                "vx": 0.5,
                "vz": -0.25,
                "history": [
                    {"t": 1000, "x": 0.0, "z": 0.0},
                    {"t": 1400, "x": 0.0, "z": 0.0},
                ],
                "age_ms": 30,
                "stale": False,
            }
        ],
        predictor=None,
        robot={"x": 0, "z": 0, "stop_radius": 1, "slow_radius": 2},
    )

    assert results[0]["source"] == "kalman"
    assert results[0]["modes"][0]["path"][0] == [0.4, 1.2, 1.9]


def test_stale_track_is_returned_without_active_prediction():
    results = predict_global_tracks(
        [{"id": 6, "history": _long_history(), "age_ms": 900, "stale": True}],
        predictor=FakePredictor(),
        robot={"x": 0, "z": 0, "stop_radius": 1, "slow_radius": 2},
    )

    assert results == [
        {
            "id": 6,
            "age_ms": 900,
            "stale": True,
            "source": "stale",
            "modes": [],
            "risk": {"stop_entry_s": None, "slow_entry_s": None},
        }
    ]
