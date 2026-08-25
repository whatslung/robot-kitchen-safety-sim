"""Pure formatting and degradation rules for global-track future prediction."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


OBSERVATION_STEPS = 8
PREDICTION_STEPS = 12
STEP_SECONDS = 0.4


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _history_rows(history: Sequence[dict[str, Any]]) -> list[tuple[int, float, float]]:
    by_time: dict[int, tuple[float, float]] = {}
    for row in history:
        try:
            timestamp = int(row["t"])
            x = float(row["x"])
            z = float(row["z"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(x) and math.isfinite(z):
            by_time[timestamp] = (x, z)
    return [(timestamp, *by_time[timestamp]) for timestamp in sorted(by_time)]


def resample_history(
    history: Sequence[dict[str, Any]],
) -> list[tuple[float, float]] | None:
    """Resample timestamped BEV history to 8 observations spaced by 0.4 seconds."""

    rows = _history_rows(history)
    required_span_ms = int((OBSERVATION_STEPS - 1) * STEP_SECONDS * 1000)
    if len(rows) < 2 or rows[-1][0] - rows[0][0] < required_span_ms:
        return None
    end_ms = rows[-1][0]
    targets = np.arange(OBSERVATION_STEPS, dtype=np.float64) * STEP_SECONDS * 1000
    targets += end_ms - required_span_ms
    timestamps = np.asarray([row[0] for row in rows], dtype=np.float64)
    xs = np.asarray([row[1] for row in rows], dtype=np.float64)
    zs = np.asarray([row[2] for row in rows], dtype=np.float64)
    sampled_x = np.interp(targets, timestamps, xs)
    sampled_z = np.interp(targets, timestamps, zs)
    return [(float(x), float(z)) for x, z in zip(sampled_x, sampled_z, strict=True)]


def risk_entry(
    modes: Sequence[dict[str, Any]],
    center: tuple[float, float],
    stop_radius: float,
    slow_radius: float,
) -> dict[str, float | None]:
    """Return the earliest predicted entry time into each robot-centred radius."""

    center_x, center_z = (_finite_float(center[0]), _finite_float(center[1]))
    radii = {
        "stop_entry_s": max(0.0, _finite_float(stop_radius)),
        "slow_entry_s": max(0.0, _finite_float(slow_radius)),
    }
    entries: dict[str, float | None] = {key: None for key in radii}
    for mode in modes:
        if _finite_float(mode.get("prob"), 0.0) <= 0.0:
            continue
        for row in mode.get("path", []):
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                continue
            timestamp, x, z = (_finite_float(value, math.nan) for value in row)
            if not all(math.isfinite(value) for value in (timestamp, x, z)):
                continue
            distance = math.hypot(x - center_x, z - center_z)
            for key, radius in radii.items():
                current = entries[key]
                if distance <= radius and (current is None or timestamp < current):
                    entries[key] = round(timestamp, 4)
    return entries


def _format_lstm_modes(raw_modes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for raw_mode in raw_modes:
        probability = float(raw_mode["w"])
        raw_path = raw_mode["path"]
        raw_sigma = raw_mode["sigma"]
        if len(raw_path) != PREDICTION_STEPS or len(raw_sigma) != PREDICTION_STEPS:
            raise ValueError("predictor returned an unexpected trajectory shape")
        path: list[list[float]] = []
        sigma: list[float] = []
        for index, ((x, z), uncertainty) in enumerate(zip(raw_path, raw_sigma, strict=True), start=1):
            values = (probability, float(x), float(z), float(uncertainty))
            if not all(math.isfinite(value) for value in values):
                raise ValueError("predictor returned a non-finite value")
            path.append([round(index * STEP_SECONDS, 4), round(float(x), 4), round(float(z), 4)])
            sigma.append(round(max(0.01, float(uncertainty)), 4))
        formatted.append({"prob": max(0.0, probability), "path": path, "sigma": sigma})
    if not formatted:
        raise ValueError("predictor returned no modes")
    probability_sum = sum(mode["prob"] for mode in formatted)
    if probability_sum <= 0.0:
        raise ValueError("predictor mode probabilities are zero")
    for mode in formatted:
        mode["prob"] = round(mode["prob"] / probability_sum, 6)
    formatted.sort(key=lambda mode: (-mode["prob"], mode["path"][0][1], mode["path"][0][2]))
    return formatted


def _constant_velocity_mode(track: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _history_rows(track.get("history", []))
    state_x = _finite_float(track.get("x"), math.nan)
    state_z = _finite_float(track.get("z"), math.nan)
    if math.isfinite(state_x) and math.isfinite(state_z):
        x, z = state_x, state_z
    elif rows:
        _, x, z = rows[-1]
    else:
        x = _finite_float(track.get("x"))
        z = _finite_float(track.get("z"))
    state_vx = _finite_float(track.get("vx"), math.nan)
    state_vz = _finite_float(track.get("vz"), math.nan)
    if math.isfinite(state_vx) and math.isfinite(state_vz):
        vx, vz = state_vx, state_vz
    elif len(rows) >= 2:
        previous, latest = rows[-2], rows[-1]
        dt = (latest[0] - previous[0]) / 1000.0
        vx = vz = 0.0
        if dt > 1e-3:
            vx = (latest[1] - previous[1]) / dt
            vz = (latest[2] - previous[2]) / dt
    else:
        vx = vz = 0.0
    speed = math.hypot(vx, vz)
    if not math.isfinite(speed):
        vx = vz = 0.0
    elif speed > 3.0:
        scale = 3.0 / speed
        vx *= scale
        vz *= scale
    path = [
        [
            round(index * STEP_SECONDS, 4),
            round(x + vx * index * STEP_SECONDS, 4),
            round(z + vz * index * STEP_SECONDS, 4),
        ]
        for index in range(1, PREDICTION_STEPS + 1)
    ]
    sigma = [round(0.15 + 0.08 * index, 4) for index in range(PREDICTION_STEPS)]
    return [{"prob": 1.0, "path": path, "sigma": sigma}]


def predict_global_tracks(
    tracks: Sequence[dict[str, Any]],
    predictor: Any,
    robot: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Predict all fresh tracks, batching eligible LSTM histories exactly once."""

    robot = robot or {}
    center = (_finite_float(robot.get("x")), _finite_float(robot.get("z")))
    stop_radius = max(0.0, _finite_float(robot.get("stop_radius"), 3.1))
    slow_radius = max(stop_radius, _finite_float(robot.get("slow_radius"), 3.9))
    ordered = sorted(tracks, key=lambda track: int(track["id"]))
    sampled_by_id: dict[int, list[tuple[float, float]]] = {}
    eligible_ids: list[int] = []
    for track in ordered:
        track_id = int(track["id"])
        if bool(track.get("stale")):
            continue
        sampled = resample_history(track.get("history", []))
        if sampled is not None:
            sampled_by_id[track_id] = sampled
            eligible_ids.append(track_id)

    raw_by_id: dict[int, Sequence[dict[str, Any]]] = {}
    if predictor is not None and eligible_ids:
        try:
            batch_output = predictor.predict_batch([sampled_by_id[track_id] for track_id in eligible_ids])
            if len(batch_output) != len(eligible_ids):
                raise ValueError("predictor batch length mismatch")
            raw_by_id = dict(zip(eligible_ids, batch_output, strict=True))
        except Exception:  # model failure is an explicit CV degradation path
            raw_by_id = {}

    results: list[dict[str, Any]] = []
    for track in ordered:
        track_id = int(track["id"])
        age_ms = max(0, int(_finite_float(track.get("age_ms"))))
        stale = bool(track.get("stale"))
        if stale:
            results.append(
                {
                    "id": track_id,
                    "age_ms": age_ms,
                    "stale": True,
                    "source": "stale",
                    "modes": [],
                    "risk": {"stop_entry_s": None, "slow_entry_s": None},
                }
            )
            continue
        source = "kalman"
        modes = _constant_velocity_mode(track)
        if track_id in raw_by_id:
            try:
                modes = _format_lstm_modes(raw_by_id[track_id])
                source = "lstm"
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
        results.append(
            {
                "id": track_id,
                "age_ms": age_ms,
                "stale": False,
                "source": source,
                "modes": modes,
                "risk": risk_entry(modes, center, stop_radius, slow_radius),
            }
        )
    return results
