"""Image-to-floor calibration and multi-camera BEV tracking primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


class CalibrationError(ValueError):
    """Raised when image/world correspondences cannot define a homography."""


def _as_points(name: str, points: Sequence[Sequence[float]], minimum: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < minimum:
        raise CalibrationError(f"{name} must contain at least {minimum} x/y pairs")
    if not np.isfinite(values).all():
        raise CalibrationError(f"{name} contains a non-finite coordinate")
    return values


def _normalise_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    centred = points - centre
    mean_distance = float(np.linalg.norm(centred, axis=1).mean())
    if mean_distance < 1e-10 or np.linalg.matrix_rank(centred) < 2:
        raise CalibrationError("calibration points are collinear or coincident")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [
            [scale, 0.0, -scale * centre[0]],
            [0.0, scale, -scale * centre[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalised = (transform @ homogeneous.T).T
    return normalised[:, :2], transform


def _solve_homography(image: np.ndarray, world: np.ndarray) -> np.ndarray:
    image_norm, image_transform = _normalise_points(image)
    world_norm, world_transform = _normalise_points(world)
    rows: list[list[float]] = []
    for (u, v), (x, z) in zip(image_norm, world_norm, strict=True):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, z * u, z * v, z])
    design = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(design) < 8:
        raise CalibrationError("calibration correspondences are degenerate")
    _, _, vh = np.linalg.svd(design)
    normalised_h = vh[-1].reshape(3, 3)
    matrix = np.linalg.inv(world_transform) @ normalised_h @ image_transform
    norm = float(matrix[2, 2])
    if abs(norm) < 1e-12:
        norm = float(np.linalg.norm(matrix))
    if norm < 1e-12 or not np.isfinite(matrix).all():
        raise CalibrationError("calibration produced a singular homography")
    matrix /= norm
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise CalibrationError("calibration produced a singular homography")
    return matrix


def _inside_polygon(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    """Return whether a point is inside or on the boundary of a simple polygon."""

    x, y = point
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(
            y1, y2
        ) - 1e-9 <= y <= max(y1, y2) + 1e-9:
            return True
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
    return inside


@dataclass(frozen=True)
class CameraCalibration:
    matrix: np.ndarray
    valid_world_polygon: tuple[tuple[float, float], ...]
    reprojection_rms: float

    @classmethod
    def from_points(
        cls,
        image: Sequence[Sequence[float]],
        world: Sequence[Sequence[float]],
        valid_world_polygon: Sequence[Sequence[float]],
    ) -> "CameraCalibration":
        image_points = _as_points("image", image, 4)
        world_points = _as_points("world", world, 4)
        if len(image_points) != len(world_points):
            raise CalibrationError("image and world point counts differ")
        polygon = _as_points("valid_world_polygon", valid_world_polygon, 3)
        matrix = _solve_homography(image_points, world_points)
        homogeneous = np.column_stack((image_points, np.ones(len(image_points))))
        projected = (matrix @ homogeneous.T).T
        if np.any(np.abs(projected[:, 2]) < 1e-10):
            raise CalibrationError("homography denominator is zero at a calibration point")
        projected_xy = projected[:, :2] / projected[:, 2, None]
        rms = float(np.sqrt(np.mean(np.sum((projected_xy - world_points) ** 2, axis=1))))
        if not np.isfinite(rms):
            raise CalibrationError("calibration reprojection error is non-finite")
        frozen_matrix = matrix.copy()
        frozen_matrix.setflags(write=False)
        return cls(
            matrix=frozen_matrix,
            valid_world_polygon=tuple((float(x), float(y)) for x, y in polygon),
            reprojection_rms=rms,
        )

    def project(self, image_xy: Sequence[float]) -> tuple[float, float] | None:
        if len(image_xy) != 2 or not np.isfinite(image_xy).all():
            raise CalibrationError("image point must contain two finite coordinates")
        projected = self.matrix @ np.array([image_xy[0], image_xy[1], 1.0], dtype=np.float64)
        if abs(float(projected[2])) < 1e-10:
            raise CalibrationError("homography denominator is zero")
        world = (float(projected[0] / projected[2]), float(projected[1] / projected[2]))
        return world if _inside_polygon(world, self.valid_world_polygon) else None


@dataclass(frozen=True)
class Measurement:
    camera_id: str
    local_id: int | str
    x: float
    z: float
    confidence: float
    timestamp_ms: int
    box_index: int

    @property
    def local_key(self) -> tuple[str, int | str]:
        return (self.camera_id, self.local_id)


@dataclass
class GlobalTrack:
    id: int
    state: np.ndarray
    covariance: np.ndarray
    created_ms: int
    last_update_ms: int
    sources: dict[str, int] = field(default_factory=dict)
    history: list[dict[str, float | int]] = field(default_factory=list)


DEFAULT_FUSION_CONFIG = {
    "gate": 0.8,
    "fusion_window_ms": 250,
    "coast_ms": 750,
    "remove_ms": 1500,
}


def _predict_cv(
    state: np.ndarray, covariance: np.ndarray, dt_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    dt = max(0.0, float(dt_seconds))
    transition = np.array(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    process_noise = 0.35
    q = process_noise * np.array(
        [
            [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
            [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
            [dt**3 / 2.0, 0.0, dt**2, 0.0],
            [0.0, dt**3 / 2.0, 0.0, dt**2],
        ],
        dtype=np.float64,
    )
    return transition @ state, transition @ covariance @ transition.T + q


class MultiViewFusion:
    """Fuse camera-local track measurements into persistent world-space IDs."""

    def __init__(
        self,
        gate: float = DEFAULT_FUSION_CONFIG["gate"],
        fusion_window_ms: int = DEFAULT_FUSION_CONFIG["fusion_window_ms"],
        coast_ms: int = DEFAULT_FUSION_CONFIG["coast_ms"],
        remove_ms: int = DEFAULT_FUSION_CONFIG["remove_ms"],
    ) -> None:
        if gate <= 0 or not (0 <= fusion_window_ms <= coast_ms <= remove_ms):
            raise ValueError("invalid multiview fusion timing or gate configuration")
        self.gate = float(gate)
        self.fusion_window_ms = int(fusion_window_ms)
        self.coast_ms = int(coast_ms)
        self.remove_ms = int(remove_ms)
        self.calibrations: dict[str, CameraCalibration] = {}
        self.tracks: dict[int, GlobalTrack] = {}
        self._local_bindings: dict[tuple[str, int | str], int] = {}
        self._next_id = 1
        self._latest_timestamp_ms: int | None = None

    def calibrate(self, camera_id: str, calibration: CameraCalibration) -> None:
        camera = str(camera_id).strip()
        if not camera:
            raise CalibrationError("camera id is empty")
        self.calibrations[camera] = calibration

    def reset_tracks(self) -> None:
        self.tracks.clear()
        self._local_bindings.clear()
        self._next_id = 1
        self._latest_timestamp_ms = None

    def _expire(self, timestamp_ms: int) -> None:
        expired = {
            track_id
            for track_id, track in self.tracks.items()
            if timestamp_ms - track.last_update_ms > self.remove_ms
        }
        for track_id in expired:
            del self.tracks[track_id]
        if expired:
            self._local_bindings = {
                key: track_id
                for key, track_id in self._local_bindings.items()
                if track_id not in expired
            }

    @staticmethod
    def _predicted(track: GlobalTrack, timestamp_ms: int) -> tuple[np.ndarray, np.ndarray]:
        return _predict_cv(
            track.state,
            track.covariance,
            (timestamp_ms - track.last_update_ms) / 1000.0,
        )

    def _distance(self, track: GlobalTrack, measurement: Measurement) -> float:
        predicted, _ = self._predicted(track, measurement.timestamp_ms)
        return float(np.linalg.norm(predicted[:2] - np.array([measurement.x, measurement.z])))

    def _new_track(self, measurement: Measurement) -> GlobalTrack:
        track_id = self._next_id
        self._next_id += 1
        track = GlobalTrack(
            id=track_id,
            state=np.array([measurement.x, measurement.z, 0.0, 0.0], dtype=np.float64),
            covariance=np.diag([0.08, 0.08, 1.0, 1.0]).astype(np.float64),
            created_ms=measurement.timestamp_ms,
            last_update_ms=measurement.timestamp_ms,
        )
        self.tracks[track_id] = track
        self._accept_measurement(track, measurement, initialise=True)
        return track

    def _accept_measurement(
        self, track: GlobalTrack, measurement: Measurement, initialise: bool = False
    ) -> None:
        if not initialise:
            predicted, covariance = self._predicted(track, measurement.timestamp_ms)
            observation = np.array([measurement.x, measurement.z], dtype=np.float64)
            observation_model = np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
            )
            variance = max(0.008, 0.06 * (1.0 - measurement.confidence))
            measurement_noise = np.eye(2, dtype=np.float64) * variance
            innovation_covariance = (
                observation_model @ covariance @ observation_model.T + measurement_noise
            )
            gain = (
                covariance
                @ observation_model.T
                @ np.linalg.inv(innovation_covariance)
            )
            track.state = predicted + gain @ (observation - observation_model @ predicted)
            identity = np.eye(4, dtype=np.float64)
            track.covariance = (identity - gain @ observation_model) @ covariance
        track.last_update_ms = measurement.timestamp_ms
        track.sources[measurement.camera_id] = measurement.timestamp_ms
        track.history.append(
            {"t": measurement.timestamp_ms, "x": measurement.x, "z": measurement.z}
        )
        cutoff = measurement.timestamp_ms - 4000
        track.history = [row for row in track.history if int(row["t"]) >= cutoff]
        self._local_bindings[measurement.local_key] = track.id

    def _assign(self, measurements: list[Measurement]) -> dict[int, int]:
        assignments: dict[int, int] = {}
        used_track_ids: set[int] = set()

        for measurement_index, measurement in enumerate(measurements):
            bound_id = self._local_bindings.get(measurement.local_key)
            bound = self.tracks.get(bound_id) if bound_id is not None else None
            if bound is not None and self._distance(bound, measurement) <= self.gate:
                assignments[measurement_index] = bound.id
                used_track_ids.add(bound.id)

        remaining_measurements = [
            index for index in range(len(measurements)) if index not in assignments
        ]
        remaining_tracks = [
            track_id for track_id in sorted(self.tracks) if track_id not in used_track_ids
        ]
        if not remaining_measurements or not remaining_tracks:
            return assignments

        costs = np.full(
            (len(remaining_measurements), len(remaining_tracks)),
            self.gate + 1.0,
            dtype=np.float64,
        )
        for row, measurement_index in enumerate(remaining_measurements):
            for column, track_id in enumerate(remaining_tracks):
                distance = self._distance(self.tracks[track_id], measurements[measurement_index])
                if distance <= self.gate:
                    costs[row, column] = distance

        try:
            from scipy.optimize import linear_sum_assignment

            row_indices, column_indices = linear_sum_assignment(costs)
            pairs = zip(row_indices.tolist(), column_indices.tolist(), strict=True)
        except (ImportError, ValueError):
            candidates = sorted(
                (costs[row, column], row, column)
                for row in range(costs.shape[0])
                for column in range(costs.shape[1])
            )
            chosen_rows: set[int] = set()
            chosen_columns: set[int] = set()
            greedy_pairs: list[tuple[int, int]] = []
            for _, row, column in candidates:
                if row not in chosen_rows and column not in chosen_columns:
                    chosen_rows.add(row)
                    chosen_columns.add(column)
                    greedy_pairs.append((row, column))
            pairs = iter(greedy_pairs)

        for row, column in pairs:
            if costs[row, column] <= self.gate:
                assignments[remaining_measurements[row]] = remaining_tracks[column]
        return assignments

    def update(
        self, camera_id: str, detections: Sequence[dict[str, Any]], timestamp_ms: int
    ) -> list[dict[str, Any]]:
        enriched = [dict(detection) for detection in detections]
        timestamp = int(timestamp_ms)
        if self._latest_timestamp_ms is not None and timestamp < self._latest_timestamp_ms:
            return enriched
        self._latest_timestamp_ms = timestamp
        self._expire(timestamp)
        calibration = self.calibrations.get(camera_id)
        if calibration is None:
            return enriched

        measurements: list[Measurement] = []
        for box_index, detection in enumerate(enriched):
            if detection.get("label") != "person" or detection.get("id") is None:
                continue
            try:
                footpoint = (
                    float(detection["cx"]),
                    float(detection["cy"]) + float(detection["h"]) / 2.0,
                )
                world = calibration.project(footpoint)
                confidence = float(detection.get("conf", 0.0))
            except (KeyError, TypeError, ValueError, CalibrationError):
                continue
            if world is None or not np.isfinite(confidence):
                continue
            measurements.append(
                Measurement(
                    camera_id=camera_id,
                    local_id=detection["id"],
                    x=world[0],
                    z=world[1],
                    confidence=min(1.0, max(0.0, confidence)),
                    timestamp_ms=timestamp,
                    box_index=box_index,
                )
            )

        assignments = self._assign(measurements)
        for measurement_index, measurement in enumerate(measurements):
            track_id = assignments.get(measurement_index)
            track = self.tracks.get(track_id) if track_id is not None else None
            if track is None:
                track = self._new_track(measurement)
            else:
                self._accept_measurement(track, measurement)
            detection = enriched[measurement.box_index]
            detection["global_id"] = track.id
            detection["world"] = {"x": measurement.x, "z": measurement.z}
        return enriched

    def snapshot(self, timestamp_ms: int) -> list[dict[str, Any]]:
        timestamp = int(timestamp_ms)
        self._expire(timestamp)
        snapshot: list[dict[str, Any]] = []
        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            state, _ = self._predicted(track, max(timestamp, track.last_update_ms))
            age_ms = max(0, timestamp - track.last_update_ms)
            snapshot.append(
                {
                    "id": track.id,
                    "x": float(state[0]),
                    "z": float(state[1]),
                    "vx": float(state[2]),
                    "vz": float(state[3]),
                    "age_ms": age_ms,
                    "stale": age_ms > self.coast_ms,
                    "sources": sorted(
                        camera
                        for camera, seen_ms in track.sources.items()
                        if timestamp - seen_ms <= self.remove_ms
                    ),
                    "history": [dict(row) for row in track.history],
                }
            )
        return snapshot
