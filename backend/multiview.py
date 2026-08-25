"""Image-to-floor calibration and multi-camera BEV tracking primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
