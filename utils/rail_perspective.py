"""从钢轨中心线估计消失点，并构造透视极坐标采样网格。"""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from math import atan2, pi, sin

import cv2
import numpy as np
from numpy.typing import NDArray

_MIN_LINE_SPAN_PX = 15
_MIN_LINE_SPAN_HEIGHT_FRACTION = 0.18
_RANSAC_MAX_TRIALS = 256
_VP_MIN_ANGLE_RAD = np.deg2rad(2)
_VP_RESIDUAL_DIAGONAL_RATIO = 0.008
_VP_MIN_INLIER_FRACTION = 0.5
_VP_MAX_DISTANCE_DIAGONALS = 4.0
_POLAR_MAX_SIZE = (2048, 1600)

FloatArray = NDArray[np.float64]


@dataclass
class VanishingPoint:
    """原图像素坐标下的消失点及内点支持情况。"""

    status: str
    line_count: int
    xy: FloatArray | None = None
    inlier_ids: tuple[int, ...] = ()
    rmse_px: float | None = None


@dataclass
class PolarGrid:
    """像素到原图的反向采样网格，列为角度，行为径向距离。"""

    map_x: NDArray[np.float32]
    map_y: NDArray[np.float32]
    theta_min_rad: float
    radius_min_px: float
    theta_step_rad: float
    radius_step_px: float


def estimate_vanishing_point(
    centerlines: Iterable[tuple[int, FloatArray]], shape: tuple[int, int]
) -> VanishingPoint:
    """RANSAC 选择共点直线并以最小二乘重估消失点。

    Args:
        centerlines: 实例 ID 和对应连续中心线。坐标形状为 (N,2)，列为 x、y。
        shape: 原图 (高度, 宽度)，单位为像素。

    Returns:
        消失点估计。直线不足或缺少共识时，xy 为 None。
    """
    height, width = shape
    ids: list[int] = []
    lines: list[tuple[float, float, float]] = []
    min_length = max(_MIN_LINE_SPAN_PX, _MIN_LINE_SPAN_HEIGHT_FRACTION * height)
    for instance_id, points in centerlines:
        if len(points) < 2 or np.linalg.norm(points[-1] - points[0]) < min_length:
            continue
        trim = len(points) // 10
        core = points[trim:len(points) - trim].astype(np.float32)
        vx, vy, x, y = cv2.fitLine(core, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((float(vy), float(-vx), float(vx * y - vy * x)))
        ids.append(instance_id)

    count = len(lines)
    if count < 2:
        return VanishingPoint("insufficient_lines", count)
    diagonal = float(np.hypot(width, height))
    center = np.array((width / 2, height / 2))
    threshold = max(3.0, diagonal * _VP_RESIDUAL_DIAGONAL_RATIO)
    hypotheses = list(combinations(range(count), 2))
    if len(hypotheses) > _RANSAC_MAX_TRIALS:
        indices = np.random.default_rng(0).choice(
            len(hypotheses), _RANSAC_MAX_TRIALS, replace=False
        )
        hypotheses = [hypotheses[index] for index in indices]

    best_inliers = np.zeros(count, dtype=np.bool_)
    best_score = (0, -np.inf)
    line_array = np.asarray(lines, dtype=np.float64)
    normals, offsets = line_array[:, :2], line_array[:, 2]
    for first, second in hypotheses:
        pair = [first, second]
        axes = normals[pair]
        if abs(np.linalg.det(axes)) < sin(_VP_MIN_ANGLE_RAD):
            continue
        point = np.linalg.solve(axes, -offsets[pair])
        if np.linalg.norm(point - center) > diagonal * _VP_MAX_DISTANCE_DIAGONALS:
            continue
        residuals = np.abs(normals @ point + offsets)
        inliers = residuals <= threshold
        score = (int(inliers.sum()), -float(np.median(residuals[inliers])))
        if score > best_score:
            best_score, best_inliers = score, inliers

    minimum = 2 if count == 2 else max(3, int(np.ceil(count * _VP_MIN_INLIER_FRACTION)))
    if best_score[0] < minimum:
        status = "degenerate_geometry" if best_score[0] == 0 else "no_consensus"
        return VanishingPoint(status, count)
    point = np.linalg.lstsq(
        normals[best_inliers], -offsets[best_inliers], rcond=None
    )[0]
    residuals = np.abs(normals @ point + offsets)
    inliers = residuals <= threshold
    if (
        inliers.sum() < minimum
        or np.linalg.norm(point - center) > diagonal * _VP_MAX_DISTANCE_DIAGONALS
    ):
        return VanishingPoint("no_consensus", count)
    inlier_ids = tuple(ids[index] for index in np.flatnonzero(inliers))
    rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
    status = "ok" if count > 2 else "two_line_support"
    return VanishingPoint(status, count, point, inlier_ids, rmse)


def image_corners(shape: tuple[int, int]) -> FloatArray:
    """返回 (高度, 宽度) 画布的左上、右上、左下、右下像素中心坐标。"""
    height, width = shape
    return np.array(
        ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)),
        dtype=np.float64,
    )


def build_polar_grid(shape: tuple[int, int], origin: FloatArray) -> PolarGrid:
    """按可见角域建立等角度、等半径的极坐标采样网格。

    Args:
        shape: 原图 (高度, 宽度)，单位为像素。
        origin: 消失点坐标 (x, y)，允许位于画面外。

    Returns:
        供 OpenCV remap 使用的原图采样坐标以及角度、半径的轴参数。
    """
    corners = image_corners(shape)
    rays = corners - origin
    nearest = np.clip(origin, corners[0], corners[-1])
    radius_min = float(np.linalg.norm(nearest - origin))
    radius_max = float(np.linalg.norm(rays, axis=1).max())
    is_inside = radius_min == 0
    if is_inside:
        center = (corners[0] + corners[-1]) / 2 - origin
        theta_min, theta_span = atan2(center[1], center[0]) - pi, 2 * pi
    else:
        angles = np.sort(np.arctan2(rays[:, 1], rays[:, 0]))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * pi])
        largest_gap = int(gaps.argmax())
        theta_min = float(angles[(largest_gap + 1) % len(angles)])
        theta_span = 2 * pi - float(gaps[largest_gap])

    max_width, max_height = _POLAR_MAX_SIZE
    width = min(max_width, max(2, int(np.ceil(radius_max * theta_span)) + 1))
    height = min(max_height, max(2, int(np.ceil(radius_max - radius_min)) + 1))
    theta_step = theta_span / (width if is_inside else width - 1)
    radius_step = (radius_max - radius_min) / (height - 1)
    theta = theta_min + np.arange(width) * theta_step
    radius = (radius_min + np.arange(height) * radius_step)[:, None]
    map_x = (origin[0] + radius * np.cos(theta)).astype(np.float32)
    map_y = (origin[1] + radius * np.sin(theta)).astype(np.float32)
    return PolarGrid(map_x, map_y, theta_min, radius_min, theta_step, radius_step)
