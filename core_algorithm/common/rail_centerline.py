"""从实例 Mask 提取逐截面中心线、法向宽度与可见段。

本模块只做几何测量：输入紧包围框内的二值 Mask 与画布包围框，输出画布坐标下的中心线
样本、逐截面宽度、曲率与可见段切片。提取时先把实例旋转到主轴竖直，再逐行取最长前景
交段的中点，因此对低曲率与平滑弯曲的钢轨都成立。

模块不依赖 torch 与 ultralytics，供训练目标、评估脚本和数据集几何报表共用。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin

import cv2
import numpy as np
from numpy.typing import NDArray

from utils.rail_perspective import image_corners

FloatArray = NDArray[np.float64]
ByteArray = NDArray[np.uint8]
Box = tuple[int, int, int, int]

_MIN_GAP_PX = 4
_MIN_CURVATURE_SAMPLES = 15
_CURVATURE_WINDOW_RADIUS_RATIO = 0.05


@dataclass
class SectionTrace:
    """单个实例的逐截面中心、宽度和曲率，间断位置单独存储。"""

    points: FloatArray
    widths: FloatArray
    curvatures: FloatArray
    segments: list[slice]
    length_px: float
    gap_count: int
    max_gap_px: int
    split_fraction: float

    @property
    def longest_segment(self) -> slice:
        """返回采样点最多的连续区段。"""
        return max(self.segments, key=lambda section: section.stop - section.start)


def _section_derivatives(
    centers: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """在连续区段内局部拟合三次曲线，估计横向一阶、二阶导数。"""
    count = len(centers)
    second = np.full(count, np.nan)
    if count < _MIN_CURVATURE_SAMPLES:
        slope = (centers[-1] - centers[0]) / max(count - 1, 1)
        return np.full(count, slope), second

    min_radius = (_MIN_CURVATURE_SAMPLES - 1) // 2
    radius = max(min_radius, round(count * _CURVATURE_WINDOW_RADIUS_RATIO))
    offsets = np.arange(-radius, radius + 1) / radius
    basis = np.polynomial.polynomial.polyvander(offsets, 3)
    weights = np.linalg.pinv(basis)
    slopes = np.correlate(centers, weights[1], mode="valid") / radius
    second[radius:-radius] = (
        2 * np.correlate(centers, weights[2], mode="valid") / radius**2
    )
    return np.pad(slopes, radius, mode="edge"), second


def measure_sections(mask: ByteArray) -> SectionTrace | None:
    """在纵向沿行扫描，取最长前景交段的中心及法向近似宽度。

    Args:
        mask: 主轴已竖直的二值掩码，取值 0/1，形状为 (H, W)。

    Returns:
        逐截面轨迹；掩码为空或没有前景交段时返回 None。
    """
    if not mask.size:
        return None
    edges = np.diff(
        mask.view(np.int8), axis=1, prepend=np.int8(0), append=np.int8(0)
    )
    section_rows, starts = np.nonzero(edges == 1)
    if not starts.size:
        return None
    ends = np.nonzero(edges == -1)[1]
    spans = ends - starts
    widest = np.zeros(mask.shape[0], dtype=np.intp)
    np.maximum.at(widest, section_rows, spans)
    best = np.flatnonzero(spans == widest[section_rows])
    rows, first = np.unique(section_rows[best], return_index=True)
    chosen = best[first]
    centers = (starts[chosen] + ends[chosen] - 1) / 2

    # 忽略旋转栅格化产生的极小交段，保留有实质宽度的分叉。
    meaningful = spans >= np.maximum(3, 0.25 * widest[section_rows])
    run_counts = np.bincount(section_rows[meaningful], minlength=mask.shape[0])
    split_fraction = float(np.count_nonzero(run_counts > 1) / len(rows))
    gaps = np.diff(rows) - 1
    bounds = np.r_[0, np.flatnonzero(gaps > 0) + 1, len(rows)]
    segments = [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    widths = spans[chosen].astype(np.float64)
    curvatures = np.full(len(rows), np.nan)
    length = 0.0
    for segment in segments:
        slopes, second = _section_derivatives(centers[segment])
        slope_factor = np.hypot(1, slopes)
        widths[segment] /= slope_factor
        curvatures[segment] = np.abs(second) / slope_factor**3
        length += float(np.hypot(1, (slopes[:-1] + slopes[1:]) / 2).sum())
    return SectionTrace(
        points=np.column_stack((centers, rows)),
        widths=widths,
        curvatures=curvatures,
        segments=segments,
        length_px=length,
        gap_count=int(np.count_nonzero(gaps >= _MIN_GAP_PX)),
        max_gap_px=int(gaps.max(initial=0)),
        split_fraction=split_fraction,
    )


def trace_instance_mask(
    instance_mask: ByteArray, box: Box
) -> SectionTrace | None:
    """沿掩码主轴采样，并把中心线还原到画布坐标。

    Args:
        instance_mask: 紧包围框内的二值掩码，取值 0/1。
        box: 实例在画布中的包围框，前两项为左上角坐标。

    Returns:
        画布坐标下的逐截面轨迹；掩码为空或无法提取时返回 None。
    """
    if not instance_mask.any():
        return None
    moments = cv2.moments(instance_mask, binaryImage=True)
    angle = atan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"]) / 2
    along = np.array((cos(angle), sin(angle)))
    if along[1] < 0:
        along = -along
    rotation = np.array(((along[1], -along[0]), along))
    corners = image_corners(instance_mask.shape)
    projected = corners @ rotation.T
    lower = np.floor(projected.min(axis=0))
    aligned_size = tuple((np.ceil(projected.max(axis=0)) - lower + 1).astype(int))
    transform = np.column_stack((rotation, -lower))
    aligned = cv2.warpAffine(
        instance_mask, transform, aligned_size, flags=cv2.INTER_NEAREST
    )
    trace = measure_sections(aligned)
    if trace is not None:
        trace.points = (trace.points + lower) @ rotation + np.array(box[:2])
    return trace
