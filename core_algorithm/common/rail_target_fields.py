"""把逐截面中心线转换成网格化的中心线、方向与宽度目标场。

目标场定义在原型网格上（训练时 ``mask_ratio=4``，即输入画布的 1/4）。中心线热力图由
到中心线的欧氏距离生成，方向用双角表示 ``(cos2φ, sin2φ)`` 消除正反向歧义，宽度取对数
视宽。三者只在实例内部与有效中心线附近有取值，其余位置标记为无效，供损失侧跳过。

本模块是纯几何计算：只依赖 numpy 与 OpenCV，不导入 torch，也不依赖数据集结构。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from core_algorithm.common.rail_centerline import SectionTrace

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_EPSILON = 1e-6
_TRUNCATE_SIGMA = 3.0
_MIN_SIGMA_CELLS = 0.5
_MIN_SECTION_SAMPLES = 3
_PROPAGATION_KERNEL = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class TargetFieldSpec:
    """目标场生成参数，长度单位统一为网格单元（原型格）。"""

    grid_ratio: int = 4
    sigma_cells: float = 0.75
    direction_band_cells: float = 2.0
    tangent_window: int = 5
    tangent_max_residual_px: float = 1.5


@dataclass
class InstanceFields:
    """单实例的目标场。

    Attributes:
        center: 中心线热力图，形状 (h, w)，取值 [0, 1]。
        center_valid: 中心线有效域，形状 (h, w)。
        direction: 双角方向场，形状 (2, h, w)，有效处为单位向量。
        direction_valid: 方向有效域，形状 (h, w)。
        width: 对数视宽，形状 (h, w)，单位是网格单元。
    """

    center: FloatArray
    center_valid: BoolArray
    direction: FloatArray
    direction_valid: BoolArray
    width: FloatArray


def _local_linear_fit(
    values: FloatArray, radius: int
) -> tuple[FloatArray, FloatArray]:
    """对一维序列做滑动窗口一次拟合，返回斜率与中心残差。

    Args:
        values: 待拟合序列，这里是逐截面中心的横向像素坐标。
        radius: 窗口单边半径（样本数）。

    Returns:
        (slope, residual)：斜率单位为“像素/样本”；残差为实际值减窗口拟合中心值，
        单位与输入相同，用于判断该处切向是否可估。
    """
    size = len(values)
    if size < _MIN_SECTION_SAMPLES:
        return np.zeros(size), np.full(size, np.inf)
    radius = max(1, min(radius, size - 1))
    offsets = np.arange(-radius, radius + 1) / radius
    basis = np.vander(offsets, 2, increasing=True)
    weights = np.linalg.pinv(basis)
    padded = np.pad(values, radius, mode="edge")
    fitted = np.convolve(padded, weights[0], mode="valid")
    slope = np.convolve(padded, weights[1], mode="valid") / radius
    return slope, np.abs(values - fitted)


def _paint_centerline(
    trace: SectionTrace, grid_shape: tuple[int, int], spec: TargetFieldSpec
) -> tuple[NDArray[np.uint8], FloatArray, FloatArray, BoolArray]:
    """把中心线样本画到网格上，并按网格单元聚合方向与对数视宽。

    逐样本 ``cv2.line`` 在长轨上会产生上万个调用，这里改为每段一条折线（只写线掩码），
    方向与宽度按落入同一网格单元的样本取均值后一次性写入。

    Returns:
        (line_mask, direction, width, usable)：网格线掩码、方向场 (2, h, w)、
        对数视宽场 (h, w)、逐样本是否可监督。
    """
    height, width = grid_shape
    ratio = spec.grid_ratio
    points_px = trace.points
    points_grid = points_px / ratio - 0.5
    width_cells = trace.widths / ratio
    usable = np.zeros(len(points_px), dtype=bool)
    angle = np.zeros(len(points_px))
    radius = max(1, spec.tangent_window // 2)
    for segment in trace.segments:
        # 样本沿主轴推进，行距不固定，所以对 x、y 分别按样本序号做局部线性拟合，
        # 再用 (dx/di, dy/di) 求切向；残差取两轴合成，单位像素。
        slope_x, residual_x = _local_linear_fit(points_px[segment, 0], radius)
        slope_y, residual_y = _local_linear_fit(points_px[segment, 1], radius)
        angle[segment] = np.arctan2(slope_y, slope_x)
        residual = np.hypot(residual_x, residual_y)
        usable[segment] = residual <= spec.tangent_max_residual_px

    # 端点处旋转栅格化会留下极窄的残余截面，宽度远小于本实例中位数时不予监督。
    if usable.any():
        median_width = float(np.median(width_cells[usable]))
        usable &= width_cells >= max(0.5, 0.25 * median_width)

    line_mask = np.zeros((height, width), dtype=np.uint8)
    direction = np.zeros((2, height, width), dtype=np.float64)
    width_field = np.zeros((height, width), dtype=np.float64)
    # 倍角表示消除钢轨正反向的歧义。
    double_angle = 2.0 * angle
    unit_direction = np.column_stack((np.cos(double_angle), np.sin(double_angle)))

    columns = np.clip(np.round(points_grid[:, 0]), 0, width - 1).astype(np.intp)
    rows = np.clip(np.round(points_grid[:, 1]), 0, height - 1).astype(np.intp)
    for segment in trace.segments:
        indices = np.flatnonzero(usable[segment]) + segment.start
        if len(indices) < 2:
            continue
        polyline = np.column_stack((columns[indices], rows[indices]))
        cv2.polylines(line_mask, [polyline], False, 1, 1)
        cells = rows[indices] * width + columns[indices]
        unique_cells, inverse = np.unique(cells, return_inverse=True)
        counts = np.maximum(np.bincount(inverse), 1)
        for channel in range(2):
            total = np.bincount(
                inverse, weights=unit_direction[indices, channel]
            )
            direction[channel].reshape(-1)[unique_cells] = total / counts
        width_sum = np.bincount(
            inverse, weights=np.log(np.maximum(width_cells[indices], _EPSILON))
        )
        width_field.reshape(-1)[unique_cells] = width_sum / counts
    return line_mask, direction, width_field, usable


def _propagate(
    values: FloatArray, known: BoolArray, limit: BoolArray, iterations: int
) -> tuple[FloatArray, BoolArray]:
    """用四邻域均值把已知值向 ``limit`` 内的未知格传播指定轮数。"""
    if values.ndim == 2:
        values = values[None]
    spread = known.copy()
    kernel = np.array(((0, 1, 0), (1, 0, 1), (0, 1, 0)), dtype=np.float64)
    for _ in range(max(iterations, 0)):
        neighbor_count = cv2.filter2D(
            spread.astype(np.float64),
            -1,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        fill = limit & ~spread & (neighbor_count > 0)
        if not fill.any():
            break
        for channel in range(values.shape[0]):
            neighbor_sum = cv2.filter2D(
                values[channel], -1, kernel, borderType=cv2.BORDER_CONSTANT
            )
            average = neighbor_sum / np.maximum(neighbor_count, 1.0)
            values[channel] = np.where(fill, average, values[channel])
        spread = spread | fill
    return values, spread


def build_instance_fields(
    trace: SectionTrace, region: BoolArray, spec: TargetFieldSpec
) -> InstanceFields | None:
    """生成一个实例的中心线、方向与宽度目标场。

    Args:
        trace: 画布像素坐标下的逐截面中心线。
        region: 该实例在网格上的前景区域，形状 (h, w)。
        spec: 生成参数。

    Returns:
        目标场；没有可用的中心线样本时返回 None。
    """
    grid_shape = region.shape
    line_mask, direction, width_field, usable = _paint_centerline(
        trace, grid_shape, spec
    )
    if not line_mask.any() or not usable.any():
        return None

    distance = cv2.distanceTransform(
        (1 - line_mask).astype(np.uint8), cv2.DIST_L2, 5
    )
    median_width = float(np.median(trace.widths)) / spec.grid_ratio
    sigma = min(spec.sigma_cells, max(_MIN_SIGMA_CELLS, median_width / 4))
    support = cv2.dilate(region.astype(np.uint8), _PROPAGATION_KERNEL).astype(bool)
    center = np.exp(-(distance**2) / (2 * sigma**2))
    center[distance > _TRUNCATE_SIGMA * sigma] = 0.0
    center[~support] = 0.0
    center_valid = center > 0
    if not center_valid.any():
        return None

    band = support & (distance <= spec.direction_band_cells)
    iterations = int(np.ceil(spec.direction_band_cells))
    known = line_mask.astype(bool)
    direction, direction_valid = _propagate(direction, known, band, iterations)
    width_values, _ = _propagate(width_field[None], known, band, iterations)
    norm = np.hypot(direction[0], direction[1])
    valid_direction = direction_valid & (norm > _EPSILON)
    direction[:, valid_direction] /= norm[valid_direction]
    direction[:, ~valid_direction] = 0.0
    return InstanceFields(
        center=center,
        center_valid=center_valid,
        direction=direction,
        direction_valid=valid_direction,
        width=width_values[0],
    )


def compose_fields(
    fields: list[InstanceFields], grid_shape: tuple[int, int]
) -> dict[str, FloatArray]:
    """按热力图峰值合成多实例目标场，避免平均造成居中方向。

    逐格只保留 ``center`` 最大的实例：相邻两根钢轨的边界格若取平均会得到两轨之间的中间
    方向，给方向损失互相矛盾的梯度。

    Args:
        fields: 各实例的目标场。
        grid_shape: 目标网格形状 (h, w)。

    Returns:
        含 center/center_valid/direction/direction_valid/width 的字典，float32。
    """
    height, width = grid_shape
    center = np.zeros((height, width), dtype=np.float64)
    center_valid = np.zeros((height, width), dtype=bool)
    direction = np.zeros((2, height, width), dtype=np.float64)
    direction_valid = np.zeros((height, width), dtype=bool)
    width_field = np.zeros((height, width), dtype=np.float64)
    for field in fields:
        better = field.center > center
        center = np.where(better, field.center, center)
        center_valid = np.where(better, field.center_valid, center_valid)
        direction = np.where(better[None], field.direction, direction)
        direction_valid = np.where(better, field.direction_valid, direction_valid)
        width_field = np.where(better, field.width, width_field)
    return {
        "center": center.astype(np.float32),
        "center_valid": center_valid.astype(np.float32),
        "direction": direction.astype(np.float32),
        "direction_valid": direction_valid.astype(np.float32),
        "width": width_field.astype(np.float32),
    }
