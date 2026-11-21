"""从逐实例 Mask 提取多截面横向位置、宽度、间距指纹与排序质量。

本模块只做测量，不做南北身份判断。输出供匹配算法（当前为 ISA-Rail）
使用，坐标系与既有算法一致：北侧 ``u = x / (W - 1)``，南侧
``u = 1 - x / (W - 1)``，即两侧都换算成"从图内左侧指向右侧"的公共方向。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from core_algorithm.common.match_result import RailInstance, Side

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Interval = tuple[int, int]

_BASE_POSITIONS = tuple(index / 100 for index in range(10, 96, 5))
_MULTI_INTERVAL_WEIGHT = 0.5
_OVERLAP_WEIGHT = 0.5
_ORDER_CONFLICT_WEIGHT = 0.4
_ORDER_TIE_PX = 0.5
_JUMP_WIDTHS = 2.0
_SEPARATION_WIDTHS = 0.35
_SCALE_MIN_GAPS = 3
_EPSILON = 1e-9


@dataclass(frozen=True)
class RailMeasurementConfig:
    """测量层参数：采样位置、有效性与大间隙判定。"""

    sample_positions: tuple[float, ...] = _BASE_POSITIONS
    min_valid_samples: int = 5
    large_gap_ratio: float = 1.8
    large_gap_persistence: float = 0.60

    def __post_init__(self) -> None:
        positions = tuple(self.sample_positions)
        object.__setattr__(self, "sample_positions", positions)
        if not 3 <= len(positions) <= 32 or not all(
            isfinite(value) and 0 <= value <= 1 for value in positions
        ) or any(a >= b for a, b in zip(positions, positions[1:])):
            raise ValueError("sample_positions 须为 3 至 32 个递增的图高比例")
        if self.min_valid_samples < 1:
            raise ValueError("min_valid_samples 须为正整数")
        if self.large_gap_ratio <= 1:
            raise ValueError("large_gap_ratio 须大于 1")
        if not 0 < self.large_gap_persistence <= 1:
            raise ValueError("large_gap_persistence 须在 (0, 1] 内")


@dataclass(frozen=True)
class RailTrace:
    """单个 Mask 在各采样行上的交段与可靠主轨迹，坐标均为原图 x 像素。"""

    intervals: tuple[tuple[Interval, ...], ...]
    centers: FloatArray
    widths: FloatArray
    weights: FloatArray
    rms_px: float | None
    width_median_px: float | None
    width_variation: float


@dataclass(frozen=True)
class MeasuredRail:
    """排序后的单根钢轨，横向坐标已按侧别换算为公共方向。"""

    record: RailInstance
    rank: int
    u: FloatArray
    widths: FloatArray
    weights: FloatArray
    valid_count: int
    quality: float
    overlap: BoolArray
    conflict: BoolArray
    order_uncertain: bool
    order_stability: float | None
    geometry_valid: bool


@dataclass(frozen=True)
class SideMeasurement:
    """一个视角的有序钢轨、行尺度与相邻间距指纹。"""

    side: Side
    rows: tuple[int, ...]
    width_px: int
    rails: list[MeasuredRail]
    u: FloatArray
    widths: FloatArray
    weights: FloatArray
    scales: FloatArray
    log_gaps: FloatArray
    gap_weights: FloatArray
    stable_large_gaps: BoolArray
    band: tuple[float, float]

    @property
    def count(self) -> int:
        """实例数量。"""
        return len(self.rails)


def _intervals(row: BoolArray) -> tuple[Interval, ...]:
    """返回一行内所有前景闭区间，保留同一行的多个不连通交段。"""
    columns = np.flatnonzero(row)
    if not len(columns):
        return ()
    cuts = np.flatnonzero(np.diff(columns) > 1)
    lefts = np.r_[columns[0], columns[cuts + 1]]
    rights = np.r_[columns[cuts], columns[-1]]
    return tuple((int(a), int(b)) for a, b in zip(lefts, rights, strict=True))


def _trace(
    mask: BoolArray, rows: tuple[int, ...], config: RailMeasurementConfig
) -> RailTrace:
    """以相邻单交段行的中心为参照，在多交段行追踪主区间。"""
    intervals = tuple(_intervals(mask[row]) for row in rows)
    centers = np.full(len(rows), np.nan, dtype=np.float64)
    widths = np.full(len(rows), np.nan, dtype=np.float64)
    weights = np.zeros(len(rows), dtype=np.float64)
    seen: set[int] = set()
    for index, (row, spans) in enumerate(zip(rows, intervals, strict=True)):
        if row in seen or len(spans) != 1:
            seen.add(row)
            continue
        left, right = spans[0]
        centers[index] = (left + right) / 2
        widths[index] = right - left + 1
        weights[index] = 1.0
        seen.add(row)

    measured = np.flatnonzero(weights > 0)
    if len(measured) >= 2:
        for index, spans in enumerate(intervals):
            if len(spans) < 2:
                continue
            expected = float(np.interp(index, measured, centers[measured]))
            width = float(np.interp(index, measured, widths[measured]))
            candidates = sorted(
                (abs((left + right) / 2 - expected), left, right)
                for left, right in spans
            )
            distance, left, right = candidates[0]
            separated = len(candidates) == 1 or (
                candidates[1][0] - distance > _SEPARATION_WIDTHS * width
            )
            if distance <= _JUMP_WIDTHS * width and separated:
                centers[index] = (left + right) / 2
                widths[index] = right - left + 1
                weights[index] = _MULTI_INTERVAL_WEIGHT

    measured = np.flatnonzero(weights > 0)
    if not len(measured):
        return RailTrace(intervals, centers, widths, weights, None, None, 0.0)
    y = np.asarray(config.sample_positions)[measured]
    degree = min(2, len(measured) - 1)
    polynomial = np.polynomial.polynomial.polyfit(
        y, centers[measured], degree, w=np.sqrt(weights[measured])
    )
    residuals = centers[measured] - np.polynomial.polynomial.polyval(y, polynomial)
    rms = float(np.sqrt(np.average(residuals ** 2, weights=weights[measured])))
    median_width = float(np.median(widths[measured]))
    variation = float(np.median(np.abs(widths[measured] - median_width)) / median_width)
    return RailTrace(intervals, centers, widths, weights, rms, median_width, variation)


def _order(
    centers: FloatArray, weights: FloatArray, tie: float
) -> tuple[list[int], BoolArray, BoolArray, FloatArray]:
    """对共见采样行作成对方向投票，并记录固定全局顺序中的冲突。

    Args:
        centers: 归一化横向位置，北侧左到右、南侧已反向。
        weights: 各采样行的证据权重，0 表示该行不可用。
        tie: 视为"中心重合"的归一化阈值，由像素阈值换算而来。
    """
    count = centers.shape[0]
    votes = np.zeros(count, dtype=np.float64)
    positions = [
        float(np.median(row[quality > 0])) if np.any(quality > 0) else np.inf
        for row, quality in zip(centers, weights, strict=True)
    ]
    for first, second in combinations(range(count), 2):
        shared = np.minimum(weights[first], weights[second])
        valid = shared > 0
        if np.any(valid):
            delta = centers[second, valid] - centers[first, valid]
            signs = np.sign(delta) * (np.abs(delta) > tie)
            vote = float(np.average(signs, weights=shared[valid]))
        else:
            vote = float(positions[first] < positions[second])
            vote -= float(positions[first] > positions[second])
        votes[first] -= vote
        votes[second] += vote
    order = sorted(
        range(count), key=lambda index: (votes[index], positions[index], index)
    )

    conflicts = np.zeros((count, centers.shape[1]), dtype=np.bool_)
    uncertain = np.zeros(count, dtype=np.bool_)
    comparisons = np.zeros(count, dtype=np.intp)
    concordant = np.zeros(count, dtype=np.intp)
    for first, second in combinations(order, 2):
        shared = (weights[first] > 0) & (weights[second] > 0)
        if not np.any(shared):
            continue
        agrees = (centers[second] - centers[first]) > tie
        conflict = shared & ~agrees
        for index in (first, second):
            conflicts[index] |= conflict
            comparisons[index] += np.count_nonzero(shared)
            concordant[index] += np.count_nonzero(shared & agrees)
            uncertain[index] |= bool(conflict.any())
    visible = [index for index in order if np.any(weights[index] > 0)]
    for first, second in zip(visible, visible[1:]):
        if not np.any((weights[first] > 0) & (weights[second] > 0)):
            uncertain[first] = True
            uncertain[second] = True
    stability = np.full(count, np.nan, dtype=np.float64)
    np.divide(concordant, comparisons, out=stability, where=comparisons > 0)
    return order, conflicts, uncertain, stability


def _scales(gaps: FloatArray) -> FloatArray:
    """逐行取本行正间距中位数；稀疏行按行插值，全部稀疏时退回整幅中位数。"""
    local = np.full(gaps.shape[1], np.nan, dtype=np.float64)
    for index, section in enumerate(gaps.T):
        valid = section[np.isfinite(section)]
        if len(valid) >= _SCALE_MIN_GAPS:
            local[index] = np.median(valid)
    available = np.flatnonzero(np.isfinite(local))
    if len(available):
        return np.interp(np.arange(len(local)), available, local[available])
    fallback = gaps[np.isfinite(gaps)]
    if len(fallback):
        return np.full(gaps.shape[1], float(np.median(fallback)), dtype=np.float64)
    return np.full(gaps.shape[1], np.nan, dtype=np.float64)


def measure_side(
    masks: BoolArray | None,
    side: Side,
    config: RailMeasurementConfig | None = None,
) -> SideMeasurement:
    """测量一个视角的逐实例掩码。

    Args:
        masks: 原图尺寸的 ``(N, H, W)`` 布尔数组；``None`` 表示该侧无实例。
        side: ``"south"`` 或 ``"north"``，决定横向坐标的换算方向。
        config: 采样位置与有效性门限，省略时使用默认值。

    Returns:
        排序后的钢轨、逐行尺度、相邻间距指纹与占用带。

    Raises:
        ValueError: 掩码不是三维布尔数组，或与配置的采样数不一致。
    """
    settings = config if config is not None else RailMeasurementConfig()
    if masks is None:
        return SideMeasurement(
            side, (), 0, [],
            np.zeros((0, len(settings.sample_positions))),
            np.zeros((0, len(settings.sample_positions))),
            np.zeros((0, len(settings.sample_positions))),
            np.zeros(len(settings.sample_positions)),
            np.zeros((0, len(settings.sample_positions))),
            np.zeros((0, len(settings.sample_positions))),
            np.zeros(0, dtype=np.bool_),
            (float("nan"), float("nan")),
        )
    if masks.ndim != 3 or masks.dtype != np.bool_:
        raise ValueError("单侧掩码须为 (N, H, W) 布尔数组")
    height, width = masks.shape[1:]
    rows = tuple(round((height - 1) * value) for value in settings.sample_positions)
    traces = [_trace(mask, rows, settings) for mask in masks]
    sample_count = len(rows)
    shape = (len(traces), sample_count)
    centers = np.asarray(
        [track.centers for track in traces], dtype=np.float64
    ).reshape(shape)
    widths = np.asarray(
        [track.widths for track in traces], dtype=np.float64
    ).reshape(shape)
    weights = np.asarray(
        [track.weights for track in traces], dtype=np.float64
    ).reshape(shape)
    u = centers / (width - 1) if width > 1 else centers * 0.0
    widths = widths / (width - 1) if width > 1 else widths * 0.0
    if side == "south":
        u = 1 - u
    # 像素阈值换算到归一化坐标，避免把小间距误判成"中心重合"。
    tie = _ORDER_TIE_PX / max(width - 1, 1)

    order, conflicts, uncertain, stability = _order(u, weights, tie)
    u, widths, weights = u[order], widths[order], weights[order]
    overlaps = np.zeros(shape, dtype=np.bool_)
    for first, second in combinations(range(len(order)), 2):
        intersect = np.abs(u[first] - u[second]) < (widths[first] + widths[second]) / 2
        overlaps[first] |= intersect
        overlaps[second] |= intersect
    weights = weights.copy()
    weights[overlaps] *= _OVERLAP_WEIGHT
    weights[conflicts[order]] *= _ORDER_CONFLICT_WEIGHT

    rails: list[MeasuredRail] = []
    for rank, index in enumerate(order, start=1):
        track = traces[index]
        valid = np.flatnonzero(track.weights > 0)
        center_xy = None
        if len(valid):
            middle = min(
                valid, key=lambda value: abs(settings.sample_positions[value] - 0.5)
            )
            center_xy = (float(track.centers[middle]), rows[middle])
        rms = track.rms_px or 0.0
        quality = float(weights[rank - 1].sum() / sample_count) / (
            1 + rms / max(track.width_median_px or 1.0, 1.0)
        )
        rails.append(MeasuredRail(
            {"instance_id": f"{side[0].upper()}{rank:02d}", "mask_index": index,
             "center_xy": center_xy},
            rank, u[rank - 1], widths[rank - 1], weights[rank - 1],
            int(np.count_nonzero(track.weights > 0)), quality,
            overlaps[rank - 1], conflicts[index],
            bool(uncertain[index]),
            float(stability[index]) if isfinite(stability[index]) else None,
            int(np.count_nonzero(track.weights > 0)) >= settings.min_valid_samples,
        ))

    gaps = np.diff(u, axis=0)
    valid_gap = np.isfinite(gaps) & (gaps > tie)
    scales = _scales(np.where(valid_gap, gaps, np.nan))
    ratios = np.where(valid_gap, gaps / (scales + _EPSILON), np.nan)
    log_gaps = np.log(ratios + _EPSILON)
    gap_weights = np.where(valid_gap, np.minimum(weights[:-1], weights[1:]), 0.0)
    total = np.count_nonzero(np.isfinite(ratios), axis=1)
    persistent = np.count_nonzero(ratios >= settings.large_gap_ratio, axis=1)
    large = (total >= settings.min_valid_samples) & (
        persistent >= total * settings.large_gap_persistence
    )
    visible = u[np.isfinite(u) & (weights > 0)]
    band = (
        (float(np.quantile(visible, 0.05)), float(np.quantile(visible, 0.95)))
        if len(visible)
        else (float("nan"), float("nan"))
    )
    return SideMeasurement(
        side, rows, int(width), rails, u, widths, weights, scales,
        log_gaps, gap_weights, large, band,
    )
