"""GMSA 的截面轨迹、物理排序和无量纲间距测量。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from core_algorithm.common.match_result import RailInstance, Side
from core_algorithm.common.preview import LabelMeInstanceRef

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.intp]
Direction = Literal["same", "reverse"]
Interval = tuple[int, int]
Value = str | int | float | bool | None
Row = dict[str, Value]

_DEFAULT_POSITIONS = tuple(index / 100 for index in range(10, 96, 5))
_MULTI_INTERVAL_WEIGHT = 0.5
_OVERLAP_WEIGHT = 0.5
_ORDER_CONFLICT_WEIGHT = 0.4
_ORDER_TIE_PX = 0.5
_EPSILON = 1e-9


@dataclass(frozen=True)
class GMSARailConfig:
    """GMSA 工程阈值。各代价使用无量纲尺度，采样位置用图高比例。"""

    sample_positions: tuple[float, ...] = _DEFAULT_POSITIONS
    longitudinal_direction: Literal["same", "reverse", "auto"] = "auto"
    dtw_window: int = 3
    dtw_max_repeat: int = 2
    min_valid_samples: int = 5
    min_dtw_coverage: float = 0.35
    dtw_warp_weight: float = 0.08
    dtw_coverage_weight: float = 0.45
    gap_merge_limit: int = 2
    gap_merge_penalty: float = 0.12
    min_local_gaps: int = 3
    large_gap_ratio: float = 1.8
    large_gap_persistence: float = 0.60
    node_weight: float = 0.30
    span_weight: float = 0.65
    gap_weight: float = 0.30
    anchor_weight: float = 0.05
    span_error_scale: float = 0.70
    gap_error_scale: float = 0.70
    max_span_cost: float = 1.10
    anchor_max_span_cost: float = 0.45
    max_span_skip_difference: int = 4
    skip_cost_min: float = 0.16
    skip_cost_max: float = 0.42
    min_span_transitions: int = 2
    path_relative_margin: float = 0.03
    pair_exclusion_margin: float = 0.08
    direction_relative_margin: float = 0.015
    track_max_jump_widths: float = 2.0
    track_min_separation_widths: float = 0.35
    k_best: int = 5
    max_candidate_nodes: int = 1024
    max_transitions: int = 120_000
    dtw_batch_size: int = 512

    def __post_init__(self) -> None:
        positions = tuple(self.sample_positions)
        object.__setattr__(self, "sample_positions", positions)
        if not 3 <= len(positions) <= 32 or not all(
            isfinite(value) and 0 <= value <= 1 for value in positions
        ) or any(a >= b for a, b in zip(positions, positions[1:])):
            raise ValueError("sample_positions 须为 3 至 32 个递增的图高比例")
        if self.longitudinal_direction not in ("same", "reverse", "auto"):
            raise ValueError("longitudinal_direction 须为 same、reverse 或 auto")
        integer_ranges = (
            (self.dtw_window, 0, len(positions) - 1),
            (self.dtw_max_repeat, 0, len(positions) - 1),
            (self.min_valid_samples, 1, len(positions)),
            (self.gap_merge_limit, 1, 2),
            (self.min_local_gaps, 1, 4096),
            (self.min_span_transitions, 2, 4096),
            (self.max_span_skip_difference, 0, 4096),
            (self.k_best, 2, 64),
            (self.max_candidate_nodes, 1, 4096),
            (self.max_transitions, 0, 1_000_000),
            (self.dtw_batch_size, 1, 2048),
        )
        if any(
            type(value) is not int or not lower <= value <= upper
            for value, lower, upper in integer_ranges
        ):
            raise ValueError("GMSA 的整数配置或搜索资源上限无效")
        fractions = (
            self.min_dtw_coverage, self.large_gap_persistence,
            self.path_relative_margin, self.direction_relative_margin,
        )
        if any(not 0 <= value <= 1 for value in fractions):
            raise ValueError("GMSA 的比例门限须在 0 至 1 内")
        nonnegative = (
            self.dtw_warp_weight, self.dtw_coverage_weight, self.gap_merge_penalty,
            self.node_weight, self.gap_weight, self.anchor_weight,
            self.skip_cost_min, self.skip_cost_max, self.pair_exclusion_margin,
            self.track_max_jump_widths, self.track_min_separation_widths,
        )
        positive = (
            self.span_weight, self.span_error_scale, self.gap_error_scale,
            self.max_span_cost, self.anchor_max_span_cost,
        )
        if any(not isfinite(value) or value < 0 for value in nonnegative) or any(
            not isfinite(value) or value <= 0 for value in positive
        ) or not isfinite(self.large_gap_ratio) or self.large_gap_ratio <= 1:
            raise ValueError("GMSA 的代价、尺度或大间隙门限无效")
        if self.skip_cost_max < self.skip_cost_min:
            raise ValueError("skip_cost_max 须大于等于 skip_cost_min")


@dataclass(frozen=True)
class RailView:
    """一侧已有解析器输出的逐 Polygon 布尔 Mask 和可选来源信息。

    masks 为原图坐标下 (N, H, W) 的布尔数组，None 表示该侧为空。
    references 按输入 Mask 顺序排列，省略时保留 mask_index 供回溯。
    """

    image_path: str | Path
    masks: BoolArray | None
    references: tuple[LabelMeInstanceRef, ...] = ()

    def __post_init__(self) -> None:
        if self.masks is None:
            if self.references:
                raise ValueError("空 Mask 视图不能附带实例来源编号")
            return
        if (
            not isinstance(self.masks, np.ndarray)
            or self.masks.ndim != 3
            or self.masks.dtype != np.bool_
            or min(self.masks.shape[1:]) < 1
        ):
            raise ValueError("视图 Mask 须为原图尺寸的 (N, H, W) 布尔数组")
        if self.references and len(self.references) != len(self.masks):
            raise ValueError("实例来源编号数量与输入 Mask 数量不一致")


@dataclass
class Track:
    """单个 Mask 在所有采样行上的交段与可靠主轨迹。"""

    intervals: tuple[tuple[Interval, ...], ...]
    centers: FloatArray
    widths: FloatArray
    weights: FloatArray
    rms_px: float | None
    width_median_px: float | None
    width_variation: float
    broken_count: int


@dataclass
class Instance:
    """物理排序后的实例及其测量质量。"""

    record: RailInstance
    shape_index: int | None
    group_id: int | None
    rank: int
    track: Track
    valid_count: int
    quality: float
    overlap: BoolArray
    conflict: BoolArray
    order_stability: float | None
    order_uncertain: bool
    geometry_valid: bool


@dataclass
class ViewFeatures:
    """一个视角的有序实例、局部尺度与相邻间距指纹。"""

    side: Side
    rows: tuple[int, ...]
    width_px: int
    instances: list[Instance]
    centers: FloatArray
    weights: FloatArray
    scales: FloatArray
    scale_sources: tuple[str, ...]
    ratios: FloatArray
    log_gaps: FloatArray
    gap_weights: FloatArray
    boundary_gaps: FloatArray
    stable_large_gaps: BoolArray


def _intervals(row: BoolArray) -> tuple[Interval, ...]:
    """返回前景的闭区间，保留同一行的多个不连通交段。"""
    columns = np.flatnonzero(row)
    if not len(columns):
        return ()
    cuts = np.flatnonzero(np.diff(columns) > 1)
    lefts = np.r_[columns[0], columns[cuts + 1]]
    rights = np.r_[columns[cuts], columns[-1]]
    return tuple((int(a), int(b)) for a, b in zip(lefts, rights, strict=True))


def _trace(
    mask: BoolArray, rows: tuple[int, ...], config: GMSARailConfig
) -> Track:
    """以相邻单交段的轨迹为参照，在多交段行追踪主区间。"""
    intervals = tuple(_intervals(mask[row]) for row in rows)
    centers = np.full(len(rows), np.nan, dtype=np.float64)
    widths = np.full(len(rows), np.nan, dtype=np.float64)
    weights = np.full(len(rows), 0.0, dtype=np.float64)
    seen: set[int] = set()
    for index, (row, spans) in enumerate(zip(rows, intervals, strict=True)):
        if row not in seen and len(spans) == 1:
            left, right = spans[0]
            centers[index] = (left + right) / 2
            widths[index] = right - left + 1
            weights[index] = 1
        seen.add(row)

    singles = np.flatnonzero(weights > 0)
    if len(singles) >= 2:
        for index, spans in enumerate(intervals):
            if len(spans) < 2 or (index and rows[index] == rows[index - 1]):
                continue
            expected = float(np.interp(index, singles, centers[singles]))
            width = float(np.interp(index, singles, widths[singles]))
            candidates = sorted(
                (abs((left + right) / 2 - expected), left, right)
                for left, right in spans
            )
            distance, left, right = candidates[0]
            if (
                distance <= config.track_max_jump_widths * width
                and candidates[1][0] - distance > config.track_min_separation_widths * width
            ):
                centers[index] = (left + right) / 2
                widths[index] = right - left + 1
                weights[index] = _MULTI_INTERVAL_WEIGHT

    measured = np.flatnonzero(weights > 0)
    visible = [index for index, spans in enumerate(intervals) if spans]
    broken = sum(not spans for spans in intervals[visible[0]:visible[-1]]) if visible else 0
    if not len(measured):
        return Track(intervals, centers, widths, weights, None, None, 0.0, broken)
    y = np.asarray(config.sample_positions)[measured]
    polynomial = np.polynomial.polynomial.polyfit(
        y, centers[measured], min(2, len(measured) - 1),
        w=np.sqrt(weights[measured]),
    )
    residuals = centers[measured] - np.polynomial.polynomial.polyval(y, polynomial)
    rms = float(np.sqrt(np.average(residuals ** 2, weights=weights[measured])))
    width = float(np.median(widths[measured]))
    variation = float(np.median(np.abs(widths[measured] - width)) / width)
    return Track(intervals, centers, widths, weights, rms, width, variation, broken)


def _order(
    centers: FloatArray, weights: FloatArray
) -> tuple[list[int], BoolArray, BoolArray, FloatArray]:
    """对共见截面作成对投票并记录固定全局顺序中的冲突。"""
    count, samples = centers.shape
    votes = np.full(count, 0.0, dtype=np.float64)
    positions = [
        float(np.median(row[quality > 0])) if quality.any() else np.inf
        for row, quality in zip(centers, weights, strict=True)
    ]
    for first, second in combinations(range(count), 2):
        shared = np.minimum(weights[first], weights[second])
        valid = shared > 0
        if valid.any():
            delta = centers[second, valid] - centers[first, valid]
            signs = np.sign(delta) * (np.abs(delta) > _ORDER_TIE_PX)
            vote = float(np.average(signs, weights=shared[valid]))
        else:
            vote = float(positions[first] < positions[second])
            vote -= float(positions[first] > positions[second])
        votes[first] -= vote
        votes[second] += vote
    order = sorted(range(count), key=lambda index: (votes[index], positions[index], index))
    conflicts = np.zeros((count, samples), dtype=np.bool_)
    uncertain = np.zeros(count, dtype=np.bool_)
    comparisons = np.zeros(count, dtype=np.intp)
    concordant = np.zeros(count, dtype=np.intp)
    for first, second in combinations(order, 2):
        shared = (weights[first] > 0) & (weights[second] > 0)
        if not shared.any():
            continue
        agrees = (centers[second] - centers[first]) > _ORDER_TIE_PX
        conflict = shared & ~agrees
        for index in (first, second):
            conflicts[index] |= conflict
            comparisons[index] += np.count_nonzero(shared)
            concordant[index] += np.count_nonzero(shared & agrees)
            uncertain[index] |= conflict.any()
    valid_order = [index for index in order if weights[index].any()]
    for first, second in zip(valid_order, valid_order[1:]):
        if not np.any((weights[first] > 0) & (weights[second] > 0)):
            uncertain[first] = uncertain[second] = True
    stability = np.full(count, np.nan, dtype=np.float64)
    np.divide(concordant, comparisons, out=stability, where=comparisons > 0)
    return order, conflicts, uncertain, stability


def _scales(
    gaps: FloatArray, min_local_gaps: int
) -> tuple[FloatArray, tuple[str, ...]]:
    """优先使用本行中位间距，稀疏时插值或回退到整图中位数。"""
    local = np.full(gaps.shape[1], np.nan, dtype=np.float64)
    for index, section in enumerate(gaps.T):
        valid = section[np.isfinite(section)]
        if len(valid) >= min_local_gaps:
            local[index] = np.median(valid)
    available = np.flatnonzero(np.isfinite(local))
    if len(available):
        sources = tuple(
            "local" if isfinite(value) else "neighbor_sections" for value in local
        )
        return np.interp(np.arange(len(local)), available, local[available]), sources
    available_gaps = gaps[np.isfinite(gaps)]
    if len(available_gaps):
        local.fill(np.median(available_gaps))
        return local, ("view_median",) * len(local)
    return local, ("unavailable",) * len(local)


def measure_view(view: RailView, side: Side, config: GMSARailConfig) -> ViewFeatures:
    """提取指定物理方向的多截面 Mask 特征。"""
    masks = view.masks
    sample_count = len(config.sample_positions)
    height, width = (1, 1) if masks is None else masks.shape[1:]
    rows = tuple(round((height - 1) * value) for value in config.sample_positions)
    tracks = [] if masks is None else [_trace(mask, rows, config) for mask in masks]
    shape = (len(tracks), sample_count)
    centers = np.asarray([track.centers for track in tracks], dtype=np.float64).reshape(shape)
    widths = np.asarray([track.widths for track in tracks], dtype=np.float64).reshape(shape)
    weights = np.asarray([track.weights for track in tracks], dtype=np.float64).reshape(shape)
    if side == "south":
        centers = width - 1 - centers
    order, conflicts, uncertain, stability = _order(centers, weights)
    centers, widths, weights = centers[order], widths[order], weights[order]
    overlaps = np.zeros(shape, dtype=np.bool_)
    for first, second in combinations(range(len(order)), 2):
        # 交叠只降低证据权重，每个 Polygon 仍有独立身份。
        intersect = np.abs(centers[first] - centers[second]) < (
            widths[first] + widths[second]
        ) / 2
        overlaps[first] |= intersect
        overlaps[second] |= intersect
    weights[overlaps] *= _OVERLAP_WEIGHT
    weights[conflicts[order]] *= _ORDER_CONFLICT_WEIGHT

    instances: list[Instance] = []
    for rank, index in enumerate(order, start=1):
        track = tracks[index]
        valid = np.flatnonzero(track.weights > 0)
        center_xy = None
        if len(valid):
            middle = min(valid, key=lambda value: abs(config.sample_positions[value] - 0.5))
            center_xy = (float(track.centers[middle]), rows[middle])
        reference = view.references[index] if view.references else None
        rms = track.rms_px or 0.0
        quality = float(weights[rank - 1].sum() / sample_count) / (
            1 + rms / max(track.width_median_px or 1.0, 1.0)
        )
        instances.append(Instance(
            {"instance_id": f"{side[0].upper()}{rank:02d}", "mask_index": index,
             "center_xy": center_xy},
            None if reference is None else reference.shape_index,
            None if reference is None else reference.group_id,
            rank, track, len(valid), quality, overlaps[rank - 1], conflicts[index],
            float(stability[index]) if isfinite(stability[index]) else None,
            bool(uncertain[index]), len(valid) >= config.min_valid_samples,
        ))

    gap_widths = np.diff(centers, axis=0)
    boundary_gaps = gap_widths - (widths[:-1] + widths[1:]) / 2
    valid_gap = np.isfinite(gap_widths) & (gap_widths > _ORDER_TIE_PX)
    gap_weights = np.where(valid_gap, np.minimum(weights[:-1], weights[1:]), 0.0)
    scales, sources = _scales(np.where(valid_gap, gap_widths, np.nan), config.min_local_gaps)
    ratios = np.where(valid_gap, gap_widths / (scales + _EPSILON), np.nan)
    log_gaps = np.log(ratios + _EPSILON)
    total = np.count_nonzero(np.isfinite(ratios), axis=1)
    persistent = np.count_nonzero(ratios >= config.large_gap_ratio, axis=1)
    large = (total >= config.min_valid_samples) & (
        persistent >= total * config.large_gap_persistence
    )
    return ViewFeatures(
        side, rows, int(width), instances, centers, weights, scales, sources,
        ratios, log_gaps, gap_weights, boundary_gaps, large,
    )


def instance_row(instance: Instance, side: Side, sample_count: int) -> Row:
    """生成用于诊断及未匹配表的实例质量记录。"""
    track = instance.track
    return {
        "side": side, "instance_id": instance.record["instance_id"],
        "mask_index": instance.record["mask_index"], "shape_index": instance.shape_index,
        "group_id": instance.group_id, "order_index": instance.rank,
        "valid_section_count": instance.valid_count,
        "valid_section_fraction": instance.valid_count / sample_count,
        "geometry_quality": instance.quality, "geometry_valid": instance.geometry_valid,
        "center_fit_rms_px": track.rms_px,
        "center_x_median_px": (
            float(np.nanmedian(track.centers)) if instance.valid_count else None
        ),
        "width_median_px": track.width_median_px,
        "relative_width_mad": track.width_variation if instance.valid_count else None,
        "multi_interval_section_count": sum(len(spans) > 1 for spans in track.intervals),
        "broken_section_count": track.broken_count,
        "overlap_section_count": int(np.count_nonzero(instance.overlap)),
        "order_conflict_section_count": int(np.count_nonzero(instance.conflict)),
        "order_stability": instance.order_stability,
        "order_uncertain": instance.order_uncertain,
    }


def geometry_rows(view: ViewFeatures, config: GMSARailConfig) -> tuple[list[Row], ...]:
    """为实例、采样行和相邻间距导出可回溯的测量记录。"""
    instance_rows = [
        instance_row(instance, view.side, len(view.rows)) for instance in view.instances
    ]
    section_rows: list[Row] = []
    gap_rows: list[Row] = []
    for index, fraction in enumerate(config.sample_positions):
        section: Row = {
            "side": view.side, "sample_index": index, "height_fraction": fraction,
            "y_px": view.rows[index],
        }
        for rail_index, instance in enumerate(view.instances):
            track = instance.track
            spans = track.intervals[index]
            weight = float(view.weights[rail_index, index])
            section_rows.append({
                **section, "mask_index": instance.record["mask_index"],
                "visible": bool(spans), "valid": weight > 0, "weight": weight,
                "intervals_px": "|".join(f"{a}:{b}" for a, b in spans),
                "center_x_px": float(track.centers[index]) if weight else None,
                "width_px": float(track.widths[index]) if weight else None,
                "overlap": bool(instance.overlap[index]),
                "order_conflict": bool(instance.conflict[index]),
            })
        for left, (first, second) in enumerate(zip(view.instances, view.instances[1:])):
            log_gap = float(view.log_gaps[left, index])
            boundary = float(view.boundary_gaps[left, index])
            scale = float(view.scales[index])
            gap_rows.append({
                **section, "left_mask_index": first.record["mask_index"],
                "right_mask_index": second.record["mask_index"],
                "scale_px": scale if isfinite(scale) else None,
                "scale_source": view.scale_sources[index],
                "normalized_gap": float(view.ratios[left, index]) if isfinite(log_gap) else None,
                "log_gap": log_gap if isfinite(log_gap) else None,
                "boundary_gap_px": boundary if isfinite(boundary) else None,
                "weight": float(view.gap_weights[left, index]),
                "stable_large_gap": bool(view.stable_large_gaps[left]),
            })
    return instance_rows, section_rows, gap_rows
