"""用多截面结构和局部路径共识匹配南北钢轨。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import ceil, isfinite
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_algorithm.common.match_result import (
    RailInstance,
    RailMatch,
    RailMatchingResult,
    Side,
    UnmatchedRail,
    make_unmatched_rail,
)
from data_preprocess.common.config import PROJECT_ROOT
from utils.instance_masks import as_mask_stack

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.engine.results import Masks

_SECTIONS = (25, 50, 75)
_MAIN_SECTION = 1
_SECTION_PRIORITY = (_MAIN_SECTION, 0, 2)
_MIN_ANCHORS = 3
_ANCHOR_MAX_STEP = 2
_ANCHOR_RATIO_TOLERANCE = 0.10
_MIN_GAP_SPREAD = 1.25
_MIN_ANCHOR_ASYMMETRY = 0.08
_SCALE_RANGE = (0.5, 2.0)
_MIN_MATCH_FRACTION = 0.5
_MIN_SPAN_FRACTION = 0.35
_MAX_POSITION_ERROR = 0.025
_GAP_ERROR_FRACTION = 0.48
_MAPPING_BIN_FRACTION = 0.25
_MAX_MAPPING_COUNT = 24
_SKIP_COST = 0.65
_MAD_SCALE = 1.4826
_MIN_BOUNDARY_SECTIONS = 2
_HIGH_SUPPORT = 0.80
_AMBIGUOUS_DECISIONS = {"ambiguous_alternative", "order_or_partner_conflict"}

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]
DiagnosticValue = str | float | int | bool | None
DiagnosticRow = dict[str, DiagnosticValue]
EvidenceTable = Literal[
    "rail_features", "section_gaps", "layout_groups",
    "alignment_candidates", "candidate_pair_support",
]

MSLC_TABLE_COLUMNS: dict[EvidenceTable, tuple[str, ...]] = {
    "rail_features": (
        "side", "mask_index", "instance_id", "shape_index", "group_id",
        "measurement_reason", "layout_group_id", "slope", "curvature",
        *(
            f"section_{section}_{field}"
            for section in _SECTIONS
            for field in (
                "y_px", "center_x_px", "u", "foreground_px", "run_count", "valid"
            )
        ),
    ),
    "section_gaps": (
        "side", "section_percent", "y_px", "left_mask_index", "right_mask_index",
        "left_instance_id", "right_instance_id", "gap_norm", "median_gap_norm",
        "mad_gap_norm", "threshold_norm", "is_large_gap", "is_distinct_row",
        "boundary_valid_sections", "boundary_support_sections", "is_stable_boundary",
    ),
    "layout_groups": (
        "side", "layout_group_id", "member_mask_indices", "member_instance_ids",
        "member_count", "start_u50", "end_u50", "boundary_before_support",
        "boundary_after_support", "median_slope", "median_curvature",
    ),
    "alignment_candidates": (
        "section_percent", "mapping_index", "scale", "offset", "path_cost",
        "path_match_count", "south_span_fraction", "north_span_fraction",
        "decision", "mask_index_pairs",
    ),
    "candidate_pair_support": (
        "south_mask_index", "north_mask_index",
        "south_instance_id", "north_instance_id",
        "comparable_section_count", "support_section_count", "combined_support",
        "top_section_eligible",
        *(f"support_{section}" for section in _SECTIONS),
        *(f"residual_{section}_norm" for section in _SECTIONS),
        "position_residual_norm", "position_cost", "gap_cost", "layout_cost",
        "trend_cost", "curvature_cost", "final_cost", "alternative_cost_margin",
        "decision", "support_level", "shared_id",
    ),
}


@dataclass(frozen=True)
class MSLCRailConfig:
    """控制结构检测、路径竞争和最终判定的参数，代价与位置均已归一化。"""

    gap_median_multiplier: float = 2.5
    gap_mad_multiplier: float = 3.0
    mapping_cost_margin: float = 0.65
    min_support: float = 0.55
    min_section_support: float = 0.35
    min_top_section_support: float = 0.40
    single_section_min_support: float = 0.80
    min_alternative_margin: float = 0.15
    slope_deadband: float = 0.003
    curvature_deadband: float = 0.001
    position_weight: float = 0.45
    gap_weight: float = 0.12
    layout_weight: float = 0.20
    trend_weight: float = 0.08
    curvature_weight: float = 0.04
    support_weight: float = 0.75

    def __post_init__(self) -> None:
        if any(not isfinite(value) or value < 0 for value in asdict(self).values()):
            raise ValueError("MSLC 配置参数须为有限非负数")
        if self.gap_median_multiplier <= 1:
            raise ValueError("gap_median_multiplier 须大于 1")
        if not 0 < self.min_section_support <= self.min_support <= 1:
            raise ValueError("截面支持阈值须大于 0 且不超过综合支持阈值")
        if not 0 <= self.min_top_section_support <= 1:
            raise ValueError("25% 截面支持阈值须在 0 与 1 之间，0 表示关闭")
        if not self.min_support <= self.single_section_min_support <= 1:
            raise ValueError("单截面支持阈值须在综合支持阈值和 1 之间")


class MSLCDiagnostics(TypedDict):
    """匹配过程中产生的标量统计和可关联的逐项证据。"""

    summary: DiagnosticRow
    rail_features: list[DiagnosticRow]
    section_gaps: list[DiagnosticRow]
    layout_groups: list[DiagnosticRow]
    alignment_candidates: list[DiagnosticRow]
    candidate_pair_support: list[DiagnosticRow]


class MSLCMatchingResult(RailMatchingResult):
    """保留项目匹配结果字段，并附带 MSLC 的几何和决策证据。"""

    diagnostics: MSLCDiagnostics


@dataclass(frozen=True)
class _View:
    """按主截面排序的实例和三截面观测。"""

    side: Side
    rows: tuple[int, ...]
    instances: list[RailInstance]
    reasons: list[str | None]
    valid_indices: IndexArray
    x: FloatArray
    u: FloatArray
    foreground: NDArray[np.int32]
    runs: NDArray[np.int32]
    slope: FloatArray
    curvature: FloatArray


@dataclass(frozen=True)
class _Layout:
    """稳定大间距、图内排列组和组边界的邻近度。"""

    relative_gaps: FloatArray
    boundary_proximity: FloatArray
    group_ids: dict[int, str]
    gaps: list[DiagnosticRow]
    groups: list[DiagnosticRow]


@dataclass
class _AlignmentCandidate:
    """某截面在一个仿射映射下的路径及其筛选状态。"""

    scale: float
    offset: float
    cost: float
    pairs: tuple[tuple[int, int], ...]
    residuals: tuple[float, ...]
    position_costs: tuple[float, ...]
    south_span: float
    north_span: float
    decision: str


@dataclass(frozen=True)
class _SectionEvidence:
    """某截面的支持率、残差和测量质量。

    三个矩阵均以主截面排序后的 (北索引, 南索引) 寻址。
    未能参与竞争的组合支持率为 NaN，可比较但未获票的组合为 0。
    """

    weight: float
    support: FloatArray
    residuals: FloatArray
    position_costs: FloatArray
    summary: DiagnosticRow
    candidates: list[DiagnosticRow]


def _finite_value(value: float) -> float | None:
    return float(value) if isfinite(value) else None


def _distinct_sections(*views_rows: tuple[int, ...]) -> set[int]:
    """每侧的同一像素行只作为一个观测，优先保留主截面。"""
    used = [set() for _ in views_rows]
    distinct = set()
    for section in _SECTION_PRIORITY:
        section_rows = [rows[section] for rows in views_rows]
        if all(row not in seen for row, seen in zip(section_rows, used, strict=True)):
            distinct.add(section)
            for row, seen in zip(section_rows, used, strict=True):
                seen.add(row)
    return distinct


def _tied_positions(positions: FloatArray) -> NDArray[np.bool_]:
    ties = np.zeros(len(positions), dtype=np.bool_)
    repeats = np.diff(positions) == 0
    ties[:-1] |= repeats
    ties[1:] |= repeats
    return ties


def _image_shape(image_path: str | Path) -> tuple[int, int]:
    """读取项目内原图的高度和宽度，供掩码坐标校验。"""
    relative_path = Path(image_path)
    absolute_path = (PROJECT_ROOT / relative_path).resolve()
    if relative_path.anchor or not absolute_path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"请提供相对于项目根目录的图片路径 {image_path}")
    with Image.open(absolute_path) as image:
        width, height = image.size
    if width < 2:
        raise ValueError(f"原图宽度至少为两个像素 {image_path}")
    return height, width


def _measure_view(masks: NDArray[np.bool_], side: Side) -> _View:
    """使用与 V1 一致的连续交段中点，额外记录宽度和交段数。"""
    count, height, width = masks.shape
    rows = tuple(round((height - 1) * percent / 100) for percent in _SECTIONS)
    x = np.full((count, len(rows)), np.nan, dtype=np.float64)
    foreground = np.full(x.shape, 0, dtype=np.int32)
    runs = np.full(x.shape, 0, dtype=np.int32)
    for mask_index, mask in enumerate(masks):
        for section, y in enumerate(rows):
            columns = np.flatnonzero(mask[y])
            foreground[mask_index, section] = len(columns)
            if len(columns):
                run_count = 1 + np.count_nonzero(np.diff(columns) > 1)
                runs[mask_index, section] = run_count
                if run_count == 1:
                    x[mask_index, section] = (columns[0] + columns[-1]) / 2

    u = x / (width - 1)
    if side == "south":
        u = 1 - u
    order = np.argsort(u[:, _MAIN_SECTION], kind="stable")
    x, u, foreground, runs = x[order], u[order], foreground[order], runs[order]
    ties = _tied_positions(u[:, _MAIN_SECTION])
    instances: list[RailInstance] = []
    reasons: list[str | None] = []
    for index, mask_index in enumerate(order):
        center = _finite_value(x[index, _MAIN_SECTION])
        main_runs = runs[index, _MAIN_SECTION]
        if main_runs == 0:
            reason = "no_measurement_intersection"
        elif main_runs > 1:
            reason = "multiple_measurement_intersections"
        elif ties[index]:
            reason = "inconsistent_instance_order"
        else:
            reason = None
        reasons.append(reason)
        instances.append({
            "instance_id": f"{side[0].upper()}{index + 1:02d}",
            "mask_index": int(mask_index),
            "center_xy": None if center is None else (center, rows[_MAIN_SECTION]),
        })
    valid_indices = np.flatnonzero([reason is None for reason in reasons])
    slope = u[:, 2] - u[:, 0] if rows[0] != rows[2] else np.full(count, np.nan)
    curvature = (
        u[:, _MAIN_SECTION] - (u[:, 0] + u[:, 2]) / 2
        if len(set(rows)) == len(rows) else np.full(count, np.nan)
    )
    return _View(
        side, rows, instances, reasons, valid_indices, x, u, foreground, runs,
        slope, curvature,
    )


def _median_finite(values: FloatArray) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def _measure_layout(view: _View, config: MSLCRailConfig) -> _Layout:
    """按主序相邻实例汇总稳健间距，并提取反复出现的大间距。"""
    indices = view.valid_indices
    gaps = np.diff(view.u[indices], axis=0)
    positive = np.greater(gaps, 0)
    medians = np.full(len(_SECTIONS), np.nan)
    mads = np.full(len(_SECTIONS), np.nan)
    thresholds = np.full(len(_SECTIONS), np.nan)
    for section in range(len(_SECTIONS)):
        values = gaps[positive[:, section], section]
        if len(values):
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            medians[section], mads[section] = median, mad
            thresholds[section] = max(
                config.gap_median_multiplier * median,
                median + config.gap_mad_multiplier * _MAD_SCALE * mad,
            )
    distinct = _distinct_sections(view.rows)
    is_distinct = np.array([
        section in distinct for section in range(len(_SECTIONS))
    ])
    large = gaps > thresholds
    supports = np.count_nonzero(large & is_distinct, axis=1)
    valid_counts = np.count_nonzero(positive & is_distinct, axis=1)
    boundaries = supports >= _MIN_BOUNDARY_SECTIONS

    relative_gaps = np.full((*view.u.shape, 2), np.nan)
    if len(indices) > 1:
        scaled = np.where(positive, gaps / medians, np.nan)
        relative_gaps[indices[:-1], :, 1] = scaled
        relative_gaps[indices[1:], :, 0] = scaled

    gap_rows: list[DiagnosticRow] = []
    for boundary_index, (left, right) in enumerate(zip(indices, indices[1:])):
        first, second = view.instances[left], view.instances[right]
        for section, percent in enumerate(_SECTIONS):
            gap_rows.append({
                "side": view.side, "section_percent": percent,
                "y_px": view.rows[section],
                "left_mask_index": first["mask_index"],
                "right_mask_index": second["mask_index"],
                "left_instance_id": first["instance_id"],
                "right_instance_id": second["instance_id"],
                "gap_norm": _finite_value(gaps[boundary_index, section]),
                "median_gap_norm": _finite_value(medians[section]),
                "mad_gap_norm": _finite_value(mads[section]),
                "threshold_norm": _finite_value(thresholds[section]),
                "is_large_gap": bool(large[boundary_index, section]),
                "is_distinct_row": bool(is_distinct[section]),
                "boundary_valid_sections": int(valid_counts[boundary_index]),
                "boundary_support_sections": int(supports[boundary_index]),
                "is_stable_boundary": bool(boundaries[boundary_index]),
            })

    proximity = np.zeros((len(view.instances), 2), dtype=np.float64)
    group_ids: dict[int, str] = {}
    group_rows: list[DiagnosticRow] = []
    cuts = [0, *(np.flatnonzero(boundaries) + 1), len(indices)]
    for index, (start, end) in enumerate(zip(cuts, cuts[1:]), start=1):
        if start == end:
            continue
        members = indices[start:end]
        group_id = f"{view.side[0].upper()}g{index:02d}"
        for member in members:
            group_ids[int(member)] = group_id
        if start:
            proximity[members, 0] = 1 / np.arange(1, len(members) + 1)
        if end < len(indices):
            proximity[members, 1] = 1 / np.arange(len(members), 0, -1)
        group_rows.append({
            "side": view.side, "layout_group_id": group_id,
            "member_mask_indices": "|".join(
                str(view.instances[member]["mask_index"]) for member in members
            ),
            "member_instance_ids": "|".join(
                view.instances[member]["instance_id"] for member in members
            ),
            "member_count": len(members),
            "start_u50": float(view.u[members[0], _MAIN_SECTION]),
            "end_u50": float(view.u[members[-1], _MAIN_SECTION]),
            "boundary_before_support": int(supports[start - 1]) if start else None,
            "boundary_after_support": (
                int(supports[end - 1]) if end < len(indices) else None
            ),
            "median_slope": _median_finite(view.slope[members]),
            "median_curvature": _median_finite(view.curvature[members]),
        })
    return _Layout(relative_gaps, proximity, group_ids, gap_rows, group_rows)


def _feature_rows(view: _View, layout: _Layout) -> list[DiagnosticRow]:
    rows: list[DiagnosticRow] = []
    for index, instance in enumerate(view.instances):
        row: DiagnosticRow = {
            "side": view.side,
            "mask_index": instance["mask_index"],
            "instance_id": instance["instance_id"],
            "measurement_reason": view.reasons[index],
            "layout_group_id": layout.group_ids.get(index),
            "slope": _finite_value(view.slope[index]),
            "curvature": _finite_value(view.curvature[index]),
        }
        for section, percent in enumerate(_SECTIONS):
            prefix = f"section_{percent}"
            row[f"{prefix}_y_px"] = view.rows[section]
            row[f"{prefix}_center_x_px"] = _finite_value(view.x[index, section])
            row[f"{prefix}_u"] = _finite_value(view.u[index, section])
            row[f"{prefix}_foreground_px"] = int(view.foreground[index, section])
            row[f"{prefix}_run_count"] = int(view.runs[index, section])
            row[f"{prefix}_valid"] = bool(view.runs[index, section] == 1)
        rows.append(row)
    return rows


def _section_order(view: _View, section: int) -> IndexArray:
    """截面独立排序，中心相同的观测缺乏局部次序信息。"""
    indices = view.valid_indices
    positions = view.u[indices, section]
    indices = indices[np.isfinite(positions)]
    order = np.argsort(view.u[indices, section], kind="stable")
    indices = indices[order]
    return indices[~_tied_positions(view.u[indices, section])]


def _anchor_triplets(
    positions: NDArray[np.float64],
) -> list[tuple[float, float, float]]:
    """返回局部三点的起点、总间距和首段间距占比。"""
    gaps = np.diff(positions)
    if gaps.max() < gaps.min() * _MIN_GAP_SPREAD:
        return []
    anchors = []
    for start in range(len(positions) - 2):
        middle_limit = min(len(positions) - 1, start + _ANCHOR_MAX_STEP + 1)
        for middle in range(start + 1, middle_limit):
            end_limit = min(len(positions), middle + _ANCHOR_MAX_STEP + 1)
            for end in range(middle + 1, end_limit):
                span = positions[end] - positions[start]
                ratio = (positions[middle] - positions[start]) / span
                if abs(ratio - 0.5) >= _MIN_ANCHOR_ASYMMETRY:
                    anchors.append((positions[start], span, ratio))
    return anchors


def _nearest_gaps(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """取每个位置到相邻钢轨的最小距离。"""
    gaps = np.diff(positions)
    return np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf])


def _position_costs(
    north: NDArray[np.float64],
    south: NDArray[np.float64],
    scale: float,
    offset: float,
) -> NDArray[np.float64]:
    """归一化位置残差，并拒绝距离超过局部轨距门限的组合。"""
    gaps = np.minimum(
        _nearest_gaps(north)[:, None], scale * _nearest_gaps(south)[None, :]
    )
    tolerance = np.minimum(_MAX_POSITION_ERROR, _GAP_ERROR_FRACTION * gaps)
    errors = np.abs(north[:, None] - (scale * south + offset)[None, :])
    return np.where(errors <= tolerance, errors / tolerance, np.inf)


def _candidate_mappings(
    north: NDArray[np.float64], south: NDArray[np.float64], minimum_count: int
) -> list[tuple[float, float]]:
    """用相似三点间距生成映射，合并近似重复的候选。"""
    north_anchors = _anchor_triplets(north)
    south_anchors = _anchor_triplets(south)
    bin_width = float(np.median(np.diff(north))) * _MAPPING_BIN_FRACTION
    candidates: dict[tuple[int, int], tuple[float, float, float]] = {}
    for north_start, north_span, north_ratio in north_anchors:
        for south_start, south_span, south_ratio in south_anchors:
            if abs(north_ratio - south_ratio) > _ANCHOR_RATIO_TOLERANCE:
                continue
            scale = float(north_span / south_span)
            if not _SCALE_RANGE[0] <= scale <= _SCALE_RANGE[1]:
                continue
            offset = float(north_start - scale * south_start)
            costs = _position_costs(north, south, scale, offset)
            nearest_north = costs.min(axis=1)
            nearest_south = costs.min(axis=0)
            if min(
                np.isfinite(nearest_north).sum(), np.isfinite(nearest_south).sum()
            ) < minimum_count:
                continue

            # 用两端映射位置分桶，减少同一映射附近的重复动态规划。
            key = (
                round((scale * south[0] + offset) / bin_width),
                round((scale * south[-1] + offset) / bin_width),
            )
            skip_pair_cost = 2 * _SKIP_COST
            score = float(
                np.minimum(nearest_north, skip_pair_cost).sum()
                + np.minimum(nearest_south, skip_pair_cost).sum()
            )
            if key not in candidates or score < candidates[key][0]:
                candidates[key] = score, scale, offset
    ranked = sorted(candidates.values())[:_MAX_MAPPING_COUNT]
    return [(scale, offset) for _, scale, offset in ranked]


def _ordered_alignment(
    costs: NDArray[np.float64],
) -> tuple[float, list[tuple[int, int]]]:
    """在配对、跳过北轨和跳过南轨之间寻找最小代价路径。"""
    north_count, south_count = costs.shape
    total = np.zeros((north_count + 1, south_count + 1), dtype=np.float64)
    total[:, 0] = _SKIP_COST * np.arange(north_count + 1)
    total[0, :] = _SKIP_COST * np.arange(south_count + 1)
    moves = np.zeros(total.shape, dtype=np.uint8)

    for i in range(1, north_count + 1):
        for j in range(1, south_count + 1):
            pair_cost = total[i - 1, j - 1] + costs[i - 1, j - 1]
            north_skip = total[i - 1, j] + _SKIP_COST
            south_skip = total[i, j - 1] + _SKIP_COST
            if pair_cost <= min(north_skip, south_skip):
                total[i, j] = pair_cost
            elif north_skip <= south_skip:
                total[i, j] = north_skip
                moves[i, j] = 1
            else:
                total[i, j] = south_skip
                moves[i, j] = 2

    pairs = []
    i, j = north_count, south_count
    while i and j:
        move = moves[i, j]
        if move == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return float(total[-1, -1]), pairs


def _alignment_candidates(
    north_u: FloatArray,
    south_u: FloatArray,
    north_order: IndexArray,
    south_order: IndexArray,
    layout_cost: FloatArray | float,
) -> list[_AlignmentCandidate]:
    """生成 MSLC 截面路径，加入软结构代价后检查路径覆盖。"""
    smaller_count = min(len(north_u), len(south_u))
    minimum_count = max(
        _MIN_ANCHORS, ceil(smaller_count * _MIN_MATCH_FRACTION)
    )
    candidates = []
    mappings = _candidate_mappings(north_u, south_u, minimum_count)
    north_range, south_range = np.ptp(north_u), np.ptp(south_u)
    for scale, offset in mappings:
        position_costs = _position_costs(north_u, south_u, scale, offset)
        costs = position_costs + layout_cost
        cost, local_pairs = _ordered_alignment(costs)
        north_span = south_span = 0.0
        if local_pairs:
            first_north, first_south = local_pairs[0]
            last_north, last_south = local_pairs[-1]
            north_span = float(
                (north_u[last_north] - north_u[first_north]) / north_range
            )
            south_span = float(
                (south_u[last_south] - south_u[first_south]) / south_range
            )
        if len(local_pairs) < minimum_count:
            decision = "too_few_matches"
        elif min(north_span, south_span) < _MIN_SPAN_FRACTION:
            decision = "insufficient_span"
        else:
            decision = "eligible"

        pairs = tuple(
            (int(north_order[n]), int(south_order[s])) for n, s in local_pairs
        )
        mapped_south = scale * south_u + offset
        residuals = tuple(
            float(abs(north_u[n] - mapped_south[s])) for n, s in local_pairs
        )
        normalized_costs = tuple(float(position_costs[pair]) for pair in local_pairs)
        candidates.append(_AlignmentCandidate(
            scale, offset, cost, pairs, residuals, normalized_costs,
            south_span, north_span, decision,
        ))
    return candidates


def _competitive_paths(
    candidates: list[_AlignmentCandidate], margin: float
) -> list[_AlignmentCandidate]:
    """端点近似映射由 V1 去重，相同实例路径在此合并为一票。"""
    eligible = sorted(
        (item for item in candidates if item.decision == "eligible"),
        key=lambda item: item.cost,
    )
    if not eligible:
        return []
    best_cost = eligible[0].cost
    seen: set[tuple[tuple[int, int], ...]] = set()
    competitive = []
    for candidate in eligible:
        if candidate.pairs in seen:
            candidate.decision = "duplicate_path"
            continue
        seen.add(candidate.pairs)
        if candidate.cost - best_cost <= margin:
            candidate.decision = "competitive"
            competitive.append(candidate)
        else:
            candidate.decision = "outside_cost_margin"
    return competitive


def _alignment_rows(
    candidates: list[_AlignmentCandidate], north: _View, south: _View, section: int
) -> list[DiagnosticRow]:
    rows: list[DiagnosticRow] = []
    for index, candidate in enumerate(candidates, start=1):
        paths = (
            f"{south.instances[s]['mask_index']}:{north.instances[n]['mask_index']}"
            for n, s in candidate.pairs
        )
        rows.append({
            "section_percent": _SECTIONS[section], "mapping_index": index,
            "scale": candidate.scale, "offset": candidate.offset,
            "path_cost": candidate.cost, "path_match_count": len(candidate.pairs),
            "south_span_fraction": candidate.south_span,
            "north_span_fraction": candidate.north_span,
            "decision": candidate.decision, "mask_index_pairs": "|".join(paths),
        })
    return rows


def _section_evidence(
    north: _View,
    south: _View,
    section: int,
    is_distinct: bool,
    layout_cost: FloatArray | None,
    config: MSLCRailConfig,
) -> _SectionEvidence:
    """在一个截面独立生成路径，并将投票写回主序身份。"""
    north_order = _section_order(north, section)
    south_order = _section_order(south, section)
    section_indices = np.ix_(north_order, south_order)
    shape = (len(north.instances), len(south.instances))
    support = np.full(shape, np.nan)
    residuals = np.full(shape, np.nan)
    position_costs = np.full(shape, np.nan)
    weight = 0.0
    candidates: list[_AlignmentCandidate] = []
    competitive: list[_AlignmentCandidate] = []
    if not is_distinct:
        status = "duplicate_row"
    elif min(len(north_order), len(south_order)) < _MIN_ANCHORS:
        status = "insufficient_positions"
    else:
        local_layout: FloatArray | float = 0.0
        if layout_cost is not None:
            local_layout = config.layout_weight * layout_cost[section_indices]
        candidates = _alignment_candidates(
            north.u[north_order, section], south.u[south_order, section],
            north_order, south_order, local_layout,
        )
        competitive = _competitive_paths(candidates, config.mapping_cost_margin)
        if competitive:
            status = "ok"
        else:
            status = "no_valid_path" if candidates else "no_mapping"

    if competitive:
        hits = np.zeros(shape, dtype=np.int32)
        residual_sums = np.zeros(shape)
        cost_sums = np.zeros(shape)
        for candidate in competitive:
            for key, residual, position_cost in zip(
                candidate.pairs, candidate.residuals, candidate.position_costs,
                strict=True,
            ):
                hits[key] += 1
                residual_sums[key] += residual
                cost_sums[key] += position_cost
        support[section_indices] = hits[section_indices] / len(competitive)
        voted = hits > 0
        residuals[voted] = residual_sums[voted] / hits[voted]
        position_costs[voted] = cost_sums[voted] / hits[voted]
        weight = min(
            len(north_order) / len(north.valid_indices),
            len(south_order) / len(south.valid_indices),
        )
    summary: DiagnosticRow = {
        "status": status,
        "south_usable_count": len(south_order), "north_usable_count": len(north_order),
        "mapping_count": len(candidates), "competitive_path_count": len(competitive),
        "valid_path_count": sum(
            candidate.decision not in ("too_few_matches", "insufficient_span")
            for candidate in candidates
        ),
        "unique_path_count": sum(
            candidate.decision in ("competitive", "outside_cost_margin")
            for candidate in candidates
        ),
        "weight": weight,
    }
    return _SectionEvidence(
        weight, support, residuals, position_costs, summary,
        _alignment_rows(candidates, north, south, section),
    )


def _gap_cost(north: FloatArray, south: FloatArray) -> float | None:
    """比较各自截面中位间距归一化后的左右邻距。"""
    common = np.isfinite(north) & np.isfinite(south)
    if not np.any(common):
        return None
    north, south = north[common], south[common]
    return float(np.mean(np.abs(north - south) / (north + south)))


def _sign_cost(north: float, south: float, deadband: float) -> float | None:
    """双侧均离开死区时才弱比较方向。"""
    if not (abs(north) > deadband and abs(south) > deadband):
        return None
    return float((north > 0) != (south > 0))


def _combine_evidence(
    north: _View,
    south: _View,
    north_layout: _Layout,
    south_layout: _Layout,
    layout_costs: FloatArray | None,
    sections: list[_SectionEvidence],
    config: MSLCRailConfig,
) -> tuple[FloatArray, dict[tuple[int, int], DiagnosticRow]]:
    """以每截面质量加权汇总局部证据，形成最终有序优化的代价矩阵。"""
    supports = np.stack([section.support for section in sections])
    residuals = np.stack([section.residuals for section in sections])
    position_costs = np.stack([section.position_costs for section in sections])
    weights = np.asarray([section.weight for section in sections])
    is_single_section = np.count_nonzero(weights) == 1
    minimum_support = config.single_section_min_support if is_single_section else (
        config.min_support
    )
    minimum_sections = 1 if is_single_section else 2
    costs = np.full(supports.shape[1:], np.inf)
    rows: dict[tuple[int, int], DiagnosticRow] = {}

    for north_index, south_index in np.argwhere(np.any(supports > 0, axis=0)):
        n, s = int(north_index), int(south_index)
        key = n, s
        pair_support = supports[:, n, s]
        pair_residuals = residuals[:, n, s]
        comparable = np.isfinite(pair_support)
        voted = pair_support > 0
        support_section_count = int(
            np.count_nonzero(pair_support >= config.min_section_support)
        )
        combined_support = float(
            np.average(pair_support[comparable], weights=weights[comparable])
        )
        residual = float(np.average(pair_residuals[voted], weights=weights[voted]))
        position_cost = float(
            np.average(position_costs[voted, n, s], weights=weights[voted])
        )
        gap_cost = _gap_cost(
            north_layout.relative_gaps[n, voted], south_layout.relative_gaps[s, voted]
        )
        layout_cost = None if layout_costs is None else float(layout_costs[key])
        trend_cost = _sign_cost(
            float(north.slope[n]), float(south.slope[s]), config.slope_deadband
        )
        curvature_cost = _sign_cost(
            float(north.curvature[n]), float(south.curvature[s]),
            config.curvature_deadband,
        )
        cost = (
            config.position_weight * position_cost
            + config.gap_weight * (gap_cost or 0.0)
            + config.layout_weight * (layout_cost or 0.0)
            + config.trend_weight * (trend_cost or 0.0)
            + config.curvature_weight * (curvature_cost or 0.0)
            + config.support_weight * (1 - combined_support)
        )
        top_support = float(pair_support[0]) if comparable[0] else float("nan")
        is_top_section_eligible = (
            config.min_top_section_support > 0
            and isfinite(top_support)
            and top_support >= config.min_top_section_support
        )
        # 25% 截面在真值上判别力最强：误配候选在密集的 50%/75% 截面反而拿到更高
        # 支持率，因此单独给 25% 截面一条可接受通道，其余判定仍交给有序 DP 与裕度。
        if (combined_support >= minimum_support
                and support_section_count >= minimum_sections):
            decision = "eligible"
            costs[key] = cost
        elif is_top_section_eligible:
            decision = "eligible_top_section"
            costs[key] = cost
        elif combined_support < minimum_support:
            decision = "low_support"
        else:
            decision = "insufficient_section_support"
        north_instance, south_instance = north.instances[n], south.instances[s]
        row: DiagnosticRow = {
            "south_mask_index": south_instance["mask_index"],
            "north_mask_index": north_instance["mask_index"],
            "south_instance_id": south_instance["instance_id"],
            "north_instance_id": north_instance["instance_id"],
            "comparable_section_count": int(np.count_nonzero(comparable)),
            "support_section_count": support_section_count,
            "combined_support": combined_support,
            "top_section_eligible": is_top_section_eligible,
            "position_residual_norm": residual, "position_cost": position_cost,
            "gap_cost": gap_cost, "layout_cost": layout_cost, "trend_cost": trend_cost,
            "curvature_cost": curvature_cost, "final_cost": cost, "decision": decision,
            "alternative_cost_margin": None, "support_level": None, "shared_id": None,
        }
        for section, percent in enumerate(_SECTIONS):
            row[f"support_{percent}"] = _finite_value(pair_support[section])
            row[f"residual_{percent}_norm"] = _finite_value(pair_residuals[section])
        rows[key] = row
    return costs, rows


def _select_matches(
    north: _View,
    south: _View,
    costs: FloatArray,
    candidates: dict[tuple[int, int], DiagnosticRow],
    config: MSLCRailConfig,
) -> tuple[list[RailMatch], float, int]:
    """检查禁用边后的最优代价，只从同一条有序路径保留可靠配对。"""
    path_cost, provisional = _ordered_alignment(costs)
    for row in candidates.values():
        if row["decision"] in ("eligible", "eligible_top_section"):
            row["decision"] = "order_or_partner_conflict"
            row["alternative_cost_margin"] = 0.0

    matches: list[RailMatch] = []
    for key in provisional:
        n, s = key
        cost = costs[key]
        costs[key] = np.inf
        alternate_cost, _ = _ordered_alignment(costs)
        costs[key] = cost
        margin = max(0.0, alternate_cost - path_cost)
        row = candidates[key]
        row["alternative_cost_margin"] = margin
        if margin <= config.min_alternative_margin:
            row["decision"] = "ambiguous_alternative"
            continue

        shared_id = f"R{len(matches) + 1:03d}"
        combined_support = cast(float, row["combined_support"])
        row["decision"] = "accepted"
        row["shared_id"] = shared_id
        row["support_level"] = (
            "high_support" if combined_support >= _HIGH_SUPPORT else "medium_support"
        )
        matches.append({
            "shared_id": shared_id,
            "south": south.instances[s], "north": north.instances[n],
            "position_error": cast(float, row["position_residual_norm"]),
        })
    return matches, path_cost, len(provisional)


def _unmatched(
    view: _View,
    other: _View,
    matches: list[RailMatch],
    candidates: dict[tuple[int, int], DiagnosticRow],
    has_alignment: bool,
) -> list[UnmatchedRail]:
    """按取点失败、候选歧义、证据不足的顺序保留原因。"""
    matched = {match[view.side]["mask_index"] for match in matches}
    side_index = 0 if view.side == "north" else 1
    ambiguous = {
        key[side_index] for key, row in candidates.items()
        if row["decision"] in _AMBIGUOUS_DECISIONS
    }
    proposed = {key[side_index] for key in candidates}
    unmatched: list[UnmatchedRail] = []
    for index, instance in enumerate(view.instances):
        if instance["mask_index"] in matched:
            continue
        reason = view.reasons[index]
        if reason is None:
            if not other.instances:
                reason = "other_side_empty"
            elif not has_alignment:
                reason = "insufficient_alignment_evidence"
            elif index in ambiguous:
                reason = "ambiguous_alignment"
            elif index in proposed:
                reason = "insufficient_section_support"
            else:
                reason = "no_reliable_partner"
        unmatched.append(make_unmatched_rail(instance, reason))
    return unmatched


def match_south_north_rail_mslc(
    south_image_path: str | Path,
    north_image_path: str | Path,
    south_masks: np.ndarray | Tensor | Masks | None,
    north_masks: np.ndarray | Tensor | Masks | None,
    *,
    config: MSLCRailConfig | None = None,
) -> MSLCMatchingResult:
    """依据多截面结构建立当前图对的一对一钢轨对应。

    Args:
        south_image_path: 南侧原图的项目相对路径。
        north_image_path: 北侧原图的项目相对路径。
        south_masks: 南侧 YOLO Results.masks、原图尺寸 (N, H, W) 数组或 Tensor。
            二维掩码表示单实例，None 表示该侧无实例。
        north_masks: 北侧掩码，约定同 south_masks。
        config: MSLC 权重和门限。省略时使用 MSLCRailConfig 的默认值。

    Returns:
        与 V1 字段兼容的匹配结果，以及多截面几何和候选决策诊断。
        position_error 是支持该配对的映射下，按截面质量加权的北侧归一化残差。
        mask_index 保留输入次序，共享 ID 的作用域为当前图对。

    Raises:
        OSError: 无法读取原图。
        ValueError: 路径、图像尺寸或掩码尺寸无效。
    """
    config = config if config is not None else MSLCRailConfig()
    south_shape = _image_shape(south_image_path)
    north_shape = _image_shape(north_image_path)
    south = _measure_view(as_mask_stack(south_masks, south_shape), "south")
    north = _measure_view(as_mask_stack(north_masks, north_shape), "north")
    south_layout = _measure_layout(south, config)
    north_layout = _measure_layout(north, config)
    layout_cost = None
    if len(south_layout.groups) > 1 and len(north_layout.groups) > 1:
        layout_cost = np.abs(
            north_layout.boundary_proximity[:, None, :]
            - south_layout.boundary_proximity[None, :, :]
        ).mean(axis=2)

    distinct = _distinct_sections(north.rows, south.rows)
    sections = [
        _section_evidence(north, south, index, index in distinct, layout_cost, config)
        for index in range(len(_SECTIONS))
    ]
    section_count = sum(section.weight > 0 for section in sections)
    costs, candidates = _combine_evidence(
        north, south, north_layout, south_layout, layout_cost, sections, config
    )
    matches, path_cost, provisional_count = _select_matches(
        north, south, costs, candidates, config
    )
    south_unmatched = _unmatched(south, north, matches, candidates, bool(section_count))
    north_unmatched = _unmatched(north, south, matches, candidates, bool(section_count))

    decisions = Counter(row["decision"] for row in candidates.values())
    accepted_supports = [
        cast(float, row["combined_support"])
        for row in candidates.values() if row["decision"] == "accepted"
    ]
    mode = "no_alignment"
    if section_count == 1:
        mode = "single_section_fallback"
    elif section_count >= 2:
        mode = "multi_section"
    summary: DiagnosticRow = {
        "mode": mode, "single_section_fallback": section_count == 1,
        "usable_section_count": section_count,
        "south_group_count": len(south_layout.groups),
        "north_group_count": len(north_layout.groups),
        "uses_layout_cost": layout_cost is not None and config.layout_weight > 0,
        "candidate_pair_count": len(candidates),
        "low_support_pair_count": decisions["low_support"],
        "insufficient_section_support_pair_count": decisions[
            "insufficient_section_support"
        ],
        "ambiguous_pair_count": sum(
            decisions[decision] for decision in _AMBIGUOUS_DECISIONS
        ),
        "provisional_match_count": provisional_count,
        "accepted_match_count": len(matches),
        "accepted_support_mean": (
            float(np.mean(accepted_supports)) if accepted_supports else None
        ),
        "accepted_support_min": min(accepted_supports, default=None),
        "ordered_path_cost": path_cost,
    }
    for percent, evidence in zip(_SECTIONS, sections, strict=True):
        summary.update({
            f"section_{percent}_{name}": value
            for name, value in evidence.summary.items()
        })
    summary.update({f"config_{name}": value for name, value in asdict(config).items()})
    return {
        "south_image_path": Path(south_image_path).as_posix(),
        "north_image_path": Path(north_image_path).as_posix(),
        "matches": matches,
        "unmatched_south": south_unmatched,
        "unmatched_north": north_unmatched,
        "diagnostics": {
            "summary": summary,
            "rail_features": (
                _feature_rows(south, south_layout) + _feature_rows(north, north_layout)
            ),
            "section_gaps": south_layout.gaps + north_layout.gaps,
            "layout_groups": south_layout.groups + north_layout.groups,
            "alignment_candidates": [
                row for section in sections for row in section.candidates
            ],
            "candidate_pair_support": list(candidates.values()),
        },
    }


def _run_demo(show_missed_matches_only: bool = False) -> None:
    """预览 MSLC 匹配，开关打开时仅显示有未匹配实例的图对。"""
    from core_algorithm.common.preview import preview_matching_results

    preview_matching_results(
        match_south_north_rail_mslc, show_missed_matches_only=show_missed_matches_only
    )


if __name__ == "__main__":
    _run_demo(show_missed_matches_only=False)
