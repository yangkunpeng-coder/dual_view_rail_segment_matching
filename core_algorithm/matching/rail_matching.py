"""根据南北逐实例 Mask 建立有序的一对一钢轨关联。"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_algorithm.common.match_result import (
    RailInstance,
    RailMatch,
    RailMatchingResult,
    UnmatchedRail,
    make_unmatched_rail,
)
from data_preprocess.common.config import PROJECT_ROOT
from utils.instance_masks import as_mask_stack

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.engine.results import Masks

_SECTIONS = (0.25, 0.5, 0.75)
_MAIN_SECTION = 1
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
_MAPPING_COST_MARGIN = 0.65
_MATCH_COST_MARGIN = 0.15


@dataclass(frozen=True)
class _Measurement:
    """图内实例及其排序坐标，异常实例保留原因供输出。"""

    instance: RailInstance
    position: float
    reason: str | None


@dataclass(frozen=True)
class _Alignment:
    """一个南到北横向映射下的有序配对路径。"""

    scale: float
    offset: float
    cost: float
    pairs: list[tuple[int, int]]


def match_south_north_rail(
    south_image_path: str | Path,
    north_image_path: str | Path,
    south_masks: np.ndarray | Tensor | Masks | None,
    north_masks: np.ndarray | Tensor | Masks | None,
) -> RailMatchingResult:
    """用当前图对的实例 Mask 推断南北钢轨对应。

    Args:
        south_image_path: 南侧原图相对于项目根目录的路径。
        north_image_path: 北侧原图相对于项目根目录的路径。
        south_masks: 南侧 YOLO Results.masks，或原图尺寸 (N, H, W)
            的 NumPy 数组或 Tensor。大于 0.5 的像素视为前景。
            单张二维掩码表示一个实例，None 表示没有实例。
        north_masks: 北侧掩码，约定同 south_masks。每个掩码只表示一根钢轨。

    Returns:
        图片路径、带共享 ID 的配对和南北未匹配实例。
        mask_index 对应输入顺序，position_error 为北侧原图宽度归一化残差。

    Raises:
        OSError: 无法读取原图。
        ValueError: 路径、图像尺寸或掩码尺寸无效。
    """
    south_shape = _image_shape(south_image_path)
    north_shape = _image_shape(north_image_path)
    south = _measure_instances(as_mask_stack(south_masks, south_shape), "south")
    north = _measure_instances(as_mask_stack(north_masks, north_shape), "north")
    valid_north = [measurement for measurement in north if measurement.reason is None]
    valid_south = [measurement for measurement in south if measurement.reason is None]
    north_positions = np.asarray(
        [measurement.position for measurement in valid_north], dtype=np.float64
    )
    south_positions = np.asarray(
        [measurement.position for measurement in valid_south], dtype=np.float64
    )

    alignments = _find_alignments(north_positions, south_positions)
    matches: list[RailMatch] = []
    ambiguous_north: set[str] = set()
    ambiguous_south: set[str] = set()
    default_reason = "insufficient_alignment_evidence"
    if alignments:
        default_reason = "no_reliable_partner"
        best = min(alignments, key=lambda alignment: alignment.cost)
        accepted, ambiguous = _select_reliable_pairs(
            north_positions, south_positions, best, alignments
        )
        for index, (north_index, south_index) in enumerate(accepted, start=1):
            residual = abs(
                north_positions[north_index]
                - (best.scale * south_positions[south_index] + best.offset)
            )
            matches.append(
                {
                    "shared_id": f"R{index:03d}",
                    "south": valid_south[south_index].instance,
                    "north": valid_north[north_index].instance,
                    "position_error": float(residual),
                }
            )
        for north_index, south_index in ambiguous:
            ambiguous_north.add(valid_north[north_index].instance["instance_id"])
            ambiguous_south.add(valid_south[south_index].instance["instance_id"])

    return {
        "south_image_path": Path(south_image_path).as_posix(),
        "north_image_path": Path(north_image_path).as_posix(),
        "matches": matches,
        "unmatched_south": _unmatched_instances(
            south, matches, "south", ambiguous_south,
            default_reason if north else "other_side_empty",
        ),
        "unmatched_north": _unmatched_instances(
            north, matches, "north", ambiguous_north,
            default_reason if south else "other_side_empty",
        ),
    }


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


def _section_center(row: NDArray[np.bool_]) -> tuple[float, str | None]:
    """读取连续交段中点，缺失或分叉时返回 NaN 及原因。"""
    columns = np.flatnonzero(row)
    if not len(columns):
        return np.nan, "no_measurement_intersection"
    if columns[-1] - columns[0] + 1 != len(columns):
        return np.nan, "multiple_measurement_intersections"
    return float((columns[0] + columns[-1]) / 2), None


def _measure_instances(
    masks: NDArray[np.bool_], side: Literal["north", "south"]
) -> list[_Measurement]:
    """取主截面中心并用其它截面检查单侧排序。"""
    count, height, width = masks.shape
    rows = [round((height - 1) * section) for section in _SECTIONS]
    centers = np.empty((count, len(rows)), dtype=np.float64)
    reasons: list[str | None] = [None] * count
    for mask_index, mask in enumerate(masks):
        for section_index, row in enumerate(rows):
            center, reason = _section_center(mask[row])
            centers[mask_index, section_index] = center
            if section_index == _MAIN_SECTION:
                reasons[mask_index] = reason

    main = centers[:, _MAIN_SECTION]
    main_deltas = main[:, None] - main[None, :]
    conflicts = (main_deltas == 0) & ~np.eye(count, dtype=np.bool_)
    for section_index in (0, len(rows) - 1):
        auxiliary = centers[:, section_index]
        deltas = auxiliary[:, None] - auxiliary[None, :]
        conflicts |= (main_deltas * deltas) < 0

    positions = main / (width - 1)
    if side == "south":
        positions = 1 - positions
    measurements: list[_Measurement] = []
    for order, mask_index in enumerate(np.argsort(positions, kind="stable"), start=1):
        center_xy = None
        if np.isfinite(main[mask_index]):
            center_xy = float(main[mask_index]), rows[_MAIN_SECTION]
        instance: RailInstance = {
            "instance_id": f"{side[0].upper()}{order:02d}",
            "mask_index": int(mask_index),
            "center_xy": center_xy,
        }
        reason = reasons[int(mask_index)]
        if conflicts[mask_index].any():
            reason = "inconsistent_instance_order"
        position = float(positions[mask_index])
        measurements.append(_Measurement(instance, position, reason))
    return measurements


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


def _find_alignments(
    north: NDArray[np.float64], south: NDArray[np.float64]
) -> list[_Alignment]:
    """筛选有足够匹配数量和横向覆盖的候选路径。"""
    smaller_count = min(len(north), len(south))
    if smaller_count < _MIN_ANCHORS:
        return []
    minimum_count = max(_MIN_ANCHORS, ceil(smaller_count * _MIN_MATCH_FRACTION))
    alignments = []
    for scale, offset in _candidate_mappings(north, south, minimum_count):
        cost, pairs = _ordered_alignment(_position_costs(north, south, scale, offset))
        if len(pairs) < minimum_count:
            continue
        north_span = north[pairs[-1][0]] - north[pairs[0][0]]
        south_span = south[pairs[-1][1]] - south[pairs[0][1]]
        if (
            north_span < _MIN_SPAN_FRACTION * (north[-1] - north[0])
            or south_span < _MIN_SPAN_FRACTION * (south[-1] - south[0])
        ):
            continue
        alignments.append(_Alignment(scale, offset, cost, pairs))
    return alignments


def _select_reliable_pairs(
    north: NDArray[np.float64],
    south: NDArray[np.float64],
    best: _Alignment,
    alignments: list[_Alignment],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """保留代价接近的映射中一致且有明显代价优势的配对。"""
    common = set(best.pairs)
    possible = set(best.pairs)
    for alignment in alignments:
        if alignment.cost - best.cost <= _MAPPING_COST_MARGIN:
            common.intersection_update(alignment.pairs)
            possible.update(alignment.pairs)

    costs = _position_costs(north, south, best.scale, best.offset)
    accepted = []
    for pair in best.pairs:
        if pair not in common:
            continue
        alternate = costs.copy()
        alternate[pair] = np.inf
        alternate_cost, _ = _ordered_alignment(alternate)
        if alternate_cost - best.cost > _MATCH_COST_MARGIN:
            accepted.append(pair)
    return accepted, possible.difference(accepted)


def _unmatched_instances(
    measurements: list[_Measurement],
    matches: list[RailMatch],
    side: Literal["north", "south"],
    ambiguous_ids: set[str],
    default_reason: str,
) -> list[UnmatchedRail]:
    """记录每个未匹配实例，并保留取点失败或几何歧义原因。"""
    matched_ids = {match[side]["instance_id"] for match in matches}
    unmatched: list[UnmatchedRail] = []
    for measurement in measurements:
        instance_id = measurement.instance["instance_id"]
        if instance_id in matched_ids:
            continue
        reason = measurement.reason
        if reason is None and instance_id in ambiguous_ids:
            reason = "ambiguous_alignment"
        unmatched.append(
            make_unmatched_rail(measurement.instance, reason or default_reason)
        )
    return unmatched


def _run_demo(show_missed_matches_only: bool = False) -> None:
    """预览 V1 匹配，开关打开时仅显示有未匹配实例的图对。"""
    from core_algorithm.common.preview import preview_matching_results

    preview_matching_results(
        match_south_north_rail, show_missed_matches_only=show_missed_matches_only
    )


if __name__ == "__main__":
    _run_demo(show_missed_matches_only=False)
