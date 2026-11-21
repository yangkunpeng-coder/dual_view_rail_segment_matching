"""使用同一批 LabelMe 实例掩码评测南北钢轨关联。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from functools import partial
from hashlib import sha256
from itertools import combinations
import json
from math import ceil, isfinite
from os.path import relpath
from pathlib import Path
from random import Random
import re
import sys
from time import perf_counter
from typing import cast
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import cv2
import numpy as np
from numpy.typing import NDArray

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_algorithm.common.match_result import (
    RailInstance,
    RailMatchingResult,
    Side,
    make_unmatched_rail,
)
from core_algorithm.common.preview import (
    SECTION_PERCENTS,
    LabelMeInstanceRef,
    LabelMeView,
    load_labelme_view_details,
    measure_section,
    render_side,
    section_rows,
)
from core_algorithm.matching import (
    rail_matching,
    rail_matching_cgm,
    rail_matching_gmsa,
    rail_matching_mslc,
    rail_matching_isa,
)
from data_preprocess.common.config import (
    PROJECT_ROOT,
    load_dataset_relative_path,
    load_yaml_mapping,
)
from data_preprocess.match_pair_image import match_pair_image
from utils.image_layout import compose_image_pair
from utils.report_archive import zip_analysis_reports

_OUTPUT_ROOT = PROJECT_ROOT / "out" / "evaluate_rail_match"
_README = PROJECT_ROOT / "doc" / "readme" / "evaluate_rail_match_readme.md"
_MSLC_README = _README.with_name("rail_matching_mslc_readme.md")
_GMSA_README = _README.with_name("rail_matching_gmsa_readme.md")
_ISA_README = _README.with_name("rail_matching_isa_readme.md")
_BASELINE_ALGORITHM = rail_matching.__name__.rsplit(".", 1)[-1]
_MSLC_ALGORITHM = rail_matching_mslc.__name__.rsplit(".", 1)[-1]
_GMSA_ALGORITHM = rail_matching_gmsa.__name__.rsplit(".", 1)[-1]
_ISA_ALGORITHM = rail_matching_isa.__name__.rsplit(".", 1)[-1]
_CGM_ALGORITHM = rail_matching_cgm.__name__.rsplit(".", 1)[-1]
_DEFAULT_ALGORITHMS = (_BASELINE_ALGORITHM, _MSLC_ALGORITHM, _GMSA_ALGORITHM)
_ALGORITHM_ALIASES = {
    "mslc_rail": _MSLC_ALGORITHM, "gmsa_rail": _GMSA_ALGORITHM,
    "isa_rail": _ISA_ALGORITHM, "cgm_rail": _CGM_ALGORITHM,
}
_SIDES = ("south", "north")
_LOW_COVERAGE = 0.50
_HIGH_RESIDUAL_MIN = 0.005
_HIGH_RESIDUAL_PERCENTILE = 90
_CONTROL_FRACTION = 0.20
_REVIEW_TAGS = (
    "input_error", "algorithm_error", "zero_matches", "low_coverage",
    "ambiguous_alignment", "measurement_failure", "order_conflict", "high_residual",
    "diagnostic_error",
)
_MEASUREMENT_FAILURES = (
    "no_measurement_intersection", "multiple_measurement_intersections",
)

_MANIFEST_COLUMNS = (
    "pair_id", "source_row", "source_pair_id", "south_image_path", "north_image_path",
    "south_label_path", "north_label_path", "north_minus_south_seconds",
)
_INPUT_MANIFEST_COLUMNS = (
    *_MANIFEST_COLUMNS, "read_status", "read_error", "any_empty_label", "both_empty_labels",
    *(f"{side}_{field}" for side in _SIDES for field in (
        "instance_count", "empty_label", "read_status", "label_sha256", "read_error",
    )),
)
_INSTANCE_COLUMNS = (
    "pair_id", "side", "image_path", "mask_index", "reference_instance_id",
    "instance_id", "match_status", "shared_id", "partner_mask_index", "reason",
    "measurement_reason", "normalized_position", "mask_area_px", "bbox_x0_px",
    "bbox_y0_px", "bbox_x1_px", "bbox_y1_px",
    *(f"section_{percent}_{name}" for percent in SECTION_PERCENTS
      for name in ("y_px", "center_x_px", "foreground_px", "run_count", "reason")),
)
_MATCH_COLUMNS = (
    "pair_id", "shared_id",
    *(f"{side}_{name}" for side in _SIDES
      for name in ("instance_id", "mask_index", "center_x_px", "center_y_px")),
    "position_error_norm", "position_error_north_px",
)
_UNMATCHED_COLUMNS = (
    "pair_id", "side", "image_path", "instance_id", "mask_index", "reason",
    "normalized_position", "section_50_center_x_px", "section_50_y_px",
)
_MSLC_MATCH_COLUMNS = (
    "combined_support", "support_25", "support_50", "support_75",
    "residual_50_norm", "support_section_count", "support_level",
    "alternative_cost_margin",
)
_MSLC_UNMATCHED_COLUMNS = (
    "candidate_count", "best_candidate_mask_index", "best_candidate_support",
    "best_candidate_decision",
)
_DIAGNOSTIC_COLUMNS = ("pair_id", "status", "metric", "value")
_REVIEW_COLUMNS = (
    "rank", "pair_id", "category", "tags", "status", "match_count", "south_coverage",
    "north_coverage", "position_error_p95_norm", "overlay_status", "overlay_error",
    "south_label", "north_label", "image_paths_file", "comparison_image",
)
_COMPARISON_COLUMNS = (
    "pair_id", "algorithm_a", "algorithm_b", "status_a", "status_b",
    "comparison_status",
    "match_count_a", "match_count_b", "shared_count", "only_a_count", "only_b_count",
    "shared_pairs", "only_a_pairs", "only_b_pairs", "south_coverage_delta_b_minus_a",
    "north_coverage_delta_b_minus_a",
)
_GMSA_ID_COLUMNS = tuple(
    f"{side}_{field}" for side in _SIDES
    for field in ("instance_id", "mask_index", "shape_index", "group_id", "order_index")
)
_GMSA_RELATION_COLUMNS = (
    "pair_id", "shared_id", *_GMSA_ID_COLUMNS, "status", "reason",
    "total_cost", "quality_cost", "span_cost", "gap_cost", "anchor_cost",
    "valid_dtw_points", "evidence_transition_count", "best_path_cost",
    "second_path_cost", "global_path_margin", "pair_exclusion_margin",
    "alternative_partner_count", "longitudinal_direction", "direction_support",
    "direction_consensus", "evidence_status", "candidate_rank", "path_cost",
    "cost_gap",
)
_GMSA_INSTANCE_COLUMNS = (
    "pair_id", "side", "instance_id", "mask_index", "shape_index", "group_id",
    "order_index", "valid_section_count", "valid_section_fraction", "geometry_quality",
    "geometry_valid", "center_fit_rms_px", "center_x_median_px", "width_median_px",
    "relative_width_mad", "multi_interval_section_count", "broken_section_count",
    "overlap_section_count", "order_conflict_section_count", "order_stability",
    "order_uncertain",
)
_GMSA_TRANSITION_COLUMNS = (
    "pair_id", "transition_id", "path_ranks", "is_best_path", "direction",
    *(f"{side}_{endpoint}_{field}" for side in _SIDES
      for endpoint in ("start", "end") for field in ("mask_index", "order_index")),
    "south_skipped_count", "north_skipped_count", "transition_cost", "span_cost",
    "gap_cost", "anchor_cost", "position_error_norm", "valid_dtw_points",
    "dtw_coverage", "dtw_warp_fraction", "dtw_distance_cost", "dtw_warp_cost",
    "dtw_coverage_cost",
)
_GMSA_DTW_COLUMNS = (
    "pair_id", "transition_id", "step", "direction", "south_sample_index",
    "north_sample_index", "south_height", "north_height", "south_y_px", "north_y_px",
    "valid", "local_cost",
)
_GMSA_TIMING_COLUMNS = (
    "feature_time_ms", "transition_time_ms", "dp_time_ms", "diagnostics_time_ms",
)
_GMSA_PAIR_COLUMNS = (
    "pair_id", "status", "error_stage", "error_type", "error_message",
    "algorithm_version", "support_status", "resource_limited", "candidate_pair_count",
    "theoretical_transition_count", "best_path_cost", "second_path_cost",
    "empty_path_cost", "global_path_margin", "path_competition_count",
    "evidence_transition_count", "best_path_transition_count",
    "effective_span_transition_fraction", "longitudinal_direction",
    "direction_ambiguous", "direction_relative_margin", "same_best_path_cost",
    "reverse_best_path_cost", "same_best_path_match_count",
    "reverse_best_path_match_count", "direction_consensus_pair_count",
    "direction_conflict_pair_count", "direction_consensus_ratio",
    "direction_decision", "accepted_count", "tentative_count",
    "ambiguous_candidate_count", "ambiguous_best_count", "unmatched_instance_count",
    "invalid_geometry_count", "order_uncertain_count", "median_dtw_coverage",
    "median_dtw_warp_fraction", "median_span_cost", "median_gap_cost",
    *_GMSA_TIMING_COLUMNS,
)
_SECTION_PREFIX = ("pair_id", "side", "sample_index", "height_fraction", "y_px")
_GMSA_TABLE_COLUMNS = {
    "matches": _GMSA_RELATION_COLUMNS,
    "tentative_matches": _GMSA_RELATION_COLUMNS,
    "ambiguous_candidates": _GMSA_RELATION_COLUMNS,
    "unmatched_instances": (
        *_GMSA_INSTANCE_COLUMNS, "status", "reason",
        *(f"candidate_{rank}_{field}" for rank in range(1, 4)
          for field in ("mask_index", "group_id", "order_index", "path_cost")),
    ),
    "transition_diagnostics": _GMSA_TRANSITION_COLUMNS,
    "dtw_alignment": _GMSA_DTW_COLUMNS,
    "instance_geometry": _GMSA_INSTANCE_COLUMNS,
    "section_geometry": (
        *_SECTION_PREFIX, "mask_index", "visible", "valid", "weight", "intervals_px",
        "center_x_px", "width_px", "overlap", "order_conflict",
    ),
    "gap_fingerprints": (
        *_SECTION_PREFIX, "left_mask_index", "right_mask_index", "scale_px",
        "scale_source", "normalized_gap", "log_gap", "boundary_gap_px", "weight",
        "stable_large_gap",
    ),
    "paths": (
        "pair_id", "path_rank", "path_cost", "direction", "match_count",
        "evidence_transition_count", "mask_index_pairs",
    ),
}

_ISA_TABLE_COLUMNS = {
    "hypotheses": (
        "pair_id", "rank", "shift_k", "scale", "offset", "path_cost",
        "match_count", "is_selected", "mask_index_pairs",
    ),
    "pair_costs": (
        "pair_id", "south_mask_index", "north_mask_index", "south_order_index",
        "north_order_index", "node_cost", "pair_margin", "pair_margin_relative",
        "hypothesis_margin", "decision", "shared_id",
    ),
    "unmatched": (
        "pair_id", "side", "instance_id", "mask_index", "order_index",
        "valid_section_count", "geometry_quality", "status", "reason",
    ),
    "instances": (
        "pair_id", "side", "instance_id", "mask_index", "order_index",
        "valid_section_count", "geometry_quality", "geometry_valid",
        "order_uncertain", "order_stability", "overlap_section_count",
        "order_conflict_section_count",
    ),
}

MaskStack = NDArray[np.bool_]
RgbImage = NDArray[np.uint8]
CsvRow = dict[str, object]
DiagnosticValue = int | float | str | bool | None
MatchFunction = Callable[[Path, Path, MaskStack, MaskStack], RailMatchingResult]
DiagnosticFunction = Callable[
    [MaskStack, MaskStack, RailMatchingResult], dict[str, DiagnosticValue]
]
PairMatchFunction = Callable[["_PairData"], RailMatchingResult]


@dataclass(frozen=True)
class RailMatchAdapter:
    """将一个匹配算法接入评测。

    match 接收南北项目相对路径和同一批只读 (N,H,W) 布尔 Mask，按
    RailMatchingResult 返回配对与未匹配实例。两侧 mask_index 以输入顺序为准。
    diagnose 可选，接收同一对 Mask 和匹配结果并返回标量诊断指标。
    match_pair 可选，还可访问同批 Mask 的原始 Polygon 编号。
    """

    match: MatchFunction
    diagnose: DiagnosticFunction | None = None
    match_pair: PairMatchFunction | None = None


@dataclass(frozen=True)
class _Pair:
    """保留评测编号、输入路径和原配对清单的定位信息。"""

    pair_id: str
    south_image_path: str
    north_image_path: str
    source_row: int
    source_pair_id: str = ""
    south_expected_sha256: str = ""
    north_expected_sha256: str = ""


@dataclass(frozen=True)
class _PairData:
    """一对原图、项目相对路径和共享的逐 Polygon 掩码。"""

    south_path: Path
    north_path: Path
    south_image: RgbImage
    north_image: RgbImage
    south_masks: MaskStack
    north_masks: MaskStack
    south_references: tuple[LabelMeInstanceRef, ...]
    north_references: tuple[LabelMeInstanceRef, ...]


@dataclass
class _PairInput:
    """共享清单及与 Mask 索引对应的标注来源。"""

    row: CsvRow
    references: dict[Side, tuple[LabelMeInstanceRef, ...]] = field(default_factory=dict)


@dataclass
class _PairSummary:
    """逐图对的输出覆盖、残差和执行状态。"""

    pair_id: str
    south_image_path: str
    north_image_path: str
    status: str = "ok"
    south_instance_count: int | None = None
    north_instance_count: int | None = None
    south_measurable_count: int | None = None
    north_measurable_count: int | None = None
    match_count: int | None = None
    south_unmatched_count: int | None = None
    north_unmatched_count: int | None = None
    south_coverage: float | None = None
    north_coverage: float | None = None
    pair_coverage: float | None = None
    ambiguous_instance_count: int | None = None
    measurement_failure_count: int | None = None
    order_conflict_count: int | None = None
    position_error_mean_norm: float | None = None
    position_error_p95_norm: float | None = None
    position_error_max_norm: float | None = None
    load_time_ms: float | None = None
    measure_time_ms: float | None = None
    match_time_ms: float | None = None
    diagnostic_time_ms: float | None = None
    diagnostic_status: str = "not_run"
    error_stage: str = ""
    error_type: str = ""
    error_message: str = ""


@dataclass
class _Evaluation:
    """供统计、复核与对照使用的小体积逐图对结果。"""

    pair: _Pair
    summary: _PairSummary
    result: RailMatchingResult | None = None
    measurements: list[CsvRow] = field(default_factory=list)
    diagnostics: dict[str, DiagnosticValue] = field(default_factory=dict)
    north_width: int | None = None
    references: dict[Side, tuple[LabelMeInstanceRef, ...]] = field(default_factory=dict)


def _diagnose_rail_matching(
    south_masks: MaskStack, north_masks: MaskStack, result: RailMatchingResult
) -> dict[str, DiagnosticValue]:
    """复用原算法的候选搜索，重算配对接受前的诊断信息。"""
    south = np.asarray([
        item.position for item in rail_matching._measure_instances(south_masks, "south")
        if item.reason is None
    ])
    north = np.asarray([
        item.position for item in rail_matching._measure_instances(north_masks, "north")
        if item.reason is None
    ])
    minimum = rail_matching._MIN_ANCHORS
    smaller_count = min(len(south), len(north))
    required = max(minimum, ceil(smaller_count * rail_matching._MIN_MATCH_FRACTION))
    alignments = rail_matching._find_alignments(north, south)
    diagnostics: dict[str, DiagnosticValue] = {
        "south_usable_position_count": len(south),
        "north_usable_position_count": len(north),
        "south_anchor_count": len(rail_matching._anchor_triplets(south))
        if len(south) >= minimum else 0,
        "north_anchor_count": len(rail_matching._anchor_triplets(north))
        if len(north) >= minimum else 0,
        "minimum_match_count": required,
        "valid_alignment_count": len(alignments),
        "accepted_match_count": len(result["matches"]),
    }
    if not alignments:
        return diagnostics

    best = min(alignments, key=lambda alignment: alignment.cost)
    competitive = [
        alignment for alignment in alignments
        if alignment.cost - best.cost <= rail_matching._MAPPING_COST_MARGIN
    ]
    common = set(best.pairs).intersection(
        *(alignment.pairs for alignment in competitive)
    )
    first, last = best.pairs[0], best.pairs[-1]
    south_span = south[last[1]] - south[first[1]]
    north_span = north[last[0]] - north[first[0]]
    diagnostics.update(
        competitive_alignment_count=len(competitive),
        best_scale=best.scale,
        best_offset=best.offset,
        best_cost=best.cost,
        best_path_match_count=len(best.pairs),
        best_south_span_fraction=float(south_span / np.ptp(south)),
        best_north_span_fraction=float(north_span / np.ptp(north)),
        consensus_match_count=len(common),
        best_path_removed_count=len(best.pairs) - len(result["matches"]),
    )
    return diagnostics


def _diagnose_mslc_rail(
    _south_masks: MaskStack, _north_masks: MaskStack, result: RailMatchingResult
) -> dict[str, DiagnosticValue]:
    """MSLC 在匹配时已记录各截面的路径，直接使用同次诊断。"""
    return cast(rail_matching_mslc.MSLCMatchingResult, result)["diagnostics"]["summary"]


def _match_gmsa(
    data: _PairData, *, config: rail_matching_gmsa.GMSARailConfig | None = None,
) -> rail_matching_gmsa.GMSAMatchingResult:
    """将与 V1/V2 相同的输入 Mask 及对应来源传给 V3。"""
    return rail_matching_gmsa.match_south_north_rail_gmsa(
        rail_matching_gmsa.RailView(
            data.south_path, data.south_masks, data.south_references
        ),
        rail_matching_gmsa.RailView(
            data.north_path, data.north_masks, data.north_references
        ),
        config,
    )


def _diagnose_gmsa(
    _south_masks: MaskStack, _north_masks: MaskStack, result: RailMatchingResult
) -> dict[str, DiagnosticValue]:
    """复用同次 GMSA 调用的标量诊断。"""
    diagnostics = cast(rail_matching_gmsa.GMSAMatchingResult, result)["diagnostics"]
    if diagnostics is None:
        raise ValueError("GMSA 评测需要同次匹配的诊断")
    return diagnostics["summary"]


def _diagnose_isa(
    _south_masks: MaskStack, _north_masks: MaskStack, result: RailMatchingResult
) -> dict[str, DiagnosticValue]:
    """复用同次 ISA-Rail 调用的标量诊断。"""
    diagnostics = cast(rail_matching_isa.ISAMatchingResult, result)["diagnostics"]
    if diagnostics is None:
        raise ValueError("ISA-Rail 评测需要同次匹配的诊断")
    return diagnostics["summary"]


def _diagnose_cgm(
    _south_masks: MaskStack, _north_masks: MaskStack, result: RailMatchingResult
) -> dict[str, DiagnosticValue]:
    """复用同次 CGM-Rail 调用的标量诊断。"""
    diagnostics = cast(rail_matching_cgm.CGMMatchingResult, result)["diagnostics"]
    if diagnostics is None:
        raise ValueError("CGM-Rail 评测需要同次匹配的诊断")
    return diagnostics["summary"]


_ALGORITHMS = {
    _BASELINE_ALGORITHM: RailMatchAdapter(
        rail_matching.match_south_north_rail, _diagnose_rail_matching
    ),
    _MSLC_ALGORITHM: RailMatchAdapter(
        rail_matching_mslc.match_south_north_rail_mslc, _diagnose_mslc_rail
    ),
    _GMSA_ALGORITHM: RailMatchAdapter(
        rail_matching_gmsa.match_south_north_rail_gmsa_masks, _diagnose_gmsa, _match_gmsa
    ),
    _ISA_ALGORITHM: RailMatchAdapter(
        rail_matching_isa.match_south_north_rail_isa, _diagnose_isa
    ),
    _CGM_ALGORITHM: RailMatchAdapter(
        rail_matching_cgm.match_south_north_rail_cgm, _diagnose_cgm
    ),
}


def register_rail_match_algorithm(name: str, adapter: RailMatchAdapter) -> None:
    """注册额外匹配算法，名称同时用于结果子目录。

    Args:
        name: 算法实现的 Python 文件名，不带 .py，使用小写字母、数字和下划线。
        adapter: 使用共享掩码和统一结果约定的算法适配器。

    Raises:
        ValueError: 算法名称无效或已注册。
    """
    if (
        re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
        or name in _ALGORITHMS
        or name in _ALGORITHM_ALIASES
    ):
        raise ValueError(f"算法名称无效或已注册 {name!r}")
    _ALGORITHMS[name] = adapter


def _has_rail_polygons(image_path: str) -> bool:
    """判断同名 LabelMe 标注是否含 rail 多边形。

    只做轻量检查，用于在默认清单里剔除"双侧都没有标注"的图对；真正的格式与尺寸
    校验仍由逐对加载完成。标注缺失或读不出时返回 True，让加载阶段照常报错。
    """
    label_path = Path(PROJECT_ROOT, image_path).with_suffix(".json")
    try:
        annotation = json.loads(label_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return True
    shapes = annotation.get("shapes") or []
    return any(
        isinstance(shape, dict) and shape.get("label") == "rail" for shape in shapes
    )


def _default_pairs(
    dataset: Path, *, include_empty: bool = False
) -> tuple[list[_Pair], list[str]]:
    """将项目的时间配对结果转为统一清单。

    默认剔除两侧都没有 rail 标注的图对：这类图对没有实例，不携带匹配信息，却会
    计入每个算法的零匹配数并进入复核队列。编号仍按原始配对顺序生成，便于和历史
    运行按 `pair_id` 对照；`include_empty=True` 可复现包含空标注的旧口径。

    Args:
        dataset: 数据集相对项目根目录的路径，须含 south 与 north 目录。
        include_empty: 为 True 时保留双侧空标注的图对。

    Returns:
        图对清单，以及被剔除图对的 `pair_id`。
    """
    pairs = match_pair_image(dataset)
    candidates = [
        _Pair(
            f"P{index:04d}", (dataset / "south" / f"{south}.png").as_posix(),
            (dataset / "north" / f"{north}.png").as_posix(), index,
        )
        for index, (south, north) in enumerate(
            zip(pairs["south"], pairs["north"], strict=True), start=1
        )
    ]
    if include_empty:
        return candidates, []
    kept: list[_Pair] = []
    excluded: list[str] = []
    for pair in candidates:
        if _has_rail_polygons(pair.south_image_path) and _has_rail_polygons(
            pair.north_image_path
        ):
            kept.append(pair)
        else:
            excluded.append(pair.pair_id)
    return kept, excluded


def _read_pairs(manifest: Path) -> list[_Pair]:
    """读取两列图片路径，保留异常行供逐对加载时记录。"""
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"south_image_path", "north_image_path"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{manifest} 需包含 south_image_path 和 north_image_path")
        fixed = {"pair_id", "south_label_sha256", "north_label_sha256"}.issubset(
            reader.fieldnames or ()
        )
        pairs: list[_Pair] = []
        seen: set[str] = set()
        for index, row in enumerate(reader, start=1):
            source_id = (row.get("pair_id") or "").strip()
            pair_id = source_id if fixed else f"P{index:04d}"
            if re.fullmatch(r"P[0-9]+", pair_id) is None or pair_id in seen:
                raise ValueError(f"{manifest} 第 {reader.line_num} 行的 Pair ID 无效或重复")
            seen.add(pair_id)
            pairs.append(_Pair(
                pair_id, (row["south_image_path"] or "").strip(),
                (row["north_image_path"] or "").strip(), reader.line_num,
                (row.get("source_pair_id") or source_id).strip(),
                (row.get("south_label_sha256") or "").strip(),
                (row.get("north_label_sha256") or "").strip(),
            ))
        return pairs


def _label_path(image_path: str) -> str:
    """返回清单中图片对应的同名 LabelMe 路径。"""
    path = Path(image_path)
    return path.with_suffix(".json").as_posix() if path.name else ""


def _manifest_row(pair: _Pair) -> CsvRow:
    """导出来源及可从文件名解析的南北时间差。"""
    delta = None
    south_stem = Path(pair.south_image_path).stem
    north_stem = Path(pair.north_image_path).stem
    if re.fullmatch(r"\d{14}", south_stem) and re.fullmatch(r"\d{14}", north_stem):
        try:
            south_time = datetime.strptime(south_stem, "%Y%m%d%H%M%S")
            north_time = datetime.strptime(north_stem, "%Y%m%d%H%M%S")
        except ValueError:
            # 自定义文件名可以没有可解析的采集时间。
            delta = None
        else:
            delta = int((north_time - south_time).total_seconds())
    return {
        **asdict(pair), "south_label_path": _label_path(pair.south_image_path),
        "north_label_path": _label_path(pair.north_image_path),
        "north_minus_south_seconds": delta,
    }


def _project_image_path(image_path: str, side: Side) -> Path:
    """规范化成原算法接受的项目相对路径。"""
    if not image_path:
        raise ValueError(f"{side} 图片路径为空")
    path = (PROJECT_ROOT / image_path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{side} 图片必须位于项目内 {path}")
    return path.relative_to(PROJECT_ROOT)


def _load_pair(pair: _Pair, source: _PairInput | None = None) -> _PairData:
    """一次解析两侧标注并保存同一份输入指纹，失败保留各侧状态。"""
    views: dict[Side, tuple[Path, LabelMeView]] = {}
    errors: list[Exception] = []
    sides: tuple[tuple[Side, str, str], ...] = (
        ("south", pair.south_image_path, pair.south_expected_sha256),
        ("north", pair.north_image_path, pair.north_expected_sha256),
    )
    for side, image_path, expected_hash in sides:
        label_path: Path | None = None
        actual_hash: str | None = None
        try:
            path = _project_image_path(image_path, side)
            absolute = PROJECT_ROOT / path
            label_path = absolute.with_suffix(".json")
            view = load_labelme_view_details(absolute)
            actual_hash = view.label_sha256
            if expected_hash and actual_hash != expected_hash.lower():
                raise ValueError(f"{side} JSON 与清单中的 SHA-256 不一致 {label_path}")
        except (OSError, ValueError, TypeError, cv2.error) as error:
            errors.append(error)
            if source is not None:
                if actual_hash is None and label_path is not None and label_path.is_file():
                    try:
                        actual_hash = sha256(label_path.read_bytes()).hexdigest()
                    except OSError as hash_error:
                        source.row[f"{side}_read_error"] = f"{error}；{hash_error}"
                source.row.setdefault(f"{side}_read_error", str(error))
                source.row[f"{side}_read_status"] = "error"
                source.row[f"{side}_label_sha256"] = actual_hash
            continue
        view.masks.flags.writeable = False
        views[side] = path, view
        if source is not None:
            source.references[side] = view.references
            source.row[f"{side}_label_path"] = path.with_suffix(".json").as_posix()
            source.row[f"{side}_label_sha256"] = actual_hash
            source.row[f"{side}_instance_count"] = len(view.masks)
            source.row[f"{side}_empty_label"] = not len(view.masks)
            source.row[f"{side}_read_status"] = "ok"
    if source is not None:
        source.row["read_status"] = "error" if errors else "ok"
    if errors:
        raise errors[0]
    south_path, south = views["south"]
    north_path, north = views["north"]
    if source is not None:
        source.row["any_empty_label"] = not len(south.masks) or not len(north.masks)
        source.row["both_empty_labels"] = not len(south.masks) and not len(north.masks)
    return _PairData(
        south_path, north_path, south.image, north.image, south.masks, north.masks,
        south.references, north.references,
    )


def _measure_masks(masks: MaskStack, side: Side, image_path: str) -> list[CsvRow]:
    """记录原图掩码面积、边界及三个参考截面的实测值。"""
    image_rows = section_rows(masks.shape[1])
    rows: list[CsvRow] = []
    for measurement in rail_matching._measure_instances(masks, side):
        mask_index = measurement.instance["mask_index"]
        mask = masks[mask_index]
        x, y, width, box_height = cv2.boundingRect(mask.view(np.uint8))
        area = int(np.count_nonzero(mask[y:y + box_height, x:x + width]))
        row: CsvRow = {
            "side": side, "image_path": image_path, "mask_index": mask_index,
            "reference_instance_id": measurement.instance["instance_id"],
            "measurement_reason": measurement.reason,
            "normalized_position": measurement.position
            if isfinite(measurement.position) else None,
            "mask_area_px": area,
            "bbox_x0_px": x if area else None, "bbox_y0_px": y if area else None,
            "bbox_x1_px": x + width if area else None,
            "bbox_y1_px": y + box_height if area else None,
        }
        for percent, section_y in zip(SECTION_PERCENTS, image_rows, strict=True):
            section = measure_section(mask[section_y])
            prefix = f"section_{percent}"
            row[f"{prefix}_y_px"] = section_y
            row[f"{prefix}_center_x_px"] = (
                section.center_x if section.reason is None else None
            )
            row[f"{prefix}_foreground_px"] = section.foreground_px
            row[f"{prefix}_run_count"] = section.run_count
            row[f"{prefix}_reason"] = section.reason
        rows.append(row)
    return rows


def _new_summary(pair: _Pair) -> _PairSummary:
    return _PairSummary(pair.pair_id, pair.south_image_path, pair.north_image_path)


def _record_error(
    summary: _PairSummary, stage: str, error: Exception, *, status: str
) -> None:
    summary.status = status
    summary.error_stage = stage
    summary.error_type = type(error).__name__
    summary.error_message = str(error)


def _validate_result(result: RailMatchingResult, data: _PairData) -> None:
    """检查适配器是否完整且唯一地报告每个输入 Mask。"""
    matches = result["matches"]
    if len({match["shared_id"] for match in matches}) != len(matches):
        raise ValueError("同一图对内 shared_id 重复")
    for match in matches:
        error = match["position_error"]
        if not isfinite(error) or error < 0:
            raise ValueError("position_error 应为非负有限数值")
    for side, masks, unmatched in (
        ("south", data.south_masks, result["unmatched_south"]),
        ("north", data.north_masks, result["unmatched_north"]),
    ):
        instances: list[RailInstance] = [
            match["south"] if side == "south" else match["north"] for match in matches
        ]
        instances.extend(unmatched)
        indices = sorted(instance["mask_index"] for instance in instances)
        if (
            indices != list(range(len(masks)))
            or any(not isinstance(index, (int, np.integer)) for index in indices)
        ):
            raise ValueError(f"{side} 算法结果须完整且唯一地引用输入 Mask")
        if len({instance["instance_id"] for instance in instances}) != len(indices):
            raise ValueError(f"{side} 图内 instance_id 重复")
        for instance in instances:
            center = instance["center_xy"]
            if center is not None and (
                len(center) != 2 or not all(map(isfinite, center))
            ):
                raise ValueError(f"{side} center_xy 应为有限 (x, y) 或 None")
        for instance in unmatched:
            if not instance["reason"]:
                raise ValueError(f"{side} 未匹配实例缺少 reason")


def _coverage(match_count: int, instance_count: int) -> float | None:
    return match_count / instance_count if instance_count else None


def _update_match_summary(summary: _PairSummary, result: RailMatchingResult) -> None:
    matches = result["matches"]
    south_unmatched = result["unmatched_south"]
    north_unmatched = result["unmatched_north"]
    match_count = len(matches)
    summary.match_count = match_count
    summary.south_unmatched_count = len(south_unmatched)
    summary.north_unmatched_count = len(north_unmatched)
    south_count = match_count + len(south_unmatched)
    north_count = match_count + len(north_unmatched)
    summary.south_coverage = _coverage(match_count, south_count)
    summary.north_coverage = _coverage(match_count, north_count)
    summary.pair_coverage = _coverage(2 * match_count, south_count + north_count)
    summary.ambiguous_instance_count = sum(
        instance["reason"] == "ambiguous_alignment"
        for instance in (*south_unmatched, *north_unmatched)
    )
    if matches:
        errors = [match["position_error"] for match in matches]
        summary.position_error_mean_norm = float(np.mean(errors))
        summary.position_error_p95_norm = float(np.percentile(errors, 95))
        summary.position_error_max_norm = max(errors)


def _evaluate_pair(
    evaluation: _Evaluation, data: _PairData, adapter: RailMatchAdapter
) -> None:
    """调用适配器，隔离单个算法和单次诊断的异常。"""
    summary = evaluation.summary
    summary.south_instance_count = len(data.south_masks)
    summary.north_instance_count = len(data.north_masks)
    evaluation.north_width = data.north_masks.shape[2]
    summary.south_measurable_count = sum(
        row["side"] == "south" and row["measurement_reason"] is None
        for row in evaluation.measurements
    )
    summary.north_measurable_count = sum(
        row["side"] == "north" and row["measurement_reason"] is None
        for row in evaluation.measurements
    )
    summary.measurement_failure_count = sum(
        row["measurement_reason"] in _MEASUREMENT_FAILURES
        for row in evaluation.measurements
    )
    summary.order_conflict_count = sum(
        row["measurement_reason"] == "inconsistent_instance_order"
        for row in evaluation.measurements
    )

    start = perf_counter()
    stage = "match"
    try:
        if adapter.match_pair is None:
            result = adapter.match(
                data.south_path, data.north_path, data.south_masks, data.north_masks
            )
        else:
            result = adapter.match_pair(data)
        summary.match_time_ms = 1000 * (perf_counter() - start)
        stage = "validate_result"
        _validate_result(result, data)
        _update_match_summary(summary, result)
    except Exception as error:
        if summary.match_time_ms is None:
            summary.match_time_ms = 1000 * (perf_counter() - start)
        _record_error(summary, stage, error, status="algorithm_error")
        return
    evaluation.result = result

    if adapter.diagnose is None:
        summary.diagnostic_status = "not_supported"
        return
    start = perf_counter()
    try:
        diagnostics = adapter.diagnose(data.south_masks, data.north_masks, result)
        if not isinstance(diagnostics, dict):
            raise TypeError("diagnose 应返回指标名到标量的字典")
        evaluation.diagnostics = diagnostics
    except Exception as error:
        summary.diagnostic_status = "error"
        _record_error(summary, "diagnostics", error, status="ok")
    else:
        summary.diagnostic_status = "ok"
    finally:
        summary.diagnostic_time_ms = 1000 * (perf_counter() - start)


def _evaluate_pairs(
    pairs: Sequence[_Pair], adapters: dict[str, RailMatchAdapter]
) -> tuple[dict[str, list[_Evaluation]], dict[str, _PairInput]]:
    """每对只加载和测量一次，再将相同输入交给各算法。"""
    evaluations: dict[str, list[_Evaluation]] = {name: [] for name in adapters}
    inputs: dict[str, _PairInput] = {}
    for pair in pairs:
        source = _PairInput(_manifest_row(pair))
        inputs[pair.pair_id] = source
        start = perf_counter()
        try:
            data = _load_pair(pair, source)
        except Exception as error:
            load_time = 1000 * (perf_counter() - start)
            source.row["read_status"] = "error"
            source.row["read_error"] = str(error)
            for rows in evaluations.values():
                summary = _new_summary(pair)
                summary.load_time_ms = load_time
                _record_error(summary, "load", error, status="input_error")
                rows.append(_Evaluation(pair, summary))
            continue

        load_time = 1000 * (perf_counter() - start)
        start = perf_counter()
        try:
            measurements = _measure_masks(
                data.south_masks, "south", pair.south_image_path
            )
            measurements.extend(
                _measure_masks(data.north_masks, "north", pair.north_image_path)
            )
        except Exception as error:
            measure_time = 1000 * (perf_counter() - start)
            for rows in evaluations.values():
                summary = _new_summary(pair)
                summary.load_time_ms = load_time
                summary.measure_time_ms = measure_time
                _record_error(summary, "measure", error, status="input_error")
                rows.append(_Evaluation(pair, summary))
        else:
            measure_time = 1000 * (perf_counter() - start)
            for name, adapter in adapters.items():
                summary = _new_summary(pair)
                summary.load_time_ms = load_time
                summary.measure_time_ms = measure_time
                evaluation = _Evaluation(
                    pair, summary, measurements=measurements, references=source.references
                )
                _evaluate_pair(evaluation, data, adapter)
                evaluations[name].append(evaluation)
        finally:
            del data
    return evaluations, inputs


def _instance_rows(evaluation: _Evaluation) -> Iterable[CsvRow]:
    """按原始 Mask 索引连接算法输出和公共几何测量。"""
    links: dict[tuple[object, object], CsvRow] = {}
    result = evaluation.result
    if result is not None:
        for match in result["matches"]:
            for side, instance, partner in (
                ("south", match["south"], match["north"]),
                ("north", match["north"], match["south"]),
            ):
                links[side, instance["mask_index"]] = {
                    "instance_id": instance["instance_id"], "match_status": "matched",
                    "shared_id": match["shared_id"],
                    "partner_mask_index": partner["mask_index"],
                }
        for side, unmatched in (
            ("south", result["unmatched_south"]), ("north", result["unmatched_north"]),
        ):
            for instance in unmatched:
                links[side, instance["mask_index"]] = {
                    "instance_id": instance["instance_id"], "match_status": "unmatched",
                    "reason": instance["reason"],
                }
    for row in evaluation.measurements:
        yield {
            **row, "pair_id": evaluation.pair.pair_id, "match_status": "unavailable",
            **links.get((row["side"], row["mask_index"]), {}),
        }


def _match_rows(evaluation: _Evaluation) -> Iterable[CsvRow]:
    if evaluation.result is None:
        return
    for match in evaluation.result["matches"]:
        error = match["position_error"]
        row: CsvRow = {
            "pair_id": evaluation.pair.pair_id,
            "shared_id": match["shared_id"],
            "position_error_norm": error,
            "position_error_north_px": (
                None if evaluation.north_width is None
                else error * (evaluation.north_width - 1)
            ),
        }
        for side in _SIDES:
            instance = match[side]
            center = instance["center_xy"]
            row[f"{side}_instance_id"] = instance["instance_id"]
            row[f"{side}_mask_index"] = instance["mask_index"]
            center_x, center_y = center or (None, None)
            row[f"{side}_center_x_px"] = center_x
            row[f"{side}_center_y_px"] = center_y
        yield row


def _diagnostic_rows(evaluation: _Evaluation) -> Iterable[CsvRow]:
    summary = evaluation.summary
    keys = {"pair_id": summary.pair_id, "status": summary.diagnostic_status}
    if summary.diagnostic_status == "error":
        yield {**keys, "metric": "error_type", "value": summary.error_type}
        yield {**keys, "metric": "error_message", "value": summary.error_message}
    elif evaluation.diagnostics:
        for metric, value in evaluation.diagnostics.items():
            yield {**keys, "metric": metric, "value": value}
    else:
        yield keys


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[CsvRow]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _mslc_labelme_ids(
    evaluation: _Evaluation, side: Side, expected_count: int
) -> tuple[LabelMeInstanceRef, ...]:
    """使用生成本次共享 Mask 的原始索引和图内 group_id。"""
    ids = evaluation.references.get(side, ())
    if len(ids) != expected_count:
        evaluation.diagnostics[f"{side}_labelme_metadata_error"] = (
            "ValueError 原始标注实例数与本次 Mask 数不一致"
        )
        return ()
    return ids


def _write_mslc_tables(
    output_dir: Path, evaluations: Sequence[_Evaluation]
) -> list[CsvRow]:
    """归档实际参与 MSLC 匹配的逐实例、逐截面和逐候选证据。"""
    columns_by_table = rail_matching_mslc.MSLC_TABLE_COLUMNS
    tables: dict[str, list[CsvRow]] = {name: [] for name in columns_by_table}
    for evaluation in evaluations:
        if evaluation.result is None:
            continue
        result = cast(rail_matching_mslc.MSLCMatchingResult, evaluation.result)
        diagnostics = result["diagnostics"]
        feature_rows = diagnostics["rail_features"]
        metadata = {
            side: _mslc_labelme_ids(
                evaluation, side, sum(row["side"] == side for row in feature_rows)
            )
            for side in _SIDES
        }
        for name in columns_by_table:
            for record in diagnostics[name]:
                row: CsvRow = {"pair_id": evaluation.pair.pair_id, **record}
                if name == "rail_features":
                    ids = metadata[cast(Side, row["side"])]
                    if ids:
                        shape_index, group_id = ids[cast(int, row["mask_index"])]
                        row["shape_index"], row["group_id"] = shape_index, group_id
                tables[name].append(row)
    for name, columns in columns_by_table.items():
        _write_csv(output_dir / f"{name}.csv", ("pair_id", *columns), tables[name])
    return tables["candidate_pair_support"]


def _mslc_match_rows(
    rows: Iterable[CsvRow], evidence: Sequence[CsvRow]
) -> Iterable[CsvRow]:
    accepted = {
        (row["pair_id"], row["shared_id"]): row for row in evidence
        if row["decision"] == "accepted"
    }
    for row in rows:
        candidate = accepted[row["pair_id"], row["shared_id"]]
        yield {**row, **{name: candidate[name] for name in _MSLC_MATCH_COLUMNS}}


def _mslc_unmatched_rows(
    rows: Iterable[CsvRow], evidence: Sequence[CsvRow]
) -> Iterable[CsvRow]:
    ranked = sorted(
        evidence,
        key=lambda row: (
            -cast(float, row["combined_support"]), cast(float, row["final_cost"])
        ),
    )
    best: dict[tuple[object, str, object], CsvRow] = {}
    counts: Counter[tuple[object, str, object]] = Counter()
    for candidate in ranked:
        for side in _SIDES:
            key = candidate["pair_id"], side, candidate[f"{side}_mask_index"]
            counts[key] += 1
            best.setdefault(key, candidate)
    for row in rows:
        side = cast(Side, row["side"])
        key = row["pair_id"], side, row["mask_index"]
        candidate = best.get(key, {})
        partner = "north" if side == "south" else "south"
        yield {
            **row, "candidate_count": counts[key],
            "best_candidate_mask_index": candidate.get(f"{partner}_mask_index"),
            "best_candidate_support": candidate.get("combined_support"),
            "best_candidate_decision": candidate.get("decision"),
        }


def _residual_threshold(evaluations: Sequence[_Evaluation]) -> float:
    errors = [
        evaluation.summary.position_error_p95_norm for evaluation in evaluations
        if evaluation.summary.position_error_p95_norm is not None
    ]
    if not errors:
        return _HIGH_RESIDUAL_MIN
    return max(
        _HIGH_RESIDUAL_MIN, float(np.percentile(errors, _HIGH_RESIDUAL_PERCENTILE))
    )


def _review_tags(summary: _PairSummary, residual_threshold: float) -> tuple[str, ...]:
    flags = {
        "input_error": summary.status == "input_error",
        "algorithm_error": summary.status == "algorithm_error",
        "zero_matches": summary.match_count == 0,
        "low_coverage": any(
            value is not None and value < _LOW_COVERAGE
            for value in (summary.south_coverage, summary.north_coverage)
        ),
        "ambiguous_alignment": bool(summary.ambiguous_instance_count),
        "measurement_failure": bool(summary.measurement_failure_count),
        "order_conflict": bool(summary.order_conflict_count),
        "high_residual": (
            summary.position_error_p95_norm is not None
            and summary.position_error_p95_norm >= residual_threshold
        ),
        "diagnostic_error": summary.diagnostic_status == "error",
    }
    return tuple(tag for tag in _REVIEW_TAGS if flags[tag])


def _select_reviews(
    evaluations: Sequence[_Evaluation], limit: int, seed: int, residual_threshold: float
) -> list[tuple[_Evaluation, str, tuple[str, ...]]]:
    """先覆盖不同异常类型，再补充异常与固定种子的普通样本。"""
    limit = min(limit, len(evaluations))
    tagged = [
        (evaluation, _review_tags(evaluation.summary, residual_threshold))
        for evaluation in evaluations
    ]
    ordinary = [evaluation for evaluation, tags in tagged if not tags]
    Random(seed).shuffle(ordinary)
    anomalies = [(evaluation, tags) for evaluation, tags in tagged if tags]
    anomalies.sort(key=lambda item: (-len(item[1]), item[0].pair.pair_id))
    reserve = max(1, int(limit * _CONTROL_FRACTION)) if limit >= 2 else 0
    anomaly_limit = limit - min(len(ordinary), reserve)
    selected: list[tuple[_Evaluation, str, tuple[str, ...]]] = []
    seen_pairs: set[str] = set()
    covered_tags: set[str] = set()
    for tag in _REVIEW_TAGS:
        if len(selected) == anomaly_limit:
            break
        if tag in covered_tags:
            continue
        for evaluation, tags in anomalies:
            if tag in tags and evaluation.pair.pair_id not in seen_pairs:
                selected.append((evaluation, tag, tags))
                seen_pairs.add(evaluation.pair.pair_id)
                covered_tags.update(tags)
                break
    for evaluation, tags in anomalies:
        if len(selected) == anomaly_limit:
            break
        if evaluation.pair.pair_id not in seen_pairs:
            selected.append((evaluation, tags[0], tags))
    selected.extend(
        (evaluation, "ordinary_control", ())
        for evaluation in ordinary[:limit - len(selected)]
    )
    return selected


def _write_review_label(image_path: Path, target: Path) -> None:
    """保留原始 LabelMe 标注，用相对路径引用原图。"""
    annotation_path = image_path.with_suffix(".json")
    with annotation_path.open(encoding="utf-8-sig") as stream:
        annotation = json.load(stream)
    if not isinstance(annotation, dict) or not isinstance(
        annotation.get("shapes"), list
    ):
        raise ValueError(f"{annotation_path} 的 shapes 必须为列表")

    annotation["imageData"] = None
    annotation["imagePath"] = Path(relpath(image_path, start=target.parent)).as_posix()
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(annotation, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _reference_render_result(data: _PairData) -> RailMatchingResult:
    """算法失败时只给预览图标参考编号，评测表仍保留异常。"""
    south = rail_matching._measure_instances(data.south_masks, "south")
    north = rail_matching._measure_instances(data.north_masks, "north")
    return {
        "south_image_path": data.south_path.as_posix(),
        "north_image_path": data.north_path.as_posix(),
        "matches": [],
        "unmatched_south": [
            make_unmatched_rail(measurement.instance, "reference_only")
            for measurement in south
        ],
        "unmatched_north": [
            make_unmatched_rail(measurement.instance, "reference_only")
            for measurement in north
        ],
    }


def _save_review_artifacts(
    output_dir: Path, evaluation: _Evaluation, row: CsvRow
) -> None:
    """归档 LabelMe 标注、原图相对路径和与 Demo 一致的对照图。"""
    pair = evaluation.pair
    review_dir = output_dir / "overlays" / pair.pair_id
    review_dir.mkdir()
    image_paths: list[str] = []
    errors: list[str] = []
    for side, raw_path in (
        ("south", pair.south_image_path), ("north", pair.north_image_path),
    ):
        source_path = ""
        try:
            image_path = _project_image_path(raw_path, side)
            source_path = image_path.as_posix()
            target = review_dir / f"{side}_label.json"
            _write_review_label(PROJECT_ROOT / image_path, target)
        except (OSError, ValueError) as error:
            errors.append(f"{side} {type(error).__name__} {error}")
        else:
            row[f"{side}_label"] = target.relative_to(output_dir).as_posix()
        image_paths.append(source_path)

    paths_file = review_dir / "image_paths.txt"
    with paths_file.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(image_paths) + "\n")
    row["image_paths_file"] = paths_file.relative_to(output_dir).as_posix()

    if evaluation.summary.status == "input_error":
        row["overlay_status"] = "unavailable"
        errors.insert(0, evaluation.summary.error_message)
        row["overlay_error"] = " | ".join(errors)
        return
    if errors:
        row["overlay_error"] = " | ".join(errors)

    data = _load_pair(pair)
    result = evaluation.result
    is_reference_only = result is None
    if result is None:
        result = _reference_render_result(data)
    south = render_side(data.south_image, data.south_masks, result, "south")
    north = render_side(data.north_image, data.north_masks, result, "north")
    if is_reference_only:
        for panel in (south, north):
            cv2.putText(
                panel, "Algorithm error / reference IDs", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 120), 2, cv2.LINE_AA,
            )
    comparison = cv2.cvtColor(compose_image_pair(south, north), cv2.COLOR_RGB2BGR)
    is_encoded, encoded = cv2.imencode(".png", comparison)
    if not is_encoded:
        raise OSError("无法编码南北对照图")
    target = review_dir / "match_overlay.png"
    encoded.tofile(target)
    row["comparison_image"] = target.relative_to(output_dir).as_posix()
    row["overlay_status"] = "reference_only" if is_reference_only else "ok"


def _review_rows(
    output_dir: Path,
    reviews: Sequence[tuple[_Evaluation, str, tuple[str, ...]]],
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for rank, (evaluation, category, tags) in enumerate(reviews, start=1):
        summary = evaluation.summary
        row: CsvRow = {
            "rank": rank, "pair_id": summary.pair_id, "category": category,
            "tags": "|".join(tags), "status": summary.status,
            "match_count": summary.match_count,
            "south_coverage": summary.south_coverage,
            "north_coverage": summary.north_coverage,
            "position_error_p95_norm": summary.position_error_p95_norm,
        }
        try:
            _save_review_artifacts(output_dir, evaluation, row)
        except Exception as error:
            row["overlay_status"] = "error"
            row["overlay_error"] = f"{type(error).__name__} {error}"
        rows.append(row)
    return rows


def _percent(value: float | None) -> str:
    return "未定义" if value is None else f"{value:.2%}"


def _match_totals(summaries: Sequence[_PairSummary]) -> tuple[int, int, int]:
    """累计成功图对的配对数、南侧实例数及北侧实例数。"""
    return (
        sum(summary.match_count or 0 for summary in summaries),
        sum(summary.south_instance_count or 0 for summary in summaries),
        sum(summary.north_instance_count or 0 for summary in summaries),
    )


def _write_report(
    path: Path,
    algorithm: str,
    timestamp: str,
    evaluations: Sequence[_Evaluation],
    reviews: Sequence[CsvRow],
    residual_threshold: float,
) -> None:
    """汇总匹配输出、覆盖率和复核队列。"""
    summaries = [evaluation.summary for evaluation in evaluations]
    successful = [summary for summary in summaries if summary.status == "ok"]
    residuals = [
        match["position_error"]
        for evaluation in evaluations if evaluation.result is not None
        for match in evaluation.result["matches"]
    ]
    match_count, south_count, north_count = _match_totals(successful)
    pair_coverage = _coverage(2 * match_count, south_count + north_count)
    tags: Counter[str] = Counter()
    problem_pairs: list[str] = []
    for summary in summaries:
        pair_tags = _review_tags(summary, residual_threshold)
        if pair_tags:
            tags.update(pair_tags)
            problem_pairs.append(summary.pair_id)
    statuses = Counter(summary.status for summary in summaries)
    reasons: Counter[str] = Counter()
    for evaluation in evaluations:
        for row in _instance_rows(evaluation):
            reason = row.get("reason")
            if row["match_status"] == "unmatched" and isinstance(reason, str):
                reasons[reason] += 1
    times = [
        summary.match_time_ms for summary in successful
        if summary.match_time_ms is not None
    ]
    lines = [
        f"# {algorithm} 钢轨关联评测报告", "",
        f"运行时间戳 `{timestamp}`。本次使用原图和 LabelMe 实例 Mask。",
        "项目暂无人工确认的南北实体对应真值，覆盖率描述算法输出的匹配范围。", "",
        "## 运行与匹配输出", "",
        "| 项目 | 数值 |", "| --- | ---: |",
        f"| 评测图对 | {len(evaluations)} |",
        f"| 成功执行 | {statuses['ok']} |",
        f"| 输入异常 | {statuses['input_error']} |",
        f"| 算法或输出异常 | {statuses['algorithm_error']} |",
        f"| 诊断异常 | {tags['diagnostic_error']} |",
        f"| 零匹配图对 | {tags['zero_matches']} |",
        f"| 低覆盖图对 | {tags['low_coverage']} |",
        f"| 南侧输入实例（成功图对） | {south_count} |",
        f"| 北侧输入实例（成功图对） | {north_count} |",
        f"| 输出匹配关系 | {match_count} |",
        f"| 南侧汇总覆盖率 | {_percent(_coverage(match_count, south_count))} |",
        f"| 北侧汇总覆盖率 | {_percent(_coverage(match_count, north_count))} |",
        f"| 双侧汇总覆盖率 | {_percent(pair_coverage)} |",
    ]
    if residuals:
        lines.extend([
            f"| 匹配位置残差均值 | {np.mean(residuals):.6f} |",
            f"| 匹配位置残差 P95 | {np.percentile(residuals, 95):.6f} |",
            f"| 匹配位置残差最大值 | {max(residuals):.6f} |",
        ])
    if times:
        lines.append(f"| 匹配函数耗时中位数 | {np.median(times):.2f} ms |")
    lines.extend([
        "", "汇总覆盖率按成功图对的实例数加权。位置残差只统计算法输出的匹配。",
        "原算法调用与候选重算分别计时，详见 `pair_summary.csv`。",
        "", "## 未匹配原因", "",
        "| 原因 | 实例数（南北合计） |", "| --- | ---: |",
    ])
    if reasons:
        lines.extend(
            f"| `{reason}` | {count} |" for reason, count in reasons.most_common()
        )
    else:
        lines.append("| 本次无未匹配实例记录 | 0 |")
    lines.extend(["", "## 问题匹配", "", *problem_pairs])
    lines.extend([
        "", "## 复核", "",
        f"选出 {len(reviews)} 对，其中 "
        f"{sum(row['category'] == 'ordinary_control' for row in reviews)} 对普通对照。",
        f"低覆盖门限为任一有实例侧低于 {_LOW_COVERAGE:.0%}。"
        f"本次高残差门限为图对残差 P95 ≥ {residual_threshold:.6f}。",
        "用 `review_queue.csv` 定位对照图、标注和原图路径清单。", "",
    ])
    if algorithm == _MSLC_ALGORITHM:
        modes = Counter(
            evaluation.diagnostics.get("mode") for evaluation in evaluations
        )
        lines.extend([
            "## MSLC 截面证据", "",
            f"多截面 {modes['multi_section']} 对，"
            f"单截面回退 {modes['single_section_fallback']} 对，"
            f"无可用路径 {modes['no_alignment']} 对。",
            "逐配对支持率反映候选路径的一致程度。", "",
        ])
    if algorithm == _GMSA_ALGORITHM:
        lines.extend(_gmsa_report_lines(evaluations))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def _export_algorithm(
    output_dir: Path,
    algorithm: str,
    timestamp: str,
    evaluations: Sequence[_Evaluation],
    review_limit: int,
    seed: int,
    gmsa_config: rail_matching_gmsa.GMSARailConfig | None = None,
) -> tuple[Path, Path, float]:
    """写入独立算法目录并打包该算法的报告与复核图片。"""
    (output_dir / "overlays").mkdir(parents=True)
    _write_csv(
        output_dir / "pair_manifest.csv", _MANIFEST_COLUMNS,
        (_manifest_row(evaluation.pair) for evaluation in evaluations),
    )
    _write_csv(
        output_dir / "pair_summary.csv",
        tuple(column.name for column in fields(_PairSummary)),
        (asdict(evaluation.summary) for evaluation in evaluations),
    )
    instance_rows = [
        row for evaluation in evaluations for row in _instance_rows(evaluation)
    ]
    _write_csv(output_dir / "instance_details.csv", _INSTANCE_COLUMNS, instance_rows)
    match_rows: Iterable[CsvRow] = (
        row for evaluation in evaluations for row in _match_rows(evaluation)
    )
    unmatched_rows: Iterable[CsvRow] = (
        row for row in instance_rows if row["match_status"] == "unmatched"
    )
    match_columns, unmatched_columns = _MATCH_COLUMNS, _UNMATCHED_COLUMNS
    if algorithm == _MSLC_ALGORITHM:
        evidence = _write_mslc_tables(output_dir, evaluations)
        match_rows = _mslc_match_rows(match_rows, evidence)
        unmatched_rows = _mslc_unmatched_rows(unmatched_rows, evidence)
        match_columns += _MSLC_MATCH_COLUMNS
        unmatched_columns += _MSLC_UNMATCHED_COLUMNS
        if _MSLC_README.is_file():
            (output_dir / _MSLC_README.name).write_bytes(_MSLC_README.read_bytes())
    if algorithm == _GMSA_ALGORITHM and gmsa_config is not None:
        _write_gmsa_tables(output_dir, evaluations, gmsa_config)
        if _GMSA_README.is_file():
            (output_dir / _GMSA_README.name).write_bytes(_GMSA_README.read_bytes())
    if algorithm == _ISA_ALGORITHM:
        _write_isa_tables(output_dir, evaluations)
        if _ISA_README.is_file():
            (output_dir / _ISA_README.name).write_bytes(_ISA_README.read_bytes())
    _write_csv(output_dir / "match_pairs.csv", match_columns, match_rows)
    _write_csv(
        output_dir / "unmatched_instances.csv", unmatched_columns, unmatched_rows
    )
    _write_csv(
        output_dir / "algorithm_diagnostics.csv", _DIAGNOSTIC_COLUMNS,
        (row for evaluation in evaluations for row in _diagnostic_rows(evaluation)),
    )
    residual_threshold = _residual_threshold(evaluations)
    reviews = _select_reviews(evaluations, review_limit, seed, residual_threshold)
    review_rows = _review_rows(output_dir, reviews)
    _write_csv(output_dir / "review_queue.csv", _REVIEW_COLUMNS, review_rows)
    report = output_dir / "report.md"
    _write_report(
        report, algorithm, timestamp, evaluations, review_rows, residual_threshold
    )
    archive = zip_analysis_reports(
        output_dir, f"{algorithm}_results.zip", _README,
        include_all_files=True,
        exclude_patterns=("overlays/*/match_overlay.png",),
    )
    return report, archive, residual_threshold


def _gmsa_result(
    evaluation: _Evaluation,
) -> rail_matching_gmsa.GMSAMatchingResult | None:
    """读取由 GMSA 适配器产生的结构化结果。"""
    return cast(rail_matching_gmsa.GMSAMatchingResult | None, evaluation.result)


def _write_gmsa_tables(
    output_dir: Path,
    evaluations: Sequence[_Evaluation],
    config: rail_matching_gmsa.GMSARailConfig,
) -> None:
    """导出三种关系状态、真实路径证据和完整配置。"""
    tables: dict[str, list[CsvRow]] = {name: [] for name in _GMSA_TABLE_COLUMNS}
    summaries: list[CsvRow] = []
    for evaluation in evaluations:
        summary = evaluation.summary
        pair_id = summary.pair_id
        result = _gmsa_result(evaluation)
        diagnostics = result["diagnostics"] if result is not None else None
        row: CsvRow = {
            "pair_id": pair_id, "status": summary.status, "error_stage": summary.error_stage,
            "error_type": summary.error_type, "error_message": summary.error_message,
        }
        if result is not None and diagnostics is not None:
            row.update(diagnostics["summary"])
            row.update(diagnostics["timings_ms"])
            for name, records in (
                ("matches", diagnostics["accepted"]),
                ("tentative_matches", result["tentative_matches"]),
                ("ambiguous_candidates", result["ambiguous_candidates"]),
                ("unmatched_instances", diagnostics["unmatched"]),
                ("transition_diagnostics", diagnostics["transitions"]),
                ("dtw_alignment", diagnostics["dtw_alignment"]),
                ("instance_geometry", diagnostics["instances"]),
                ("section_geometry", diagnostics["sections"]),
                ("gap_fingerprints", diagnostics["gap_fingerprints"]),
                ("paths", diagnostics["paths"]),
            ):
                tables[name].extend({"pair_id": pair_id, **record} for record in records)
        summaries.append(row)
    _write_csv(output_dir / "gmsa_pair_diagnostics.csv", _GMSA_PAIR_COLUMNS, summaries)
    for name, columns in _GMSA_TABLE_COLUMNS.items():
        _write_csv(output_dir / f"gmsa_{name}.csv", columns, tables[name])
    (output_dir / "gmsa_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def _write_isa_tables(output_dir: Path, evaluations: Sequence[_Evaluation]) -> None:
    """导出 ISA-Rail 的候选假设、逐对代价、未匹配状态与实例几何。"""
    tables: dict[str, list[CsvRow]] = {name: [] for name in _ISA_TABLE_COLUMNS}
    for evaluation in evaluations:
        result = evaluation.result
        if result is None:
            continue
        diagnostics = cast(rail_matching_isa.ISAMatchingResult, result)["diagnostics"]
        if not diagnostics:
            continue
        pair_id = evaluation.summary.pair_id
        for name in _ISA_TABLE_COLUMNS:
            records = diagnostics.get(name, [])
            tables[name].extend({"pair_id": pair_id, **record} for record in records)
    for name, columns in _ISA_TABLE_COLUMNS.items():
        _write_csv(output_dir / f"isa_{name}.csv", columns, tables[name])


def _edge_set(result: RailMatchingResult) -> set[tuple[int, int]]:
    return {
        (match["south"]["mask_index"], match["north"]["mask_index"])
        for match in result["matches"]
    }


def _format_edges(edges: set[tuple[int, int]]) -> str:
    return "|".join(f"{south}:{north}" for south, north in sorted(edges))


def _gmsa_diagnostics(evaluation: _Evaluation) -> rail_matching_gmsa.GMSADiagnostics | None:
    result = _gmsa_result(evaluation)
    return result["diagnostics"] if result is not None else None


def _gmsa_report_lines(evaluations: Sequence[_Evaluation]) -> list[str]:
    """汇总仅由匹配结构与路径产生的可观测证据。"""
    diagnostics = [
        details for item in evaluations if (details := _gmsa_diagnostics(item)) is not None
    ]
    evidence = [detail["summary"] for detail in diagnostics]
    transitions = [
        row for detail in diagnostics for row in detail["transitions"] if row["is_best_path"]
    ]
    states = Counter(row["support_status"] for row in evidence)
    span_count = sum(row["dtw_coverage"] is not None for row in transitions)
    violations = 0
    for detail in diagnostics:
        ranks = [
            (row["south_order_index"], row["north_order_index"]) for row in detail["accepted"]
        ]
        violations += any(
            not all(isinstance(value, int) for value in (p, q, i, j))
            or not (p < i and q < j)
            for (p, q), (i, j) in zip(ranks, ranks[1:])
        )
    lines = [
        "## GMSA 结构与歧义", "",
        f"- 单路径无竞争 {states['single_path_no_competition']} 对。"
        f"无可行非空路径 {states['no_feasible_path']} 对。"
        f"触及资源上限 {states['resource_limit']} 对。",
        f"- 方向歧义 {sum(row['direction_ambiguous'] is True for row in evidence)} 对。"
        f"已接受关系的非单调图对 {violations} 对。",
        f"- 最优路径使用 {len(transitions)} 次锚点转移，其中 {span_count} 次有有效跨度。",
        "", "| 证据字段 | 计数 | 中位数 | P95 |", "| --- | ---: | ---: | ---: |",
    ]
    for field_name, rows in (
        ("global_path_margin", evidence), ("dtw_coverage", transitions),
        ("dtw_warp_fraction", transitions), ("span_cost", transitions),
        ("gap_cost", transitions),
    ):
        values = [
            value for row in rows if isinstance(value := row.get(field_name), (int, float))
        ]
        if values:
            lines.append(
                f"| {field_name} | {len(values)} | {np.median(values):.4f} | "
                f"{np.percentile(values, 95):.4f} |"
            )
        else:
            lines.append(f"| {field_name} | 0 | 未定义 | 未定义 |")
    lines.extend(["", "逐项解释见 [GMSA 字段说明](rail_matching_gmsa_readme.md)。", ""])
    return lines


def _summary_row(
    name: str, evaluations: Sequence[_Evaluation], inputs: Mapping[str, _PairInput]
) -> CsvRow:
    """统计成功图对上的覆盖率及 V3 证据状态。"""
    statuses = Counter(evaluation.summary.status for evaluation in evaluations)
    successful = [item.summary for item in evaluations if item.summary.status == "ok"]
    match_count, south_count, north_count = _match_totals(successful)
    times = [item.match_time_ms for item in successful if item.match_time_ms is not None]
    rows = [
        details["summary"] for evaluation in evaluations
        if name == _GMSA_ALGORITHM and (details := _gmsa_diagnostics(evaluation)) is not None
    ]

    def count_status(field: str) -> int | None:
        if name != _GMSA_ALGORITHM:
            return None
        return sum(value for row in rows if isinstance(value := row.get(field), int))

    return {
        "algorithm": name, "pair_count": len(evaluations),
        "successful_pair_count": len(successful), "input_error_count": statuses["input_error"],
        "algorithm_error_count": statuses["algorithm_error"],
        "empty_label_pair_count": sum(source.row.get("any_empty_label") is True
                                      for source in inputs.values()),
        "both_empty_pair_count": sum(source.row.get("both_empty_labels") is True
                                     for source in inputs.values()),
        "south_instance_count": south_count, "north_instance_count": north_count,
        "match_count": match_count, "south_coverage": _coverage(match_count, south_count),
        "north_coverage": _coverage(match_count, north_count),
        "pair_coverage": _coverage(2 * match_count, south_count + north_count),
        "south_unmatched_count": south_count - match_count,
        "north_unmatched_count": north_count - match_count,
        "tentative_count": count_status("tentative_count"),
        "ambiguous_candidate_count": count_status("ambiguous_candidate_count"),
        "ambiguous_best_count": count_status("ambiguous_best_count"),
        "invalid_geometry_count": count_status("invalid_geometry_count"),
        "single_path_pair_count": (
            sum(row["support_status"] == "single_path_no_competition" for row in rows)
            if name == _GMSA_ALGORITHM else None
        ),
        "direction_ambiguous_pair_count": count_status("direction_ambiguous"),
        "resource_limited_pair_count": count_status("resource_limited"),
        "zero_match_pair_count": sum(item.match_count == 0 for item in successful),
        "median_match_time_ms": float(np.median(times)) if times else None,
        "p95_match_time_ms": float(np.percentile(times, 95)) if times else None,
    }


_PAIR_LEVEL_FIELDS = (
    "status", "match_count", "south_coverage", "north_coverage", "pair_coverage",
    "south_unmatched_count", "north_unmatched_count", "ambiguous_instance_count",
    "match_time_ms",
)
_PREFIX_BY_ALGORITHM = {
    _BASELINE_ALGORITHM: "v1", _MSLC_ALGORITHM: "v2", _GMSA_ALGORITHM: "gmsa",
    _ISA_ALGORITHM: "isa", _CGM_ALGORITHM: "cgm",
}
_RELATION_DIFF_COLUMNS = (
    "pair_id", "south_mask_index", "north_mask_index", "south_shape_index",
    "north_shape_index", "south_group_id", "north_group_id",
    "v1_output", "v2_output", "gmsa_status", "relation_change_type",
)
_RUNTIME_COLUMNS = (
    "pair_id", "algorithm", "status", "load_time_ms", "measure_time_ms",
    "match_time_ms", *_GMSA_TIMING_COLUMNS, "adapter_diagnostic_time_ms",
)


def _pair_level_row(
    items: Mapping[str, _Evaluation], source: _PairInput,
) -> CsvRow:
    row: CsvRow = {
        "pair_id": source.row["pair_id"], "read_status": source.row.get("read_status"),
        "south_instance_count": source.row.get("south_instance_count"),
        "north_instance_count": source.row.get("north_instance_count"),
        "any_empty_label": source.row.get("any_empty_label"),
    }
    for name, evaluation in items.items():
        prefix = _PREFIX_BY_ALGORITHM.get(name, f"custom_{name}")
        for field_name in _PAIR_LEVEL_FIELDS:
            row[f"{prefix}_{field_name}"] = getattr(evaluation.summary, field_name)
    gmsa = items.get(_GMSA_ALGORITHM)
    details = _gmsa_diagnostics(gmsa) if gmsa is not None else None
    row.update(
        gmsa_tentative_count=details["summary"]["tentative_count"] if details else None,
        gmsa_ambiguous_candidate_count=(
            details["summary"]["ambiguous_candidate_count"] if details else None
        ),
        gmsa_support_status=details["summary"]["support_status"] if details else None,
        gmsa_main_rejection_reason=None,
    )
    if details:
        reasons = Counter(item["reason"] for item in details["unmatched"])
        if reasons:
            row["gmsa_main_rejection_reason"] = reasons.most_common(1)[0][0]
    elif gmsa is not None:
        row["gmsa_main_rejection_reason"] = gmsa.summary.error_stage or None
    return row


def _gmsa_relation_states(result: rail_matching_gmsa.GMSAMatchingResult) -> dict[
    tuple[int, int], str
]:
    states = {edge: "accepted" for edge in _edge_set(result)}
    for rows, state in (
        (result["tentative_matches"], "tentative"),
        (result["ambiguous_candidates"], "ambiguous"),
    ):
        for row in rows:
            south, north = row["south_mask_index"], row["north_mask_index"]
            if not isinstance(south, int) or not isinstance(north, int):
                raise ValueError("GMSA 关系诊断缺少原始 Mask 索引")
            states.setdefault((south, north), state)
    return states


def _relation_differences(
    items: Mapping[str, _Evaluation], source: _PairInput
) -> Iterable[CsvRow]:
    """只在同一个 Pair 内用原始 Mask 索引联查各算法关系。"""
    baseline_edges = []
    for name in (_BASELINE_ALGORITHM, _MSLC_ALGORITHM):
        item = items.get(name)
        baseline_edges.append(
            _edge_set(item.result) if item is not None and item.result is not None else None
        )
    first, second = baseline_edges
    gmsa = items.get(_GMSA_ALGORITHM)
    result = _gmsa_result(gmsa) if gmsa is not None else None
    states = _gmsa_relation_states(result) if result is not None else {}
    edges = set(states).union(first or (), second or ())
    for south, north in sorted(edges):
        edge = south, north
        v1 = None if first is None else edge in first
        v2 = None if second is None else edge in second
        status = "unavailable" if result is None else states.get(edge, "unmatched")
        if status == "unavailable" or (v1 is None and v2 is None):
            change = "comparison_unavailable"
        elif status == "accepted":
            if v1 and v2:
                change = "all_accepted"
            else:
                change = "gmsa_retained" if v1 or v2 else "gmsa_only_accepted"
        elif v1 or v2:
            change = f"baseline_to_{status}"
        else:
            change = f"gmsa_only_{status}"
        row: CsvRow = {
            "pair_id": source.row["pair_id"], "south_mask_index": south,
            "north_mask_index": north, "v1_output": v1, "v2_output": v2,
            "gmsa_status": status, "relation_change_type": change,
        }
        for side, index in (("south", south), ("north", north)):
            refs = source.references.get(side, ())
            if 0 <= index < len(refs):
                row[f"{side}_shape_index"] = refs[index].shape_index
                row[f"{side}_group_id"] = refs[index].group_id
        yield row


def _runtime_rows(evaluations: Mapping[str, Sequence[_Evaluation]]) -> Iterable[CsvRow]:
    for name, items in evaluations.items():
        for item in items:
            summary = item.summary
            details = _gmsa_diagnostics(item) if name == _GMSA_ALGORITHM else None
            yield {
                "pair_id": summary.pair_id, "algorithm": name, "status": summary.status,
                "load_time_ms": summary.load_time_ms, "measure_time_ms": summary.measure_time_ms,
                "match_time_ms": summary.match_time_ms,
                "adapter_diagnostic_time_ms": summary.diagnostic_time_ms,
                **(details["timings_ms"] if details is not None else {}),
            }


def _write_v3_comparison_tables(
    directory: Path, evaluations: dict[str, list[_Evaluation]],
    inputs: Mapping[str, _PairInput],
) -> None:
    """写入统一运行层汇总、逐 Pair、逐关系和分阶段耗时。"""
    directory.mkdir()
    summaries = [_summary_row(name, items, inputs) for name, items in evaluations.items()]
    _write_csv(directory / "algorithm_summary.csv", tuple(summaries[0]), summaries)
    names = tuple(evaluations)
    aligned = [
        dict(zip(names, items, strict=True))
        for items in zip(*evaluations.values(), strict=True)
    ]
    sources = tuple(inputs.values())
    columns = (
        "pair_id", "read_status", "south_instance_count", "north_instance_count",
        "any_empty_label",
        *(f"{_PREFIX_BY_ALGORITHM.get(name, f'custom_{name}')}_{field_name}"
          for name in names for field_name in _PAIR_LEVEL_FIELDS),
        "gmsa_tentative_count", "gmsa_ambiguous_candidate_count",
        "gmsa_support_status", "gmsa_main_rejection_reason",
    )
    _write_csv(
        directory / "pair_level_comparison.csv", columns,
        (_pair_level_row(items, source) for items, source in zip(aligned, sources, strict=True)),
    )
    _write_csv(
        directory / "match_relation_diff.csv", _RELATION_DIFF_COLUMNS,
        (row for items, source in zip(aligned, sources, strict=True)
         for row in _relation_differences(items, source)),
    )
    _write_csv(directory / "runtime_comparison.csv", _RUNTIME_COLUMNS, _runtime_rows(evaluations))


def _difference_rows(
    evaluations: dict[str, list[_Evaluation]],
) -> Iterable[CsvRow]:
    """按输入 Mask 索引比较各算法输出的南北关联边。"""
    for name_a, name_b in combinations(evaluations, 2):
        for first, second in zip(evaluations[name_a], evaluations[name_b], strict=True):
            row: CsvRow = {
                "pair_id": first.pair.pair_id,
                "algorithm_a": name_a, "algorithm_b": name_b,
                "status_a": first.summary.status, "status_b": second.summary.status,
                "match_count_a": first.summary.match_count,
                "match_count_b": second.summary.match_count,
                "comparison_status": "unavailable",
            }
            if first.result is not None and second.result is not None:
                edges_a, edges_b = _edge_set(first.result), _edge_set(second.result)
                row["comparison_status"] = "comparable"
                for label, edges in (
                    ("shared", edges_a & edges_b),
                    ("only_a", edges_a - edges_b),
                    ("only_b", edges_b - edges_a),
                ):
                    row[f"{label}_count"] = len(edges)
                    row[f"{label}_pairs"] = _format_edges(edges)
                for side in _SIDES:
                    a = getattr(first.summary, f"{side}_coverage")
                    b = getattr(second.summary, f"{side}_coverage")
                    row[f"{side}_coverage_delta_b_minus_a"] = (
                        None if a is None or b is None else b - a
                    )
            yield row


def _comparison_metrics(evaluations: Sequence[_Evaluation]) -> dict[str, str]:
    """用各算法自己的成功图对汇总覆盖率、配对数和耗时。"""
    statuses = Counter(evaluation.summary.status for evaluation in evaluations)
    successful = [
        evaluation.summary for evaluation in evaluations
        if evaluation.summary.status == "ok"
    ]
    match_count, south_count, north_count = _match_totals(successful)
    times = [
        summary.match_time_ms for summary in successful
        if summary.match_time_ms is not None
    ]
    return {
        "成功图对": f"{len(successful)}/{len(evaluations)}",
        "输入异常图对": str(statuses["input_error"]),
        "算法或输出异常图对": str(statuses["algorithm_error"]),
        "零匹配图对": str(sum(summary.match_count == 0 for summary in successful)),
        "输出匹配关系": str(match_count),
        "南侧输入实例": str(south_count),
        "北侧输入实例": str(north_count),
        "南侧未匹配实例": str(south_count - match_count),
        "北侧未匹配实例": str(north_count - match_count),
        "南侧覆盖率": _percent(_coverage(match_count, south_count)),
        "北侧覆盖率": _percent(_coverage(match_count, north_count)),
        "双侧覆盖率": _percent(_coverage(2 * match_count, south_count + north_count)),
        "匹配函数耗时中位数": f"{np.median(times):.2f} ms" if times else "未定义",
    }


def _pairwise_comparison_lines(
    evaluations: dict[str, list[_Evaluation]], rows: Sequence[CsvRow]
) -> Iterable[str]:
    """按双方均成功的图对统计关联差异与双侧覆盖率。"""
    comparable: dict[tuple[str, str], list[CsvRow]] = defaultdict(list)
    for row in rows:
        if row["comparison_status"] == "comparable":
            key = cast(str, row["algorithm_a"]), cast(str, row["algorithm_b"])
            comparable[key].append(row)

    for name_a, name_b in combinations(evaluations, 2):
        shared_rows = comparable.get((name_a, name_b), [])
        total_count = len(evaluations[name_a])
        prefix = f"| `{name_a}` | `{name_b}` | {len(shared_rows)}/{total_count} |"
        if not shared_rows:
            yield f"{prefix} — | — | — | — | 未定义 | 未定义 |"
            continue
        shared_count = sum(cast(int, row["shared_count"]) for row in shared_rows)
        only_a_count = sum(cast(int, row["only_a_count"]) for row in shared_rows)
        only_b_count = sum(cast(int, row["only_b_count"]) for row in shared_rows)
        different_count = sum(
            row["only_a_count"] != 0 or row["only_b_count"] != 0
            for row in shared_rows
        )
        pair_ids = {row["pair_id"] for row in shared_rows}
        _, south_count, north_count = _match_totals([
            evaluation.summary for evaluation in evaluations[name_a]
            if evaluation.pair.pair_id in pair_ids
        ])
        instance_count = south_count + north_count
        coverage_a = _coverage(2 * (shared_count + only_a_count), instance_count)
        coverage_b = _coverage(2 * (shared_count + only_b_count), instance_count)
        yield (
            f"{prefix} {different_count} | {shared_count} | {only_a_count} | "
            f"{only_b_count} | {_percent(coverage_a)} | {_percent(coverage_b)} |"
        )


def _write_comparison_report(
    path: Path,
    timestamp: str,
    evaluations: dict[str, list[_Evaluation]],
    comparison_rows: Sequence[CsvRow],
) -> None:
    """生成时间戳目录中的算法汇总和可比图对差异。"""
    metrics = {name: _comparison_metrics(rows) for name, rows in evaluations.items()}
    headings = " | ".join(f"[{name}]({name}/report.md)" for name in evaluations)
    alignment = " | ".join(["---:"] * len(evaluations))
    pair_count = len(next(iter(evaluations.values())))
    lines = [
        "# 钢轨匹配算法对比", "",
        f"运行时间戳 `{timestamp}`，输入 {pair_count} 对原图及 LabelMe Mask。",
        "覆盖率衡量算法输出的匹配范围，实际对应的正确性需要人工配对真值。",
        "", "## 各算法汇总", "",
        f"| 指标 | {headings} |", f"| --- | {alignment} |",
    ]
    for metric_name in next(iter(metrics.values())):
        values = " | ".join(
            algorithm_metrics[metric_name] for algorithm_metrics in metrics.values()
        )
        lines.append(f"| {metric_name} | {values} |")
    lines.extend([
        "", "上表的实例与覆盖率按各算法成功图对累计，耗时取匹配调用的中位数。", "",
    ])

    if len(evaluations) > 1:
        lines.extend([
            "## 同一图对上的输出差异", "",
            "| 算法 A | 算法 B | 可比较图对 | 有差异图对 | 共同关系 | 仅 A | 仅 B | "
            "A 双侧覆盖率 | B 双侧覆盖率 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *_pairwise_comparison_lines(evaluations, comparison_rows),
            "", "本表仅统计两个算法都成功的图对，关联按同一图对内的南北 Mask 索引比较。",
            "有差异的图对至少包含一条仅由其中一个算法输出的关联。",
            "[逐图对差异](algorithm_comparison.csv)", "",
        ])
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def _write_comparison_archive(
    archive_path: Path, algorithm_zips: Iterable[Path]
) -> None:
    """归档运行层文件与各算法已生成的 ZIP，保留相对目录。"""
    output_dir = archive_path.parent
    pending_path = archive_path.with_suffix(".zip.part")
    run_files = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path not in (archive_path, pending_path)
    )
    comparison_files = sorted((output_dir / "comparison").glob("*.csv"))
    stream = pending_path.open("xb")
    try:
        with stream, ZipFile(stream, "w", ZIP_DEFLATED) as archive:
            for path in run_files:
                archive.write(path, path.name)
            for path in comparison_files:
                archive.write(path, path.relative_to(output_dir).as_posix())
            for path in algorithm_zips:
                relative_name = path.relative_to(output_dir).as_posix()
                archive.write(path, relative_name, compress_type=ZIP_STORED)
        pending_path.rename(archive_path)
    finally:
        pending_path.unlink(missing_ok=True)


def _write_run_readme(
    path: Path, timestamp: str, inputs: Mapping[str, _PairInput], names: Sequence[str]
) -> None:
    """在字段手册前加上本次输入口径、空标注和异常信息。"""
    empty = [name for name, item in inputs.items() if item.row.get("both_empty_labels")]
    one_empty = [
        name for name, item in inputs.items()
        if item.row.get("any_empty_label") and not item.row.get("both_empty_labels")
    ]
    errors = [name for name, item in inputs.items() if item.row.get("read_status") != "ok"]
    header = (
        f"# 本次统一评测 {timestamp}\n\n"
        f"执行算法 {', '.join(names)}。共享清单包含 {len(inputs)} 对，"
        f"双侧空标注 {len(empty)} 对，单侧空标注 {len(one_empty)} 对，"
        f"读取异常 {len(errors)} 对。\n\n"
        f"双侧空标注 Pair 为 {', '.join(empty) or '无'}。\n\n"
        f"单侧空标注 Pair 为 {', '.join(one_empty) or '无'}。\n\n"
        f"读取异常 Pair 为 {', '.join(errors) or '无'}。"
        "各侧错误和 JSON 哈希见 pair_manifest.csv。\n\n"
    )
    manuals = (
        manual.read_text(encoding="utf-8")
        for manual in (_README, _GMSA_README) if manual.is_file()
    )
    path.write_text("\n\n---\n\n".join((header, *manuals)), encoding="utf-8", newline="\n")


def evaluate_rail_match(
    algorithms: Sequence[str] = _DEFAULT_ALGORITHMS,
    pair_manifest: str | Path | None = None,
    max_pairs: int | None = None,
    review_limit: int = 30,
    seed: int = 42,
    *,
    gmsa_config: rail_matching_gmsa.GMSARailConfig | Mapping[str, object] | None = None,
    include_empty_pairs: bool = False,
) -> dict[str, object]:
    """评测同一批原图和 LabelMe 掩码上的南北关联输出。

    Args:
        algorithms: 已注册的算法名称，按传入顺序执行，默认评测 V1、MSLC 和 GMSA。
        pair_manifest: CSV 配对清单，可传项目相对路径或绝对路径。包含
            south_image_path 和 north_image_path。省略时按项目逻辑配对。
        max_pairs: 抽样图对数量上限，None 表示使用整份清单。
        review_limit: 每个算法保留的待复核图对数量上限。
        seed: 选取评测子集与普通对照样本时使用的随机种子。
        gmsa_config: GMSA 的配置对象或字段映射。省略时使用工程默认值。
        include_empty_pairs: 默认清单是否保留两侧都没有 rail 标注的图对。
            默认 False，只评测有标注的图对。

    Returns:
        运行时间戳、图对数量和清单、算法目录、报告及 ZIP 的绝对路径。
        包含运行层对比报告与总 ZIP 路径，多算法运行时另含差异 CSV 路径。

    Raises:
        ValueError: 算法名称、数量限制或输入清单表头无效。
        OSError: 无法获取输入清单或写入本次评测结果。
    """
    requested_names = (algorithms,) if isinstance(algorithms, str) else algorithms
    names = tuple(_ALGORITHM_ALIASES.get(name, name) for name in requested_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("至少指定一个算法，且算法名称不可重复")
    unknown = set(names) - _ALGORITHMS.keys()
    if unknown:
        raise ValueError(f"未注册的算法 {', '.join(sorted(unknown))}")
    if (max_pairs is not None and max_pairs < 0) or review_limit < 0:
        raise ValueError("max_pairs 和 review_limit 必须为非负整数")

    if pair_manifest is None:
        dataset = load_dataset_relative_path()
        source = {"kind": "project_pairing", "dataset_path": dataset.as_posix()}
        pairs, excluded_empty = _default_pairs(
            dataset, include_empty=include_empty_pairs
        )
    else:
        manifest = Path(PROJECT_ROOT, pair_manifest).resolve()
        source = {"kind": "csv", "manifest_path": str(manifest)}
        pairs = _read_pairs(manifest)
        excluded_empty = []
    available_count = len(pairs)
    if max_pairs is not None and max_pairs < available_count:
        indices = sorted(Random(seed).sample(range(available_count), max_pairs))
        pairs = [pairs[index] for index in indices]

    settings = rail_matching_gmsa._options(gmsa_config) if _GMSA_ALGORITHM in names else None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = _OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    adapters = {name: _ALGORITHMS[name] for name in names}
    if settings is not None:
        adapters[_GMSA_ALGORITHM] = replace(
            adapters[_GMSA_ALGORITHM], match_pair=partial(_match_gmsa, config=settings)
        )
    evaluations, inputs = _evaluate_pairs(pairs, adapters)
    pair_manifest_path = output_dir / "pair_manifest.csv"
    _write_csv(
        pair_manifest_path, _INPUT_MANIFEST_COLUMNS,
        (source.row for source in inputs.values()),
    )
    algorithm_dirs: dict[str, str] = {}
    report_paths: dict[str, str] = {}
    zip_paths: dict[str, str] = {}
    thresholds: dict[str, float] = {}
    for name, algorithm_evaluations in evaluations.items():
        algorithm_dir = output_dir / name
        report, archive, threshold = _export_algorithm(
            algorithm_dir, name, timestamp, algorithm_evaluations, review_limit, seed,
            settings,
        )
        algorithm_dirs[name] = str(algorithm_dir)
        report_paths[name] = str(report)
        zip_paths[name] = str(archive)
        thresholds[name] = threshold

    manifest_path = output_dir / "run_manifest.json"
    comparison_report = output_dir / "compare_match.md"
    archive_name = (
        "evaluate_rail_match_results.zip" if settings is not None
        else "evaluate_rail_match_compare.zip"
    )
    comparison_archive = output_dir / archive_name
    result: dict[str, object] = {
        "timestamp": timestamp,
        "output_dir": str(output_dir),
        "run_manifest_path": str(manifest_path),
        "pair_manifest_path": str(pair_manifest_path),
        "pair_count": len(pairs),
        "algorithm_dirs": algorithm_dirs,
        "report_paths": report_paths,
        "zip_paths": zip_paths,
        "comparison_report_path": str(comparison_report),
        "comparison_zip_path": str(comparison_archive),
    }
    comparison_rows = list(_difference_rows(evaluations))
    if len(names) > 1:
        comparison_path = output_dir / "algorithm_comparison.csv"
        _write_csv(comparison_path, _COMPARISON_COLUMNS, comparison_rows)
        result["comparison_path"] = str(comparison_path)
    _write_comparison_report(comparison_report, timestamp, evaluations, comparison_rows)
    if settings is not None:
        comparison_dir = output_dir / "comparison"
        _write_v3_comparison_tables(comparison_dir, evaluations, inputs)
        result["comparison_dir"] = str(comparison_dir)
        readme_path = output_dir / _README.name
        _write_run_readme(readme_path, timestamp, inputs, names)
        result["readme_path"] = str(readme_path)

    metadata = {
        **result,
        "algorithms": list(names),
        "pair_source": source,
        "available_pair_count": available_count,
        "include_empty_pairs": include_empty_pairs,
        "excluded_empty_pair_ids": excluded_empty,
        "selected_pair_ids": [pair.pair_id for pair in pairs],
        "max_pairs": max_pairs,
        "review_limit": review_limit,
        "seed": seed,
        "mask_source": "labelme_rail_polygons",
        "pair_manifest_sha256": sha256(pair_manifest_path.read_bytes()).hexdigest(),
        "reference_section_fractions": [percent / 100 for percent in SECTION_PERCENTS],
        "low_coverage_threshold": _LOW_COVERAGE,
        "high_residual_thresholds": thresholds,
    }
    if settings is not None:
        metadata["gmsa_config"] = asdict(settings)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    _write_comparison_archive(
        comparison_archive, (Path(path) for path in zip_paths.values())
    )
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms", nargs="+",
        choices=sorted(_ALGORITHMS.keys() | _ALGORITHM_ALIASES.keys()),
        default=_DEFAULT_ALGORITHMS,
        help="要评测的已注册算法",
    )
    parser.add_argument("--pair-manifest", type=Path, help="可选南北配对 CSV")
    parser.add_argument("--max-pairs", type=int, help="抽样图对上限")
    parser.add_argument("--review-limit", type=int, default=30, help="每算法复核图对上限")
    parser.add_argument("--seed", type=int, default=42, help="固定抽样种子")
    parser.add_argument("--gmsa-config", type=Path, help="可选 GMSA YAML 或 JSON 配置")
    parser.add_argument(
        "--include-empty-pairs", action="store_true",
        help="默认清单保留两侧都没有 rail 标注的图对（复现旧口径）",
    )
    args = parser.parse_args()
    if (args.max_pairs is not None and args.max_pairs < 0) or args.review_limit < 0:
        parser.error("--max-pairs 和 --review-limit 必须为非负整数")
    result = evaluate_rail_match(
        algorithms=args.algorithms, pair_manifest=args.pair_manifest,
        max_pairs=args.max_pairs, review_limit=args.review_limit, seed=args.seed,
        include_empty_pairs=args.include_empty_pairs,
        gmsa_config=(
            load_yaml_mapping(Path(PROJECT_ROOT, args.gmsa_config))
            if args.gmsa_config is not None else None
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
