"""比较同图的 YOLO GT 与预测 Mask，并检查展开前后的几何差异。"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from colorsys import hsv_to_rgb
import csv
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
import json
import os
from pathlib import Path
from shutil import copy2
import sys
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preprocess.common.config import PROJECT_ROOT, load_yaml_mapping
from data_preprocess.common.image import show_rgb_image
from experiments.yolo_seg.valid_rail_seg import latest_experiment
from utils.instance_masks import as_mask_stack, load_yolo_polygons
from utils.rail_perspective import (
    PolarGrid,
    VanishingPoint,
    build_polar_grid,
    estimate_vanishing_point,
)
from utils.report_archive import zip_analysis_reports
from utils.segmentation_predict import (
    PREDICT_CONFIG,
    iter_segmentation_predictions,
    load_predict_options,
    load_split_image_dir,
)

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.engine.results import Masks

_MIN_INTERSECTION_PX = 8
_MIN_GT_COVERAGE = 0.05
_MIN_PRED_COVERAGE = 0.15
_STRONG_GT_COVERAGE = 0.30
_MERGE_GT_COVERAGE = 0.22
_MERGE_PRED_COVERAGE = 0.08
_SECTION_STEP_PX = 4
_MIN_SECTION_COVERAGE = 0.25
_MIN_GAP_SECTIONS = 3
_CENTER_SHIFT_FRACTION = 0.25
_WIDTH_CHANGE_FRACTION = 0.30
_PANEL_WIDTH = 900
_HEADER_HEIGHT = 50
_LABEL_HEIGHT_FRACTIONS = (0.58, 0.69, 0.80, 0.90)
_PREVIEW_JPEG_QUALITY = 92
_BEST_WEIGHT_NAME = "rail_yolo26m-seg.pt"

_INSTANCE_COLUMNS = (
    "space", "gt_id", "pred_ids", "gt_area_px", "gt_continuous", "gt_coverage",
    "iou", "pred_support_components", "section_coverage", "internal_gap_count",
    "largest_gap_fraction",
    "p95_center_error_over_gt_width", "p95_width_error_over_gt_width",
    "native_flags",
)
_SECTION_COLUMNS = (
    "space", "gt_id", "image_x_px", "image_y_px", "longitudinal_px", "cross_unit",
    "gt_center", "pred_center", "gt_width", "pred_width", "center_delta",
    "width_delta", "center_delta_over_gt_width", "width_delta_over_gt_width",
    "section_coverage", "flags",
)
_PREDICTION_COLUMNS = ("pred_id", "area_px", "gt_ids", "merged_gt_ids", "status")

ByteArray = NDArray[np.uint8]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class MaskComparisonResult:
    """单图对比产物路径及原图空间中筛出的实例编号。"""

    output_dir: Path
    missed_gt_ids: tuple[int, ...]
    broken_gt_ids: tuple[int, ...]
    split_gt_ids: tuple[int, ...]
    merged_pred_ids: tuple[int, ...]
    vp_status: str


@dataclass(frozen=True)
class _Mask:
    """用开区间 xyxy 和局部 0/1 掩码表示独立实例。"""

    box: Box
    pixels: ByteArray
    area: int


@dataclass
class _Section:
    """同一纵向位置的 GT 横截面及最相关的预测横截面。"""

    row: int
    gt_center: float
    gt_width: int
    pred_center: float | None
    pred_width: int | None
    coverage: float
    is_internal_gap: bool = False

    @property
    def center_error(self) -> float | None:
        """返回以 GT 宽度归一化的有符号中心偏差。"""
        if self.pred_center is None:
            return None
        return (self.pred_center - self.gt_center) / self.gt_width

    @property
    def width_error(self) -> float | None:
        """返回以 GT 宽度归一化的有符号宽度偏差。"""
        if self.pred_width is None:
            return None
        return (self.pred_width - self.gt_width) / self.gt_width


@dataclass
class _Measurement:
    """单实例在一个坐标空间的覆盖与几何测量。"""

    sections: list[_Section]
    coverage: float | None
    iou: float | None
    support_components: int
    section_coverage: float | None
    gap_count: int
    largest_gap_fraction: float | None
    center_p95: float | None
    width_p95: float | None


def _crop_mask(image: ByteArray, origin: tuple[int, int] = (0, 0)) -> _Mask:
    """把二值图裁成紧包围区域，并记下画布偏移。"""
    x, y, width, height = cv2.boundingRect(image) if image.size else (0, 0, 0, 0)
    crop = image[y:y + height, x:x + width].copy()
    x, y = x + origin[0], y + origin[1]
    return _Mask((x, y, x + width, y + height), crop, int(np.count_nonzero(crop)))


def _rasterize(polygon: FloatArray, shape: tuple[int, int]) -> _Mask:
    """用和原图一致的像素坐标栅格化单个 Polygon。"""
    height, width = shape
    points = np.rint(polygon).astype(np.int32)
    x0, y0 = map(int, np.clip(points.min(axis=0), (0, 0), (width, height)))
    x1, y1 = map(int, np.clip(points.max(axis=0) + 1, (0, 0), (width, height)))
    mask = np.full((y1 - y0, x1 - x0), 0, dtype=np.uint8)
    if mask.size:
        cv2.fillPoly(mask, [points - np.array((x0, y0), dtype=np.int32)], 1)
    return _crop_mask(mask, (x0, y0))


def _overlap_slices(
    first: _Mask, second: _Mask
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    """返回两份局部画布中对应同一相交区域的切片。"""
    ax, ay, ax1, ay1 = first.box
    bx, by, bx1, by1 = second.box
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return None
    return (
        (slice(y0 - ay, y1 - ay), slice(x0 - ax, x1 - ax)),
        (slice(y0 - by, y1 - by), slice(x0 - bx, x1 - bx)),
    )


def _intersection(first: _Mask, second: _Mask) -> int:
    """计算同一画布下的实例重叠面积。"""
    overlap = _overlap_slices(first, second)
    if overlap is None:
        return 0
    return int(np.count_nonzero(first.pixels[overlap[0]] & second.pixels[overlap[1]]))


def _prediction_support_components(gt: _Mask, prediction: _Mask) -> int:
    """统计与 GT 有显著重叠的预测连通区域，区分断裂和连续偏移。"""
    overlap = _overlap_slices(gt, prediction)
    if overlap is None:
        return 0
    _, labels = cv2.connectedComponents(prediction.pixels, connectivity=8)
    covered = labels[overlap[1]][gt.pixels[overlap[0]] != 0]
    counts = np.bincount(covered)
    threshold = max(_MIN_INTERSECTION_PX, _MIN_GT_COVERAGE * gt.area)
    return int(np.count_nonzero(counts[1:] >= threshold))


def _union(masks: Sequence[_Mask]) -> _Mask:
    """在局部画布上合成关联预测，以便测量断裂和覆盖。"""
    masks = [mask for mask in masks if mask.area]
    if not masks:
        return _Mask((0, 0, 0, 0), np.full((0, 0), 0, dtype=np.uint8), 0)
    if len(masks) == 1:
        return masks[0]
    x0 = min(mask.box[0] for mask in masks)
    y0 = min(mask.box[1] for mask in masks)
    x1 = max(mask.box[2] for mask in masks)
    y1 = max(mask.box[3] for mask in masks)
    pixels = np.full((y1 - y0, x1 - x0), 0, dtype=np.uint8)
    for mask in masks:
        left, top, right, bottom = mask.box
        pixels[top - y0:bottom - y0, left - x0:right - x0] |= mask.pixels
    return _Mask((x0, y0, x1, y1), pixels, int(np.count_nonzero(pixels)))


def _associate(
    gt_masks: list[_Mask], pred_masks: list[_Mask]
) -> tuple[BoolArray, BoolArray]:
    """按双向覆盖率关联，保留一对多碎片和多对一合并。"""
    overlaps = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.int64)
    for gt_index, gt in enumerate(gt_masks):
        for pred_index, prediction in enumerate(pred_masks):
            overlaps[gt_index, pred_index] = _intersection(gt, prediction)
    gt_areas = np.array([mask.area for mask in gt_masks])[:, None]
    pred_areas = np.array([mask.area for mask in pred_masks])[None, :]
    gt_coverage = overlaps / np.maximum(gt_areas, 1)
    pred_coverage = overlaps / np.maximum(pred_areas, 1)
    links = (overlaps >= _MIN_INTERSECTION_PX) & (gt_coverage >= _MIN_GT_COVERAGE)
    links &= (
        (pred_coverage >= _MIN_PRED_COVERAGE) | (gt_coverage >= _STRONG_GT_COVERAGE)
    )
    merges = links & (gt_coverage >= _MERGE_GT_COVERAGE)
    merges &= pred_coverage >= _MERGE_PRED_COVERAGE
    merges &= np.count_nonzero(merges, axis=0) >= 2
    return links, merges


def _runs(row: ByteArray) -> NDArray[np.intp]:
    """返回每个前景交段的 [起点, 终点) 列索引。"""
    edges = np.diff(row, prepend=np.uint8(0), append=np.uint8(0))
    return np.flatnonzero(edges).reshape(-1, 2)


def _measure_sections(gt: _Mask, prediction: _Mask) -> list[_Section]:
    """沿画布逐行采样。预测宽度取与 GT 交集最大的完整前景段。"""
    gt_x, gt_y, _, gt_end = gt.box
    pred_x, pred_y, _, pred_end = prediction.box
    sections: list[_Section] = []
    for row in range(gt_y, gt_end, _SECTION_STEP_PX):
        gt_runs = _runs(gt.pixels[row - gt_y]) + gt_x
        if not len(gt_runs):
            continue
        left, right = gt_runs[np.argmax(gt_runs[:, 1] - gt_runs[:, 0])]
        width = int(right - left)
        section = _Section(row, float(left + (width - 1) / 2), width, None, None, 0.0)
        if pred_y <= row < pred_end:
            pred_runs = _runs(prediction.pixels[row - pred_y]) + pred_x
            overlap_widths = np.minimum(pred_runs[:, 1], right) - np.maximum(
                pred_runs[:, 0], left
            )
            overlap = np.maximum(0, overlap_widths)
            if overlap.any():
                start, stop = pred_runs[int(overlap.argmax())]
                section.pred_center = float((start + stop - 1) / 2)
                section.pred_width = int(stop - start)
                section.coverage = float(overlap.sum() / width)
        sections.append(section)
    return sections


def _mark_internal_gaps(sections: list[_Section]) -> tuple[int, int, int]:
    """标记两端有预测覆盖的内部空段，跳过 GT 自带空段和端部截断。"""
    rows = np.array([section.row for section in sections])
    covered = np.array([
        section.coverage >= _MIN_SECTION_COVERAGE for section in sections
    ], dtype=np.uint8)
    breaks = np.flatnonzero(np.diff(rows) > _SECTION_STEP_PX) + 1
    boundaries = np.r_[0, breaks, len(rows)]
    gap_count, max_gap = 0, 0
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        covered_rows = np.flatnonzero(covered[start:stop])
        if len(covered_rows) < 2:
            continue
        first, last = start + covered_rows[[0, -1]]
        for gap_start, gap_stop in _runs(1 - covered[first:last + 1]):
            count = int(gap_stop - gap_start)
            if count >= _MIN_GAP_SECTIONS:
                gap_count += 1
                max_gap = max(max_gap, count)
                for section in sections[first + gap_start:first + gap_stop]:
                    section.is_internal_gap = True
    return gap_count, max_gap, int(covered.sum())


def _p95_error(values: Iterable[float | None]) -> float | None:
    """计算有可比预测截面的绝对相对误差 95 分位。"""
    errors = [abs(value) for value in values if value is not None]
    return float(np.percentile(errors, 95)) if errors else None


def _measure(gt: _Mask, prediction: _Mask) -> _Measurement:
    """计算像素覆盖和纵向截面统计。"""
    intersection = _intersection(gt, prediction)
    sections = _measure_sections(gt, prediction)
    gaps, max_gap, covered = _mark_internal_gaps(sections)
    union_area = gt.area + prediction.area - intersection
    return _Measurement(
        sections, intersection / gt.area if gt.area else None,
        intersection / union_area if union_area else None,
        _prediction_support_components(gt, prediction) if intersection else 0,
        covered / len(sections) if sections else None, gaps,
        max_gap / len(sections) if sections else None,
        _p95_error(section.center_error for section in sections),
        _p95_error(section.width_error for section in sections),
    )


def _has_unique_pieces(gt: _Mask, predictions: list[_Mask], union: _Mask) -> bool:
    """区分同一 GT 的独立预测碎片和几乎重复的预测 Mask。"""
    if len(predictions) < 2:
        return False
    total = _intersection(gt, union)
    unique_count = 0
    for index in range(len(predictions)):
        others = _union(predictions[:index] + predictions[index + 1:])
        if total - _intersection(gt, others) >= _MIN_GT_COVERAGE * gt.area:
            unique_count += 1
    return unique_count >= 2


def _warp_mask(mask: _Mask, grid: PolarGrid) -> _Mask:
    """重采样单实例 Mask，保留各实例在极坐标画布中的独立性。"""
    if not mask.area:
        return mask
    x0, y0, _, _ = mask.box
    transformed = cv2.remap(
        mask.pixels, grid.map_x - x0, grid.map_y - y0, cv2.INTER_NEAREST
    )
    return _crop_mask(transformed)


def _gt_vanishing_point(
    measurements: list[_Measurement], continuous: list[bool], shape: tuple[int, int]
) -> VanishingPoint:
    """只用 GT 的长中心线确定两组掩码共用的展开原点。"""
    lines: list[tuple[int, FloatArray]] = []
    for gt_id, (measurement, connected) in enumerate(
        zip(measurements, continuous, strict=True), start=1
    ):
        if not connected or len(measurement.sections) < 2:
            continue
        points = np.array([(s.gt_center, s.row) for s in measurement.sections])
        median_width = np.median([s.gt_width for s in measurement.sections])
        if np.linalg.norm(points[-1] - points[0]) >= 5 * median_width:
            lines.append((gt_id, points))
    return estimate_vanishing_point(lines, shape)


def _section_flags(section: _Section) -> list[str]:
    """标记局部漏覆盖、内部空段及偏移和宽度变化。"""
    flags = []
    if section.is_internal_gap:
        flags.append("internal_gap")
    elif section.coverage < _MIN_SECTION_COVERAGE:
        flags.append("low_coverage")
    center_error = section.center_error
    width_error = section.width_error
    if center_error is not None and abs(center_error) >= _CENTER_SHIFT_FRACTION:
        flags.append("center_shift")
    if width_error is not None and abs(width_error) >= _WIDTH_CHANGE_FRACTION:
        flags.append("width_change")
    return flags


def _section_records(
    measurements: list[_Measurement], grid: PolarGrid | None, vp: VanishingPoint
) -> Iterable[dict[str, object]]:
    """输出逐截面原图位置和对应空间的横向量。"""
    scale = grid.theta_step_rad * 1000 if grid is not None else 1.0
    offset = grid.theta_min_rad * 1000 if grid is not None else 0.0
    for gt_id, measurement in enumerate(measurements, start=1):
        for section in measurement.sections:
            longitudinal = section.row
            image_x, image_y = section.gt_center, float(section.row)
            gt_center = section.gt_center * scale + offset
            if grid is not None and vp.xy is not None:
                longitudinal = grid.radius_min_px + section.row * grid.radius_step_px
                angle = gt_center / 1000
                image_x = float(vp.xy[0] + longitudinal * np.cos(angle))
                image_y = float(vp.xy[1] + longitudinal * np.sin(angle))
            gt_width = section.gt_width * scale
            pred_center = (
                None if section.pred_center is None
                else section.pred_center * scale + offset
            )
            pred_width = (
                None if section.pred_width is None else section.pred_width * scale
            )
            center_delta = None if pred_center is None else pred_center - gt_center
            yield {
                "space": "polar" if grid is not None else "image", "gt_id": gt_id,
                "image_x_px": image_x, "image_y_px": image_y,
                "longitudinal_px": longitudinal,
                "cross_unit": "mrad" if grid is not None else "px",
                "gt_center": gt_center, "pred_center": pred_center,
                "gt_width": gt_width, "pred_width": pred_width,
                "center_delta": center_delta,
                "width_delta": None if pred_width is None else pred_width - gt_width,
                "center_delta_over_gt_width": section.center_error,
                "width_delta_over_gt_width": section.width_error,
                "section_coverage": section.coverage,
                "flags": "|".join(_section_flags(section)),
            }


def _instance_records(
    space: str,
    masks: list[_Mask],
    measurements: list[_Measurement],
    links: BoolArray,
    continuous: list[bool],
    flags: list[list[str]],
) -> Iterable[dict[str, object]]:
    """写出实例匹配和坐标空间统计，诊断标签取原图结论。"""
    for index, (mask, measurement) in enumerate(zip(masks, measurements, strict=True)):
        yield {
            "space": space, "gt_id": index + 1,
            "pred_ids": _format_ids(np.flatnonzero(links[index]) + 1),
            "gt_area_px": mask.area, "gt_continuous": continuous[index],
            "gt_coverage": measurement.coverage, "iou": measurement.iou,
            "pred_support_components": measurement.support_components,
            "section_coverage": measurement.section_coverage,
            "internal_gap_count": measurement.gap_count,
            "largest_gap_fraction": measurement.largest_gap_fraction,
            "p95_center_error_over_gt_width": measurement.center_p95,
            "p95_width_error_over_gt_width": measurement.width_p95,
            "native_flags": "|".join(flags[index]),
        }


def _format_ids(ids: Iterable[int]) -> str:
    """将实例编号写成 CSV 单元格。"""
    return "|".join(str(value) for value in ids)


def _mask_color(instance_id: int) -> tuple[int, int, int]:
    """给关联的 GT 和预测分配一致的 BGR 颜色。"""
    red, green, blue = hsv_to_rgb((instance_id * 0.61803398875) % 1, 0.75, 1.0)
    return round(blue * 255), round(green * 255), round(red * 255)


def _draw_mask(
    canvas: ByteArray,
    mask: _Mask,
    color: tuple[int, int, int],
    label: str,
    label_index: int,
) -> None:
    """在原图像素上叠加独立实例颜色和编号。"""
    if not mask.area:
        return
    x0, y0, x1, y1 = mask.box
    patch = canvas[y0:y1, x0:x1]
    foreground = mask.pixels != 0
    patch[foreground] = np.rint(
        0.55 * patch[foreground] + 0.45 * np.asarray(color)
    ).astype(np.uint8)
    # 错开相邻实例的编号高度，在同一截面拥挤时也能追溯实例 ID。
    occupied_rows = np.flatnonzero(foreground.any(axis=1))
    fraction = _LABEL_HEIGHT_FRACTIONS[label_index % len(_LABEL_HEIGHT_FRACTIONS)]
    row = occupied_rows[round(fraction * (len(occupied_rows) - 1))]
    column = int(np.median(np.flatnonzero(foreground[row])))
    origin = (x0 + column, y0 + int(row))
    font_scale = max(0.5, canvas.shape[1] / 1750)
    cv2.putText(
        canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 5
    )
    cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)


def _render_comparison(
    image: ByteArray,
    gt_masks: list[_Mask],
    pred_masks: list[_Mask],
    measurements: list[_Measurement],
    links: BoolArray,
    merges: BoolArray,
    flags: list[list[str]],
    space: str,
) -> ByteArray:
    """左侧标注，右侧预测，红色标出断裂，黄色标出横向偏离。"""
    gt_panel, pred_panel = image.copy(), image.copy()
    red, yellow = (45, 65, 255), (25, 215, 255)
    for index, mask in enumerate(gt_masks):
        notes = flags[index]
        status = [
            short for flag, short in (("missed", "M"), ("broken", "B"), ("split", "S"))
            if flag in notes
        ]
        label = f"G{index + 1}" + (" " + "".join(status) if status else "")
        color = red if status else _mask_color(index + 1)
        _draw_mask(gt_panel, mask, color, label, index)
    for index, mask in enumerate(pred_masks):
        gt_ids = np.flatnonzero(links[:, index]) + 1
        merged = bool(merges[:, index].any())
        color = _mask_color(int(gt_ids[0])) if len(gt_ids) == 1 else (175, 175, 175)
        label = f"P{index + 1}" + (" MERGE" if merged else "")
        _draw_mask(pred_panel, mask, red if merged else color, label, index)

    radius = max(2, image.shape[1] // 550)
    for measurement in measurements:
        for section in measurement.sections[::3]:
            if section.is_internal_gap:
                point = (round(section.gt_center), section.row)
                cv2.circle(pred_panel, point, radius, red, -1, cv2.LINE_AA)
            elif section.pred_center is not None:
                flags_at_row = _section_flags(section)
                if "center_shift" in flags_at_row or "width_change" in flags_at_row:
                    point = (round(section.pred_center), section.row)
                    cv2.circle(pred_panel, point, radius, yellow, -1, cv2.LINE_AA)

    height, width = image.shape[:2]
    panel_width = min(width, _PANEL_WIDTH)
    panel_height = max(1, round(height * panel_width / width))
    size = panel_width, panel_height
    left = cv2.resize(gt_panel, size, interpolation=cv2.INTER_AREA)
    right = cv2.resize(pred_panel, size, interpolation=cv2.INTER_AREA)
    canvas = np.full((panel_height + _HEADER_HEIGHT, panel_width * 2, 3), 24, np.uint8)
    canvas[_HEADER_HEIGHT:, :panel_width] = left
    canvas[_HEADER_HEIGHT:, panel_width:] = right
    titles = (f"GT  {len(gt_masks)} masks  |  {space}",
              f"YOLO  {len(pred_masks)} masks  |  {space}")
    legends = ("M missed   B broken   S split", "Red gap or merge   Yellow drift")
    for index, (title, legend) in enumerate(zip(titles, legends, strict=True)):
        x = 9 + index * panel_width
        cv2.putText(canvas, title, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, legend, (x, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (190, 190, 190), 1, cv2.LINE_AA)
    return canvas


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: Iterable[dict[str, object]]
) -> None:
    """写入带固定列名的 CSV。"""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_preview(path: Path, image: ByteArray) -> None:
    """保存可包含中文路径的对照图。

    对照图只供人工复核，JPG 足够；PNG 仅用于需要无损的场景。
    """
    if path.suffix.lower() in (".jpg", ".jpeg"):
        parameters = [cv2.IMWRITE_JPEG_QUALITY, _PREVIEW_JPEG_QUALITY]
    else:
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    success, encoded = cv2.imencode(path.suffix, image, parameters)
    if not success:
        raise OSError(f"无法保存掩码对照图 {path}")
    encoded.tofile(path)


def _relative_under(path: str | Path, root: Path) -> Path:
    """将用户相对路径定位到指定目录内。"""
    relative = Path(path)
    resolved = (root / relative).resolve()
    if relative.anchor or not resolved.is_relative_to(root):
        raise ValueError(f"请提供 {root} 下的相对路径，收到 {path}")
    return resolved


def compare_mask(
    image_relative_path: str | Path,
    label_relative_path: str | Path,
    predicted_masks: np.ndarray | Tensor | Masks | None,
    output_subdir: str | Path = "mask_compare",
    should_show: bool = False,
    *,
    include_polar: bool = True,
    write_sections: bool = True,
    render_preview: bool = True,
) -> MaskComparisonResult:
    """比较一张验证图的 GT 和预测掩码，并保存可追溯的几何报告。

    Args:
        image_relative_path: 相对于项目根目录的验证图片路径。
        label_relative_path: 相对于项目根目录的 YOLO Polygon TXT 路径。
        predicted_masks: YOLO Results.masks 或原图尺寸的 (N,H,W) 实例掩码。
            单实例也可传 (H,W) 数组。None 表示没有检测结果。
        output_subdir: 相对于 out 的自定义目录。每张图占用一个子目录。
        should_show: 为 True 时用 OpenCV 显示原图对照，关闭窗口后返回。
        include_polar: 为 False 时跳过消失点估计和极坐标对照。
        write_sections: 为 False 时不写逐截面 sections.csv。
        render_preview: 为 False 时不生成 comparison.jpg。

    Returns:
        报告目录及筛出的漏检、断裂、拆分、合并实例编号。

    Raises:
        OSError: 无法读取输入或创建输出目录。
        ValueError: 输入路径、图片、标签或掩码坐标无效。
    """
    image_path = _relative_under(image_relative_path, PROJECT_ROOT)
    label_path = _relative_under(label_relative_path, PROJECT_ROOT)
    output_root = _relative_under(output_subdir, (PROJECT_ROOT / "out").resolve())
    image = cv2.imdecode(
        np.fromfile(image_path, dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None:
        raise ValueError(f"无法读取验证图片 {image_path}")
    height, width = image.shape[:2]
    shape = height, width
    polygons = load_yolo_polygons(label_path, (width, height))
    gt_masks = [_rasterize(points, shape) for points in polygons]
    pred_masks = [
        _crop_mask(np.ascontiguousarray(mask, dtype=np.uint8))
        for mask in as_mask_stack(predicted_masks, shape)
    ]
    links, merges = _associate(gt_masks, pred_masks)
    associations = [np.flatnonzero(row) for row in links]
    continuous = [
        bool(mask.area and cv2.connectedComponents(mask.pixels, connectivity=8)[0] == 2)
        for mask in gt_masks
    ]

    measurements: list[_Measurement] = []
    flags: list[list[str]] = []
    for index, gt in enumerate(gt_masks):
        matched = [pred_masks[pred_index] for pred_index in associations[index]]
        union = _union(matched)
        measurement = _measure(gt, union)
        measurements.append(measurement)
        flags.append(
            _native_flags(
                gt, matched, union, measurement, continuous[index],
                bool(merges[index].any()),
            )
        )

    if should_show and not render_preview:
        raise ValueError("显示预览需要 render_preview=True")
    vp = (
        _gt_vanishing_point(measurements, continuous, shape)
        if include_polar else VanishingPoint("not_computed", 0)
    )
    output_dir = output_root / image_path.stem
    output_dir.mkdir(parents=True)
    native_preview: ByteArray | None = None
    if render_preview:
        native_preview = _render_comparison(
            image, gt_masks, pred_masks, measurements, links, merges, flags, "image"
        )
        _save_preview(output_dir / "comparison.jpg", native_preview)
    instance_rows = list(_instance_records(
        "image", gt_masks, measurements, links, continuous, flags
    ))
    section_rows = _section_records(measurements, None, vp)

    if include_polar and vp.xy is not None:
        grid = build_polar_grid(shape, vp.xy)
        polar_gt = [_warp_mask(mask, grid) for mask in gt_masks]
        polar_pred = [_warp_mask(mask, grid) for mask in pred_masks]
        polar_measurements = [
            _measure(gt, _union([polar_pred[index] for index in related]))
            for gt, related in zip(polar_gt, associations, strict=True)
        ]
        if render_preview:
            polar_image = cv2.remap(
                image, grid.map_x, grid.map_y, cv2.INTER_LINEAR
            )
            polar_preview = _render_comparison(
                polar_image, polar_gt, polar_pred, polar_measurements, links,
                merges, flags, "polar",
            )
            _save_preview(output_dir / "polar_comparison.jpg", polar_preview)
        instance_rows.extend(_instance_records(
            "polar", polar_gt, polar_measurements, links, continuous, flags
        ))
        polar_sections = _section_records(polar_measurements, grid, vp)
        section_rows = chain(section_rows, polar_sections)

    _write_csv(output_dir / "instances.csv", _INSTANCE_COLUMNS, instance_rows)
    if write_sections:
        _write_csv(output_dir / "sections.csv", _SECTION_COLUMNS, section_rows)
    _write_csv(output_dir / "predictions.csv", _PREDICTION_COLUMNS,
               _prediction_records(pred_masks, links, merges))
    result = MaskComparisonResult(
        output_dir,
        tuple(i for i, tags in enumerate(flags, start=1) if "missed" in tags),
        tuple(i for i, tags in enumerate(flags, start=1) if "broken" in tags),
        tuple(i for i, tags in enumerate(flags, start=1) if "split" in tags),
        tuple(int(i) + 1 for i in np.flatnonzero(merges.any(axis=0))),
        vp.status,
    )
    _save_summary(image_path, label_path, result, vp)
    if should_show:
        if native_preview is None:
            raise ValueError("显示预览需要 render_preview=True")
        show_rgb_image(cv2.cvtColor(native_preview, cv2.COLOR_BGR2RGB), image_path.name)
    return result


def _native_flags(
    gt: _Mask,
    predictions: list[_Mask],
    union: _Mask,
    measurement: _Measurement,
    continuous: bool,
    merged: bool,
) -> list[str]:
    """按原图掩码给出诊断，极坐标重采样沿用这些结论。"""
    if not gt.area:
        return ["empty_gt"]
    flags = [] if continuous else ["gt_discontinuous"]
    if not predictions:
        return [*flags, "missed"]
    if measurement.gap_count:
        flags.append("coverage_gap")
    if continuous and measurement.support_components >= 2:
        flags.append("broken")
    if continuous and _has_unique_pieces(gt, predictions, union):
        flags.append("split")
    if merged:
        flags.append("merged")
    if (
        measurement.center_p95 is not None
        and measurement.center_p95 >= _CENTER_SHIFT_FRACTION
    ):
        flags.append("center_shift")
    if (
        measurement.width_p95 is not None
        and measurement.width_p95 >= _WIDTH_CHANGE_FRACTION
    ):
        flags.append("width_change")
    return flags


def _prediction_records(
    masks: list[_Mask], links: BoolArray, merges: BoolArray
) -> Iterable[dict[str, object]]:
    """记录每条预测的 GT 对应及合并候选。"""
    for index, mask in enumerate(masks):
        related = np.flatnonzero(links[:, index]) + 1
        merged = np.flatnonzero(merges[:, index]) + 1
        status = "matched" if len(related) else "unmatched"
        if len(merged):
            status = "merged"
        yield {
            "pred_id": index + 1, "area_px": mask.area,
            "gt_ids": _format_ids(related), "merged_gt_ids": _format_ids(merged),
            "status": status,
        }


def _save_summary(
    image: Path, label: Path, result: MaskComparisonResult, vp: VanishingPoint
) -> None:
    """记录输入来源、诊断阈值和极坐标估计状态。"""
    metadata = {
        "image": image.relative_to(PROJECT_ROOT).as_posix(),
        "label": label.relative_to(PROJECT_ROOT).as_posix(),
        "missed_gt_ids": result.missed_gt_ids,
        "broken_gt_ids": result.broken_gt_ids,
        "split_gt_ids": result.split_gt_ids,
        "merged_pred_ids": result.merged_pred_ids,
        "vp_status": vp.status,
        "vp_xy_px": vp.xy.tolist() if vp.xy is not None else None,
        "vp_line_count": vp.line_count,
        "vp_inlier_gt_ids": vp.inlier_ids,
        "vp_rmse_px": vp.rmse_px,
        "polar_input": "same_masks_resampled",
        "thresholds": {
            "min_intersection_px": _MIN_INTERSECTION_PX,
            "min_gt_coverage": _MIN_GT_COVERAGE,
            "min_prediction_coverage": _MIN_PRED_COVERAGE,
            "strong_gt_coverage": _STRONG_GT_COVERAGE,
            "merge_gt_coverage": _MERGE_GT_COVERAGE,
            "merge_prediction_coverage": _MERGE_PRED_COVERAGE,
            "section_step_canvas_px": _SECTION_STEP_PX,
            "min_section_coverage": _MIN_SECTION_COVERAGE,
            "min_gap_sections": _MIN_GAP_SECTIONS,
            "center_shift_over_gt_width": _CENTER_SHIFT_FRACTION,
            "width_change_over_gt_width": _WIDTH_CHANGE_FRACTION,
        },
    }
    text = json.dumps(metadata, ensure_ascii=False, indent=2)
    (result.output_dir / "summary.json").write_text(text + "\n", encoding="utf-8")


def _predict_config_snapshot(run_dir: Path) -> tuple[Path, str]:
    """选出该实验当时使用的预测配置。

    优先训练归档快照 `configs/rail_predict.yaml`，其次是独立验证目录里的
    快照，最后才回退到当前配置。只有回退时结果才可能与历史记录不一致。

    Args:
        run_dir: `out/yolo_seg` 下的实验目录。

    Returns:
        (预测配置文件路径, 来源说明)。
    """
    archived = run_dir / "configs" / PREDICT_CONFIG.name
    if archived.is_file():
        return archived, "实验归档"
    valid_root = run_dir / "valid"
    if valid_root.is_dir():
        snapshots = sorted(
            path / PREDICT_CONFIG.name
            for path in valid_root.iterdir()
            if (path / PREDICT_CONFIG.name).is_file()
        )
        if snapshots:
            return snapshots[-1], "独立验证快照"
    return PREDICT_CONFIG, "警告：实验未归档预测配置，回退到当前配置"


def _run_validation_example(experiment_id: str, should_show: bool = False) -> None:
    """对指定实验的验证集逐图检查，并归档非图片报告。"""
    out_root = (PROJECT_ROOT / "out").resolve()
    yolo_root = out_root / "yolo_seg"
    run_dir = _relative_under(experiment_id, yolo_root)
    if run_dir.parent != yolo_root:
        raise ValueError("experiment_id 必须是 out/yolo_seg 下的实验目录名")

    weight_path = run_dir / "weights" / _BEST_WEIGHT_NAME
    if not weight_path.is_file():
        raise FileNotFoundError(f"缺少实验最佳权重 {weight_path}")
    train_config = run_dir / "logs" / "args.yaml"
    if not train_config.is_file():
        train_config = run_dir / "configs" / "train_rail_seg.yaml"
    data_setting = load_yaml_mapping(train_config).get("data")
    if not isinstance(data_setting, str) or not data_setting.strip():
        raise ValueError(f"{train_config} 的 data 必须指向数据集配置")
    data_config = run_dir / "configs" / Path(data_setting).name
    validation_dir = load_split_image_dir(data_config, "val")
    if not validation_dir.is_dir():
        raise NotADirectoryError(f"缺少实验验证集 {validation_dir}")
    if not validation_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"实验验证集必须位于项目内 {validation_dir}")

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO
    from ultralytics.data.utils import img2label_paths

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = out_root / "compare_mask" / run_dir.name / timestamp
    output_dir.mkdir(parents=True)
    output_subdir = output_dir.relative_to(out_root)
    predict_source, source_label = _predict_config_snapshot(run_dir)
    snapshot = output_dir / PREDICT_CONFIG.name
    copy2(predict_source, snapshot)
    print(f"预测配置：{source_label} {predict_source}")
    options = load_predict_options(snapshot)
    model = YOLO(str(weight_path))
    predictions = iter_segmentation_predictions(model, validation_dir, options)
    columns = (
        "image", "missed_gt_ids", "broken_gt_ids", "split_gt_ids", "merged_pred_ids",
        "vp_status", "report_dir",
    )
    index_path = output_dir / "index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for prediction in predictions:
            image_path = Path(prediction.path).resolve()
            label_path = Path(img2label_paths([image_path])[0])
            result = compare_mask(
                image_path.relative_to(PROJECT_ROOT),
                label_path.relative_to(PROJECT_ROOT),
                prediction.masks, output_subdir, should_show,
            )
            writer.writerow({
                "image": image_path.name,
                "missed_gt_ids": _format_ids(result.missed_gt_ids),
                "broken_gt_ids": _format_ids(result.broken_gt_ids),
                "split_gt_ids": _format_ids(result.split_gt_ids),
                "merged_pred_ids": _format_ids(result.merged_pred_ids),
                "vp_status": result.vp_status,
                "report_dir": result.output_dir.relative_to(output_dir).as_posix(),
            })
            print(f"已分析 {image_path.name}")
    archive_path = zip_analysis_reports(
        output_dir,
        f"{run_dir.name}_compare_{timestamp}.zip",
        PROJECT_ROOT / "doc" / "readme" / "mask_compare_readme.md",
    )
    print(f"掩码几何对照已保存到 {output_dir}")
    print(f"非图片报告和说明已打包到 {archive_path}")


def _resolve_experiment(run: Path | None) -> str:
    """把命令行参数归一为 out/yolo_seg 下的实验目录名。

    Args:
        run: `--run` 传入的绝对路径、项目相对路径或实验目录名；None 表示
            取 `out/yolo_seg` 下最新一次实验。

    Returns:
        实验目录名，用于定位 `out/yolo_seg/<实验>`。

    Raises:
        SystemExit: 路径不在 `out/yolo_seg` 下，或没有可用的实验目录。
    """
    if run is None:
        experiment = latest_experiment()
        if not (experiment / "weights" / _BEST_WEIGHT_NAME).is_file():
            raise SystemExit(
                f"最新实验 {experiment.name} 缺少归档权重 {_BEST_WEIGHT_NAME}，"
                "请用 --run 指定其它实验"
            )
        return experiment.name
    yolo_root = (PROJECT_ROOT / "out" / "yolo_seg").resolve()
    if run.is_absolute():
        resolved = run.resolve()
    elif len(run.parts) == 1:
        resolved = (yolo_root / run).resolve()
    else:
        resolved = (PROJECT_ROOT / run).resolve()
    if resolved.parent != yolo_root:
        raise SystemExit(f"--run 必须是 {yolo_root} 下的实验目录，收到 {run}")
    return resolved.name


def main() -> None:
    """命令行入口：默认分析 `out/yolo_seg` 下最新一次实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path,
        help="实验目录（项目相对路径或目录名）；省略时取 out/yolo_seg 下最新一次",
    )
    parser.add_argument(
        "--show", action="store_true", help="额外打开原图对照的 OpenCV 窗口"
    )
    args = parser.parse_args()
    experiment_id = _resolve_experiment(args.run)
    print(f"实验目录：out/yolo_seg/{experiment_id}")
    _run_validation_example(experiment_id, should_show=args.show)


if __name__ == "__main__":
    main()
