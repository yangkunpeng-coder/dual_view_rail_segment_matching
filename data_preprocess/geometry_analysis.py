"""从原始图片和 LabelMe 标注分析钢轨的二维几何与标注质量。

geometry.npz 中的掩码为独立的 0/1 裁片，按对应的 boxes_xyxy 和画布尺寸
放回即可复原。中心线列由 centerline_columns 给出，曲率单位为每像素，
邻域采样不足处用 NaN 表示。
极坐标列为角度，行为半径，坐标等于对应轴的最小值加索引乘步长。
"""

from collections.abc import Iterable
from colorsys import hsv_to_rgb
import csv
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from itertools import combinations
from math import cos
from pathlib import Path
import sys

import cv2
import numpy as np
from numpy.typing import NDArray

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_algorithm.common.rail_centerline import (
    SectionTrace,
    measure_sections,
    trace_instance_mask,
)
from data_preprocess.common.config import PROJECT_ROOT, load_dataset_relative_path
from data_preprocess.common.labelme import RailPolygon, load_rail_polygons
from utils.rail_perspective import (
    PolarGrid,
    VanishingPoint,
    build_polar_grid,
    estimate_vanishing_point,
)
from utils.report_archive import zip_analysis_reports

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_CAMERAS = ("north", "south")
_MIN_FRAGMENT_SPAN_PX = 15
_MIN_ANGULAR_RADIUS_PX = 4
_MIN_RAIL_SPAN_RATIO = 0.18
_MIN_RAIL_ASPECT_RATIO = 5.0
_MIN_SPLIT_SECTION_FRACTION = 0.05
_PREVIEW_MAX_WIDTH = 1400
_BREAK_MAX_ANGLE_RAD = np.deg2rad(8)
_BREAK_MAX_GAP_HEIGHT_RATIO = 0.08
_BREAK_ENDPOINT_SAMPLE_COUNT = 12
_BREAK_MAX_WIDTH_RATIO = 2.5
_MIN_OUTLIER_PEERS = 4
_MAD_SCALE = 1.4826
_TRACE_COLUMNS = ("x_px", "y_px", "width_px", "curvature_inv_px", "segment_id")
_INSTANCE_COLUMNS = (
    "camera", "image", "instance_id", "group_id", "area_px", "component_count",
    "fragment_area_fraction", "centerline_sample_count", "arc_length_px",
    "width_p10_px", "width_p50_px", "width_p90_px", "curvature_p95_inv_px",
    "gap_count", "max_gap_px", "multiple_section_fraction", "angular_width_mrad",
    "polar_centerline_sample_count", "polar_arc_length_px", "polar_width_p50_px",
    "polar_curvature_p95_inv_px", "flags",
)
_BREAK_COLUMNS = (
    "camera", "image", "first_instance_id", "second_instance_id", "gap_px",
    "lateral_offset_px", "endpoint_width_ratio",
)
_STABILITY_COLUMNS = (
    "camera", "image_count", "estimate_count", "two_line_count", "reference_width_px",
    "reference_height_px", "median_x_px", "median_y_px", "std_x_px", "std_y_px",
    "median_deviation_px", "p95_deviation_px", "max_deviation_px",
    "outlier_threshold_px", "outlier_count",
)

FloatArray = NDArray[np.float64]
ByteArray = NDArray[np.uint8]
Box = tuple[int, int, int, int]


@dataclass
class _Rail:
    """原图中的独立实例 Mask，box 右边界和下边界为开区间。"""

    instance_id: int
    group_id: int | None
    box: Box
    mask: ByteArray
    component_count: int
    fragment_area_fraction: float
    trace: SectionTrace | None
    flags: list[str] = field(default_factory=list)


@dataclass
class _ImageReport:
    """用于逐图 CSV 和同相机稳定性统计的测量结果。"""

    camera: str
    image: str
    width_px: int | None = None
    height_px: int | None = None
    instance_count: int = 0
    flagged_instance_count: int = 0
    vp_status: str = "input_error"
    vp_x_px: float | None = None
    vp_y_px: float | None = None
    vp_line_count: int = 0
    vp_inlier_count: int = 0
    vp_rmse_px: float | None = None
    vp_deviation_px: float | None = None
    vp_outlier: bool | None = None
    error: str = ""


def _rasterize_rails(
    polygons: list[RailPolygon], image_shape: tuple[int, int]
) -> list[_Rail]:
    """按 group_id 合并多边形，重叠像素仍保留在各自的独立掩码中。"""
    grouped_ids: dict[int, int] = {}
    groups: list[tuple[int | None, list[NDArray[np.int32]]]] = []
    for polygon in polygons:
        group_id = polygon["group_id"]
        index = len(groups)
        if group_id is not None:
            index = grouped_ids.setdefault(group_id, index)
        if index == len(groups):
            groups.append((group_id, []))
        points = np.asarray(polygon["points"], dtype=np.float64)
        if points.shape != (len(points), 2) or not np.isfinite(points).all():
            raise ValueError("rail 多边形顶点坐标无效")
        groups[index][1].append(np.rint(points).astype(np.int32))

    height, width = image_shape
    rails: list[_Rail] = []
    for instance_id, (group_id, contours) in enumerate(groups, start=1):
        vertices = np.concatenate(contours)
        x0, y0 = map(int, np.clip(vertices.min(axis=0), (0, 0), (width, height)))
        x1, y1 = map(int, np.clip(vertices.max(axis=0) + 1, (0, 0), (width, height)))
        mask = np.full((y1 - y0, x1 - x0), 0, dtype=np.uint8)
        if mask.size:
            offset = np.array((x0, y0), dtype=np.int32)
            for contour in contours:
                cv2.fillPoly(mask, [contour - offset], 1)
        box = (x0, y0, x1, y1)
        areas = np.empty(0, dtype=np.int32)
        if mask.any():
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA]
        fragment_fraction = 1 - float(areas.max() / areas.sum()) if areas.size else 0.0
        rail = _Rail(
            instance_id, group_id, box, mask, len(areas), fragment_fraction,
            trace_instance_mask(mask, box),
        )
        _flag_mask_geometry(rail, height)
        rails.append(rail)
    return rails


def _flag_mask_geometry(rail: _Rail, image_height: int) -> None:
    """记录断裂、分叉、碎片及过短或过宽的形状。"""
    trace = rail.trace
    if trace is None:
        rail.flags.append("empty_mask")
        return
    if rail.component_count > 1:
        rail.flags.append("disconnected_components")
    if trace.gap_count:
        rail.flags.append("internal_gap")
    if trace.split_fraction >= _MIN_SPLIT_SECTION_FRACTION:
        rail.flags.append("multiple_cross_sections")
    if trace.length_px < _MIN_RAIL_SPAN_RATIO * image_height:
        rail.flags.append("short_visible_span")
    if trace.length_px < _MIN_RAIL_ASPECT_RATIO * np.median(trace.widths):
        rail.flags.append("low_aspect_ratio")


def _estimate_vp(rails: list[_Rail], shape: tuple[int, int]) -> VanishingPoint:
    """用各长实例的最长连续中心线估计消失点。"""
    centerlines: list[tuple[int, FloatArray]] = []
    for rail in rails:
        trace = rail.trace
        if trace is not None and "low_aspect_ratio" not in rail.flags:
            centerlines.append((rail.instance_id, trace.points[trace.longest_segment]))
    return estimate_vanishing_point(centerlines, shape)


def _warp_rail_mask(
    rail: _Rail, grid: PolarGrid
) -> tuple[ByteArray, Box, SectionTrace | None]:
    """用最近邻映射一个独立 Mask，并在极坐标画布上测量中心线。"""
    if not rail.mask.any():
        return np.full((0, 0), 0, dtype=np.uint8), (0, 0, 0, 0), None
    x0, y0 = rail.box[:2]
    transformed = cv2.remap(
        rail.mask, grid.map_x - x0, grid.map_y - y0, cv2.INTER_NEAREST
    )
    x, y, width, height = cv2.boundingRect(transformed)
    box = (x, y, x + width, y + height)
    mask = transformed[y:y + height, x:x + width].copy()
    trace = measure_sections(mask)
    if trace is not None:
        trace.points += (x, y)
    return mask, box, trace


def _curvature_p95(trace: SectionTrace) -> float | None:
    """返回有足够采样支持的曲率 95 分位值。"""
    values = trace.curvatures[np.isfinite(trace.curvatures)]
    return float(np.percentile(values, 95)) if values.size else None


def _angular_width(trace: SectionTrace, vp: VanishingPoint) -> float | None:
    """按中心线半径估算透视归一的角宽中位数，单位为毫弧度。"""
    if vp.xy is None:
        return None
    radii = np.linalg.norm(trace.points - vp.xy, axis=1)
    valid = radii >= _MIN_ANGULAR_RADIUS_PX
    if not np.any(valid):
        return None
    angles = 2 * np.arctan2(trace.widths[valid] / 2, radii[valid])
    return float(np.median(angles) * 1000)


def _flag_peer_outliers(rails: list[_Rail], vp: VanishingPoint) -> None:
    """以同图长实例的中位数和 MAD 筛查异常角宽及弯曲程度。"""
    widths: list[tuple[_Rail, float]] = []
    bends: list[tuple[_Rail, float]] = []
    for rail in rails:
        trace = rail.trace
        if trace is None or "short_visible_span" in rail.flags:
            continue
        width = _angular_width(trace, vp)
        if width is not None:
            widths.append((rail, float(np.log(width))))
        curvature = _curvature_p95(trace)
        if curvature is not None:
            # 乘可见弧长得到无量纲的弯曲程度，便于同图比较。
            bends.append((rail, curvature * trace.length_px))

    for measurements, minimum_deviation, flag in (
        (widths, float(np.log(1.75)), "unusual_angular_width"),
        (bends, 0.25, "unusual_curvature"),
    ):
        if len(measurements) < _MIN_OUTLIER_PEERS:
            continue
        values = np.array([value for _, value in measurements])
        center = np.median(values)
        mad = np.median(np.abs(values - center))
        threshold = max(minimum_deviation, 3.5 * _MAD_SCALE * mad)
        for rail, value in measurements:
            deviation = value - center
            if flag == "unusual_angular_width":
                deviation = abs(deviation)
            if deviation > threshold:
                rail.flags.append(flag)


def _find_possible_breaks(
    rails: list[_Rail], image_height: int
) -> list[dict[str, object]]:
    """筛选纵向不重叠且端点方向、位置和宽度接近的实例对。"""
    endpoints: list[tuple[_Rail, FloatArray, FloatArray, FloatArray, FloatArray]] = []
    for rail in rails:
        trace = rail.trace
        if trace is None or "low_aspect_ratio" in rail.flags:
            continue
        segment = trace.longest_segment
        points, widths = trace.points[segment], trace.widths[segment]
        direction = points[-1] - points[0]
        span = np.linalg.norm(direction)
        if span >= _MIN_FRAGMENT_SPAN_PX:
            endpoints.append((rail, points[0], points[-1], direction / span, widths))

    candidates: list[dict[str, object]] = []
    for first, second in combinations(endpoints, 2):
        if first[3] @ second[3] < cos(_BREAK_MAX_ANGLE_RAD):
            continue
        axis = first[3] + second[3]
        axis /= np.linalg.norm(axis)
        earlier, later = sorted((first, second), key=lambda item: item[1] @ axis)
        delta = later[1] - earlier[2]
        gap = float(delta @ axis)
        if not 0 < gap <= image_height * _BREAK_MAX_GAP_HEIGHT_RATIO:
            continue
        lateral = abs(float(delta @ np.array((-axis[1], axis[0]))))
        end_width = float(np.median(earlier[4][-_BREAK_ENDPOINT_SAMPLE_COUNT:]))
        start_width = float(np.median(later[4][:_BREAK_ENDPOINT_SAMPLE_COUNT]))
        width_ratio = max(end_width, start_width) / min(end_width, start_width)
        if (
            lateral > 2 + 0.5 * min(end_width, start_width)
            or width_ratio > _BREAK_MAX_WIDTH_RATIO
        ):
            continue
        for rail in (earlier[0], later[0]):
            if "possible_split_instance" not in rail.flags:
                rail.flags.append("possible_split_instance")
        candidates.append(
            {
                "first_instance_id": earlier[0].instance_id,
                "second_instance_id": later[0].instance_id,
                "gap_px": gap,
                "lateral_offset_px": lateral,
                "endpoint_width_ratio": width_ratio,
            }
        )
    return candidates


def _trace_array(trace: SectionTrace | None) -> NDArray[np.float32]:
    """将中心线、宽度、曲率和连续段编号写为独立数值列。"""
    if trace is None:
        return np.full((0, len(_TRACE_COLUMNS)), 0, dtype=np.float32)
    segment_ids = np.empty(len(trace.points), dtype=np.int32)
    for segment_id, segment in enumerate(trace.segments, start=1):
        segment_ids[segment] = segment_id
    return np.column_stack(
        (trace.points, trace.widths, trace.curvatures, segment_ids)
    ).astype(np.float32)


def _rail_measurements(
    rail: _Rail, polar_trace: SectionTrace | None, vp: VanishingPoint
) -> dict[str, object]:
    """整理原图及极坐标画布内的实例统计，空值代表缺少可测几何。"""
    trace = rail.trace
    row: dict[str, object] = {
        "instance_id": rail.instance_id,
        "group_id": rail.group_id,
        "area_px": int(np.count_nonzero(rail.mask)),
        "component_count": rail.component_count,
        "fragment_area_fraction": rail.fragment_area_fraction,
        "flags": "|".join(rail.flags),
    }
    if trace is not None:
        low, median, high = np.percentile(trace.widths, (10, 50, 90))
        row.update(
            centerline_sample_count=len(trace.points),
            arc_length_px=trace.length_px,
            width_p10_px=low,
            width_p50_px=median,
            width_p90_px=high,
            curvature_p95_inv_px=_curvature_p95(trace),
            gap_count=trace.gap_count,
            max_gap_px=trace.max_gap_px,
            multiple_section_fraction=trace.split_fraction,
            angular_width_mrad=_angular_width(trace, vp),
        )
    if polar_trace is not None:
        row.update(
            polar_centerline_sample_count=len(polar_trace.points),
            polar_arc_length_px=polar_trace.length_px,
            polar_width_p50_px=float(np.median(polar_trace.widths)),
            polar_curvature_p95_inv_px=_curvature_p95(polar_trace),
        )
    return row


def _draw_instance(
    canvas: ByteArray,
    mask: ByteArray,
    box: Box,
    trace: SectionTrace | None,
    rail: _Rail,
) -> None:
    """为同一实例使用一致的颜色，异常实例的中心线用红色标出。"""
    x0, y0, x1, y1 = box
    patch = canvas[y0:y1, x0:x1]
    pixels = mask != 0
    rgb = hsv_to_rgb((rail.instance_id * 0.61803398875) % 1.0, 0.75, 1.0)
    bgr = np.asarray(rgb[::-1]) * 255
    patch[pixels] = np.rint(patch[pixels] * 0.55 + bgr * 0.45).astype(np.uint8)
    if trace is None:
        return

    points = np.rint(trace.points).astype(np.int32)
    color = (35, 55, 255) if rail.flags else (255, 255, 255)
    thickness = max(1, canvas.shape[1] // 1200)
    lines = [
        points[segment]
        for segment in trace.segments
        if segment.stop - segment.start > 1
    ]
    if lines:
        cv2.polylines(canvas, lines, False, color, thickness, cv2.LINE_AA)
    segment = trace.longest_segment
    center = tuple(points[(segment.start + segment.stop) // 2])
    label = f"{rail.instance_id}{'!' if rail.flags else ''}"
    scale = max(0.45, canvas.shape[1] / 2100)
    cv2.putText(canvas, label, center, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(canvas, label, center, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def _write_image(path: Path, image: ByteArray) -> None:
    """保存 BGR 图像，支持 Windows 下包含中文的项目路径。"""
    if path.suffix == ".jpg":
        height, width = image.shape[:2]
        if width > _PREVIEW_MAX_WIDTH:
            new_height = max(1, round(height * _PREVIEW_MAX_WIDTH / width))
            size = (_PREVIEW_MAX_WIDTH, new_height)
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 90]
    else:
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    success, encoded = cv2.imencode(path.suffix, image, parameters)
    if not success:
        raise OSError(f"无法写入图像 {path}")
    encoded.tofile(path)


def _save_image_analysis(
    sample_dir: Path, image: ByteArray, rails: list[_Rail], vp: VanishingPoint
) -> list[dict[str, object]]:
    """保存可复原的实例掩码、中心线及两种坐标下的叠加图。"""
    sample_dir.mkdir(parents=True)
    shape = image.shape[:2]
    ids = [rail.instance_id for rail in rails]
    boxes = np.asarray([rail.box for rail in rails], dtype=np.int32)
    arrays: dict[str, NDArray[np.generic]] = {
        "image_shape_hw": np.asarray(shape),
        "instance_ids": np.asarray(ids, dtype=np.int32),
        "gt_boxes_xyxy": boxes.reshape(-1, 4),
        "centerline_columns": np.asarray(_TRACE_COLUMNS),
    }
    grid, polar_overlay = None, None
    if vp.xy is not None:
        grid = build_polar_grid(shape, vp.xy)
        polar_image = cv2.remap(image, grid.map_x, grid.map_y, cv2.INTER_LINEAR)
        _write_image(sample_dir / "polar.jpg", polar_image)
        polar_overlay = polar_image
        arrays.update(
            vp_xy_px=vp.xy,
            polar_shape_hw=np.asarray(polar_image.shape[:2]),
            polar_theta_min_rad=np.asarray(grid.theta_min_rad),
            polar_radius_min_px=np.asarray(grid.radius_min_px),
            polar_theta_step_rad=np.asarray(grid.theta_step_rad),
            polar_radius_step_px=np.asarray(grid.radius_step_px),
        )

    overlay = image.copy()
    measurements: list[dict[str, object]] = []
    polar_boxes: list[Box] = []
    for rail in rails:
        suffix = f"{rail.instance_id:03d}"
        arrays[f"gt_mask_{suffix}"] = rail.mask
        arrays[f"gt_centerline_{suffix}"] = _trace_array(rail.trace)
        polar_trace = None
        if grid is not None and polar_overlay is not None:
            polar_mask, polar_box, polar_trace = _warp_rail_mask(rail, grid)
            if polar_trace is None and rail.trace is not None:
                rail.flags.append("empty_polar_mask")
            polar_boxes.append(polar_box)
            arrays[f"polar_mask_{suffix}"] = polar_mask
            arrays[f"polar_centerline_{suffix}"] = _trace_array(polar_trace)
            _draw_instance(polar_overlay, polar_mask, polar_box, polar_trace, rail)
        _draw_instance(overlay, rail.mask, rail.box, rail.trace, rail)
        measurements.append(_rail_measurements(rail, polar_trace, vp))

    height, width = shape
    status = f"VP {vp.status} | lines {len(vp.inlier_ids)}/{vp.line_count}"
    if vp.xy is not None:
        point = np.clip(vp.xy, (0, 0), (width - 1, height - 1))
        marker = tuple(np.rint(point).astype(int))
        cv2.drawMarker(overlay, marker, (0, 220, 255), cv2.MARKER_CROSS, 32, 3)
        status += f" | ({vp.xy[0]:.1f}, {vp.xy[1]:.1f}) px"
        if not np.array_equal(point, vp.xy):
            status += " | outside frame"
    origin = (16, min(height - 1, 38))
    cv2.putText(overlay, status, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(
        overlay, status, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2
    )
    _write_image(sample_dir / "overlay.jpg", overlay)
    if polar_overlay is not None:
        polar_boxes_array = np.asarray(polar_boxes, dtype=np.int32)
        arrays["polar_boxes_xyxy"] = polar_boxes_array.reshape(-1, 4)
        _write_image(sample_dir / "polar_overlay.jpg", polar_overlay)
    np.savez_compressed(sample_dir / "geometry.npz", **arrays)
    return measurements


def _analyse_image(
    image_path: Path, camera: str, output_dir: Path
) -> tuple[_ImageReport, list[dict[str, object]], list[dict[str, object]]]:
    """处理一张图，读取失败时写入错误状态并继续处理其它图片。"""
    report = _ImageReport(camera, image_path.relative_to(PROJECT_ROOT).as_posix())
    try:
        image = cv2.imdecode(
            np.fromfile(image_path, dtype=np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if image is None:
            raise ValueError("无法解码原始图片")
        height, width = image.shape[:2]
        report.height_px, report.width_px = height, width
        polygons = load_rail_polygons(image_path, (width, height))
        rails = _rasterize_rails(polygons, image.shape[:2])
    except (OSError, ValueError, TypeError, KeyError, cv2.error) as error:
        report.error = str(error)
        return report, [], []

    vp = VanishingPoint("no_rails", 0)
    if rails:
        vp = _estimate_vp(rails, image.shape[:2])
    _flag_peer_outliers(rails, vp)
    breaks = _find_possible_breaks(rails, height)
    sample_dir = output_dir / "samples" / camera / image_path.name
    measurements = _save_image_analysis(sample_dir, image, rails, vp)
    report.instance_count = len(rails)
    report.flagged_instance_count = sum(bool(rail.flags) for rail in rails)
    report.vp_status = vp.status
    report.vp_line_count = vp.line_count
    report.vp_inlier_count = len(vp.inlier_ids)
    report.vp_rmse_px = vp.rmse_px
    if vp.xy is not None:
        report.vp_x_px, report.vp_y_px = map(float, vp.xy)
    source = {"camera": camera, "image": report.image}
    return report, [{**source, **row} for row in measurements], [
        {**source, **row} for row in breaks
    ]


def _summarize_stability(
    reports: list[_ImageReport], output_dir: Path
) -> list[dict[str, object]]:
    """按相机汇总消失点离散度，并绘制统一参考分辨率下的散点。"""
    from matplotlib.figure import Figure

    cameras = sorted({report.camera for report in reports})
    figure = Figure(figsize=(5.4 * len(cameras), 4.2), layout="constrained")
    axes = figure.subplots(1, len(cameras), squeeze=False)[0]
    summary: list[dict[str, object]] = []
    for camera, axis in zip(cameras, axes, strict=True):
        frames = [report for report in reports if report.camera == camera]
        estimates = [report for report in frames if report.vp_status == "ok"]
        row: dict[str, object] = {
            "camera": camera,
            "image_count": len(frames),
            "estimate_count": len(estimates),
            "two_line_count": sum(r.vp_status == "two_line_support" for r in frames),
        }
        summary.append(row)
        axis.set_title(f"{camera} | {len(estimates)}/{len(frames)} supported estimates")
        axis.set(xlabel="VP x (reference px)", ylabel="VP y (reference px)")
        axis.invert_yaxis()
        axis.grid(alpha=0.25)
        if not estimates:
            axis.text(
                0.5, 0.5, "No supported estimates", transform=axis.transAxes,
                ha="center", va="center",
            )
            continue

        sizes = np.array([(r.width_px, r.height_px) for r in estimates], dtype=float)
        reference_size = np.median(sizes, axis=0)
        points = np.array([(r.vp_x_px, r.vp_y_px) for r in estimates], dtype=float)
        points = points / sizes * reference_size
        median_point = np.median(points, axis=0)
        deviations = np.linalg.norm(points - median_point, axis=1)
        median_deviation = float(np.median(deviations))
        row.update(
            reference_width_px=reference_size[0],
            reference_height_px=reference_size[1],
            median_x_px=median_point[0],
            median_y_px=median_point[1],
            median_deviation_px=median_deviation,
            p95_deviation_px=float(np.percentile(deviations, 95)),
            max_deviation_px=float(deviations.max()),
        )
        if len(estimates) > 1:
            row["std_x_px"], row["std_y_px"] = points.std(axis=0, ddof=1)
        threshold = None
        if len(estimates) >= _MIN_OUTLIER_PEERS:
            mad = float(np.median(np.abs(deviations - median_deviation)))
            threshold = max(
                0.01 * np.linalg.norm(reference_size),
                median_deviation + 3 * _MAD_SCALE * mad,
            )
            row["outlier_threshold_px"] = threshold
            row["outlier_count"] = int(np.count_nonzero(deviations > threshold))
        for report, deviation in zip(estimates, deviations, strict=True):
            report.vp_deviation_px = float(deviation)
            if threshold is not None:
                report.vp_outlier = bool(deviation > threshold)

        colors = [
            "tab:red" if report.vp_outlier else "tab:blue" for report in estimates
        ]
        axis.scatter(points[:, 0], points[:, 1], c=colors, s=15, alpha=0.75)
        axis.scatter(*median_point, marker="+", c="black", s=85, label="Median")
        axis.legend(loc="best")
    figure.savefig(output_dir / "vp_stability.png", dpi=140)
    return summary


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: Iterable[dict[str, object]]
) -> None:
    """写入带表头的 UTF-8 CSV，方便 Windows 表格软件直接打开。"""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _collect_images(dataset_dir: Path) -> list[tuple[str, Path]]:
    """按相机和文件名排序，支持数据集根目录及单相机图片目录。"""
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"找不到原始数据集目录 {dataset_dir}")
    cameras = [(name, dataset_dir / name) for name in _CAMERAS]
    cameras = [(name, directory) for name, directory in cameras if directory.is_dir()]
    if not cameras:
        cameras = [(dataset_dir.name, dataset_dir)]
    return [
        (name, image_path)
        for name, directory in cameras
        for image_path in sorted(directory.iterdir())
        if image_path.suffix.lower() in _IMAGE_SUFFIXES and image_path.is_file()
    ]


def analyse_rail_geometry(dataset_relative_path: str | Path) -> Path:
    """分析原图中的钢轨几何和标注异常，在 out 下归档本次结果。

    Args:
        dataset_relative_path: 相对于项目根目录的原始数据集目录，包含
            north、south 子目录或直接包含图片与同名 LabelMe JSON。

    Returns:
        本次分析输出目录的绝对路径。报告记录无法读取的样本及消失点
        支持不足的图片。后者保留原图几何，极坐标产物按有效估计生成。

    Raises:
        ValueError: 输入路径不在项目内或使用了绝对路径。
        OSError: 找不到输入目录或图片，或无法写入结果。
    """
    relative_path = Path(dataset_relative_path)
    if relative_path.anchor:
        raise ValueError("请输入相对于项目根目录的数据集路径")
    dataset_dir = (PROJECT_ROOT / relative_path).resolve()
    if not dataset_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("原始数据集路径必须位于项目内")
    images = _collect_images(dataset_dir)
    if not images:
        raise FileNotFoundError(f"原始数据集没有图片 {dataset_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT_ROOT / "out" / "rail_geometry" / timestamp
    output_dir.mkdir(parents=True)
    print(f"开始分析 {len(images)} 张原始图片，输出到 {output_dir}")
    image_reports: list[_ImageReport] = []
    instance_rows: list[dict[str, object]] = []
    break_rows: list[dict[str, object]] = []
    for index, (camera, image_path) in enumerate(images, start=1):
        report, measurements, breaks = _analyse_image(image_path, camera, output_dir)
        image_reports.append(report)
        instance_rows.extend(measurements)
        break_rows.extend(breaks)
        if report.error:
            print(f"读取失败 {report.image} | {report.error}")
        if index % 20 == 0 or index == len(images):
            print(f"已处理 {index}/{len(images)} 张图片")

    stability = _summarize_stability(image_reports, output_dir)
    image_columns = tuple(column.name for column in fields(_ImageReport))
    _write_csv(output_dir / "images.csv", image_columns, map(asdict, image_reports))
    _write_csv(output_dir / "instances.csv", _INSTANCE_COLUMNS, instance_rows)
    _write_csv(output_dir / "possible_breaks.csv", _BREAK_COLUMNS, break_rows)
    _write_csv(output_dir / "vp_stability.csv", _STABILITY_COLUMNS, stability)
    archive_path = zip_analysis_reports(
        output_dir,
        f"rail_geometry_{timestamp}.zip",
        PROJECT_ROOT / "doc" / "readme" / "geometry_analysis_readme.md",
    )
    print(f"几何报告与可视化已保存到 {output_dir}")
    print(f"非图片报告和说明已打包到 {archive_path}")
    return output_dir


if __name__ == "__main__":
    analyse_rail_geometry(load_dataset_relative_path())
