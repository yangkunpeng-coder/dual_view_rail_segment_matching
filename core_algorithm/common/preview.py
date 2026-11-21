"""南北匹配结果的标注读取、截面统计与 OpenCV 预览。"""

from collections.abc import Callable
from colorsys import hsv_to_rgb
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from core_algorithm.common.match_result import (
    RailInstance,
    RailMatchingResult,
    Side,
    UnmatchedRail,
)
from data_preprocess.common.config import PROJECT_ROOT, load_dataset_relative_path
from data_preprocess.common.image import review_rgb_image
from data_preprocess.common.labelme import (
    extract_rail_polygons,
    parse_labelme_annotation,
)
from data_preprocess.match_pair_image import match_pair_image
from utils.image_layout import compose_image_pair

SECTION_PERCENTS = (25, 50, 75)
MAIN_SECTION_INDEX = 1

_PREVIEW_PANEL_WIDTH = 640
_UNMATCHED_COLOR = (172, 172, 172)
_MASK_OPACITY = 0.45
_COLOR_HUE_STEP = 0.61803398875
_LABEL_ROW_FRACTIONS = (0.33, 0.45, 0.57, 0.69, 0.81)
_PREVIEW_BACKGROUND = 255
_SUMMARY_FONT_NAME = "msyhbd.ttc"
_SUMMARY_FONT_SIZE = 20
_SUMMARY_LINE_HEIGHT = 27
_SUMMARY_MARGIN = 16
_SUMMARY_TEXT_COLOR = (24, 31, 40)
_SUMMARY_MISSED_COLOR = (192, 0, 0)


class MaskSection(NamedTuple):
    """单行 Mask 的中心横坐标、前景像素数、交段数和测量失败原因。"""

    center_x: float
    foreground_px: int
    run_count: int
    reason: str | None


class LabelMeInstanceRef(NamedTuple):
    """图内原始 shapes 索引和可空分组编号。"""

    shape_index: int
    group_id: int | None


@dataclass(frozen=True)
class LabelMeView:
    """逐 Polygon Mask 及与输入索引一一对应的来源信息。"""

    image: NDArray[np.uint8]
    masks: NDArray[np.bool_]
    references: tuple[LabelMeInstanceRef, ...]
    label_sha256: str


def load_labelme_view(
    image_path: Path,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """读取原图，将同名 LabelMe 文件的各钢轨 Polygon 独立栅格化。

    Args:
        image_path: 磁盘上的原图路径。

    Returns:
        形状为 (H, W, 3) 的 uint8 RGB 原图和形状为 (N, H, W) 的布尔 Mask。
        Mask 按原钢轨 Polygon 顺序排列。
    Raises:
        OSError: 无法读取原图或标注。
        ValueError: LabelMe 标注与图像尺寸不符或多边形无效。
    """
    view = load_labelme_view_details(image_path)
    return view.image, view.masks


def load_labelme_view_details(image_path: Path) -> LabelMeView:
    """读取原图、独立栅格化钢轨 Polygon 并保留来源编号和哈希。

    Args:
        image_path: 磁盘上的原图路径，旁边须有同名 JSON。

    Returns:
        RGB 图像、按原 Polygon 顺序排列的布尔 Mask、来源编号及 JSON 哈希。

    Raises:
        OSError: 无法读取原图或 JSON。
        ValueError: JSON 或钢轨 Polygon 无效。
    """
    with Image.open(image_path) as image:
        width, height = image.size
        image_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    label_path = image_path.with_suffix(".json")
    content = label_path.read_bytes()
    annotation = parse_labelme_annotation(
        content.decode("utf-8-sig"), label_path, (width, height)
    )
    polygons = extract_rail_polygons(annotation, label_path)
    shapes = annotation.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"{label_path} 的 shapes 必须为列表")
    shape_indices = (
        index for index, shape in enumerate(shapes)
        if isinstance(shape, dict) and shape.get("label") == "rail"
    )
    references = tuple(
        LabelMeInstanceRef(index, polygon["group_id"])
        for index, polygon in zip(shape_indices, polygons, strict=True)
    )
    masks = np.zeros((len(polygons), height, width), dtype=np.bool_)
    for mask, polygon in zip(masks, polygons, strict=True):
        vertices = np.rint(polygon["points"]).astype(np.int32)
        cv2.fillPoly(mask.view(np.uint8), [vertices], 1)
    return LabelMeView(image_rgb, masks, references, sha256(content).hexdigest())


def section_rows(height: int) -> tuple[int, ...]:
    """计算 25%、50%、75% 截面的像素行号。

    Args:
        height: 原图高度，单位为像素。

    Returns:
        依次对应三个截面的像素行号。
    """
    return tuple(round((height - 1) * (percent / 100)) for percent in SECTION_PERCENTS)


def measure_section(row: NDArray[np.bool_]) -> MaskSection:
    """统计一行 Mask 的前景，并在只有一个连续段时取其中点。

    Args:
        row: 一维布尔 Mask 行，True 表示前景。

    Returns:
        截面测量。缺失或分段时保留原因，中心横坐标为 NaN。
    """
    columns = np.flatnonzero(row)
    foreground_px = len(columns)
    if not foreground_px:
        return MaskSection(np.nan, 0, 0, "no_measurement_intersection")
    if columns[-1] - columns[0] + 1 == foreground_px:
        center = float((columns[0] + columns[-1]) / 2)
        return MaskSection(center, foreground_px, 1, None)
    run_count = 1 + int(np.count_nonzero(np.diff(columns) > 1))
    return MaskSection(
        np.nan, foreground_px, run_count, "multiple_measurement_intersections"
    )


def render_side(
    image_rgb: NDArray[np.uint8],
    masks: NDArray[np.bool_],
    result: RailMatchingResult,
    side: Side,
) -> NDArray[np.uint8]:
    """给同一匹配叠加同色 Mask 和共享 ID，未匹配保留图内编号。

    Args:
        image_rgb: 形状为 (H, W, 3) 的 uint8 RGB 图像。
        masks: 原图坐标下形状为 (N, H, W) 的布尔 Mask。
        result: 两侧匹配和未匹配实例。
        side: 要绘制的相机侧。

    Returns:
        带匹配标注的 RGB 预览面板。
    """
    annotations: list[tuple[RailInstance, str, tuple[int, int, int]]] = []
    for index, match in enumerate(result["matches"]):
        hue = (index * _COLOR_HUE_STEP) % 1
        red, green, blue = hsv_to_rgb(hue, 0.8, 1)
        color = round(red * 255), round(green * 255), round(blue * 255)
        annotations.append((match[side], match["shared_id"], color))
    unmatched = (
        result["unmatched_south"] if side == "south" else result["unmatched_north"]
    )
    annotations.extend(
        (instance, instance["instance_id"], _UNMATCHED_COLOR) for instance in unmatched
    )
    overlay = image_rgb.copy()
    for instance, _, color in annotations:
        mask = masks[instance["mask_index"]]
        pixels = (1 - _MASK_OPACITY) * overlay[mask] + _MASK_OPACITY * np.asarray(color)
        overlay[mask] = pixels.astype(np.uint8)

    height, width = overlay.shape[:2]
    factor = _PREVIEW_PANEL_WIDTH / width
    panel_size = _PREVIEW_PANEL_WIDTH, max(1, round(height * factor))
    preview = cv2.resize(overlay, panel_size, interpolation=cv2.INTER_AREA)
    main_y = preview.shape[0] // 2
    cv2.line(preview, (0, main_y), (_PREVIEW_PANEL_WIDTH - 1, main_y), (220, 220, 220))
    for index, (instance, label, color) in enumerate(annotations):
        mask = masks[instance["mask_index"]]
        center = instance["center_xy"]
        fraction = _LABEL_ROW_FRACTIONS[index % len(_LABEL_ROW_FRACTIONS)]
        label_row = round((height - 1) * fraction)
        section = measure_section(mask[label_row])
        label_x, reason = section.center_x, section.reason
        if center is not None:
            x, y = center
            point = round(x * factor), main_y
            cv2.circle(preview, point, 3, color, -1, cv2.LINE_AA)
            if reason is not None:
                label_x, label_row = x, y
        elif reason is not None:
            rows = np.flatnonzero(mask.any(axis=1))
            if not len(rows):
                continue
            label_row = int(rows[len(rows) // 2])
            columns = np.flatnonzero(mask[label_row])
            label_x = float(columns[len(columns) // 2])

        label_point = round(label_x * factor), round(label_row * factor)
        cv2.putText(
            preview, label, label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0),
            3, cv2.LINE_AA,
        )
        cv2.putText(
            preview, label, label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color,
            1, cv2.LINE_AA,
        )
    cv2.putText(
        preview, side.capitalize(), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return preview


def _unmatched_without_partner(
    masks: NDArray[np.bool_], side: Side
) -> list[UnmatchedRail]:
    """对侧为空时，只测主截面供本侧编号和显示。"""
    main_y = section_rows(masks.shape[1])[MAIN_SECTION_INDEX]
    sections = [measure_section(mask[main_y]) for mask in masks]
    centers = np.asarray([section.center_x for section in sections], dtype=np.float64)
    positions = -centers if side == "south" else centers
    unmatched: list[UnmatchedRail] = []
    for order, sorted_index in enumerate(np.argsort(positions, kind="stable"), start=1):
        mask_index = int(sorted_index)
        section = sections[mask_index]
        unmatched.append({
            "instance_id": f"{side[0].upper()}{order:02d}",
            "mask_index": mask_index,
            "center_xy": None if section.reason else (section.center_x, main_y),
            "reason": section.reason or "other_side_empty",
        })
    return unmatched


def _compose_matching_preview(
    south_panel: NDArray[np.uint8],
    north_panel: NDArray[np.uint8],
    result: RailMatchingResult,
) -> NDArray[np.uint8]:
    """在白底对照图的左上方留出独立统计区。"""
    match_count = len(result["matches"])
    south_unmatched_count = len(result["unmatched_south"])
    north_unmatched_count = len(result["unmatched_north"])
    south_count = match_count + south_unmatched_count
    north_count = match_count + north_unmatched_count
    try:
        font = ImageFont.truetype(_SUMMARY_FONT_NAME, _SUMMARY_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default(size=_SUMMARY_FONT_SIZE)
        lines = (
            (
                f"South: {south_count} instances   {match_count} matched",
                _SUMMARY_TEXT_COLOR,
            ),
            (
                f"North: {north_count} instances   {match_count} matched",
                _SUMMARY_TEXT_COLOR,
            ),
            (
                f"Unpaired: South: {south_unmatched_count}   "
                f"North: {north_unmatched_count}",
                _SUMMARY_MISSED_COLOR,
            ),
            ("Wrong matches: Unavailable without pair labels", _SUMMARY_TEXT_COLOR),
        )
    else:
        lines = (
            (f"South: 实例 {south_count} 个   匹配 {match_count} 个", _SUMMARY_TEXT_COLOR),
            (f"North: 实例 {north_count} 个   匹配 {match_count} 个", _SUMMARY_TEXT_COLOR),
            (
                f"漏配对: South: {south_unmatched_count} 个   "
                f"North: {north_unmatched_count} 个",
                _SUMMARY_MISSED_COLOR,
            ),
            ("误配对: 当前图对缺少配对标注，无法统计", _SUMMARY_TEXT_COLOR),
        )
    panels = compose_image_pair(
        south_panel, north_panel, background_value=_PREVIEW_BACKGROUND
    )
    height, width = panels.shape[:2]
    header_height = _SUMMARY_MARGIN * 2 + _SUMMARY_LINE_HEIGHT * len(lines)
    canvas = Image.new(
        "RGB", (width, height + header_height), (_PREVIEW_BACKGROUND,) * 3
    )
    canvas.paste(Image.fromarray(panels), (0, header_height))
    drawer = ImageDraw.Draw(canvas)
    for index, (line, color) in enumerate(lines):
        top = _SUMMARY_MARGIN + index * _SUMMARY_LINE_HEIGHT
        drawer.text((_SUMMARY_MARGIN, top), line, fill=color, font=font)
    return np.asarray(canvas)


def preview_matching_results(
    matcher: Callable[
        [Path, Path, NDArray[np.bool_], NDArray[np.bool_]], RailMatchingResult
    ],
    *,
    show_missed_matches_only: bool,
) -> None:
    """用统一的 OpenCV 预览显示南北匹配结果。

    任一侧没有 Mask 实例时，直接预览已有实例。

    Args:
        matcher: 接收南北相对路径及对应 Mask 的匹配函数。
        show_missed_matches_only: 为 True 时仅显示至少一侧有未匹配实例的图对。
    """
    dataset_path = load_dataset_relative_path()
    pairs = match_pair_image(dataset_path)
    names = zip(pairs["south"], pairs["north"], strict=True)
    print("关闭窗口或按空格查看下一对，按 Esc 退出。灰色为未匹配实例。")
    if show_missed_matches_only:
        print("当前只显示至少一侧有未匹配钢轨的图对。")
    has_displayed_pair = False
    result: RailMatchingResult
    for pair_index, (south_name, north_name) in enumerate(names, start=1):
        south_path = dataset_path / "south" / f"{south_name}.png"
        north_path = dataset_path / "north" / f"{north_name}.png"
        south_image, south_masks = load_labelme_view(PROJECT_ROOT / south_path)
        north_image, north_masks = load_labelme_view(PROJECT_ROOT / north_path)
        if len(south_masks) and len(north_masks):
            result = matcher(south_path, north_path, south_masks, north_masks)
        else:
            result = {
                "south_image_path": south_path.as_posix(),
                "north_image_path": north_path.as_posix(),
                "matches": [],
                "unmatched_south": _unmatched_without_partner(south_masks, "south"),
                "unmatched_north": _unmatched_without_partner(north_masks, "north"),
            }
        unmatched = [*result["unmatched_south"], *result["unmatched_north"]]
        if show_missed_matches_only and not unmatched:
            continue

        south_panel = render_side(south_image, south_masks, result, "south")
        north_panel = render_side(north_image, north_masks, result, "north")
        preview = _compose_matching_preview(south_panel, north_panel, result)
        title = (
            f"{pair_index}/{len(pairs['south'])} | {south_name} | {north_name}"
            f" | matched {len(result['matches'])}"
            f" | S unmatched {len(result['unmatched_south'])}"
            f" | N unmatched {len(result['unmatched_north'])}"
        )
        for instance in unmatched:
            print(f"{instance['instance_id']} {instance['reason']}")
        has_displayed_pair = True
        if not review_rgb_image(
            preview, title, should_maximize=True, background_value=_PREVIEW_BACKGROUND,
            should_align_top=True,
        ):
            break
    if show_missed_matches_only and not has_displayed_pair:
        print("本次没有含未匹配钢轨的图对。")
