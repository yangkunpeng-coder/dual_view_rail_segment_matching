"""按已标注钢轨的逐行边界遮除有效区域之外的原图像素。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from data_preprocess.common.labelme import load_labelme_annotation
from data_preprocess.common.rail_corridor import make_corridor_mask
from data_preprocess.common.roi_config import CorridorSettings, load_roi_settings

_BORDER_TOLERANCE_PX = 0.001
_MIN_DOUBLE_AREA_PX = 1e-9
_MIN_POLYGON_VERTICES = 3


@dataclass(frozen=True)
class CropSample:
    """供导出和审核使用的 RGB 图像、有效区域和独立实例标注。"""

    original_image: NDArray[np.uint8]
    processed_image: NDArray[np.uint8]
    valid_mask: NDArray[np.uint8]
    polygons: list[NDArray[np.float64]]
    annotation: dict[str, object]
    margins: tuple[int, int]


def crop_valid_image(
    image_path: str | Path, json_path: str | Path
) -> NDArray[np.uint8]:
    """按 LabelMe 多边形提取带状有效区域，保持原画布尺寸。

    Args:
        image_path: 原始图片路径。
        json_path: 与图片对应的 LabelMe JSON 路径。

    Returns:
        形状为 (H, W, 3) 的 RGB uint8 数组，区域外填充配置中的颜色。

    Raises:
        OSError: 无法读取图片或标注。
        ValueError: 标注尺寸或多边形无效，或没有目标实例。
    """
    settings = load_roi_settings().corridor
    return load_crop_sample(Path(image_path), Path(json_path), settings).processed_image


def load_crop_sample(
    image_path: Path, json_path: Path, settings: CorridorSettings
) -> CropSample:
    """读取单张样本，生成有效区域并校验每个实例的像素覆盖。

    Args:
        image_path: 原始图片路径。
        json_path: 对应标注文件路径。
        settings: 有效区域计算参数。

    Returns:
        原图、处理后 RGB 图、0/1 有效区域和原图像素坐标下的独立多边形。

    Raises:
        OSError: 无法读取图片或标注。
        ValueError: 标注、有效区域或填充结果校验失败。
    """
    with Image.open(image_path) as image:
        image_size = image.size
        original_image = np.array(image.convert("RGB"), dtype=np.uint8)
    annotation = load_labelme_annotation(json_path, image_size)
    polygons = _read_polygons(annotation, json_path, image_size, settings.target_label)
    if not polygons:
        raise ValueError(f"{json_path} 没有 {settings.target_label} 多边形")

    union = np.full(original_image.shape[:2], 0, dtype=np.uint8)
    instance_masks: list[NDArray[np.uint8]] = []
    for instance_index, points in enumerate(polygons, start=1):
        instance_mask = np.zeros_like(union)
        vertices = np.rint(points).astype(np.int32)
        cv2.fillPoly(instance_mask, [vertices], 1)
        if not cv2.countNonZero(instance_mask):
            raise ValueError(f"{json_path} 的第 {instance_index} 个实例无可见像素")
        instance_masks.append(instance_mask)
        np.bitwise_or(union, instance_mask, out=union)

    image_key = f"{image_path.parent.name}/{image_path.name}"
    margins = settings.margin_overrides.get(image_key, settings.margins)
    valid_mask = make_corridor_mask(union, settings, margins)
    outside = valid_mask == 0
    for instance_index, mask in enumerate(instance_masks, start=1):
        lost_count = cv2.countNonZero(cv2.bitwise_and(mask, outside.view(np.uint8)))
        if lost_count:
            raise ValueError(f"{json_path} 的第 {instance_index} 个实例丢失 {lost_count} 像素")

    processed_image = original_image.copy()
    processed_image[outside] = settings.outside_fill_rgb
    if np.any(processed_image[outside] != settings.outside_fill_rgb):
        raise ValueError(f"{image_path} 的区域外像素填充校验失败")
    return CropSample(
        original_image, processed_image, valid_mask, polygons, annotation, margins
    )


def _read_polygons(
    annotation: dict[str, object],
    json_path: Path,
    image_size: tuple[int, int],
    target_label: str,
) -> list[NDArray[np.float64]]:
    """校验目标类别的原图坐标，每个 Polygon 独立保留。"""
    polygons: list[NDArray[np.float64]] = []
    image_bounds = np.asarray(image_size, dtype=np.float64)
    shapes = annotation["shapes"]
    if not isinstance(shapes, list):
        raise ValueError(f"{json_path} 的 shapes 必须为列表")
    for shape_index, shape in enumerate(shapes, start=1):
        context = f"{json_path} 的第 {shape_index} 条标注"
        if not isinstance(shape, dict) or not isinstance(shape.get("label"), str):
            raise ValueError(f"{context} 缺少有效的 label")
        if shape["label"] != target_label:
            continue
        if shape.get("shape_type") != "polygon":
            raise ValueError(f"{context} 必须是 Polygon")
        try:
            points = np.asarray(shape.get("points"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{context} 的坐标无效") from error
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < _MIN_POLYGON_VERTICES
            or not np.isfinite(points).all()
            or (points < -_BORDER_TOLERANCE_PX).any()
            or (points > image_bounds + _BORDER_TOLERANCE_PX).any()
        ):
            raise ValueError(f"{context} 的顶点不足、无效或越界")

        # 原始 JSON 仍保留原值，只消除用于栅格化和归一化的边缘浮点误差。
        points = np.clip(points, 0, image_bounds)
        x, y = points.T
        double_area = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        if abs(double_area) <= _MIN_DOUBLE_AREA_PX:
            raise ValueError(f"{context} 的多边形面积为零")
        polygons.append(points)
    return polygons
