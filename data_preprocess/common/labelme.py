"""读取原图并将 LabelMe 钢轨多边形转换成实例 Mask。"""

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw


class RailPolygon(TypedDict):
    """一个 LabelMe 钢轨多边形及其图内分组编号。"""

    points: list[tuple[float, float]]
    group_id: int | None


def load_labelme_annotation(
    annotation_path: Path, image_size: tuple[int, int]
) -> dict[str, object]:
    """读取 LabelMe 标注并核对画布尺寸和 shapes 字段。

    Args:
        annotation_path: JSON 文件路径。
        image_size: 原图宽和高，单位为像素。

    Returns:
        保留原始元信息的标注字典。

    Raises:
        OSError: 无法读取 JSON。
        ValueError: JSON 无效或标注尺寸与原图不一致。
    """
    content = annotation_path.read_text(encoding="utf-8-sig")
    return parse_labelme_annotation(content, annotation_path, image_size)


def parse_labelme_annotation(
    content: str, annotation_path: Path, image_size: tuple[int, int]
) -> dict[str, object]:
    """解析 LabelMe 文本，并校验其画布和 shapes 字段。

    Args:
        content: 解码后的 LabelMe JSON 文本。
        annotation_path: 标注路径，用于错误定位。
        image_size: 原图宽和高，单位为像素。

    Returns:
        保留原始元信息的标注字典。

    Raises:
        ValueError: JSON 无效或标注尺寸与原图不一致。
    """
    annotation = json.loads(content)
    if not isinstance(annotation, dict) or not isinstance(
        annotation.get("shapes"), list
    ):
        raise ValueError(f"{annotation_path} 的 shapes 必须为列表")
    annotation_size = (annotation.get("imageWidth"), annotation.get("imageHeight"))
    if annotation_size != image_size:
        raise ValueError(
            f"{annotation_path} 的标注尺寸 {annotation_size}"
            f"与图片尺寸 {image_size} 不一致"
        )
    return annotation


def load_rail_polygons(
    image_path: Path, image_size: tuple[int, int]
) -> list[RailPolygon]:
    """读取与图片同名的 LabelMe 钢轨多边形。

    Args:
        image_path: 原始图片路径。
        image_size: 原始图片的宽度和高度，单位为像素。

    Returns:
        按标注顺序排列的多边形，顶点为原图坐标。

    Raises:
        FileNotFoundError: 缺少同名 JSON 文件。
        ValueError: 标注尺寸不符或钢轨多边形无效。
    """
    annotation_path = image_path.with_suffix(".json")
    annotation = load_labelme_annotation(annotation_path, image_size)
    return extract_rail_polygons(annotation, annotation_path)


def extract_rail_polygons(
    annotation: dict[str, object], annotation_path: Path
) -> list[RailPolygon]:
    """从已校验的 LabelMe 标注提取钢轨多边形。

    Args:
        annotation: 由 load_labelme_annotation 读取的标注字典。
        annotation_path: 标注路径，用于定位无效标注。

    Returns:
        按原标注顺序排列的钢轨多边形。
    Raises:
        ValueError: 钢轨标注不是有效的 Polygon。
    """
    shapes = annotation.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError(f"{annotation_path} 的 shapes 必须为列表")

    polygons: list[RailPolygon] = []
    for shape in shapes:
        if not isinstance(shape, dict) or not isinstance(shape.get("label"), str):
            raise ValueError(f"{annotation_path} 的标注必须包含字符串类型的 label")
        if shape["label"] != "rail":
            continue
        points = shape.get("points")
        if (
            shape.get("shape_type", "polygon") != "polygon"
            or not isinstance(points, (list, tuple))
            or len(points) < 3
        ):
            raise ValueError(f"{annotation_path} 的 rail 标注需要至少三个多边形顶点")

        vertices: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"{annotation_path} 的 rail 顶点必须包含 x、y 坐标")
            x, y = point
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError(f"{annotation_path} 的 rail 顶点坐标必须为数值")
            vertices.append((x, y))

        group_id = shape.get("group_id")
        if group_id is not None and not isinstance(group_id, int):
            raise ValueError(f"{annotation_path} 的 group_id 必须为整数或 null")
        polygons.append({"points": vertices, "group_id": group_id})
    return polygons


def load_rail_image_and_mask(
    image_path: Path,
) -> tuple[NDArray[np.uint8], NDArray[np.int32]]:
    """加载一张图片及其同名 LabelMe 标注。

    Args:
        image_path: 磁盘上的图片路径。标注是同目录下同名的 JSON 文件。

    Returns:
        RGB 图像为形状 (H, W, 3) 的 uint8 数组，像素值为 0 至 255。
        实例 Mask 为形状 (H, W) 的 int32 数组，0 表示背景，正整数
        表示图内实例。相同非空 group_id 共用一个 ID。多边形重叠时，
        按标注顺序使用最后绘制的实例 ID。

    Raises:
        FileNotFoundError: 缺少图片或对应标注文件。
        ValueError: JSON 无法解析、标注尺寸不符或钢轨多边形无效。
    """
    with Image.open(image_path) as image:
        image_size = image.size
        image_data = np.array(image.convert("RGB"), dtype=np.uint8)

    polygons = load_rail_polygons(image_path, image_size)
    mask_image = Image.new("I", image_size)
    drawer = ImageDraw.Draw(mask_image)
    grouped_ids: dict[int, int] = {}
    instance_count = 0

    for polygon in polygons:
        group_id = polygon["group_id"]
        if group_id is not None and group_id in grouped_ids:
            instance_id = grouped_ids[group_id]
        else:
            instance_count += 1
            instance_id = instance_count
            if group_id is not None:
                grouped_ids[group_id] = instance_id

        drawer.polygon(polygon["points"], fill=instance_id)

    return image_data, np.array(mask_image, dtype=np.int32)
