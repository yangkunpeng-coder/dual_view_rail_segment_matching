"""生成钢轨有效区域对照图。"""

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from data_preprocess.crop_image import CropSample

_PANEL_MAX_WIDTH = 1500
_HEADER_HEIGHT = 42
_OUTSIDE_SHADE = 112
_POLYGON_COLOR = (30, 240, 90)
_LEFT_EDGE_COLOR = (40, 120, 255)
_RIGHT_EDGE_COLOR = (255, 75, 60)


def render_crop_preview(sample: "CropSample", image_name: str) -> Image.Image:
    """绘制原图边界标记和处理后图像的左右对照。

    Args:
        sample: 已通过像素完整性检查的裁剪样本。
        image_name: 用于预览标题的文件名。

    Returns:
        RGB 预览图，左侧显示原图、绿色轮廓及蓝红边界，右侧显示处理结果。
    """
    height, width = sample.valid_mask.shape
    panel_width = min(width, _PANEL_MAX_WIDTH)
    panel_height = max(1, round(height * panel_width / width))
    panel_size = (panel_width, panel_height)
    original = Image.fromarray(sample.original_image).resize(panel_size).convert("RGBA")
    region = Image.fromarray(sample.valid_mask * 255).resize(
        panel_size, Image.Resampling.NEAREST
    )
    shade = Image.new("RGBA", panel_size)
    shade.putalpha(region.point(lambda value: 0 if value else _OUTSIDE_SHADE))
    original.alpha_composite(shade)

    draw = ImageDraw.Draw(original)
    for polygon in sample.polygons:
        points = [
            (x * panel_width / width, y * panel_height / height) for x, y in polygon
        ]
        draw.line([*points, points[0]], fill=_POLYGON_COLOR, width=2)
    _draw_boundaries(draw, region)

    processed = Image.fromarray(sample.processed_image).resize(panel_size)
    canvas = Image.new("RGB", (2 * panel_width, panel_height + _HEADER_HEIGHT))
    canvas.paste(original, (0, _HEADER_HEIGHT))
    canvas.paste(processed, (panel_width, _HEADER_HEIGHT))
    title = (
        f"{image_name} | instances={len(sample.polygons)}"
        f" | margin L/R={sample.margins[0]}/{sample.margins[1]} px"
    )
    header = ImageDraw.Draw(canvas)
    header.text((8, 4), title, fill="white")
    header.text((8, 22), "Original + ROI", fill="white")
    header.text((panel_width + 8, 22), "Processed", fill="white")
    return canvas


def _draw_boundaries(draw: ImageDraw.ImageDraw, region: Image.Image) -> None:
    """沿每段连续有效行绘制左右边界曲线。"""
    mask = np.asarray(region)
    width = mask.shape[1]
    rows = np.flatnonzero(mask.any(axis=1))
    breaks = np.flatnonzero(np.diff(rows) > 1) + 1
    for segment in np.split(rows, breaks):
        if segment.size == 0:
            continue
        band = mask[segment]
        left = band.argmax(axis=1)
        right = width - 1 - band[:, ::-1].argmax(axis=1)
        draw.line(
            list(zip(left.tolist(), segment.tolist())), fill=_LEFT_EDGE_COLOR, width=2
        )
        draw.line(
            list(zip(right.tolist(), segment.tolist())), fill=_RIGHT_EDGE_COLOR, width=2
        )
