"""预测 Mask 的只读质量特征：供诊断和匹配参考，不修改 Mask 本身。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

_CONNECTIVITY = 8


@dataclass(frozen=True)
class MaskQuality:
    """单个预测实例的结构特征，坐标均为原图像素。"""

    area_px: int
    component_count: int
    main_component_ratio: float
    bbox: tuple[int, int, int, int]
    touches_border: bool


def measure_mask_quality(
    mask: NDArray[np.bool_],
    box: tuple[float, float, float, float] | None = None,
) -> MaskQuality:
    """统计一个实例 Mask 的连通域与包围盒特征。

    Args:
        mask: 原图尺寸的二维布尔 Mask。
        box: 可选的紧包围盒 xyxy。传入时只在盒内统计，要求盒子覆盖整个 Mask；
            省略时扫描整图求包围盒。

    Returns:
        面积、连通域数量、主连通域面积占比、紧包围盒和是否接触图像边界。
        空 Mask 返回全零特征。

    Raises:
        ValueError: 输入不是二维布尔数组。
    """
    if mask.ndim != 2:
        raise ValueError("mask 必须为二维数组")
    if mask.dtype != np.bool_:
        raise ValueError("mask 必须为布尔数组")
    height, width = mask.shape
    if box is None:
        region, x_offset, y_offset = mask, 0, 0
    else:
        x0, y0, x1, y1 = _clip_box(box, width, height)
        region = mask[y0:y1, x0:x1]
        x_offset, y_offset = x0, y0
    area = int(np.count_nonzero(region))
    if area == 0:
        return MaskQuality(0, 0, 0.0, (0, 0, 0, 0), False)

    rows = np.flatnonzero(region.any(axis=1))
    columns = np.flatnonzero(region.any(axis=0))
    x0 = x_offset + int(columns[0])
    y0 = y_offset + int(rows[0])
    x1 = x_offset + int(columns[-1]) + 1
    y1 = y_offset + int(rows[-1]) + 1
    crop = region[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        crop.astype(np.uint8), _CONNECTIVITY
    )
    component_count = max(component_count - 1, 0)
    if component_count == 0:
        return MaskQuality(area, 0, 0.0, (x0, y0, x1, y1), False)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    main_ratio = float(component_areas.max()) / area
    touches_border = x0 == 0 or y0 == 0 or x1 == width or y1 == height
    return MaskQuality(
        area_px=area,
        component_count=component_count,
        main_component_ratio=main_ratio,
        bbox=(x0, y0, x1, y1),
        touches_border=touches_border,
    )


def measure_mask_qualities(
    masks: NDArray[np.bool_],
    boxes: NDArray[np.floating] | None = None,
) -> tuple[MaskQuality, ...]:
    """逐实例统计质量特征。

    Args:
        masks: 原图尺寸的 (N, H, W) 布尔实例掩码。
        boxes: 可选的 (N, 4) xyxy 包围盒，传入时只在盒内统计。

    Returns:
        与实例顺序一致的只读特征元组。

    Raises:
        ValueError: 输入不是三维布尔数组，或包围盒数量不一致。
    """
    if masks.ndim != 3:
        raise ValueError("masks 必须为 (N, H, W) 的三维数组")
    if masks.dtype != np.bool_:
        raise ValueError("masks 必须为布尔数组")
    if boxes is None:
        return tuple(measure_mask_quality(mask) for mask in masks)
    if boxes.shape != (len(masks), 4):
        raise ValueError("boxes 必须为 (N, 4) 的 xyxy 数组")
    return tuple(
        measure_mask_quality(mask, tuple(box))
        for mask, box in zip(masks, boxes, strict=True)
    )


def _clip_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """把浮点 xyxy 包围盒裁剪到图像范围内并转为整数开区间。"""
    x0 = max(0, min(int(np.floor(box[0])), width))
    y0 = max(0, min(int(np.floor(box[1])), height))
    x1 = max(0, min(int(np.ceil(box[2])), width))
    y1 = max(0, min(int(np.ceil(box[3])), height))
    return x0, y0, max(x1, x0), max(y1, y0)

