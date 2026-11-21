"""Mask 级实例去重：抑制同一张图中同一根钢轨的重复预测。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MaskNmsResult:
    """去重结果，索引均指向调用方传入的原始实例数组。"""

    keep_indices: NDArray[np.intp]
    suppressed_indices: NDArray[np.intp]
    duplicate_of: dict[int, int]


def suppress_duplicate_masks(
    masks: NDArray[np.bool_],
    scores: NDArray[np.floating],
    *,
    iou_threshold: float = 0.60,
    min_area_px: int = 8,
    boxes: NDArray[np.floating] | None = None,
) -> MaskNmsResult:
    """按置信度贪心抑制高 Mask IoU 的重复实例。

    去重只使用预测 Mask、分数和包围盒，不使用 GT。包围盒只用于预筛：两个 Mask 的
    IoU 不可能大于其包围盒 IoU，因此框 IoU 不超过阈值时可以安全跳过 Mask 求交。

    Args:
        masks: 原图尺寸的布尔实例掩码，形状为 (N, H, W)。
        scores: 各实例置信度，形状为 (N,)，取值在 [0, 1]。
        iou_threshold: Mask IoU 抑制阈值，取值在 [0, 1]。
        min_area_px: 小于该像素数的实例不参与去重，始终保留。
        boxes: 可选的实例包围盒，形状为 (N, 4)，xyxy 格式。调用方已有检测框时传入可
            避免重新扫描整图；包围盒必须覆盖对应 Mask。

    Returns:
        keep_indices 为按分数降序保留的原始索引；suppressed_indices 为被抑制的原始
        索引；duplicate_of 记录被抑制实例对应的保留实例。

    Raises:
        ValueError: 掩码、分数、包围盒或阈值无效。
    """
    _validate_inputs(masks, scores, iou_threshold, min_area_px, boxes)
    count = len(masks)
    if count == 0:
        empty = np.empty(0, dtype=np.intp)
        return MaskNmsResult(empty, empty.copy(), {})

    areas = _resolve_areas(masks, boxes)
    bounds = _resolve_boxes(masks, boxes)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    kept: list[int] = []
    suppressed: list[int] = []
    duplicate_of: dict[int, int] = {}

    for index in order.tolist():
        if areas[index] < min_area_px:
            kept.append(index)
            continue
        duplicate_of_index: int | None = None
        for keeper in kept:
            if areas[keeper] < min_area_px:
                continue
            if _box_iou(bounds[index], bounds[keeper]) <= iou_threshold:
                continue
            if _mask_iou(
                masks[index], masks[keeper], bounds[index], bounds[keeper],
                int(areas[index]), int(areas[keeper]),
            ) <= iou_threshold:
                continue
            duplicate_of_index = keeper
            break
        if duplicate_of_index is None:
            kept.append(index)
        else:
            suppressed.append(index)
            duplicate_of[index] = duplicate_of_index

    return MaskNmsResult(
        keep_indices=np.asarray(kept, dtype=np.intp),
        suppressed_indices=np.asarray(suppressed, dtype=np.intp),
        duplicate_of=duplicate_of,
    )


def _validate_inputs(
    masks: NDArray[np.bool_],
    scores: NDArray[np.floating],
    iou_threshold: float,
    min_area_px: int,
    boxes: NDArray[np.floating] | None,
) -> None:
    """校验去重输入，错误信息直接指出调用方需要修正的字段。"""
    if masks.ndim != 3:
        raise ValueError("masks 必须为 (N, H, W) 的三维数组")
    if masks.dtype != np.bool_:
        raise ValueError("masks 必须为布尔数组")
    if scores.shape != (len(masks),):
        raise ValueError("scores 长度必须与 masks 实例数一致")
    if not np.isfinite(scores).all():
        raise ValueError("scores 必须为有限值")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores 必须位于 [0, 1]")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold 必须位于 [0, 1]")
    if min_area_px < 0:
        raise ValueError("min_area_px 不能为负数")
    if boxes is not None:
        if boxes.shape != (len(masks), 4):
            raise ValueError("boxes 必须为 (N, 4) 的 xyxy 数组")
        if not np.isfinite(boxes).all():
            raise ValueError("boxes 必须为有限值")


def _resolve_boxes(
    masks: NDArray[np.bool_], boxes: NDArray[np.floating] | None
) -> NDArray[np.float64]:
    """返回用于预筛和求交的 xyxy 包围盒。"""
    if boxes is None:
        return _mask_boxes(masks)
    return np.asarray(boxes, dtype=np.float64)


def _resolve_areas(
    masks: NDArray[np.bool_], boxes: NDArray[np.floating] | None
) -> NDArray[np.int64]:
    """统计实例像素数；有包围盒时只在盒内统计以减少全图扫描。"""
    if boxes is None:
        return np.count_nonzero(masks, axis=(1, 2)).astype(np.int64)
    height, width = masks.shape[1:]
    areas = np.zeros(len(masks), dtype=np.int64)
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = _clip_box(box, width, height)
        areas[index] = np.count_nonzero(masks[index, y0:y1, x0:x1])
    return areas


def _clip_box(
    box: NDArray[np.floating], width: int, height: int
) -> tuple[int, int, int, int]:
    """把浮点 xyxy 包围盒裁剪到图像范围内并转为整数开区间。"""
    x0 = max(0, min(int(np.floor(box[0])), width))
    y0 = max(0, min(int(np.floor(box[1])), height))
    x1 = max(0, min(int(np.ceil(box[2])), width))
    y1 = max(0, min(int(np.ceil(box[3])), height))
    return x0, y0, max(x1, x0), max(y1, y0)


def _mask_boxes(masks: NDArray[np.bool_]) -> NDArray[np.float64]:
    """扫描实例 Mask 得到紧包围盒，格式为 xyxy 的开区间坐标。"""
    count = len(masks)
    boxes = np.zeros((count, 4), dtype=np.float64)
    for index, mask in enumerate(masks):
        rows = np.flatnonzero(mask.any(axis=1))
        if rows.size == 0:
            continue
        columns = np.flatnonzero(mask.any(axis=0))
        boxes[index] = (
            float(columns[0]), float(rows[0]),
            float(columns[-1] + 1), float(rows[-1] + 1),
        )
    return boxes


def _box_iou(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """计算两个 xyxy 包围盒的 IoU。"""
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    first_area = max(first[2] - first[0], 0.0) * max(first[3] - first[1], 0.0)
    second_area = max(second[2] - second[0], 0.0) * max(
        second[3] - second[1], 0.0
    )
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _mask_iou(
    first: NDArray[np.bool_],
    second: NDArray[np.bool_],
    first_box: NDArray[np.float64],
    second_box: NDArray[np.float64],
    first_area: int,
    second_area: int,
) -> float:
    """在包围盒交集内计算两个 Mask 的 IoU。"""
    height, width = first.shape
    first_x0, first_y0, first_x1, first_y1 = _clip_box(
        first_box, width, height
    )
    second_x0, second_y0, second_x1, second_y1 = _clip_box(
        second_box, width, height
    )
    x0 = max(first_x0, second_x0)
    y0 = max(first_y0, second_y0)
    x1 = min(first_x1, second_x1)
    y1 = min(first_y1, second_y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    first_crop = first[y0:y1, x0:x1]
    second_crop = second[y0:y1, x0:x1]
    intersection = int(np.count_nonzero(first_crop & second_crop))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0
