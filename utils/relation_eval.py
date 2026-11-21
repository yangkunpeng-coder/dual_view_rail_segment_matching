"""关系级匹配评测：不要求预测实例下标与 GT 下标一致。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Relation = tuple[int, int]


@dataclass(frozen=True)
class RelationScore:
    """一个图对在一个门限下的关系级计数与覆盖情况。"""

    gt_relations: int
    pred_relations: int
    true_positive: int
    false_positive: int
    false_negative: int
    extra_on_single: int
    unmapped_extra: int
    pair_exact: bool
    south_gt_total: int
    south_gt_covered: int
    north_gt_total: int
    north_gt_covered: int
    south_pred_total: int
    south_pred_covered: int
    north_pred_total: int
    north_pred_covered: int
    matched_relations: tuple[
        tuple[int, int, int, int, float, float], ...
    ] = ()

    @property
    def precision(self) -> float:
        """预测关系中被 GT 支持的比例。"""
        return _safe_div(self.true_positive, self.pred_relations)

    @property
    def recall(self) -> float:
        """GT 关系中被预测找回的比例。"""
        return _safe_div(self.true_positive, self.gt_relations)

    @property
    def f1(self) -> float:
        """关系级 F1。"""
        denominator = self.precision + self.recall
        return (
            2 * self.precision * self.recall / denominator
            if denominator else 0.0
        )


def score_relations(
    gt_relations: set[Relation],
    pred_relations: set[Relation],
    south_iou: NDArray[np.float64],
    north_iou: NDArray[np.float64],
    single_south: set[int],
    single_north: set[int],
    threshold: float,
) -> RelationScore:
    """在预测关系与 GT 关系之间做一对一最大匹配并计分。

    两个关系满足两侧 Mask IoU 都达到门限时才能连边。匹配只要求“预测 Mask 与 GT
    Mask 是同一根钢轨”，不要求两边实例下标相等。

    Args:
        gt_relations: GT 的 (south_mask_index, north_mask_index) 集合。
        pred_relations: 预测的 (south_mask_index, north_mask_index) 集合。
        south_iou: 南侧预测 × GT 的 Mask IoU，形状为 (P_s, G_s)。
        north_iou: 北侧预测 × GT 的 Mask IoU，形状为 (P_n, G_n)。
        single_south: 南侧单侧 GT 实例下标集合。
        single_north: 北侧单侧 GT 实例下标集合。
        threshold: 关系连边的 IoU 门限。

    Returns:
        关系级 TP/FP/FN、单侧误配、未映射预测和实例覆盖计数。

    Raises:
        ValueError: IoU 矩阵维度、门限或关系下标无效。
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须位于 [0, 1]")
    _validate_matrix(south_iou, "south_iou")
    _validate_matrix(north_iou, "north_iou")
    gt_list = sorted(gt_relations)
    pred_list = sorted(pred_relations)
    _validate_indices(
        gt_list, south_iou.shape[1], north_iou.shape[1], "GT"
    )
    _validate_indices(
        pred_list, south_iou.shape[0], north_iou.shape[0], "预测"
    )

    adjacency: list[list[tuple[float, int]]] = [[] for _ in gt_list]
    for gt_index, (gt_south, gt_north) in enumerate(gt_list):
        for pred_index, (pred_south, pred_north) in enumerate(pred_list):
            south_value = south_iou[pred_south, gt_south]
            north_value = north_iou[pred_north, gt_north]
            if south_value >= threshold and north_value >= threshold:
                adjacency[gt_index].append(
                    (min(south_value, north_value), pred_index)
                )

    match_pred = [-1] * len(pred_list)
    for gt_index in range(len(gt_list)):
        _augment(gt_index, adjacency, match_pred, set())

    true_positive = sum(1 for value in match_pred if value >= 0)
    matched_pred = {value for value in match_pred if value >= 0}
    false_negative = len(gt_list) - true_positive
    false_positive = len(pred_list) - true_positive
    matched_relations: list[tuple[int, int, int, int, float, float]] = []
    for pred_index, gt_index in enumerate(match_pred):
        if gt_index < 0:
            continue
        gt_south, gt_north = gt_list[gt_index]
        pred_south, pred_north = pred_list[pred_index]
        matched_relations.append((
            gt_south, gt_north, pred_south, pred_north,
            float(south_iou[pred_south, gt_south]),
            float(north_iou[pred_north, gt_north]),
        ))
    extra_on_single = 0
    unmapped_extra = 0
    for pred_index, (pred_south, pred_north) in enumerate(pred_list):
        if pred_index in matched_pred:
            continue
        touches_single = any(
            south_iou[pred_south, gt_index] >= threshold
            for gt_index in single_south
        ) or any(
            north_iou[pred_north, gt_index] >= threshold
            for gt_index in single_north
        )
        if touches_single:
            extra_on_single += 1
        else:
            unmapped_extra += 1

    return RelationScore(
        gt_relations=len(gt_list),
        pred_relations=len(pred_list),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        extra_on_single=extra_on_single,
        unmapped_extra=unmapped_extra,
        pair_exact=(
            true_positive == len(gt_list) and false_positive == 0
        ),
        south_gt_total=int(south_iou.shape[1]),
        south_gt_covered=_covered_count(south_iou, axis=0, threshold=threshold),
        north_gt_total=int(north_iou.shape[1]),
        north_gt_covered=_covered_count(north_iou, axis=0, threshold=threshold),
        south_pred_total=int(south_iou.shape[0]),
        south_pred_covered=_covered_count(south_iou, axis=1, threshold=threshold),
        north_pred_total=int(north_iou.shape[0]),
        north_pred_covered=_covered_count(north_iou, axis=1, threshold=threshold),
        matched_relations=tuple(matched_relations),
    )


def mask_iou_matrix(
    pred_masks: NDArray[np.bool_],
    gt_masks: NDArray[np.bool_],
    threshold: float = 0.5,
) -> NDArray[np.float64]:
    """计算预测 Mask 与 GT Mask 的 IoU 矩阵。

    Args:
        pred_masks: 预测实例 Mask，形状为 (P, H, W)。
        gt_masks: GT 实例 Mask，形状为 (G, H, W)。
        threshold: 用于包围盒预筛的最低关注 IoU；默认 0.5。

    Returns:
        形状为 (P, G) 的 IoU 矩阵。包围盒交集不可能达到门限的对直接置 0。

    Raises:
        ValueError: 输入不是三维布尔数组或尺寸不一致。
    """
    _validate_masks(pred_masks, "pred_masks")
    _validate_masks(gt_masks, "gt_masks")
    if pred_masks.shape[1:] != gt_masks.shape[1:]:
        raise ValueError("预测与 GT Mask 尺寸不一致")
    matrix = np.zeros(
        (len(pred_masks), len(gt_masks)), dtype=np.float64
    )
    if not len(pred_masks) or not len(gt_masks):
        return matrix
    pred_boxes = _mask_boxes(pred_masks)
    gt_boxes = _mask_boxes(gt_masks)
    pred_areas = np.count_nonzero(pred_masks, axis=(1, 2)).astype(np.int64)
    gt_areas = np.count_nonzero(gt_masks, axis=(1, 2)).astype(np.int64)
    for pred_index, pred_box in enumerate(pred_boxes):
        for gt_index, gt_box in enumerate(gt_boxes):
            if _box_overlap_area(pred_box, gt_box) < threshold * min(
                pred_areas[pred_index], gt_areas[gt_index]
            ):
                continue
            intersection = _intersection(
                pred_masks[pred_index], gt_masks[gt_index],
                pred_box, gt_box,
            )
            union = (
                pred_areas[pred_index] + gt_areas[gt_index] - intersection
            )
            if union:
                matrix[pred_index, gt_index] = intersection / union
    return matrix


def _augment(
    gt_index: int,
    adjacency: list[list[tuple[float, int]]],
    match_pred: list[int],
    seen: set[int],
) -> bool:
    """Kuhn 增广路：为 GT 关系寻找一个未占用的预测关系。"""
    for _, pred_index in sorted(
        adjacency[gt_index], key=lambda item: (-item[0], item[1])
    ):
        if pred_index in seen:
            continue
        seen.add(pred_index)
        if match_pred[pred_index] == -1 or _augment(
            match_pred[pred_index], adjacency, match_pred, seen
        ):
            match_pred[pred_index] = gt_index
            return True
    return False


def _covered_count(
    matrix: NDArray[np.float64], axis: int, threshold: float
) -> int:
    """统计每行或每列是否存在达到门限的 IoU。"""
    if matrix.size == 0:
        return 0
    return int(np.count_nonzero((matrix >= threshold).any(axis=axis)))


def _mask_boxes(masks: NDArray[np.bool_]) -> NDArray[np.int64]:
    """求实例 Mask 的 xyxy 紧包围盒。"""
    boxes = np.zeros((len(masks), 4), dtype=np.int64)
    for index, mask in enumerate(masks):
        rows = np.flatnonzero(mask.any(axis=1))
        if not rows.size:
            continue
        columns = np.flatnonzero(mask.any(axis=0))
        boxes[index] = (
            columns[0], rows[0], columns[-1] + 1, rows[-1] + 1
        )
    return boxes


def _box_overlap_area(
    first: NDArray[np.int64], second: NDArray[np.int64]
) -> int:
    """返回两个 xyxy 包围盒的交集面积。"""
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    return width * height if width > 0 and height > 0 else 0


def _intersection(
    first: NDArray[np.bool_],
    second: NDArray[np.bool_],
    first_box: NDArray[np.int64],
    second_box: NDArray[np.int64],
) -> int:
    """在包围盒交集内统计两个 Mask 的共同像素。"""
    x0 = max(int(first_box[0]), int(second_box[0]))
    y0 = max(int(first_box[1]), int(second_box[1]))
    x1 = min(int(first_box[2]), int(second_box[2]))
    y1 = min(int(first_box[3]), int(second_box[3]))
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(
        first[y0:y1, x0:x1] & second[y0:y1, x0:x1]
    ))


def _validate_matrix(matrix: NDArray[np.float64], name: str) -> None:
    """校验 IoU 矩阵为二维数值数组。"""
    if matrix.ndim != 2:
        raise ValueError(f"{name} 必须为二维数组")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} 必须为有限值")


def _validate_masks(masks: NDArray[np.bool_], name: str) -> None:
    """校验 Mask 栈为三维布尔数组。"""
    if masks.ndim != 3:
        raise ValueError(f"{name} 必须为 (N, H, W) 的三维数组")
    if masks.dtype != np.bool_:
        raise ValueError(f"{name} 必须为布尔数组")


def _validate_indices(
    relations: list[Relation],
    south_count: int,
    north_count: int,
    name: str,
) -> None:
    """校验关系中的实例下标没有越界。"""
    for south, north in relations:
        if not 0 <= south < south_count:
            raise ValueError(f"{name} 南侧下标越界 {south}")
        if not 0 <= north < north_count:
            raise ValueError(f"{name} 北侧下标越界 {north}")


def _safe_div(numerator: int, denominator: int) -> float:
    """安全除法，分母为 0 时返回 0。"""
    return numerator / denominator if denominator else 0.0
