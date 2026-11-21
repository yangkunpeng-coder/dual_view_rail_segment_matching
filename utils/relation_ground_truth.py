"""读取已复核图对及其原图坐标下的关系真值。"""

from collections.abc import Collection, Mapping
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from data_preprocess.common.config import PROJECT_ROOT
from data_preprocess.common.labelme import load_rail_polygons
from utils.match_gt import Payload, read_pair_annotation
from utils.relation_eval import (
    Relation,
    RelationScore,
    mask_iou_matrix,
    score_relations,
)


@dataclass(frozen=True)
class ReviewedPair:
    """已纳入关系评测的图对，图片路径相对于项目根目录。"""

    pair_id: str
    split: str
    south_image: Path
    north_image: Path
    association: Payload


@dataclass(frozen=True)
class RelationTruth:
    """原图布尔 Mask 和基于数组下标的关系真值。"""

    gt_south: NDArray[np.bool_]
    gt_north: NDArray[np.bool_]
    gt_relations: set[Relation]
    single_south: set[int]
    single_north: set[int]

    def score(
        self,
        south_masks: NDArray[np.bool_],
        north_masks: NDArray[np.bool_],
        pred_relations: set[Relation],
        thresholds: tuple[float, ...],
    ) -> list[tuple[float, RelationScore]]:
        """按两端 Mask IoU 评分，每个门限复用同一组 IoU 矩阵。

        Args:
            south_masks: 南侧预测布尔掩码，形状为 (N, H, W)。
            north_masks: 北侧预测布尔掩码，形状为 (N, H, W)。
            pred_relations: 预测关系的 (南侧下标, 北侧下标) 集合。
            thresholds: 非空的 Mask IoU 门限。

        Returns:
            按输入顺序排列的门限和关系评测结果。
        """
        minimum = min(thresholds)
        south_iou = mask_iou_matrix(south_masks, self.gt_south, minimum)
        north_iou = mask_iou_matrix(north_masks, self.gt_north, minimum)
        return [
            (
                threshold,
                score_relations(
                    self.gt_relations, pred_relations, south_iou, north_iou,
                    self.single_south, self.single_north, threshold,
                ),
            )
            for threshold in thresholds
        ]


def excluded_pairs(config: Mapping[str, object]) -> dict[str, str]:
    """从真值配置返回排除的图对编号及对应原因。"""
    excluded = config.get("excluded_pairs", {})
    if not isinstance(excluded, dict):
        return {}
    return {str(key): str(value) for key, value in excluded.items()}


def read_reviewed_pairs(
    gt_root: Path,
    splits: Collection[str],
    excluded: Mapping[str, str],
) -> tuple[list[ReviewedPair], list[dict[str, str]]]:
    """按清单顺序筛选图对并记录排除原因。

    Args:
        gt_root: 真值根目录，包含 manifest.csv 与逐划分 association JSON。
        splits: 要读取的划分。
        excluded: 要排除的图对编号和原因。

    Returns:
        已复核的图对，以及未通过筛选的图对编号、划分与原因。
    """
    pairs: list[ReviewedPair] = []
    skipped: list[dict[str, str]] = []
    with (gt_root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            pair_id = row["pair_id"]
            if split not in splits:
                continue
            annotation = _load_reviewed_annotation(gt_root, split, pair_id, excluded)
            if isinstance(annotation, str):
                skipped.append({
                    "pair_id": pair_id, "split": split, "reason": annotation,
                })
                continue
            pairs.append(ReviewedPair(
                pair_id=pair_id,
                split=split,
                south_image=Path(row["south_image_path"]),
                north_image=Path(row["north_image_path"]),
                association=annotation,
            ))
    return pairs, skipped


def _load_reviewed_annotation(
    gt_root: Path, split: str, pair_id: str, excluded: Mapping[str, str]
) -> Payload | str:
    """返回可用于评测的关联标注，或跳过当前图对的原因。"""
    if pair_id in excluded:
        return f"excluded: {excluded[pair_id]}"
    payload = read_pair_annotation(gt_root / split / f"{pair_id}_association.json")
    review = payload.get("review") or {}
    if review.get("status") != "done":
        return f"review_status={review.get('status')}"
    if payload.get("ambiguous"):
        return "ambiguous_annotations"
    return payload


def load_relation_truth(pair: ReviewedPair) -> RelationTruth:
    """栅格化图对的 LabelMe 多边形并解析对应的实例下标。

    Args:
        pair: 已复核图对，关联标注的 Mask 下标沿用 LabelMe 顺序。

    Returns:
        两侧原图 Mask、配对关系和单侧实例下标。

    Raises:
        OSError: 原图或标注无法读取。
        ValueError: 标注无效或 Mask 下标越界。
    """
    south_masks = _load_gt_masks(PROJECT_ROOT / pair.south_image)
    north_masks = _load_gt_masks(PROJECT_ROOT / pair.north_image)
    payload = pair.association
    instances = payload["instances"]
    south_map = _instance_indices(instances["south"], len(south_masks), "南")
    north_map = _instance_indices(instances["north"], len(north_masks), "北")
    relations = {
        (
            south_map[str(item["south_instance_id"])],
            north_map[str(item["north_instance_id"])],
        )
        for item in payload.get("matches", [])
    }
    single_south: set[int] = set()
    single_north: set[int] = set()
    for item in payload.get("unmatched", []):
        camera = item.get("camera")
        identifier = str(item.get("instance_id"))
        if camera == "south":
            single_south.add(south_map[identifier])
        elif camera == "north":
            single_north.add(north_map[identifier])
    return RelationTruth(
        south_masks, north_masks, relations, single_south, single_north
    )


def _instance_indices(
    instances: list[dict[str, object]], mask_count: int, side_name: str
) -> dict[str, int]:
    """把真值实例编号映射到有效的 Mask 下标。"""
    indices = {
        str(record["instance_id"]): int(record["mask_index"]) for record in instances
    }
    for index in indices.values():
        if not 0 <= index < mask_count:
            raise ValueError(f"GT {side_name}侧 mask_index 越界 {index}")
    return indices


def _load_gt_masks(image_path: Path) -> NDArray[np.bool_]:
    """按 LabelMe 标注顺序栅格化 GT Mask。"""
    with Image.open(image_path) as image:
        width, height = image.size
    polygons = load_rail_polygons(image_path, (width, height))
    masks = np.zeros((len(polygons), height, width), dtype=np.bool_)
    for mask, polygon in zip(masks, polygons, strict=True):
        vertices = np.rint(np.asarray(polygon["points"])).astype(np.int32)
        cv2.fillPoly(mask.view(np.uint8), [vertices], 1)
    return masks
