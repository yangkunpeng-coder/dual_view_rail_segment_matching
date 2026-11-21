"""在原图坐标下统一评价不同分割模型，并保存可比较的验证产物。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from data_preprocess.common.config import PROJECT_ROOT
from utils.instance_masks import load_yolo_polygons
from utils.segmentation_predict import (
    OVERLAY_JPEG_QUALITY,
    prediction_image_path,
)
from utils.segmentation_review import ImageMaskCounts

EVALUATOR_ID = "rail_mask_original_v1"
METRIC_COLUMNS = ("precision", "recall", "mask_map50", "mask_map50_95")
_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_IMAGE_COLUMNS = (
    "image", "width", "height", "target_count", "prediction_count",
    "display_prediction_count", "matched_count", "missed_count",
    "false_positive_count", "inference_ms",
)
_OVERLAY_ALPHA = 0.4
# 按钢轨截面宽度分组，单位原图像素。边界固定便于跨实验比较。
_WIDTH_BUCKETS = (
    (0.0, 23.0, "<=23px"),
    (23.0, 28.0, "23-28px"),
    (28.0, 37.0, "28-37px"),
    (37.0, float("inf"), ">37px"),
)
_WIDTH_SAMPLE_RATIOS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class MaskPrediction:
    """单张图的实例预测，masks 为原图大小的 (N,H,W) 布尔数组。"""

    masks: NDArray[np.bool_]
    scores: NDArray[np.float32]
    inference_ms: float


class MaskEvaluation:
    """按置信度贪心匹配 GT，用 Ultralytics AP 积分规则汇总指标，并按钢轨宽度分组。"""

    def __init__(self, count_confidence: float) -> None:
        """设置逐图漏检计数的置信度门限。

        Args:
            count_confidence: 用于逐图 TP、FP、FN 的最低实例分数。
        """
        if not 0 <= count_confidence <= 1:
            raise ValueError("计数置信度必须在 [0, 1] 内")
        self.count_confidence = count_confidence
        self.image_counts: dict[str, ImageMaskCounts] = {}
        self._rows: list[dict[str, object]] = []
        self._inputs: list[dict[str, object]] = []
        self._correct: list[NDArray[np.bool_]] = []
        self._scores: list[NDArray[np.float32]] = []
        self._matched: list[NDArray[np.int64]] = []
        self._buckets: list[NDArray[np.int64]] = []
        self._target_count = 0

    def update(self, image_path: Path, prediction: MaskPrediction) -> None:
        """读取同名 YOLO Polygon 标签并累积一张图的原图 Mask 评价。

        Args:
            image_path: YOLO images 目录下的源图路径。
            prediction: 各实例掩码、分数和推理耗时。

        Raises:
            OSError: 无法读取源图或标签。
            ValueError: 掩码、分数或单类别标签无效，或图片重复。
        """
        from ultralytics.data.utils import img2label_paths

        masks, scores = prediction.masks, prediction.scores
        if masks.ndim != 3 or masks.dtype != np.bool_:
            raise ValueError("预测 Mask 必须为原图尺寸的 (N,H,W) 布尔数组")
        if scores.shape != (len(masks),) or not np.isfinite(scores).all():
            raise ValueError("预测分数必须为与实例数量一致的有限一维数组")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("预测分数必须在 [0, 1] 内")
        if image_path.name in self.image_counts:
            raise ValueError(f"验证清单重复出现图片 {image_path.name}")

        label_path = Path(img2label_paths([str(image_path)])[0])
        label_bytes = label_path.read_bytes()
        for line in label_bytes.decode("utf-8-sig").splitlines():
            if line.strip() and int(line.split()[0]) != 0:
                raise ValueError(f"当前评价只支持类别 0 rail，检查 {label_path}")
        height, width = masks.shape[1:]
        polygons = load_yolo_polygons(label_path, (width, height))
        order = np.argsort(-scores, kind="stable")
        ordered_scores = scores[order]
        ious = _polygon_mask_ious(polygons, masks, order)
        correct, matched_index = _match_masks(ious)
        widths = np.asarray([_polygon_width(polygon) for polygon in polygons])
        buckets = np.asarray(
            [_width_bucket_index(float(value)) for value in widths],
            dtype=np.int64,
        )
        is_counted = ordered_scores >= self.count_confidence
        matched_count = int(correct[is_counted, 0].sum())
        predicted_count = int(is_counted.sum())
        target_count = len(polygons)
        self.image_counts[image_path.name] = ImageMaskCounts(
            target_count, matched_count
        )
        self._correct.append(correct)
        self._scores.append(ordered_scores)
        self._matched.append(matched_index)
        self._buckets.append(buckets)
        self._target_count += target_count
        self._rows.append({
            "image": image_path.name, "width": width, "height": height,
            "target_count": target_count, "prediction_count": len(masks),
            "display_prediction_count": predicted_count,
            "matched_count": matched_count,
            "missed_count": target_count - matched_count,
            "false_positive_count": predicted_count - matched_count,
            "inference_ms": prediction.inference_ms,
        })
        self._inputs.append({
            "image": _project_path(image_path),
            "image_sha256": sha256(image_path.read_bytes()).hexdigest(),
            "label": _project_path(label_path),
            "label_sha256": sha256(label_bytes).hexdigest(),
        })

    def save(
        self, output_dir: Path, split: str, model_name: str,
        weights_path: Path, options: dict[str, object],
    ) -> Path:
        """保存指标、逐图统计和包含实际输入哈希的评估清单。

        Args:
            output_dir: 本次 valid、train 或 test 目录。
            split: CSV 中的数据划分，val、train 或 test。
            model_name: 生成预测的模型名称。
            weights_path: 实际使用的权重。
            options: 实际预测配置。

        Returns:
            metrics/evaluation.csv 的路径。
        """
        from ultralytics.utils.metrics import ap_per_class

        if not self._rows:
            raise ValueError("没有可评价的验证图片")
        correct = np.concatenate(self._correct)
        scores = np.concatenate(self._scores)
        values = ap_per_class(
            correct, scores, np.zeros(len(scores)),
            np.zeros(self._target_count), names={0: "rail"},
        )
        precision, recall, ap = values[2], values[3], values[5]
        metrics = (
            float(precision.mean()) if precision.size else 0.0,
            float(recall.mean()) if recall.size else 0.0,
            float(ap[:, 0].mean()) if ap.size else 0.0,
            float(ap.mean()) if ap.size else 0.0,
        )
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir()
        csv_path = metrics_dir / "evaluation.csv"
        with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("split", *METRIC_COLUMNS))
            writer.writerow((split, *metrics))
        with (metrics_dir / "per_image.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=_IMAGE_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(self._rows)

        with (metrics_dir / "width_buckets.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("split", "bucket", "gt_count", *METRIC_COLUMNS))
            for row in self._width_bucket_rows():
                writer.writerow((split, *row))

        inputs = sorted(self._inputs, key=lambda row: str(row["image"]))
        signature = sha256(json.dumps(
            inputs, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")).hexdigest()
        metadata = {
            "evaluator": EVALUATOR_ID, "model": model_name, "split": split,
            "image_count": len(inputs), "target_count": self._target_count,
            "width_bucket_labels": [bucket[2] for bucket in _WIDTH_BUCKETS],
            "width_bucket_counts": [
                sum(int(np.count_nonzero(items == index)) for items in self._buckets)
                for index in range(len(_WIDTH_BUCKETS))
            ],
            "dataset_sha256": signature, "images": inputs,
            "weights_path": _project_path(weights_path),
            "predict_options": options,
            "confidence_floor": options["conf"],
            "count_confidence": self.count_confidence,
            "mask_iou_thresholds": _IOU_THRESHOLDS.tolist(),
            "precision_recall_rule": "Ultralytics max-F1 confidence",
            "mask_space": "original_image_independent_instances",
        }
        (metrics_dir / "evaluation_manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8", newline="\n",
        )
        return csv_path

    def _width_bucket_rows(self) -> list[tuple[object, ...]]:
        """按钢轨宽度分组计算四项指标，缺失分组返回空值。

        分组口径参照按尺寸分层的 AP：以 IoU 0.50 门限下的归属确定每个预测属于
        哪个宽度分组；归属到其它分组的预测被忽略，未匹配的预测计为负样本，
        因此分组指标同时反映该宽度区间的误检代价。

        Returns:
            每行依次为分组名、GT 数量、Precision、Recall、mAP50、mAP50–95。
        """
        from ultralytics.utils.metrics import ap_per_class

        rows: list[tuple[object, ...]] = []
        for bucket_index, (_, _, label) in enumerate(_WIDTH_BUCKETS):
            target_count = sum(
                int(np.count_nonzero(items == bucket_index)) for items in self._buckets
            )
            if not target_count:
                rows.append((label, 0, None, None, None, None))
                continue
            tp_parts: list[NDArray[np.bool_]] = []
            score_parts: list[NDArray[np.float32]] = []
            for correct, matched, buckets, scores in zip(
                self._correct, self._matched, self._buckets, self._scores
            ):
                if not len(scores):
                    continue
                assigned = matched[:, 0]
                assigned_bucket = np.full(len(scores), -1, dtype=np.int64)
                valid = assigned >= 0
                assigned_bucket[valid] = buckets[assigned[valid]]
                keep = (assigned_bucket == bucket_index) | (assigned_bucket < 0)
                if not keep.any():
                    continue
                owned = (assigned_bucket == bucket_index)[:, None]
                tp_parts.append(correct[keep] & owned[keep])
                score_parts.append(scores[keep])
            if not tp_parts:
                rows.append((label, target_count, None, None, None, None))
                continue
            tp = np.concatenate(tp_parts)
            conf = np.concatenate(score_parts)
            values = ap_per_class(
                tp, conf, np.zeros(len(conf)), np.zeros(target_count),
                names={0: "rail"},
            )
            precision, recall, ap = values[2], values[3], values[5]
            rows.append((
                label, target_count,
                float(precision.mean()) if precision.size else 0.0,
                float(recall.mean()) if recall.size else 0.0,
                float(ap[:, 0].mean()) if ap.size else 0.0,
                float(ap.mean()) if ap.size else 0.0,
            ))
        return rows


def save_mask_prediction(
    output_dir: Path, image_path: Path, image: NDArray[np.uint8],
    prediction: MaskPrediction, display_confidence: float,
    *, save_overlay: bool = True,
) -> None:
    """保存分数、原图二值 Mask 和按南北分组的掩码叠加图。

    Args:
        output_dir: 当前验证目录。
        image_path: 源图路径，仅用于命名。
        image: 原图 H×W×3、uint8、BGR 数据。
        prediction: 当前图片的实例预测。
        display_confidence: 叠加图显示门限，NPZ 保存全部保留的预测。
        save_overlay: 为 False 时只保存 NPZ，不生成全尺寸叠加图。
    """
    from ultralytics.utils.plotting import colors

    if prediction.masks.shape[1:] != image.shape[:2]:
        raise ValueError(f"预测 Mask 与原图尺寸不一致 {image_path}")
    if save_overlay:
        destination = prediction_image_path(output_dir, image_path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        overlay = image.copy()
        for index in np.flatnonzero(prediction.scores >= display_confidence):
            mask = prediction.masks[index]
            color = np.asarray(colors(int(index), bgr=True), dtype=np.float32)
            overlay[mask] = (
                overlay[mask] * (1 - _OVERLAY_ALPHA) + color * _OVERLAY_ALPHA
            ).astype(np.uint8)
        encode_params = (
            [cv2.IMWRITE_JPEG_QUALITY, OVERLAY_JPEG_QUALITY]
            if destination.suffix.lower() in (".jpg", ".jpeg")
            else []
        )
        success, encoded = cv2.imencode(
            destination.suffix, overlay, encode_params
        )
        if not success:
            raise OSError(f"无法编码预测图 {destination}")
        encoded.tofile(destination)
    mask_path = prediction_image_path(
        output_dir / "masks", image_path.name
    ).with_suffix(".npz")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        mask_path, masks=prediction.masks, scores=prediction.scores
    )


def _polygon_mask_ious(
    polygons: list[NDArray[np.float64]], masks: NDArray[np.bool_],
    order: NDArray[np.intp],
) -> NDArray[np.float64]:
    """在 GT 紧包围区域内计算交集，避开整图浮点矩阵的高内存消耗。"""
    height, width = masks.shape[1:]
    areas = np.count_nonzero(masks, axis=(1, 2))
    ious = np.zeros((len(polygons), len(masks)), dtype=np.float64)
    for target_index, polygon in enumerate(polygons):
        points = np.rint(polygon).astype(np.int32)
        x_min, y_min = np.clip(points.min(0), (0, 0), (width, height))
        x_max, y_max = np.clip(points.max(0) + 1, (0, 0), (width, height))
        target = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        if not target.size:
            continue
        offset = np.asarray((x_min, y_min), dtype=np.int32)
        cv2.fillPoly(target, [points - offset], 1)
        target_area = int(np.count_nonzero(target))
        for output_index, mask_index in enumerate(order):
            intersection = int(np.count_nonzero(
                masks[mask_index, y_min:y_max, x_min:x_max] & target
            ))
            union = target_area + int(areas[mask_index]) - intersection
            if union:
                ious[target_index, output_index] = intersection / union
    return ious


def _match_masks(
    ious: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """沿用当前 Ultralytics 的按分数顺序、逐 IoU 门限一对一匹配规则。

    Args:
        ious: GT 与预测的 IoU 矩阵，形状为 (N_gt, N_pred)，列按分数降序。

    Returns:
        correct 为 (N_pred, 门限数) 的布尔矩阵；matched_index 记录每个门限下
        命中的 GT 索引，未命中为 -1。
    """
    correct = np.zeros((ious.shape[1], len(_IOU_THRESHOLDS)), dtype=np.bool_)
    matched_index = np.full(
        (ious.shape[1], len(_IOU_THRESHOLDS)), -1, dtype=np.int64
    )
    if not ious.shape[0]:
        return correct, matched_index
    matched = np.zeros((ious.shape[0], len(_IOU_THRESHOLDS)), dtype=np.bool_)
    for index in np.flatnonzero((ious >= _IOU_THRESHOLDS[0]).any(0)):
        available = np.where(matched, 0, ious[:, index, None])
        targets = available.argmax(0)
        hit = available[targets, np.arange(len(_IOU_THRESHOLDS))] >= _IOU_THRESHOLDS
        correct[index] = hit
        matched[targets, np.arange(len(_IOU_THRESHOLDS))] |= hit
        matched_index[index, hit] = targets[hit]
    return correct, matched_index


def _polygon_width(points: NDArray[np.float64]) -> float:
    """用若干水平截面估计 Polygon 表示的钢轨宽度，单位为原图像素。

    Args:
        points: Polygon 顶点数组，形状为 (N, 2)，原图像素坐标。

    Returns:
        各截面弦宽的中位数；无法求交时回退为外接框宽度。
    """
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())
    if y_max <= y_min:
        return float(points[:, 0].max() - points[:, 0].min())
    chords: list[float] = []
    for ratio in _WIDTH_SAMPLE_RATIOS:
        row = y_min + (y_max - y_min) * ratio
        crossings: list[float] = []
        for index in range(len(points)):
            start, end = points[index - 1], points[index]
            if (start[1] <= row < end[1]) or (end[1] <= row < start[1]):
                span = end[1] - start[1]
                if span:
                    crossings.append(
                        start[0] + (row - start[1]) / span * (end[0] - start[0])
                    )
        if len(crossings) >= 2:
            crossings.sort()
            chords.append(crossings[-1] - crossings[0])
    if chords:
        return float(np.median(chords))
    return float(points[:, 0].max() - points[:, 0].min())


def _width_bucket_index(width: float) -> int:
    """返回钢轨宽度所属的分组索引，越界值归入首组或末组。"""
    for index, (_, high, _) in enumerate(_WIDTH_BUCKETS):
        if width <= high:
            return index
    return len(_WIDTH_BUCKETS) - 1


def _project_path(path: Path) -> str:
    """优先记录项目相对路径，兼容项目外绝对路径输入。"""
    path = path.resolve()
    return (
        path.relative_to(PROJECT_ROOT).as_posix()
        if path.is_relative_to(PROJECT_ROOT) else str(path)
    )
