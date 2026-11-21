"""在固定预测上消融分割后处理：raw 与不同 Mask NMS 阈值。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core_algorithm.postprocess import suppress_duplicate_masks
from data_preprocess.common.config import PROJECT_ROOT, config_path, load_yaml_mapping
from utils.instance_masks import load_yolo_polygons
from utils.segmentation_evaluation import _match_masks, _polygon_mask_ious

_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_DEFAULT_OUTPUT_ROOT = "out/seg_postprocess"
_DEFAULT_CONFIG = "config/eval/postprocess_ablation.yaml"
_DEFAULT_YOLO_ROOT = "out/yolo_seg"
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
_CONFIG_SNAPSHOTS = (
    "config/eval/postprocess_ablation.yaml",
    "config/eval/seg_postprocess.yaml",
    "config/predict/rail_predict.yaml",
    "config/data/clop_seg.yaml",
)


@dataclass(frozen=True)
class AblationConfig:
    """消融参数。"""

    conf: float = 0.25
    duplicate_iou: float = 0.50
    merge_gt_coverage: float = 0.22
    merge_pred_coverage: float = 0.08
    min_area_px: int = 8
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70)


@dataclass(frozen=True)
class Variant:
    """一个消融变体；threshold 为 None 表示 raw。"""

    name: str
    threshold: float | None


def main() -> None:
    parser = argparse.ArgumentParser(description="分割后处理固定预测消融")
    parser.add_argument("--valid-pred-dir", default=None)
    parser.add_argument("--valid-label-dir", default="data/clop_seg/labels/val")
    parser.add_argument("--test-pred-dir", default=None)
    parser.add_argument("--test-label-dir", default="data/clop_seg/labels/test")
    parser.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = _load_config(config_path(args.config))
    splits = {
        "valid": (
            _raw_prediction_dir(
                _resolve_prediction_dir(args.valid_pred_dir, "valid")
            ),
            Path(PROJECT_ROOT, args.valid_label_dir),
        )
    }
    if args.test_pred_dir:
        splits["test"] = (
            _raw_prediction_dir(
                _resolve_prediction_dir(args.test_pred_dir, "test")
            ),
            Path(PROJECT_ROOT, args.test_label_dir),
        )
    for pred_dir, _ in splits.values():
        _warn_prediction_mode(pred_dir)
    variants = [Variant("raw", None)] + [
        Variant(f"nms_{threshold:.2f}", threshold)
        for threshold in config.thresholds
    ]

    output_root = Path(PROJECT_ROOT, args.output_root)
    output_dir = output_root / datetime.now().strftime(_TIMESTAMP_FORMAT)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "configs").mkdir()
    (output_dir / "metrics").mkdir()
    (output_dir / "tables").mkdir()
    (output_dir / "figures").mkdir()
    for relative in _CONFIG_SNAPSHOTS:
        try:
            source = config_path(relative)
        except FileNotFoundError:
            continue
        shutil.copy2(source, output_dir / "configs" / source.name)
    _write_change(output_dir, config, splits)

    rows: list[dict[str, object]] = []
    side_rows: list[dict[str, object]] = []
    per_image_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    split_inputs: dict[str, object] = {}
    for split, (pred_dir, label_dir) in splits.items():
        files = sorted(pred_dir.rglob("*.npz"))
        if args.limit:
            files = files[: args.limit]
        if not files:
            raise ValueError(f"{pred_dir} 下没有 NPZ 预测")
        split_inputs[split] = {
            "predictions": _project_path(pred_dir),
            "labels": _project_path(label_dir),
            "image_count": len(files),
            "fingerprint": _fingerprint(files),
        }
        split_rows, split_side, split_image, split_pairs = _evaluate_split(
            split, files, label_dir, variants, config
        )
        rows.extend(split_rows)
        side_rows.extend(split_side)
        per_image_rows.extend(split_image)
        pair_rows.extend(split_pairs)

    _write_csv(
        output_dir / "metrics" / "variants.csv", _VARIANT_COLUMNS, rows
    )
    _write_csv(
        output_dir / "metrics" / "variants_by_side.csv",
        _SIDE_COLUMNS, side_rows,
    )
    _write_csv(
        output_dir / "metrics" / "per_image.csv",
        _IMAGE_COLUMNS, per_image_rows,
    )
    _write_csv(
        output_dir / "metrics" / "duplicate_pairs.csv",
        _PAIR_COLUMNS, pair_rows,
    )
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "evaluator": "rail_mask_original_v1",
                "conf": config.conf,
                "mask_iou_thresholds": _IOU_THRESHOLDS.tolist(),
                "splits": split_inputs,
                "variants": [variant.name for variant in variants],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    recommendation = _recommend(rows)
    (output_dir / "metrics" / "summary.json").write_text(
        json.dumps(
            {
                "recommendation": recommendation,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_tables(output_dir, rows, side_rows)
    if not args.no_figures:
        _write_figures(output_dir, rows, side_rows)
    _write_report(output_dir, rows, side_rows, recommendation)
    print(f"消融结果 {output_dir}")


def _evaluate_split(
    split: str,
    files: list[Path],
    label_dir: Path,
    variants: list[Variant],
    config: AblationConfig,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """逐图评价所有变体，返回全局、分侧、逐图和重复对行。"""
    accumulator: dict[str, dict[str, object]] = {}
    for variant in variants:
        accumulator[variant.name] = _new_accumulator()
    image_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for path in files:
        data = np.load(path)
        masks = np.asarray(data["masks"], dtype=np.bool_)
        scores = np.asarray(data["scores"], dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        masks = masks[order]
        scores = scores[order]
        height, width = masks.shape[1:]
        side = path.parent.name
        label_name = f"{side}_{path.stem}.txt"
        polygons = load_yolo_polygons(label_dir / label_name, (width, height))
        gt_areas = [_polygon_area(polygon, width, height) for polygon in polygons]
        ious = _polygon_mask_ious(polygons, masks, np.arange(len(masks)))
        boxes = _mask_boxes(masks)
        pred_areas = np.count_nonzero(masks, axis=(1, 2)).astype(np.int64)
        pair_iou = _pair_iou_matrix(masks, boxes, pred_areas)
        best_gt = ious.argmax(0) if len(polygons) else np.full(len(masks), -1)
        best_iou = ious.max(0) if len(polygons) else np.zeros(len(masks))
        raw_correct, raw_matched = _match_masks(ious)
        raw_matched_set = _matched_gt_set(raw_matched)
        raw_pairs, raw_same = _count_pairs(
            pair_iou, best_gt, best_iou, config.duplicate_iou
        )
        _append_pairs(
            pair_rows, split, path.name, side, pair_iou, best_gt, best_iou,
            config.duplicate_iou,
        )
        for variant in variants:
            if variant.threshold is None:
                keep = np.arange(len(masks), dtype=np.intp)
                correct, matched = raw_correct, raw_matched
                latency_ms = 0.0
                pairs_after, same_after = raw_pairs, raw_same
            else:
                started = perf_counter()
                result = suppress_duplicate_masks(
                    masks, scores,
                    iou_threshold=variant.threshold,
                    min_area_px=config.min_area_px,
                    boxes=boxes,
                )
                latency_ms = (perf_counter() - started) * 1000.0
                keep = result.keep_indices
                correct, matched = _match_masks(ious[:, keep])
                pairs_after, same_after = _count_pairs(
                    pair_iou[np.ix_(keep, keep)], best_gt[keep],
                    best_iou[keep], config.duplicate_iou,
                )
            kept_scores = scores[keep]
            gt_count = len(polygons)
            tp, pred_count = _counts(correct, kept_scores, config.conf)
            matched_set = _matched_gt_set(matched)
            tp_lost = len(raw_matched_set - matched_set)
            merged = _merged_count(
                ious[:, keep], pred_areas[keep], gt_areas, config
            )
            entry = accumulator[variant.name]
            entry["correct"].append(correct)
            entry["scores"].append(kept_scores)
            entry["gt_count"] += gt_count
            entry["side_correct"].setdefault(side, []).append(correct)
            entry["side_scores"].setdefault(side, []).append(kept_scores)
            entry["side_gt"][side] = entry["side_gt"].get(side, 0) + gt_count
            counts = entry["counts"].setdefault(side, {"gt": 0, "pred": 0, "tp": 0})
            counts["gt"] += gt_count
            counts["pred"] += pred_count
            counts["tp"] += tp
            entry["pairs_before"] += raw_pairs
            entry["same_gt_before"] += raw_same
            entry["pairs_after"] += pairs_after
            entry["suppressed"] += len(masks) - len(keep)
            entry["tp_lost"] += tp_lost
            entry["merged"] += merged
            entry["latency"].append(latency_ms)
            side_entry = entry["side_stats"].setdefault(side, {
                "tp_lost": 0, "pairs_before": 0, "same_gt_before": 0,
                "pairs_after": 0, "suppressed": 0, "merged": 0,
            })
            side_entry["tp_lost"] += tp_lost
            side_entry["pairs_before"] += raw_pairs
            side_entry["same_gt_before"] += raw_same
            side_entry["pairs_after"] += pairs_after
            side_entry["suppressed"] += len(masks) - len(keep)
            side_entry["merged"] += merged
            image_rows.append({
                "split": split, "side": side, "image": path.name,
                "variant": variant.name, "gt": gt_count,
                "pred": pred_count, "tp": tp,
                "fp": pred_count - tp, "missed": gt_count - tp,
                "suppressed": len(masks) - len(keep),
                "tp_lost": tp_lost, "merged": merged,
                "latency_ms": round(latency_ms, 3),
            })
    rows, side_rows = _aggregate(split, accumulator)
    return rows, side_rows, image_rows, pair_rows


def _aggregate(
    split: str, accumulator: dict[str, dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """把逐图结果汇总成全局和分侧指标。"""
    rows: list[dict[str, object]] = []
    side_rows: list[dict[str, object]] = []
    for name, entry in accumulator.items():
        correct = np.concatenate(entry["correct"], axis=0)
        scores = np.concatenate(entry["scores"], axis=0)
        gt_count = int(entry["gt_count"])
        precision, recall, map50, map5095, ap75 = _ap(correct, scores, gt_count)
        counts = entry["counts"]
        pred_count = sum(item["pred"] for item in counts.values())
        tp = sum(item["tp"] for item in counts.values())
        row = {
            "variant": name, "split": split,
            "threshold": _variant_threshold(name),
            "precision": precision, "recall": recall,
            "map50": map50, "map50_95": map5095, "ap75": ap75,
            "instances": int(len(scores)), "gt": gt_count,
            "pred": pred_count, "tp": tp,
            "fp": pred_count - tp, "missed": gt_count - tp,
            "count_precision": _safe_div(tp, pred_count),
            "count_recall": _safe_div(tp, gt_count),
            "count_f1": _safe_div(2 * tp, pred_count + gt_count),
            "duplicate_pairs_before": int(entry["pairs_before"]),
            "same_gt_pairs_before": int(entry["same_gt_before"]),
            "duplicate_pairs_after": int(entry["pairs_after"]),
            "suppressed": int(entry["suppressed"]),
            "tp_lost": int(entry["tp_lost"]),
            "missed_added": int(entry["tp_lost"]),
            "merged_predictions": int(entry["merged"]),
            "latency_median_ms": _percentile(entry["latency"], 50),
            "latency_p95_ms": _percentile(entry["latency"], 95),
        }
        rows.append(row)
        for side in sorted(counts):
            side_counts = counts[side]
            side_correct = np.concatenate(entry["side_correct"][side], axis=0)
            side_scores = np.concatenate(entry["side_scores"][side], axis=0)
            side_gt = int(entry["side_gt"][side])
            _, _, side_map50, side_map5095, side_ap75 = _ap(
                side_correct, side_scores, side_gt
            )
            side_rows.append({
                "variant": name, "split": split, "side": side,
                "gt": side_gt, "pred": side_counts["pred"],
                "tp": side_counts["tp"],
                "fp": side_counts["pred"] - side_counts["tp"],
                "missed": side_gt - side_counts["tp"],
                "count_precision": _safe_div(
                    side_counts["tp"], side_counts["pred"]
                ),
                "count_recall": _safe_div(side_counts["tp"], side_gt),
                "count_f1": _safe_div(
                    2 * side_counts["tp"], side_counts["pred"] + side_gt
                ),
                "map50": side_map50, "map50_95": side_map5095,
                "ap75": side_ap75,
                "tp_lost": int(entry["side_stats"][side]["tp_lost"]),
                "duplicate_pairs_before": int(
                    entry["side_stats"][side]["pairs_before"]
                ),
                "duplicate_pairs_after": int(
                    entry["side_stats"][side]["pairs_after"]
                ),
                "merged_predictions": int(
                    entry["side_stats"][side]["merged"]
                ),
            })
    return rows, side_rows


def _ap(
    correct: np.ndarray, scores: np.ndarray, gt_count: int
) -> tuple[float, float, float, float, float]:
    """按项目评价口径计算 P、R、mAP50、mAP50–95、AP75。"""
    if not len(scores) or not gt_count:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    from ultralytics.utils.metrics import ap_per_class

    values = ap_per_class(
        correct, scores, np.zeros(len(scores)), np.zeros(gt_count),
        names={0: "rail"},
    )
    precision, recall, ap = values[2], values[3], values[5]
    return (
        float(precision.mean()) if precision.size else 0.0,
        float(recall.mean()) if recall.size else 0.0,
        float(ap[:, 0].mean()) if ap.size else 0.0,
        float(ap.mean()) if ap.size else 0.0,
        float(ap[:, 5].mean()) if ap.size else 0.0,
    )


def _counts(
    correct: np.ndarray, scores: np.ndarray, confidence: float
) -> tuple[int, int]:
    """返回 conf 门限下的 TP 和参与计数的预测数。"""
    counted = scores >= confidence
    tp = int(np.count_nonzero(correct[:, 0] & counted))
    return tp, int(counted.sum())


def _matched_gt_set(matched: np.ndarray) -> set[int]:
    """返回 IoU 0.50 下被匹配的 GT 索引集合。"""
    values = matched[:, 0]
    return {int(value) for value in values if value >= 0}


def _count_pairs(
    pair_iou: np.ndarray,
    best_gt: np.ndarray,
    best_iou: np.ndarray,
    threshold: float,
) -> tuple[int, int]:
    """统计超过阈值的实例对及其中同 GT 的数量。"""
    count = 0
    same_gt = 0
    for first in range(len(pair_iou)):
        for second in range(first + 1, len(pair_iou)):
            if pair_iou[first, second] <= threshold:
                continue
            count += 1
            if (
                best_gt[first] == best_gt[second]
                and best_iou[first] >= 0.5
                and best_iou[second] >= 0.5
            ):
                same_gt += 1
    return count, same_gt


def _append_pairs(
    rows: list[dict[str, object]],
    split: str,
    image: str,
    side: str,
    pair_iou: np.ndarray,
    best_gt: np.ndarray,
    best_iou: np.ndarray,
    threshold: float,
) -> None:
    """保存 raw 预测中的高 IoU 对，供人工检查误抑制。"""
    for first in range(len(pair_iou)):
        for second in range(first + 1, len(pair_iou)):
            value = pair_iou[first, second]
            if value <= threshold:
                continue
            rows.append({
                "split": split, "image": image, "side": side,
                "pred_a": first, "pred_b": second,
                "iou": round(float(value), 6),
                "same_gt": int(
                    best_gt[first] == best_gt[second]
                    and best_iou[first] >= 0.5
                    and best_iou[second] >= 0.5
                ),
                "gt_a": int(best_gt[first]),
                "gt_b": int(best_gt[second]),
            })


def _merged_count(
    ious: np.ndarray,
    pred_areas: np.ndarray,
    gt_areas: list[int],
    config: AblationConfig,
) -> int:
    """统计同时覆盖多个 GT 的预测数。"""
    count = 0
    for column, pred_area in enumerate(pred_areas):
        if pred_area <= 0:
            continue
        covered = 0
        for row, gt_area in enumerate(gt_areas):
            if gt_area <= 0:
                continue
            iou = ious[row, column]
            if iou <= 0:
                continue
            intersection = iou * (pred_area + gt_area) / (1.0 + iou)
            if (
                intersection / pred_area >= config.merge_pred_coverage
                and intersection / gt_area >= config.merge_gt_coverage
            ):
                covered += 1
        if covered >= 2:
            count += 1
    return count


def _pair_iou_matrix(
    masks: np.ndarray, boxes: np.ndarray, areas: np.ndarray
) -> np.ndarray:
    """计算预测之间的 Mask IoU，用于重复对诊断。"""
    count = len(masks)
    matrix = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            value = _mask_iou(
                masks[first], masks[second], boxes[first], boxes[second],
                int(areas[first]), int(areas[second]),
            )
            matrix[first, second] = matrix[second, first] = value
    return matrix


def _mask_iou(
    first: np.ndarray,
    second: np.ndarray,
    first_box: np.ndarray,
    second_box: np.ndarray,
    first_area: int,
    second_area: int,
) -> float:
    """在包围盒交集内求两个 Mask 的 IoU。"""
    x0 = max(int(first_box[0]), int(second_box[0]))
    y0 = max(int(first_box[1]), int(second_box[1]))
    x1 = min(int(first_box[2]), int(second_box[2]))
    y1 = min(int(first_box[3]), int(second_box[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = int(np.count_nonzero(
        first[y0:y1, x0:x1] & second[y0:y1, x0:x1]
    ))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _mask_boxes(masks: np.ndarray) -> np.ndarray:
    """扫描实例 Mask 得到 xyxy 紧包围盒。"""
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


def _polygon_area(points: np.ndarray, width: int, height: int) -> int:
    """栅格化单个 GT Polygon 并返回像素数。"""
    rounded = np.rint(points).astype(np.int32)
    x0, y0 = np.clip(rounded.min(0), (0, 0), (width, height))
    x1, y1 = np.clip(rounded.max(0) + 1, (0, 0), (width, height))
    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(local, [rounded - np.asarray((x0, y0), np.int32)], 1)
    return int(np.count_nonzero(local))


def _new_accumulator() -> dict[str, object]:
    """创建一个变体的聚合容器。"""
    return {
        "correct": [], "scores": [], "gt_count": 0,
        "side_correct": {}, "side_scores": {}, "side_gt": {},
        "counts": {},
        "pairs_before": 0, "same_gt_before": 0, "pairs_after": 0,
        "suppressed": 0, "tp_lost": 0, "merged": 0, "latency": [],
        "side_stats": {},
    }


def _recommend(rows: list[dict[str, object]]) -> dict[str, object]:
    """按 valid 硬门槛选择 FP 最低的变体。"""
    valid = [row for row in rows if row["split"] == "valid"]
    if not valid:
        valid = rows
    baseline = next(
        (row for row in valid if row["variant"] == "raw"), valid[0]
    )
    candidates = [
        row for row in valid
        if row["variant"] != "raw"
        and row["tp_lost"] <= 5
        and row["missed_added"] <= 5
        and row["map50"] >= float(baseline["map50"])
        and row["map50_95"] >= float(baseline["map50_95"]) - 0.2
        and row["latency_median_ms"] <= 50
    ]
    zero_loss = [row for row in candidates if row["tp_lost"] == 0]
    pool = zero_loss or candidates
    chosen = min(pool, key=lambda row: row["fp"]) if pool else baseline
    aggressive = (
        min(candidates, key=lambda row: row["fp"]) if candidates else baseline
    )
    return {
        "variant": chosen["variant"],
        "mask_nms_iou": chosen["threshold"],
        "reason": (
            "在丢失 TP 为 0 的变体中 FP 最低；漏检一根轨的代价高于多保留重复候选"
            if zero_loss else "没有零丢失变体，取 TP 损失不超过 5 且 FP 最低者"
        ),
        "tp_lost": chosen["tp_lost"],
        "fp": chosen["fp"],
        "aggressive": {
            "variant": aggressive["variant"],
            "mask_nms_iou": aggressive["threshold"],
            "tp_lost": aggressive["tp_lost"],
            "fp": aggressive["fp"],
        },
    }


def _write_tables(
    output_dir: Path,
    rows: list[dict[str, object]],
    side_rows: list[dict[str, object]],
) -> None:
    """生成主表和阈值敏感性表。"""
    lines = ["# 后处理消融主表", ""]
    for split in sorted({str(row["split"]) for row in rows}):
        lines.extend([
            f"## {split}", "",
            "| 变体 | TP | FP | 漏检 | 丢失TP | 计数P | 计数R | "
            "mAP50 | mAP50–95 | 重复对(后) | 耗时中位 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: |",
        ])
        for row in rows:
            if row["split"] != split:
                continue
            lines.append(
                f"| {row['variant']} | {row['tp']} | {row['fp']} | "
                f"{row['missed']} | {row['tp_lost']} | "
                f"{_pct(row['count_precision'])} | "
                f"{_pct(row['count_recall'])} | {_pct(row['map50'])} | "
                f"{_pct(row['map50_95'])} | {row['duplicate_pairs_after']} | "
                f"{row['latency_median_ms']:.1f} ms |"
            )
        lines.append("")
    (output_dir / "tables" / "main_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )

    lines = ["# 分侧与阈值敏感性", ""]
    for split in sorted({str(row["split"]) for row in side_rows}):
        lines.extend([
            f"## {split}", "",
            "| 变体 | 侧 | TP | FP | 漏检 | 计数P | 计数R | mAP50 | mAP50–95 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in side_rows:
            if row["split"] != split:
                continue
            lines.append(
                f"| {row['variant']} | {row['side']} | {row['tp']} | "
                f"{row['fp']} | {row['missed']} | "
                f"{_pct(row['count_precision'])} | "
                f"{_pct(row['count_recall'])} | {_pct(row['map50'])} | "
                f"{_pct(row['map50_95'])} |"
            )
        lines.append("")
    (output_dir / "tables" / "sensitivity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _write_report(
    output_dir: Path,
    rows: list[dict[str, object]],
    side_rows: list[dict[str, object]],
    recommendation: dict[str, object],
) -> None:
    """写出简要结论。"""
    lines = [
        "# 分割后处理消融报告", "",
        f"推荐变体：**{recommendation['variant']}**"
        f"（Mask NMS IoU {recommendation['mask_nms_iou']}）", "",
        str(recommendation["reason"]), "",
    ]
    aggressive = recommendation.get("aggressive")
    if (
        isinstance(aggressive, dict)
        and aggressive["variant"] != recommendation["variant"]
    ):
        lines.extend([
            f"更激进的去重候选：**{aggressive['variant']}**"
            f"（FP {aggressive['fp']}，丢失 TP {aggressive['tp_lost']}）；"
            "若匹配阶段证明重复候选代价更低，可改用它。",
            "",
        ])
    lines.extend([
        "## 结果", "",
    ])
    for split in sorted({str(row["split"]) for row in rows}):
        lines.extend([f"### {split}", ""])
        for row in rows:
            if row["split"] != split:
                continue
            lines.append(
                f"- {row['variant']}：TP {row['tp']}，FP {row['fp']}，"
                f"漏检 {row['missed']}，丢失TP {row['tp_lost']}，"
                f"mAP50 {_pct(row['map50'])}，"
                f"mAP50–95 {_pct(row['map50_95'])}，"
                f"耗时中位 {row['latency_median_ms']:.1f} ms"
            )
        lines.append("")
    lines.extend([
        "## 说明", "",
        "- 评价使用与项目一致的原图 Mask 匹配和 AP 积分口径。",
        "- one-to-one 下 mAP 变化通常很小，主要看 FP、丢失 TP 和重复对。",
        "- 测试集只应在参数冻结后运行一次。",
    ])
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _write_figures(
    output_dir: Path,
    rows: list[dict[str, object]],
    side_rows: list[dict[str, object]],
) -> None:
    """生成阈值与 FP、TP、mAP 的关系图。"""
    splits = sorted({str(row["split"]) for row in rows})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for split in splits:
        subset = [row for row in rows if row["split"] == split]
        axes[0].plot(
            [row["fp"] for row in subset], [row["tp"] for row in subset],
            marker="o", label=split,
        )
        axes[1].plot(
            [row["threshold"] for row in subset],
            [row["map50_95"] for row in subset],
            marker="o", label=split,
        )
        axes[2].plot(
            [row["threshold"] for row in subset],
            [row["duplicate_pairs_after"] for row in subset],
            marker="o", label=split,
        )
    axes[0].set_xlabel("FP")
    axes[0].set_ylabel("TP")
    axes[0].set_title("TP / FP trade-off")
    axes[1].set_xlabel("Mask NMS IoU")
    axes[1].set_ylabel("mAP50-95")
    axes[1].set_title("mAP50-95")
    axes[2].set_xlabel("Mask NMS IoU")
    axes[2].set_ylabel("duplicate pairs")
    axes[2].set_title("duplicate pairs after")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "figures" / "tradeoff.png", dpi=150)
    plt.close(figure)


def _write_change(
    output_dir: Path,
    config: AblationConfig,
    splits: dict[str, tuple[Path, Path]],
) -> None:
    """写入本次消融说明。"""
    lines = [
        "# 后处理消融", "",
        "## 本次方案", "",
        f"- 变体：raw + Mask NMS {config.thresholds}", "",
        "## 实验思路", "",
        "在固定预测上只切换后处理，隔离模型和推理差异。", "",
        "## 要解决的问题", "",
        "去重能减少多少 FP，代价是丢失多少 TP 和 mAP。", "",
        "## 输入", "",
    ]
    for split, (pred_dir, label_dir) in splits.items():
        lines.append(
            f"- {split}: {_project_path(pred_dir)} / "
            f"{_project_path(label_dir)}"
        )
    (output_dir / "change.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _raw_prediction_dir(pred_dir: Path) -> Path:
    """优先使用同次预测保存的未去重 masks_raw。"""
    raw_dir = pred_dir.parent / "masks_raw"
    if raw_dir.is_dir():
        return raw_dir
    print(
        f"提示：{pred_dir} 旁没有 masks_raw，raw 变体可能已被去重"
    )
    return pred_dir


def _warn_prediction_mode(pred_dir: Path) -> None:
    """当预测快照不是 one-to-one 时提示，避免混用推理模式。"""
    snapshot = pred_dir.parent / "rail_predict.yaml"
    if not snapshot.is_file():
        return
    try:
        mapping = load_yaml_mapping(snapshot)
    except (OSError, ValueError):
        return
    if mapping.get("nms") is not False:
        print(
            f"警告：{pred_dir} 的预测配置不是 nms: false，"
            "消融结果不能与当前 one-to-one 配置直接比较"
        )


def _resolve_prediction_dir(value: str | None, split: str) -> Path:
    """解析预测目录；省略时取 out/yolo_seg 下最新的对应划分。"""
    if value:
        return Path(PROJECT_ROOT, value)
    root = PROJECT_ROOT / _DEFAULT_YOLO_ROOT
    candidates = [
        path for path in root.glob(f"*/{split}/*/masks")
        if (path / "north").is_dir() or (path / "south").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"out/yolo_seg 下没有 {split} 预测目录")
    return max(candidates, key=lambda path: path.parent.name)


def _load_config(path: Path) -> AblationConfig:
    """读取并校验消融配置。"""
    mapping = load_yaml_mapping(path)
    thresholds = mapping.get("thresholds", [0.50, 0.55, 0.60, 0.65, 0.70])
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError(f"{path} 的 thresholds 必须是非空列表")
    values = tuple(float(value) for value in thresholds)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError(f"{path} 的 thresholds 必须位于 [0, 1]")
    return AblationConfig(
        conf=float(mapping.get("conf", 0.25)),
        duplicate_iou=float(mapping.get("duplicate_iou", 0.50)),
        merge_gt_coverage=float(mapping.get("merge_gt_coverage", 0.22)),
        merge_pred_coverage=float(mapping.get("merge_pred_coverage", 0.08)),
        min_area_px=int(mapping.get("min_area_px", 8)),
        thresholds=values,
    )


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    """写出固定列顺序的 CSV。"""
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: object) -> str:
    """把比例格式化为百分数。"""
    return f"{float(value) * 100:.2f}%"


def _safe_div(numerator: float, denominator: float) -> float:
    """安全除法，分母为 0 时返回 0。"""
    return float(numerator / denominator) if denominator else 0.0


def _percentile(values: list[float], percent: float) -> float:
    """返回百分位，空列表返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(
        int(np.ceil(len(ordered) * percent / 100.0)) - 1, len(ordered) - 1
    ))
    return float(ordered[index])


def _variant_threshold(name: str) -> float | None:
    """从变体名解析阈值。"""
    if name == "raw":
        return None
    return float(name.split("_", 1)[1])


def _fingerprint(files: list[Path]) -> str:
    """用文件名和大小生成输入指纹。"""
    digest = sha256()
    for path in files:
        digest.update(f"{path.name}:{path.stat().st_size}\n".encode())
    return digest.hexdigest()


def _project_path(path: Path) -> str:
    """优先输出项目相对路径。"""
    resolved = path.resolve()
    return (
        resolved.relative_to(PROJECT_ROOT).as_posix()
        if resolved.is_relative_to(PROJECT_ROOT) else str(resolved)
    )


_VARIANT_COLUMNS = (
    "variant", "split", "threshold", "precision", "recall", "map50",
    "map50_95", "ap75", "instances", "gt", "pred", "tp", "fp", "missed",
    "count_precision", "count_recall", "count_f1",
    "duplicate_pairs_before", "same_gt_pairs_before",
    "duplicate_pairs_after", "suppressed", "tp_lost", "missed_added",
    "merged_predictions", "latency_median_ms", "latency_p95_ms",
)
_SIDE_COLUMNS = (
    "variant", "split", "side", "gt", "pred", "tp", "fp", "missed",
    "count_precision", "count_recall", "count_f1", "map50", "map50_95",
    "ap75", "tp_lost", "duplicate_pairs_before", "duplicate_pairs_after",
    "merged_predictions",
)
_IMAGE_COLUMNS = (
    "split", "side", "image", "variant", "gt", "pred", "tp", "fp",
    "missed", "suppressed", "tp_lost", "merged", "latency_ms",
)
_PAIR_COLUMNS = (
    "split", "image", "side", "pred_a", "pred_b", "iou", "same_gt",
    "gt_a", "gt_b",
)


if __name__ == "__main__":
    main()
