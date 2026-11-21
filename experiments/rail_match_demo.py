"""从南北相机原图分割和匹配钢轨，可用已复核的图对评测。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_algorithm.common.match_result import RailMatch
from core_algorithm.common.preview import render_side
from core_algorithm.matching.rail_matching_mslc import match_south_north_rail_mslc
from data_preprocess.common.config import (
    PROJECT_ROOT,
    config_path,
    load_dataset_relative_path,
    load_yaml_mapping,
)
from data_preprocess.common.roi_config import load_roi_settings
from data_preprocess.match_pair_image import match_pair_image
from utils.relation_ground_truth import (
    ReviewedPair,
    excluded_pairs,
    load_relation_truth,
    read_reviewed_pairs,
)
from utils.segmentation_instances import PredictedViewLoader
from utils.segmentation_postprocess import load_postprocess_config

_DEFAULT_PREDICT_CONFIG = "config/predict/rail_predict.yaml"
_DEFAULT_POSTPROCESS_CONFIG = "config/eval/seg_postprocess.yaml"
_DEFAULT_ROI_CONFIG = "config/data/rail_roi.yaml"
_DEFAULT_GT_CONFIG = "config/eval/rail_match_gt.yaml"
_DEFAULT_OUTPUT_ROOT = "out/rail_match_demo"
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
_IOU_THRESHOLDS = (0.50, 0.75)
_MATCH_COLUMNS = (
    "pair_id", "south_image", "north_image", "shared_id",
    "south_instance_id", "north_instance_id",
    "south_mask_index", "north_mask_index", "position_error",
)
_UNMATCHED_COLUMNS = ("pair_id", "side", "instance_id", "mask_index", "reason")
_PAIR_COLUMNS = (
    "pair_id", "south_image", "north_image", "status",
    "south_masks", "north_masks", "south_suppressed", "north_suppressed",
    "match_count", "south_unmatched", "north_unmatched",
    "south_postprocess_ms", "north_postprocess_ms", "elapsed_ms",
    "south_roi_status", "north_roi_status", "south_roi_bbox", "north_roi_bbox",
)
_RELATION_COLUMNS = (
    "split", "iou_threshold", "gt_relations", "pred_relations",
    "true_positive", "false_positive", "false_negative", "precision", "recall", "f1",
)
_PAIR_RELATION_COLUMNS = (
    "pair_id", *_RELATION_COLUMNS,
    "south_gt_total", "south_gt_covered", "north_gt_total", "north_gt_covered",
)


def main() -> None:
    """运行原图分割、南北匹配和可选的关系评测。"""
    parser = argparse.ArgumentParser(description="南北相机原图分割与匹配 Demo")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--predict-config", default=_DEFAULT_PREDICT_CONFIG)
    parser.add_argument("--postprocess-config", default=_DEFAULT_POSTPROCESS_CONFIG)
    parser.add_argument(
        "--roi-mode", choices=("raw", "predicted"), default="raw",
        help="raw 直接分割原图，predicted 用首遍 Mask 生成走廊后再分割",
    )
    parser.add_argument("--roi-config", default=_DEFAULT_ROI_CONFIG)
    parser.add_argument(
        "--gt-split", choices=("valid", "test"),
        help="使用指定划分的已复核图对，输出关系级指标",
    )
    parser.add_argument("--gt-config", default=_DEFAULT_GT_CONFIG)
    parser.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--limit", type=int, default=5,
        help="处理的图对数量；0 表示全部",
    )
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--save-masks", action="store_true", help="保存原图 Mask NPZ")
    args = parser.parse_args()

    weights = (
        Path(PROJECT_ROOT, args.weights).resolve()
        if args.weights else _latest_weights()
    )
    predict_config = config_path(args.predict_config)
    postprocess_config = config_path(args.postprocess_config)
    roi_config = config_path(args.roi_config) if args.roi_mode == "predicted" else None
    gt_config = config_path(args.gt_config) if args.gt_split else None
    output_root = Path(PROJECT_ROOT, args.output_root).resolve()

    gt_root = None
    reviewed = {}
    skipped = []
    if gt_config is not None:
        gt_settings = load_yaml_mapping(gt_config)
        root = gt_settings.get("root")
        if not isinstance(root, str) or not root:
            raise ValueError(f"{gt_config} 的 root 必须是真值目录")
        gt_root = Path(PROJECT_ROOT, root).resolve()
        gt_pairs, skipped = read_reviewed_pairs(
            gt_root, (args.gt_split,), excluded_pairs(gt_settings)
        )
        reviewed = {pair.pair_id: pair for pair in gt_pairs}
        pairs = [
            (pair.pair_id, pair.south_image, pair.north_image) for pair in gt_pairs
        ]
    else:
        pairs = _source_pairs()
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("没有可运行的南北图对")

    loader = PredictedViewLoader(
        weights, predict_config, load_postprocess_config(postprocess_config),
        corridor=load_roi_settings(roi_config).corridor if roi_config else None,
    )
    output_dir = output_root / datetime.now().strftime(_TIMESTAMP_FORMAT)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "configs").mkdir()
    if not args.no_previews:
        (output_dir / "previews").mkdir()
    if args.save_masks:
        for side in ("south", "north"):
            (output_dir / "masks" / side).mkdir(parents=True)
    snapshots = (
        (_DEFAULT_PREDICT_CONFIG, predict_config),
        (_DEFAULT_POSTPROCESS_CONFIG, postprocess_config),
        (_DEFAULT_ROI_CONFIG, roi_config),
        (_DEFAULT_GT_CONFIG, gt_config),
    )
    for name, source in snapshots:
        if source is not None:
            target = output_dir / "configs" / Path(name).name
            target.write_bytes(source.read_bytes())

    match_rows: list[dict[str, object]] = []
    unmatched_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []
    started_all = perf_counter()
    for index, (pair_id, south_image, north_image) in enumerate(pairs, start=1):
        artifact_stem = pair_id.zfill(4)
        started = perf_counter()
        south_view = loader.load(PROJECT_ROOT / south_image)
        north_view = loader.load(PROJECT_ROOT / north_image)
        result = match_south_north_rail_mslc(
            south_image, north_image, south_view.masks, north_view.masks,
        )
        elapsed_ms = (perf_counter() - started) * 1000.0
        matches = result["matches"]
        south_unmatched = result["unmatched_south"]
        north_unmatched = result["unmatched_north"]
        for match in matches:
            match_rows.append({
                "pair_id": pair_id,
                "south_image": south_image.name,
                "north_image": north_image.name,
                "shared_id": match["shared_id"],
                "south_instance_id": match["south"]["instance_id"],
                "north_instance_id": match["north"]["instance_id"],
                "south_mask_index": match["south"]["mask_index"],
                "north_mask_index": match["north"]["mask_index"],
                "position_error": match["position_error"],
            })
        for side, unmatched in (("south", south_unmatched), ("north", north_unmatched)):
            for instance in unmatched:
                unmatched_rows.append({
                    "pair_id": pair_id, "side": side,
                    "instance_id": instance["instance_id"],
                    "mask_index": instance["mask_index"],
                    "reason": instance["reason"],
                })
        status = "matched" if matches else "no_match"
        if not len(south_view.masks) or not len(north_view.masks):
            status = "empty_side"
        pair_row: dict[str, object] = {
            "pair_id": pair_id,
            "south_image": south_image.name,
            "north_image": north_image.name,
            "status": status,
            "match_count": len(matches),
            "south_unmatched": len(south_unmatched),
            "north_unmatched": len(north_unmatched),
            "elapsed_ms": round(elapsed_ms, 2),
        }
        for side, view in (("south", south_view), ("north", north_view)):
            pair_row[f"{side}_masks"] = len(view.masks)
            pair_row[f"{side}_suppressed"] = view.suppressed_count
            pair_row[f"{side}_postprocess_ms"] = round(view.postprocess_ms, 2)
            pair_row[f"{side}_roi_status"] = view.roi_status
            pair_row[f"{side}_roi_bbox"] = (
                json.dumps(view.roi_bbox) if view.roi_bbox else ""
            )
            if args.save_masks:
                np.savez_compressed(
                    output_dir / "masks" / side / f"{artifact_stem}.npz",
                    masks=view.masks, scores=view.scores,
                )
        pair_rows.append(pair_row)
        if reviewed:
            relation_rows.extend(
                _score_pair(
                    reviewed[pair_id], south_view.masks, north_view.masks, matches
                )
            )
        if not args.no_previews:
            south_panel = render_side(
                south_view.image_rgb, south_view.masks, result, "south"
            )
            north_panel = render_side(
                north_view.image_rgb, north_view.masks, result, "north"
            )
            preview = _stack_panels(south_panel, north_panel)
            preview_path = output_dir / "previews" / f"{artifact_stem}.png"
            success, encoded = cv2.imencode(
                ".png", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
            )
            if not success:
                raise OSError(f"无法保存预览图 {preview_path}")
            preview_path.write_bytes(encoded.tobytes())
        print(
            f"[{index}/{len(pairs)}] {pair_id} {south_image.name} | "
            f"匹配 {len(matches)} | 南侧未匹配 {len(south_unmatched)} | "
            f"北侧未匹配 {len(north_unmatched)}",
            flush=True,
        )

    _write_csv(output_dir / "matches.csv", _MATCH_COLUMNS, match_rows)
    _write_csv(output_dir / "unmatched.csv", _UNMATCHED_COLUMNS, unmatched_rows)
    _write_csv(output_dir / "pair_summary.csv", _PAIR_COLUMNS, pair_rows)
    summary: dict[str, object] = {
        "weights": str(weights),
        "predict_config": str(predict_config),
        "postprocess_config": str(postprocess_config),
        "roi_mode": args.roi_mode,
        "roi_config": str(roi_config) if roi_config else None,
        "pair_count": len(pair_rows),
        "total_matches": len(match_rows),
        "total_unmatched": len(unmatched_rows),
        "elapsed_ms": round((perf_counter() - started_all) * 1000.0, 2),
    }
    if relation_rows:
        metrics = _aggregate_relations(relation_rows, args.gt_split)
        _write_csv(
            output_dir / "relation_pair_metrics.csv",
            _PAIR_RELATION_COLUMNS,
            relation_rows,
        )
        _write_csv(
            output_dir / "relation_metrics.csv", ("pairs", *_RELATION_COLUMNS), metrics
        )
        summary["relation_eval"] = {
            "gt_config": str(gt_config),
            "gt_root": str(gt_root),
            "eligible_pairs": len(reviewed),
            "skipped": skipped,
            "metrics": metrics,
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"结果目录 {output_dir}")


def _source_pairs() -> list[tuple[str, Path, Path]]:
    """按相机采集顺序返回普通 Demo 的南北原图路径。"""
    dataset_path = load_dataset_relative_path()
    pairs = match_pair_image(dataset_path)
    return [
        (str(index), dataset_path / "south" / f"{south}.png",
         dataset_path / "north" / f"{north}.png")
        for index, (south, north) in enumerate(
            zip(pairs["south"], pairs["north"], strict=True), start=1
        )
    ]


def _relation_counts(
    true_positive: int, pred_relations: int, gt_relations: int
) -> dict[str, int | float]:
    """由关系数和正确配对数计算指标。"""
    total = pred_relations + gt_relations
    return {
        "gt_relations": gt_relations,
        "pred_relations": pred_relations,
        "true_positive": true_positive,
        "false_positive": pred_relations - true_positive,
        "false_negative": gt_relations - true_positive,
        "precision": true_positive / pred_relations if pred_relations else 0.0,
        "recall": true_positive / gt_relations if gt_relations else 0.0,
        "f1": 2 * true_positive / total if total else 0.0,
    }


def _score_pair(
    pair: ReviewedPair,
    south_masks: NDArray[np.bool_],
    north_masks: NDArray[np.bool_],
    matches: list[RailMatch],
) -> list[dict[str, object]]:
    """在原图坐标下评分，并记录每侧 GT Mask 的覆盖。"""
    pred_relations = {
        (match["south"]["mask_index"], match["north"]["mask_index"])
        for match in matches
    }
    scores = load_relation_truth(pair).score(
        south_masks, north_masks, pred_relations, _IOU_THRESHOLDS,
    )
    return [
        {
            "pair_id": pair.pair_id, "split": pair.split, "iou_threshold": threshold,
            **_relation_counts(
                score.true_positive, score.pred_relations, score.gt_relations
            ),
            "south_gt_total": score.south_gt_total,
            "south_gt_covered": score.south_gt_covered,
            "north_gt_total": score.north_gt_total,
            "north_gt_covered": score.north_gt_covered,
        }
        for threshold, score in scores
    ]


def _aggregate_relations(
    rows: list[dict[str, object]], split: str
) -> list[dict[str, object]]:
    """按 IoU 门限汇总计数，再计算全划分的精确率、召回率和 F1。"""
    metrics = []
    for threshold in _IOU_THRESHOLDS:
        group = [row for row in rows if row["iou_threshold"] == threshold]
        metrics.append({
            "split": split, "iou_threshold": threshold, "pairs": len(group),
            **_relation_counts(
                sum(int(row["true_positive"]) for row in group),
                sum(int(row["pred_relations"]) for row in group),
                sum(int(row["gt_relations"]) for row in group),
            ),
        })
    return metrics


def _latest_weights() -> Path:
    """取 out/yolo_seg 下最新的实例分割权重。"""
    root = PROJECT_ROOT / "out" / "yolo_seg"
    candidates = [
        path for path in root.glob("*/weights/*.pt")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"{root} 下没有可用权重，请用 --weights 指定")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _stack_panels(
    south_panel: np.ndarray, north_panel: np.ndarray
) -> np.ndarray:
    """把两侧预览缩放到同一高度后水平拼接。"""
    height = max(south_panel.shape[0], north_panel.shape[0])
    resized = []
    for panel in (south_panel, north_panel):
        width = max(1, round(panel.shape[1] * height / panel.shape[0]))
        resized.append(
            cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
        )
    return np.hstack(resized)


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    """写出固定列顺序的 CSV，使用 LF 换行。"""
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
