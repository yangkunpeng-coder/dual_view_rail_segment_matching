"""关系级匹配评测：比较 raw 与去重后预测 Mask 的南北匹配效果。"""

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

import numpy as np

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_algorithm.matching.rail_matching_mslc import match_south_north_rail_mslc
from core_algorithm.postprocess import PostprocessConfig, run_postprocess
from data_preprocess.common.config import PROJECT_ROOT, config_path, load_yaml_mapping
from utils.relation_ground_truth import (
    ReviewedPair,
    excluded_pairs,
    load_relation_truth,
    read_reviewed_pairs,
)

_DEFAULT_CONFIG = "config/eval/postprocess_matching.yaml"
_DEFAULT_GT_CONFIG = "config/eval/rail_match_gt.yaml"
_DEFAULT_GT_ROOT = "data/rail_match_gt"
_DEFAULT_OUTPUT_ROOT = "out/postprocess_matching"
_DEFAULT_YOLO_ROOT = "out/yolo_seg"
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
_CONFIG_SNAPSHOTS = (
    "config/eval/postprocess_matching.yaml",
    "config/eval/seg_postprocess.yaml",
    "config/predict/rail_predict.yaml",
    "config/eval/rail_match_gt.yaml",
)


@dataclass(frozen=True)
class MatchingConfig:
    """关系级评测参数。"""

    algorithm: str
    thresholds: tuple[float, ...]
    variants: tuple[tuple[str, float | None], ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="关系级匹配评测")
    parser.add_argument("--valid-pred-dir", default=None)
    parser.add_argument("--test-pred-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=["valid"])
    parser.add_argument("--gt-root", default=_DEFAULT_GT_ROOT)
    parser.add_argument("--gt-config", default=_DEFAULT_GT_CONFIG)
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-excluded", action="store_true")
    args = parser.parse_args()

    config = _load_config(config_path(args.config))
    gt_root = Path(PROJECT_ROOT, args.gt_root)
    gt_config = load_yaml_mapping(config_path(args.gt_config))
    gt_excluded = {} if args.include_excluded else excluded_pairs(gt_config)
    prediction_dirs = {
        "valid": _raw_prediction_dir(
            _resolve_prediction_dir(args.valid_pred_dir, "valid")
        ),
        "test": _raw_prediction_dir(
            _resolve_prediction_dir(args.test_pred_dir, "test")
        ),
    }
    for split in args.splits:
        _warn_prediction_mode(prediction_dirs[split])
    pairs, load_skipped = read_reviewed_pairs(gt_root, args.splits, gt_excluded)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("没有可评测的图对")

    output_dir = (
        Path(PROJECT_ROOT, args.output_root) / datetime.now().strftime(
            _TIMESTAMP_FORMAT
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "configs").mkdir()
    (output_dir / "metrics").mkdir()
    (output_dir / "tables").mkdir()
    for relative in _CONFIG_SNAPSHOTS:
        try:
            source = config_path(relative)
        except FileNotFoundError:
            continue
        shutil.copy2(source, output_dir / "configs" / source.name)
    _write_change(output_dir, config, pairs)

    metric_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = list(load_skipped)
    for pair in pairs:
        split_pred_dir = prediction_dirs[pair.split]
        try:
            south_raw = _load_npz(split_pred_dir, "south", pair.south_image.stem)
            north_raw = _load_npz(split_pred_dir, "north", pair.north_image.stem)
        except FileNotFoundError as error:
            skipped.append({
                "pair_id": pair.pair_id, "split": pair.split,
                "reason": str(error),
            })
            continue
        truth = load_relation_truth(pair)
        if south_raw[0].shape[1:] != truth.gt_south.shape[1:]:
            raise ValueError(f"{pair.pair_id} 南侧预测与 GT 尺寸不一致")
        if north_raw[0].shape[1:] != truth.gt_north.shape[1:]:
            raise ValueError(f"{pair.pair_id} 北侧预测与 GT 尺寸不一致")
        variant_masks = _variant_masks(
            config, south_raw, north_raw
        )
        for name, (south_masks, north_masks) in variant_masks.items():
            pred_relations = _match_relations(
                pair, south_masks, north_masks
            )
            scores = truth.score(
                south_masks, north_masks, pred_relations, config.thresholds
            )
            for threshold, score in scores:
                pair_rows.append(_pair_row(
                    name, pair, threshold, score
                ))
                _append_relations(
                    relation_rows, name, pair, threshold, score,
                    pred_relations, truth.gt_relations,
                )
                coverage_rows.append(_coverage_row(
                    name, pair, threshold, score
                ))
                _accumulate(metric_rows, name, pair, threshold, score)

    _write_csv(output_dir / "metrics" / "metrics.csv", _METRIC_COLUMNS, metric_rows)
    _write_csv(
        output_dir / "metrics" / "pair_metrics.csv", _PAIR_COLUMNS, pair_rows
    )
    _write_csv(
        output_dir / "metrics" / "relations.csv",
        _RELATION_COLUMNS, relation_rows,
    )
    _write_csv(
        output_dir / "metrics" / "mapping_coverage.csv",
        _COVERAGE_COLUMNS, coverage_rows,
    )
    summary = _build_summary(metric_rows, pairs, skipped)
    (output_dir / "metrics" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_comparison(output_dir, metric_rows, config)
    _write_report(output_dir, metric_rows, summary, config)
    (output_dir / "run_manifest.json").write_text(
        json.dumps({
            "config": {
                "algorithm": config.algorithm,
                "thresholds": config.thresholds,
                "variants": config.variants,
            },
            "pairs": [pair.pair_id for pair in pairs],
            "skipped": skipped,
            "splits": sorted({pair.split for pair in pairs}),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"关系级评测结果 {output_dir}")


def _match_relations(
    pair: ReviewedPair,
    south_masks: object,
    north_masks: object,
) -> set[tuple[int, int]]:
    """在预测 Mask 上跑 MSLC，返回预测关系的下标集合。"""
    if not len(south_masks) or not len(north_masks):
        return set()
    result = match_south_north_rail_mslc(
        pair.south_image, pair.north_image, south_masks, north_masks
    )
    return {
        (int(match["south"]["mask_index"]), int(match["north"]["mask_index"]))
        for match in result["matches"]
    }


def _variant_masks(
    config: MatchingConfig,
    south_raw: tuple[object, object],
    north_raw: tuple[object, object],
) -> dict[str, tuple[object, object]]:
    """生成 raw 与各去重阈值下的两侧 Mask。"""
    variants: dict[str, tuple[object, object]] = {}
    for name, threshold in config.variants:
        if threshold is None:
            variants[name] = (south_raw[0], north_raw[0])
            continue
        post_config = PostprocessConfig(
            enabled=True, mask_nms_iou=threshold, min_area_px=8,
            quality_enabled=False,
        )
        variants[name] = (
            run_postprocess(south_raw[0], south_raw[1], post_config).masks,
            run_postprocess(north_raw[0], north_raw[1], post_config).masks,
        )
    return variants


def _load_npz(
    pred_dir: Path, side: str, stem: str
) -> tuple[object, object]:
    """读取一侧预测 NPZ，兼容带前缀和不带前缀的文件名。"""
    for name in (f"{stem}.npz", f"{side}_{stem}.npz"):
        path = pred_dir / side / name
        if path.is_file():
            data = np.load(path)
            return data["masks"], data["scores"]
    raise FileNotFoundError(f"{pred_dir / side} 缺少 {stem}.npz")


def _accumulate(
    rows: list[dict[str, object]],
    variant: str,
    pair: ReviewedPair,
    threshold: float,
    score: object,
) -> None:
    """按 (variant, split, threshold) 累加计数。"""
    key = (variant, pair.split, threshold)
    row = next(
        (item for item in rows if (
            item["variant"], item["split"], item["iou_threshold"]
        ) == key), None
    )
    if row is None:
        row = {
            "variant": variant, "split": pair.split,
            "iou_threshold": threshold,
            "pairs": 0, "gt_relations": 0, "pred_relations": 0,
            "true_positive": 0, "false_positive": 0, "false_negative": 0,
            "pair_exact": 0, "extra_on_single": 0, "unmapped_extra": 0,
            "south_gt_total": 0, "south_gt_covered": 0,
            "north_gt_total": 0, "north_gt_covered": 0,
            "south_pred_total": 0, "south_pred_covered": 0,
            "north_pred_total": 0, "north_pred_covered": 0,
        }
        rows.append(row)
    row["pairs"] = int(row["pairs"]) + 1
    row["gt_relations"] = int(row["gt_relations"]) + score.gt_relations
    row["pred_relations"] = int(row["pred_relations"]) + score.pred_relations
    row["true_positive"] = int(row["true_positive"]) + score.true_positive
    row["false_positive"] = int(row["false_positive"]) + score.false_positive
    row["false_negative"] = int(row["false_negative"]) + score.false_negative
    row["pair_exact"] = int(row["pair_exact"]) + int(score.pair_exact)
    row["extra_on_single"] = int(row["extra_on_single"]) + score.extra_on_single
    row["unmapped_extra"] = int(row["unmapped_extra"]) + score.unmapped_extra
    for key in (
        "south_gt_total", "south_gt_covered", "north_gt_total",
        "north_gt_covered", "south_pred_total", "south_pred_covered",
        "north_pred_total", "north_pred_covered",
    ):
        row[key] = int(row[key]) + int(getattr(score, key))


def _pair_row(
    variant: str, pair: ReviewedPair, threshold: float, score: object
) -> dict[str, object]:
    """生成单图对指标行。"""
    return {
        "variant": variant, "split": pair.split, "pair_id": pair.pair_id,
        "iou_threshold": threshold,
        "gt_relations": score.gt_relations,
        "pred_relations": score.pred_relations,
        "true_positive": score.true_positive,
        "false_positive": score.false_positive,
        "false_negative": score.false_negative,
        "precision": score.precision, "recall": score.recall, "f1": score.f1,
        "pair_exact": int(score.pair_exact),
        "extra_on_single": score.extra_on_single,
        "unmapped_extra": score.unmapped_extra,
        "south_gt_covered": score.south_gt_covered,
        "south_gt_total": score.south_gt_total,
        "north_gt_covered": score.north_gt_covered,
        "north_gt_total": score.north_gt_total,
        "south_pred_covered": score.south_pred_covered,
        "south_pred_total": score.south_pred_total,
        "north_pred_covered": score.north_pred_covered,
        "north_pred_total": score.north_pred_total,
    }


def _coverage_row(
    variant: str, pair: ReviewedPair, threshold: float, score: object
) -> dict[str, object]:
    """生成实例映射覆盖行。"""
    return {
        "variant": variant, "split": pair.split, "pair_id": pair.pair_id,
        "iou_threshold": threshold,
        "south_gt_covered": score.south_gt_covered,
        "south_gt_total": score.south_gt_total,
        "north_gt_covered": score.north_gt_covered,
        "north_gt_total": score.north_gt_total,
        "south_pred_covered": score.south_pred_covered,
        "south_pred_total": score.south_pred_total,
        "north_pred_covered": score.north_pred_covered,
        "north_pred_total": score.north_pred_total,
    }


def _append_relations(
    rows: list[dict[str, object]],
    variant: str,
    pair: ReviewedPair,
    threshold: float,
    score: object,
    pred_relations: set[tuple[int, int]],
    gt_relations: set[tuple[int, int]],
) -> None:
    """写出 TP、FN、FP 关系明细。"""
    matched = {
        (item[0], item[1]): item for item in score.matched_relations
    }
    for gt_south, gt_north in sorted(gt_relations):
        item = matched.get((gt_south, gt_north))
        if item is None:
            rows.append({
                "variant": variant, "split": pair.split,
                "pair_id": pair.pair_id, "iou_threshold": threshold,
                "status": "FN", "gt_south": gt_south, "gt_north": gt_north,
                "pred_south": None, "pred_north": None,
                "south_iou": None, "north_iou": None,
            })
        else:
            rows.append({
                "variant": variant, "split": pair.split,
                "pair_id": pair.pair_id, "iou_threshold": threshold,
                "status": "TP", "gt_south": gt_south, "gt_north": gt_north,
                "pred_south": item[2], "pred_north": item[3],
                "south_iou": item[4], "north_iou": item[5],
            })
    matched_pairs = {(item[2], item[3]) for item in score.matched_relations}
    for pred_south, pred_north in sorted(pred_relations - matched_pairs):
        rows.append({
            "variant": variant, "split": pair.split,
            "pair_id": pair.pair_id, "iou_threshold": threshold,
            "status": "FP", "gt_south": None, "gt_north": None,
            "pred_south": pred_south, "pred_north": pred_north,
            "south_iou": None, "north_iou": None,
        })


def _build_summary(
    rows: list[dict[str, object]],
    pairs: list[ReviewedPair],
    skipped: list[dict[str, object]],
) -> dict[str, object]:
    """汇总指标并给出去重是否保留的建议。"""
    summary_rows: list[dict[str, object]] = []
    for row in rows:
        gt = int(row["gt_relations"])
        pred = int(row["pred_relations"])
        tp = int(row["true_positive"])
        f1 = (
            2 * tp / (gt + pred) if gt + pred else 0.0
        )
        exact = int(row["pair_exact"]) / int(row["pairs"]) if row["pairs"] else 0.0
        south_gt = int(row["south_gt_total"])
        north_gt = int(row["north_gt_total"])
        south_pred = int(row["south_pred_total"])
        north_pred = int(row["north_pred_total"])
        summary_rows.append({
            "variant": row["variant"], "split": row["split"],
            "iou_threshold": row["iou_threshold"],
            "pairs": row["pairs"], "gt_relations": gt,
            "pred_relations": pred, "true_positive": tp,
            "false_positive": row["false_positive"],
            "false_negative": row["false_negative"],
            "precision": tp / pred if pred else 0.0,
            "recall": tp / gt if gt else 0.0,
            "f1": f1,
            "pair_exact_rate": exact,
            "south_gt_coverage": (
                int(row["south_gt_covered"]) / south_gt if south_gt else 0.0
            ),
            "north_gt_coverage": (
                int(row["north_gt_covered"]) / north_gt if north_gt else 0.0
            ),
            "south_pred_coverage": (
                int(row["south_pred_covered"]) / south_pred
                if south_pred else 0.0
            ),
            "north_pred_coverage": (
                int(row["north_pred_covered"]) / north_pred
                if north_pred else 0.0
            ),
            "extra_on_single": row["extra_on_single"],
            "unmapped_extra": row["unmapped_extra"],
        })
    return {
        "pairs": [pair.pair_id for pair in pairs],
        "skipped": skipped,
        "metrics": summary_rows,
        "warning": _mapping_warning(summary_rows),
    }


def _mapping_warning(rows: list[dict[str, object]]) -> str:
    """当 GT 覆盖不足时给出提示。"""
    values = [
        float(row["south_gt_coverage"]) for row in rows
        if row["variant"] != "raw" and float(row["iou_threshold"]) == 0.50
    ] + [
        float(row["north_gt_coverage"]) for row in rows
        if row["variant"] != "raw" and float(row["iou_threshold"]) == 0.50
    ]
    if values and min(values) < 0.80:
        return "去重后 GT 映射覆盖率低于 80%，F1 只能作参考"
    return ""


def _write_comparison(
    output_dir: Path,
    rows: list[dict[str, object]],
    config: MatchingConfig,
) -> None:
    """生成 raw 与去重变体的对比表。"""
    lines = ["# 关系级匹配对比", ""]
    summary = _build_summary(rows, [], [])
    for split in sorted({str(row["split"]) for row in rows}):
        lines.extend([
            f"## {split}", "",
            "| 变体 | IoU | 图对 | GT | 预测 | TP | FP | FN | P | R | F1 | "
            "整对正确 | GT覆盖(S/N) | 预测覆盖(S/N) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in summary["metrics"]:
            if row["split"] != split:
                continue
            lines.append(
                f"| {row['variant']} | {row['iou_threshold']:.2f} | "
                f"{row['pairs']} | {row['gt_relations']} | "
                f"{row['pred_relations']} | {row['true_positive']} | "
                f"{row['false_positive']} | {row['false_negative']} | "
                f"{_pct(row['precision'])} | {_pct(row['recall'])} | "
                f"{_pct(row['f1'])} | {_pct(row['pair_exact_rate'])} | "
                f"{_pct(row['south_gt_coverage'])}/"
                f"{_pct(row['north_gt_coverage'])} | "
                f"{_pct(row['south_pred_coverage'])}/"
                f"{_pct(row['north_pred_coverage'])} |"
            )
        lines.append("")
    (output_dir / "tables" / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _write_report(
    output_dir: Path,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    config: MatchingConfig,
) -> None:
    """生成文字结论。"""
    lines = [
        "# 关系级匹配评测报告", "",
        f"算法：`{config.algorithm}`；变体："
        f"{', '.join(name for name, _ in config.variants)}", "",
    ]
    warning = summary.get("warning")
    if warning:
        lines.extend([f"> {warning}", ""])
    rows = summary.get("metrics", [])
    lookup = {
        (str(row["variant"]), str(row["split"]), float(row["iou_threshold"])): row
        for row in rows
    }
    lines.extend(["## 对比", ""])
    for split in sorted({str(row["split"]) for row in rows}):
        for threshold in config.thresholds:
            baseline = lookup.get(("raw", split, threshold))
            if baseline is None:
                continue
            for name, _ in config.variants:
                if name == "raw":
                    continue
                candidate = lookup.get((name, split, threshold))
                if candidate is None:
                    continue
                lines.append(
                    f"- {split} τ={threshold:.2f}：{name} F1 "
                    f"{_pct(candidate['f1'])}（raw "
                    f"{_pct(baseline['f1'])}），召回 "
                    f"{_pct(candidate['recall'])}（raw "
                    f"{_pct(baseline['recall'])}），GT 覆盖 "
                    f"{_pct(candidate['south_gt_coverage'])}/"
                    f"{_pct(candidate['north_gt_coverage'])}"
                )
    lines.extend(["", "## 结论", ""])
    conclusion = _conclusion(lookup, config)
    lines.extend(conclusion)
    skipped = summary.get("skipped", [])
    if skipped:
        lines.extend(["", "## 跳过的图对", ""])
        for item in skipped[:20]:
            lines.append(
                f"- {item.get('pair_id')}（{item.get('split')}）："
                f"{item.get('reason')}"
            )
    lines.extend([
        "", "## 说明", "",
        "- 关系级评测不要求预测下标与 GT 下标一致；"
        "两侧 Mask IoU 同时达到门限才连边。",
        "- 一个预测关系最多认领一个 GT 关系；"
        "一个预测覆盖两根 GT 时，另一个会记为 FN。",
        "- 该评测只比较匹配输入的作用，不改匹配算法。",
    ])
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _conclusion(
    lookup: dict[tuple[str, str, float], dict[str, object]],
    config: MatchingConfig,
) -> list[str]:
    """根据 raw 与去重变体差值生成结论。"""
    lines: list[str] = []
    best_ratio = 0.0
    for split in sorted({key[1] for key in lookup}):
        for threshold in config.thresholds:
            baseline = lookup.get(("raw", split, threshold))
            if baseline is None:
                continue
            for name, _ in config.variants:
                if name == "raw":
                    continue
                candidate = lookup.get((name, split, threshold))
                if candidate is None:
                    continue
                delta = float(candidate["f1"]) - float(baseline["f1"])
                coverage_kept = (
                    float(candidate["south_gt_coverage"])
                    >= float(baseline["south_gt_coverage"]) - 0.01
                    and float(candidate["north_gt_coverage"])
                    >= float(baseline["north_gt_coverage"]) - 0.01
                )
                best_ratio = max(best_ratio, float(candidate["f1"]))
                if delta >= -0.01 and coverage_kept:
                    lines.append(
                        f"- {split} τ={threshold:.2f}：{name} 关系 F1 "
                        f"提升 {delta * 100:+.2f} 点，覆盖未下降，"
                        "可以保留该去重阈值。"
                    )
                else:
                    lines.append(
                        f"- {split} τ={threshold:.2f}：{name} 关系 F1 "
                        f"变化 {delta * 100:+.2f} 点，覆盖"
                        f"{'未下降' if coverage_kept else '下降'}，"
                        "建议回退或检查误抑制。"
                    )
    if best_ratio < 0.50:
        lines.append(
            "- 当前关系 F1 仍低于 50%，说明瓶颈不只是重复实例；"
            "预测 Mask 的合并/缺失和匹配拒绝同样是主因。"
        )
    return lines


def _write_change(
    output_dir: Path, config: MatchingConfig, pairs: list[ReviewedPair]
) -> None:
    """写入本次评测说明。"""
    lines = [
        "# 关系级匹配评测", "",
        "## 本次方案", "",
        f"- 算法：{config.algorithm}",
        f"- 变体：{config.variants}",
        f"- 图对：{len(pairs)}", "",
        "## 目的", "",
        "比较同侧去重是否改善南北匹配，不要求预测下标与 GT 下标一致。",
    ]
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
            "关系级结果不能与当前 one-to-one 配置直接比较"
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


def _load_config(path: Path) -> MatchingConfig:
    """读取关系级评测配置。"""
    mapping = load_yaml_mapping(path)
    algorithm = str(mapping.get("algorithm", "rail_matching_mslc"))
    if algorithm != "rail_matching_mslc":
        raise ValueError(f"当前只支持 rail_matching_mslc，收到 {algorithm}")
    thresholds = mapping.get("iou_thresholds", [0.50, 0.75])
    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("iou_thresholds 必须是非空列表")
    values = tuple(float(value) for value in thresholds)
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("iou_thresholds 必须位于 [0, 1]")
    variants = mapping.get("variants", {"raw": None, "post_065": 0.65})
    if not isinstance(variants, dict) or "raw" not in variants:
        raise ValueError("variants 必须包含 raw")
    parsed: list[tuple[str, float | None]] = []
    for name, threshold in variants.items():
        if threshold is None:
            parsed.append((str(name), None))
        else:
            value = float(threshold)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} 的阈值必须在 [0, 1]")
            parsed.append((str(name), value))
    return MatchingConfig(algorithm, values, tuple(parsed))


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
    """比例格式化为百分数。"""
    return f"{float(value) * 100:.2f}%"


_METRIC_COLUMNS = (
    "variant", "split", "iou_threshold", "pairs", "gt_relations",
    "pred_relations", "true_positive", "false_positive", "false_negative",
    "pair_exact", "extra_on_single", "unmapped_extra",
    "south_gt_total", "south_gt_covered", "north_gt_total",
    "north_gt_covered", "south_pred_total", "south_pred_covered",
    "north_pred_total", "north_pred_covered",
)
_PAIR_COLUMNS = (
    "variant", "split", "pair_id", "iou_threshold", "gt_relations",
    "pred_relations", "true_positive", "false_positive", "false_negative",
    "precision", "recall", "f1", "pair_exact", "extra_on_single",
    "unmapped_extra", "south_gt_covered", "south_gt_total",
    "north_gt_covered", "north_gt_total", "south_pred_covered",
    "south_pred_total", "north_pred_covered", "north_pred_total",
)
_RELATION_COLUMNS = (
    "variant", "split", "pair_id", "iou_threshold", "status",
    "gt_south", "gt_north", "pred_south", "pred_north",
    "south_iou", "north_iou",
)
_COVERAGE_COLUMNS = (
    "variant", "split", "pair_id", "iou_threshold",
    "south_gt_covered", "south_gt_total", "north_gt_covered",
    "north_gt_total", "south_pred_covered", "south_pred_total",
    "north_pred_covered", "north_pred_total",
)


if __name__ == "__main__":
    main()
