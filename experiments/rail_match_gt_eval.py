"""用人工确认的南北匹配真值评测匹配算法。

两种入口：

- ``--run <评测目录>``：读取既有评测运行（每个含 ``match_pairs.csv`` 的子目录算一个
  算法），只做真值评分，不重新跑算法。算法名取目录名。
- ``--algorithms <名称...>``：在当前进程中直接跑已注册算法，掩码用真值清单里的
  LabelMe 标注，用于改动前后快速 A/B。

产物默认写到 ``out/rail_match_label/gt/<时间戳>/``：``gt_metrics.csv``、
``gt_pair_level.csv``、``gt_errors.csv``、``gt_threshold_curve.csv``、``gt_report.md``、
``run_manifest.json``。真值口径与 ``config/eval/rail_match_gt.yaml`` 一致：只统计
``review.status = done`` 的图对，``uncertain`` 不进分子分母；配置里
``excluded_pairs`` 列出的"标注有问题"图对整对剔除（原因写入运行清单），
``--include-excluded`` 可恢复。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preprocess.common.config import (
    PROJECT_ROOT,
    config_path as resolve_config,
    load_yaml_mapping,
)
from experiments import rail_match_eval as ev
from utils.match_gt import read_pair_annotation, wilson_interval

_GT_ROOT = PROJECT_ROOT / "data" / "rail_match_gt"
_GT_MANIFEST = _GT_ROOT / "manifest.csv"
_GT_CONFIG = resolve_config("rail_match_gt.yaml")
_GT_SPLITS = ("valid", "test")
_OUTPUT_ROOT = PROJECT_ROOT / "out" / "rail_match_label" / "gt"
_WILSON_LEVEL = 0.95
_SPLIT_ORDER = ("all", *_GT_SPLITS)
_CURVE_LIMIT = 200

Relation = tuple[int, int]
RelationRow = tuple[int, int, float]
_METRIC_COLUMNS = (
    "algorithm", "split", "pairs", "gold", "predicted", "true_positive",
    "false_positive", "false_negative", "missing", "swap", "extra_on_single",
    "unmapped_extra", "precision", "recall", "f1", "false_match_rate",
    "coverage", "instances", "covered_instances", "abstain_rate",
    "pair_exact_count", "pair_exact_rate", "precision_ci_low",
    "precision_ci_high", "recall_ci_low", "recall_ci_high",
)
_PAIR_COLUMNS = (
    "algorithm", "pair_id", "split", "gold", "predicted", "true_positive",
    "false_positive", "false_negative", "precision", "recall", "f1",
    "pair_exact",
)
_ERROR_COLUMNS = (
    "algorithm", "pair_id", "split", "category", "south_mask_index",
    "north_mask_index", "partner_north_mask_index", "note",
)
_CURVE_COLUMNS = (
    "algorithm", "threshold", "relations", "true_positive", "false_positive",
    "precision", "coverage",
)


@dataclass
class _Truth:
    """一个图对的真值，坐标均为该侧输入的 mask_index。"""

    split: str
    paired: set[Relation] = field(default_factory=set)
    south_only: set[int] = field(default_factory=set)
    north_only: set[int] = field(default_factory=set)
    south_instances: set[int] = field(default_factory=set)
    north_instances: set[int] = field(default_factory=set)


@dataclass
class _Score:
    """一个算法在一个划分上的混淆计数与派生指标。"""

    pairs: int = 0
    gold: int = 0
    predicted: int = 0
    true_positive: int = 0
    missing: int = 0
    swap: int = 0
    extra_on_single: int = 0
    unmapped_extra: int = 0
    instances: int = 0
    covered_instances: int = 0
    exact_pairs: int = 0

    @property
    def false_positive(self) -> int:
        """错误输出的关系数。"""
        return self.swap + self.extra_on_single + self.unmapped_extra

    @property
    def false_negative(self) -> int:
        """漏掉的真值关系数。"""
        return self.missing

    def absorb(self, other: "_Score") -> None:
        """累加另一个计分。"""
        for name in (
            "pairs", "gold", "predicted", "true_positive", "missing", "swap",
            "extra_on_single", "unmapped_extra", "instances", "covered_instances",
            "exact_pairs",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def ratios(self) -> tuple[float, float, float]:
        """返回精确率、召回率与 F1。"""
        precision = (
            self.true_positive / self.predicted if self.predicted else float("nan")
        )
        recall = self.true_positive / self.gold if self.gold else float("nan")
        if not precision or not recall:
            return precision, recall, 0.0
        return precision, recall, 2 * precision * recall / (precision + recall)


def _mask_by_instance(payload: dict, side: str) -> dict[str, int]:
    """真值实例编号到输入 mask_index 的映射。"""
    records = (payload.get("instances") or {}).get(side, [])
    return {
        str(record["instance_id"]): int(record["mask_index"])
        for record in records
        if record.get("mask_index") is not None
    }


def _load_truth() -> dict[str, _Truth]:
    """读取已确认的真值关系，按图对分组。"""
    if not _GT_MANIFEST.is_file():
        raise FileNotFoundError(f"缺少真值清单：{_GT_MANIFEST}")
    truths: dict[str, _Truth] = {}
    for split in _GT_SPLITS:
        directory = _GT_ROOT / split
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*_association.json")):
            payload = read_pair_annotation(path)
            review = payload.get("review") or {}
            if review.get("status") != "done":
                continue
            pair_id = str(payload["pair_id"])
            truth = _Truth(str(payload.get("split") or split))
            south = _mask_by_instance(payload, "south")
            north = _mask_by_instance(payload, "north")
            truth.south_instances = set(south.values())
            truth.north_instances = set(north.values())
            for match in payload.get("matches") or []:
                south_index = south.get(str(match["south_instance_id"]))
                north_index = north.get(str(match["north_instance_id"]))
                if south_index is not None and north_index is not None:
                    truth.paired.add((south_index, north_index))
            for entry in payload.get("unmatched") or []:
                side = str(entry.get("camera", ""))
                table = south if side == "south" else north
                index = table.get(str(entry["instance_id"]))
                if index is None:
                    continue
                if side == "south":
                    truth.south_only.add(index)
                else:
                    truth.north_only.add(index)
            truths[pair_id] = truth
    return truths


def _excluded_pairs(config_path: Path) -> dict[str, str]:
    """读取"标注有问题"的图对与原因。"""
    if not config_path.is_file():
        return {}
    config = load_yaml_mapping(config_path)
    raw = config.get("excluded_pairs") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} 的 excluded_pairs 须为 pair_id 到原因的映射")
    return {str(pair_id): str(reason) for pair_id, reason in raw.items()}


def _apply_exclusions(
    truths: dict[str, _Truth], excluded: dict[str, str]
) -> tuple[dict[str, _Truth], dict[str, str], list[str]]:
    """整对剔除标注有问题的图对，返回保留项、剔除项与清单里不存在的编号。"""
    kept = {pair_id: truth for pair_id, truth in truths.items()
            if pair_id not in excluded}
    dropped = {pair_id: reason for pair_id, reason in excluded.items()
               if pair_id in truths}
    unknown = sorted(set(excluded) - set(truths))
    return kept, dropped, unknown


def _score_pair(
    truth: _Truth, predicted: set[Relation]
) -> tuple[_Score, list[dict[str, object]], bool]:
    """把一个图对的输出与真值比较，返回计分、逐条误差与整对是否正确。"""
    score = _Score(pairs=1, gold=len(truth.paired), predicted=len(predicted))
    score.true_positive = len(predicted & truth.paired)
    score.missing = len(truth.paired - predicted)
    south_paired = {south for south, _ in truth.paired}
    north_paired = {north for _, north in truth.paired}
    partner_by_south = {south: north for south, north in truth.paired}
    score.instances = len(truth.south_instances) + len(truth.north_instances)
    covered_south = {south for south, _ in predicted} & truth.south_instances
    covered_north = {north for _, north in predicted} & truth.north_instances
    score.covered_instances = len(covered_south) + len(covered_north)
    errors: list[dict[str, object]] = []
    for south, north in sorted(truth.paired - predicted):
        errors.append({
            "category": "missing", "south_mask_index": south,
            "north_mask_index": north, "partner_north_mask_index": None,
            "note": "",
        })
    for south, north in sorted(predicted - truth.paired):
        if south in truth.south_only or north in truth.north_only:
            category = "extra_on_single"
        elif south in south_paired or north in north_paired:
            category = "swap"
        else:
            category = "unmapped_extra"
        setattr(score, category, getattr(score, category) + 1)
        errors.append({
            "category": category, "south_mask_index": south,
            "north_mask_index": north,
            "partner_north_mask_index": partner_by_south.get(south),
            "note": "",
        })
    exact = predicted == truth.paired
    score.exact_pairs = 1 if exact else 0
    return score, errors, exact


def _score_algorithm(
    truths: dict[str, _Truth], relations: dict[str, list[RelationRow]]
) -> tuple[dict[str, _Score], list[dict], list[dict], list[tuple[float, bool]]]:
    """按划分汇总一个算法在全部真值图对上的表现。"""
    buckets: dict[str, _Score] = {}
    errors: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    scored: list[tuple[float, bool]] = []
    for pair_id, truth in sorted(truths.items()):
        rows = relations.get(pair_id, [])
        predicted = {(south, north) for south, north, _ in rows}
        score, pair_errors, exact = _score_pair(truth, predicted)
        for bucket in ("all", truth.split):
            if bucket:
                buckets.setdefault(bucket, _Score()).absorb(score)
        precision, recall, f1 = score.ratios()
        pair_rows.append({
            "pair_id": pair_id, "split": truth.split, "gold": score.gold,
            "predicted": score.predicted, "true_positive": score.true_positive,
            "false_positive": score.false_positive,
            "false_negative": score.false_negative,
            "precision": precision, "recall": recall, "f1": f1,
            "pair_exact": exact,
        })
        for entry in pair_errors:
            errors.append({"pair_id": pair_id, "split": truth.split, **entry})
        for south, north, value in rows:
            scored.append((value, (south, north) in truth.paired))
    return buckets, errors, pair_rows, scored


def _relation_rows_from_csv(
    path: Path, pairs: set[str]
) -> dict[str, list[RelationRow]]:
    """读取一次评测运行的 match_pairs.csv，保留位置残差作为门限代理。"""
    relations: dict[str, list[RelationRow]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            pair_id = row.get("pair_id", "")
            if pair_id not in pairs:
                continue
            try:
                south = int(row["south_mask_index"])
                north = int(row["north_mask_index"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                score = float(row.get("position_error_norm") or "nan")
            except ValueError:
                score = float("nan")
            relations.setdefault(pair_id, []).append((south, north, score))
    return relations


def _metric_row(algorithm: str, split: str, score: _Score) -> dict[str, object]:
    """把计分展开成一行指标。"""
    precision, recall, f1 = score.ratios()
    precision_low, precision_high = wilson_interval(
        score.true_positive, score.predicted, _WILSON_LEVEL
    ) if score.predicted else (float("nan"), float("nan"))
    recall_low, recall_high = wilson_interval(
        score.true_positive, score.gold, _WILSON_LEVEL
    ) if score.gold else (float("nan"), float("nan"))
    return {
        "algorithm": algorithm,
        "split": split,
        "pairs": score.pairs,
        "gold": score.gold,
        "predicted": score.predicted,
        "true_positive": score.true_positive,
        "false_positive": score.false_positive,
        "false_negative": score.false_negative,
        "missing": score.missing,
        "swap": score.swap,
        "extra_on_single": score.extra_on_single,
        "unmapped_extra": score.unmapped_extra,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_match_rate": 1 - precision if score.predicted else float("nan"),
        "coverage": score.predicted / score.gold if score.gold else float("nan"),
        "instances": score.instances,
        "covered_instances": score.covered_instances,
        "abstain_rate": (
            1 - score.covered_instances / score.instances
            if score.instances else float("nan")
        ),
        "pair_exact_count": score.exact_pairs,
        "pair_exact_rate": (
            score.exact_pairs / score.pairs if score.pairs else float("nan")
        ),
        "precision_ci_low": precision_low,
        "precision_ci_high": precision_high,
        "recall_ci_low": recall_low,
        "recall_ci_high": recall_high,
    }


def _curve_rows(
    algorithm: str, scored: list[tuple[float, bool]], gold: int
) -> list[dict[str, object]]:
    """按位置残差从松到严累计精度与覆盖率。"""
    values = sorted(
        (score, label) for score, label in scored if score == score
    )
    if not values or not gold:
        return []
    total = len(values)
    step = max(1, total // _CURVE_LIMIT)
    rows: list[dict[str, object]] = []
    true_positive = false_positive = 0
    for index, (threshold, label) in enumerate(values, start=1):
        if label:
            true_positive += 1
        else:
            false_positive += 1
        if index % step and index != total:
            continue
        relations = true_positive + false_positive
        rows.append({
            "algorithm": algorithm,
            "threshold": threshold,
            "relations": relations,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "precision": true_positive / relations if relations else float("nan"),
            "coverage": relations / gold,
        })
    return rows


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    """写出 CSV，空值保留为空单元格。"""
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: "" if row.get(name) is None else row.get(name)
                for name in columns
            })


def _report(scores: dict[str, dict[str, _Score]]) -> str:
    """生成 Markdown 报告。"""
    header = (
        "| 算法 | 图对 | 真值 | 输出 | TP | FP | FN | P | R | F1 | "
        "FMR | 覆盖率 | 弃答率 | 整对正确 |"
    )
    divider = "| " + " | ".join(["---", *[":---:"] * 13]) + " |"
    lines = [
        "# 匹配算法真值评测",
        "",
        "真值：`data/rail_match_gt`（人工确认的 paired / south_only / north_only）。",
        "P、R 的方括号为 Wilson 95% 置信区间；uncertain 关系不计入。",
        "",
    ]
    for split in _SPLIT_ORDER:
        rows = [
            (name, buckets[split]) for name, buckets in scores.items()
            if split in buckets
        ]
        if not rows:
            continue
        lines.extend([f"## {split}", "", header, divider])
        for name, score in sorted(rows, key=lambda item: -item[1].ratios()[2]):
            precision, recall, f1 = score.ratios()
            precision_low, precision_high = wilson_interval(
                score.true_positive, score.predicted, _WILSON_LEVEL
            ) if score.predicted else (float("nan"), float("nan"))
            recall_low, recall_high = wilson_interval(
                score.true_positive, score.gold, _WILSON_LEVEL
            ) if score.gold else (float("nan"), float("nan"))
            coverage = score.predicted / score.gold if score.gold else float("nan")
            abstain = (
                1 - score.covered_instances / score.instances
                if score.instances else float("nan")
            )
            exact = score.exact_pairs / score.pairs if score.pairs else float("nan")
            lines.append(
                f"| {name} | {score.pairs} | {score.gold} | {score.predicted} | "
                f"{score.true_positive} | {score.false_positive} | "
                f"{score.false_negative} | "
                f"{precision:.3f} [{precision_low:.3f}, {precision_high:.3f}] | "
                f"{recall:.3f} [{recall_low:.3f}, {recall_high:.3f}] | {f1:.3f} | "
                f"{1 - precision:.3f} | {coverage:.3f} | {abstain:.3f} | "
                f"{exact:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_artifacts(
    scores: dict[str, dict[str, _Score]],
    pair_rows: list[dict], errors: list[dict], curves: list[dict],
    metadata: dict[str, object],
) -> Path:
    """把指标、逐对结果、误差、门限曲线与运行信息写到时间戳目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = _OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    metric_rows = [
        _metric_row(name, split, buckets[split])
        for name, buckets in scores.items() for split in _SPLIT_ORDER
        if split in buckets
    ]
    _write_csv(output_dir / "gt_metrics.csv", _METRIC_COLUMNS, metric_rows)
    _write_csv(output_dir / "gt_pair_level.csv", _PAIR_COLUMNS, pair_rows)
    _write_csv(output_dir / "gt_errors.csv", _ERROR_COLUMNS, errors)
    _write_csv(output_dir / "gt_threshold_curve.csv", _CURVE_COLUMNS, curves)
    (output_dir / "gt_report.md").write_text(
        _report(scores), encoding="utf-8", newline="\n"
    )
    manifest = {
        **metadata,
        "timestamp": timestamp,
        "output_dir": str(output_dir),
        "algorithms": sorted(scores),
        "metric_path": str(output_dir / "gt_metrics.csv"),
        "pair_level_path": str(output_dir / "gt_pair_level.csv"),
        "error_path": str(output_dir / "gt_errors.csv"),
        "curve_path": str(output_dir / "gt_threshold_curve.csv"),
        "report_path": str(output_dir / "gt_report.md"),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return output_dir


def _evaluate_run(
    run_dir: Path, truths: dict[str, _Truth]
) -> tuple[dict[str, dict[str, _Score]], list[dict], list[dict], list[dict]]:
    """对既有评测运行逐算法评分。"""
    scores: dict[str, dict[str, _Score]] = {}
    pair_rows: list[dict] = []
    errors: list[dict] = []
    curves: list[dict] = []
    pair_ids = set(truths)
    for directory in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        summary = directory / "match_pairs.csv"
        if not summary.is_file():
            continue
        relations = _relation_rows_from_csv(summary, pair_ids)
        if not relations:
            continue
        buckets, algorithm_errors, algorithm_pairs, scored = _score_algorithm(
            truths, relations
        )
        gold = buckets["all"].gold
        scores[directory.name] = buckets
        pair_rows.extend(
            {"algorithm": directory.name, **row} for row in algorithm_pairs
        )
        errors.extend(
            {"algorithm": directory.name, **row} for row in algorithm_errors
        )
        curves.extend(_curve_rows(directory.name, scored, gold))
    return scores, pair_rows, errors, curves


def _evaluate_process(
    algorithms: tuple[str, ...], truths: dict[str, _Truth]
) -> tuple[dict[str, dict[str, _Score]], list[dict], list[dict], list[dict]]:
    """直接运行算法并评分（每个图对每个算法只跑一次）。"""
    pairs = {pair.pair_id: pair for pair in ev._read_pairs(_GT_MANIFEST)}
    scores: dict[str, dict[str, _Score]] = {name: {} for name in algorithms}
    pair_rows: list[dict] = []
    errors: list[dict] = []
    scored: dict[str, list[tuple[float, bool]]] = {name: [] for name in algorithms}
    for pair_id, truth in sorted(truths.items()):
        data = ev._load_pair(pairs[pair_id])
        for name in algorithms:
            result = ev._ALGORITHMS[name].match(
                data.south_path, data.north_path, data.south_masks, data.north_masks
            )
            rows = [
                (match["south"]["mask_index"], match["north"]["mask_index"],
                 float(match["position_error"]))
                for match in result["matches"]
            ]
            predicted = {(south, north) for south, north, _ in rows}
            score, pair_errors, exact = _score_pair(truth, predicted)
            for bucket in ("all", truth.split):
                if bucket:
                    scores[name].setdefault(bucket, _Score()).absorb(score)
            precision, recall, f1 = score.ratios()
            pair_rows.append({
                "algorithm": name, "pair_id": pair_id, "split": truth.split,
                "gold": score.gold, "predicted": score.predicted,
                "true_positive": score.true_positive,
                "false_positive": score.false_positive,
                "false_negative": score.false_negative,
                "precision": precision, "recall": recall, "f1": f1,
                "pair_exact": exact,
            })
            errors.extend(
                {"algorithm": name, "pair_id": pair_id, "split": truth.split, **entry}
                for entry in pair_errors
            )
            scored[name].extend(
                (value, (south, north) in truth.paired)
                for south, north, value in rows
            )
    curves: list[dict] = []
    for name in algorithms:
        curves.extend(
            _curve_rows(name, scored[name], scores[name]["all"].gold)
        )
    return scores, pair_rows, errors, curves


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="既有评测运行目录（含各算法子目录）")
    parser.add_argument(
        "--algorithms", nargs="+", choices=sorted(ev._ALGORITHMS),
        help="直接运行的已注册算法名",
    )
    parser.add_argument("--output", type=Path, help="自定义产物根目录")
    parser.add_argument(
        "--config", type=Path, default=_GT_CONFIG,
        help="真值配置路径，默认 config/eval/rail_match_gt.yaml",
    )
    parser.add_argument(
        "--include-excluded", action="store_true",
        help="仍然评测配置中标注有问题的图对（默认整对剔除）",
    )
    args = parser.parse_args()
    if (args.run is None) == (args.algorithms is None):
        parser.error("--run 与 --algorithms 必须且只能指定一个")

    global _OUTPUT_ROOT
    if args.output is not None:
        _OUTPUT_ROOT = Path(PROJECT_ROOT, args.output)
    config_path = resolve_config(args.config)
    excluded = {} if args.include_excluded else _excluded_pairs(config_path)
    truths = _load_truth()
    truths, dropped, unknown = _apply_exclusions(truths, excluded)
    if args.run is not None:
        run_dir = Path(PROJECT_ROOT, args.run).resolve()
        if not run_dir.is_dir():
            parser.error(f"评测目录不存在：{run_dir}")
        scores, pair_rows, errors, curves = _evaluate_run(run_dir, truths)
        metadata = {"source": "run", "run_dir": str(run_dir)}
    else:
        scores, pair_rows, errors, curves = _evaluate_process(
            tuple(args.algorithms), truths
        )
        metadata = {"source": "in_process", "algorithms": list(args.algorithms)}
    if not scores:
        parser.error("没有找到可评分的算法输出")
    metadata["truth_pairs"] = len(truths)
    metadata["truth_relations"] = sum(len(truth.paired) for truth in truths.values())
    metadata["excluded_pairs"] = dropped
    metadata["unknown_excluded_pairs"] = unknown
    metadata["config_path"] = str(config_path)
    output_dir = _write_artifacts(
        scores, pair_rows, errors, curves, metadata
    )
    print(_report(scores), end="")
    if dropped:
        detail = "、".join(f"{pair_id}（{reason}）" for pair_id, reason in dropped.items())
        print(f"已整对剔除 {len(dropped)} 个标注有问题图对：{detail}")
    if unknown:
        print(f"配置中的图对不在真值清单里，已忽略：{', '.join(unknown)}")
    print(f"产物目录：{output_dir}")


if __name__ == "__main__":
    main()
