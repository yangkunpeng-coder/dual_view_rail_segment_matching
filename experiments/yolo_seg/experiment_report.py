"""汇总 YOLO 与 SAM3 的实验结果，生成 Markdown 与 CSV 报告。

只读取实验目录，不修改任何已有产物。清单来自
config/eval/experiment_report.yaml，未列出的实验只在报告末尾提示。

用法（项目根目录，或在 PyCharm 中直接运行）：

    python experiments/yolo_seg/experiment_report.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

import yaml

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preprocess.common.config import PROJECT_ROOT, config_path, load_yaml_mapping

_REPORT_CONFIG = config_path("experiment_report.yaml")
_OUTPUT_ROOT = PROJECT_ROOT / "out" / "seg_report"
_YOLO_ROOT = PROJECT_ROOT / "out" / "yolo_seg"
_SAM3_ROOT = PROJECT_ROOT / "out" / "sam3_seg"
_WEIGHT_NAMES = ("rail_yolo26m-seg.pt", "best.pt", "last.pt")
_RUNNING_WINDOW_SECONDS = 1800


def _is_running(run_dir: Path) -> bool:
    """实验目录最近半小时内是否仍有写入，用于区分训练中与历史未完成。"""
    mtimes = [
        path.stat().st_mtime
        for path in (
            run_dir / "results.csv",
            run_dir / "logs" / "results.csv",
            run_dir / "weights" / "last.pt",
        )
        if path.is_file()
    ]
    return bool(mtimes) and (time.time() - max(mtimes)) < _RUNNING_WINDOW_SECONDS


@dataclass
class Metrics:
    """一次评价的四项 Mask 指标，缺失项为 None。"""

    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None


@dataclass
class RunResult:
    """一个实验在报告中的一行。"""

    source: str
    experiment: str
    status: str = ""
    dataset: str = "clop_seg"
    structure: str = ""
    prompt: str = ""
    epochs: str = ""
    best_epoch: str = ""
    mask_ratio: str = ""
    imgsz: str = ""
    batch: str = ""
    optimizer: str = ""
    lr0: str = ""
    cos_lr: str = ""
    conf: str = ""
    native: Metrics = field(default_factory=Metrics)
    valid: Metrics = field(default_factory=Metrics)
    test: Metrics = field(default_factory=Metrics)
    evaluator: str = ""
    dataset_sha256: str = ""
    image_count: str = ""
    weights: str = ""
    plan: str = ""


def _read_metrics(path: Path) -> Metrics | None:
    """读取 evaluation.csv，返回四项指标。"""
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    row = rows[-1]
    return Metrics(
        precision=float(row.get("precision", "") or 0.0),
        recall=float(row.get("recall", "") or 0.0),
        map50=float(row.get("mask_map50", "") or 0.0),
        map50_95=float(row.get("mask_map50_95", "") or 0.0),
    )


def _find_metrics(run_dir: Path, split: str) -> Metrics | None:
    """在 run_dir/<split> 下找最新的 evaluation.csv。"""
    candidates = sorted(
        (run_dir / split).glob("**/metrics/evaluation.csv"),
        key=lambda path: path.parent.parent.name,
    )
    if not candidates:
        return None
    return _read_metrics(candidates[-1])


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    return loaded if isinstance(loaded, dict) else {}


def _read_yaml(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    return loaded if isinstance(loaded, dict) else {}


def _weight_path(run_dir: Path) -> Path | None:
    for name in _WEIGHT_NAMES:
        candidate = run_dir / "weights" / name
        if candidate.is_file():
            return candidate
    return None


def _locate(run_dir: Path, name: str) -> Path | None:
    """实验根目录或 logs 子目录下的文件，训练中尚未归档时在根目录。"""
    for candidate in (run_dir / "logs" / name, run_dir / name):
        if candidate.is_file():
            return candidate
    return None


def _best_epoch(results_csv: Path) -> str:
    if not results_csv.is_file():
        return ""
    with results_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    best_row, best_value = None, -1.0
    for row in rows:
        try:
            value = float(row.get("metrics/mAP50-95(M)", "") or "")
        except ValueError:
            continue
        if value > best_value:
            best_row, best_value = row, value
    if best_row is None:
        return str(len(rows))
    return f"{best_row.get('epoch', '')} / {len(rows)}"


def _change_plan(change_md: Path) -> str:
    """取 change.md 中「本次方案」一节的完整正文，合并为一段。"""
    if not change_md.is_file():
        return ""
    lines = change_md.read_text(encoding="utf-8-sig").splitlines()
    collecting = False
    parts: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## 本次方案"):
            collecting = True
            continue
        if collecting:
            if stripped.startswith("## "):
                break
            if stripped and stripped != "待填写":
                parts.append(stripped)
    return " ".join(parts)


def _relative(path: Path | str | None) -> str:
    """把实验内的文件路径转成项目相对路径，便于直接定位。"""
    if not path:
        return ""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _status(has_weight: bool, *metrics: Metrics | None) -> str:
    if not has_weight:
        return "无权重"
    missing = [
        name for name, value in zip(("valid", "test"), metrics) if value is None
    ]
    if len(missing) == 2 and all(item is None for item in metrics):
        return "缺 valid、test"
    if missing:
        return "缺 " + "、".join(missing)
    return "完整"


def _collect_yolo(run_id: str) -> RunResult:
    run_dir = _YOLO_ROOT / run_id
    result = RunResult(source="yolo", experiment=run_id)
    if not run_dir.is_dir():
        result.status = "目录不存在"
        return result
    args_path = _locate(run_dir, "args.yaml")
    args = _read_yaml(args_path) if args_path else {}
    config = _read_yaml(run_dir / "configs" / "train_rail_seg.yaml")
    p2_block = config.get("p2")
    p2_enabled = isinstance(p2_block, dict) and p2_block.get("enabled") is True
    model_path = str(args.get("model", ""))
    result.structure = "P2" if p2_enabled or "p2" in model_path.lower() else "基线"
    result.dataset = Path(str(args.get("data", "clop_seg"))).stem
    result.epochs = str(args.get("epochs", ""))
    result.mask_ratio = str(args.get("mask_ratio", ""))
    result.imgsz = str(args.get("imgsz", ""))
    result.batch = str(args.get("batch", ""))
    result.optimizer = str(args.get("optimizer", ""))
    result.lr0 = str(args.get("lr0", ""))
    result.cos_lr = str(args.get("cos_lr", ""))
    results_path = _locate(run_dir, "results.csv")
    result.best_epoch = _best_epoch(results_path) if results_path else ""

    result.native = _read_metrics(run_dir / "metrics" / "evaluation.csv") or Metrics()
    valid_metrics = _find_metrics(run_dir, "valid")
    test_metrics = _find_metrics(run_dir, "test")
    result.valid = valid_metrics or Metrics()
    result.test = test_metrics or Metrics()

    result.plan = _change_plan(run_dir / "change.md")
    weight = _weight_path(run_dir)
    result.weights = _relative(weight)
    if weight is None:
        result.status = "无权重"
    elif valid_metrics is None and test_metrics is None and _is_running(run_dir):
        result.status = "训练中"
    else:
        result.status = _status(True, valid_metrics, test_metrics)

    manifest = {}
    for split in ("valid", "test"):
        found = sorted((run_dir / split).glob("**/evaluation_manifest.json"))
        if found:
            manifest = _read_json(found[-1])
            break
    result.evaluator = str(manifest.get("evaluator", ""))
    result.dataset_sha256 = str(manifest.get("dataset_sha256", ""))
    result.image_count = str(manifest.get("image_count", ""))
    confidence = manifest.get("confidence_floor")
    result.conf = "" if confidence is None else f"{float(confidence):.2f}"
    return result


def _collect_sam3(run_id: str) -> RunResult:
    run_dir = _SAM3_ROOT / run_id
    result = RunResult(source="sam3", experiment=run_id)
    if not run_dir.is_dir():
        result.status = "目录不存在"
        return result
    manifest = _read_json(run_dir / "run_manifest.json")
    options = manifest.get("actual_options")
    options = options if isinstance(options, dict) else _read_yaml(
        run_dir / "configs" / "sam3_predict.yaml"
    )
    result.structure = "SAM3 预训练"
    result.prompt = str(options.get("prompt", ""))
    result.weights = _relative(manifest.get("weights_path"))
    result.conf = "" if options.get("conf") is None else f"{float(options['conf']):.2f}"
    result.image_count = str(manifest.get("image_count", ""))

    result.valid = _find_metrics(run_dir, "valid") or Metrics()
    result.test = _find_metrics(run_dir, "test") or Metrics()
    has_valid = _find_metrics(run_dir, "valid") is not None
    has_test = _find_metrics(run_dir, "test") is not None
    result.status = _status(True, result.valid if has_valid else None,
                            result.test if has_test else None)

    evaluation = {}
    for split in ("valid", "test"):
        candidate = run_dir / split / "metrics" / "evaluation_manifest.json"
        if candidate.is_file():
            evaluation = _read_json(candidate)
            break
    result.evaluator = str(evaluation.get("evaluator", ""))
    result.dataset_sha256 = str(evaluation.get("dataset_sha256", ""))
    if not result.image_count:
        result.image_count = str(evaluation.get("image_count", ""))
    floor = evaluation.get("confidence_floor")
    if floor is not None:
        result.conf = f"{float(floor):.2f}"
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _cell(value: object) -> str:
    text = str(value)
    return text if text else "—"


def _render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    numeric_columns: Iterable[int] = (),
) -> list[str]:
    """渲染 Markdown 表格，并把每列最大值加粗。"""
    columns = list(numeric_columns)
    best: dict[int, float] = {}
    for index in columns:
        values = []
        for row in rows:
            text = row[index].rstrip("%")
            if text != "—":
                try:
                    values.append(float(text))
                except ValueError:
                    continue
        if values:
            best[index] = max(values)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        rendered = []
        for index, text in enumerate(row):
            if index in best and text.rstrip("%") != "—":
                try:
                    if abs(float(text.rstrip("%")) - best[index]) < 1e-9:
                        text = f"**{text}**"
                except ValueError:
                    pass
            rendered.append(text)
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def _missing_section(runs: Sequence[RunResult]) -> list[str]:
    broken = [run for run in runs if run.status not in ("完整",)]
    if not broken:
        return []
    lines = ["## 缺失", "", "| 实验 | 状态 |", "| --- | --- |"]
    for run in broken:
        lines.append(f"| {run.experiment} | {run.status} |")
    lines.append("")
    return lines


def _plan_section(runs: Sequence[RunResult]) -> list[str]:
    """列出各实验 change.md 中「本次方案」的完整原文。"""
    lines = ["## 实验方案", ""]
    for run in runs:
        if run.plan:
            lines.append(f"- **{run.experiment}**：{run.plan}")
    if len(lines) == 2:
        return []
    lines.append("")
    return lines


def _unlisted_section(config: dict[str, object]) -> list[str]:
    listed_yolo = set(_read_entries(config, "yolo"))
    listed_sam3 = set(_read_entries(config, "sam3"))
    unlisted = []
    for root, listed in ((_YOLO_ROOT, listed_yolo), (_SAM3_ROOT, listed_sam3)):
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if path.is_dir() and path.name not in listed:
                unlisted.append(f"{root.name}/{path.name}")
    if not unlisted:
        return []
    lines = ["## 未列入", ""]
    lines.extend(f"- `{name}`" for name in unlisted)
    lines.append("")
    return lines


def _cross_model_section(yolo: Sequence[RunResult], sam3: Sequence[RunResult]) -> list[str]:
    """同口径（评价版本、数据哈希、门限一致）时给出跨模型差值。"""

    def key(run: RunResult) -> tuple[str, str, str] | None:
        if not run.evaluator or not run.dataset_sha256 or not run.conf:
            return None
        if run.valid.map50_95 is None:
            return None
        return (run.evaluator, run.dataset_sha256, run.conf)

    groups: dict[tuple[str, str, str], dict[str, list[RunResult]]] = {}
    for item in yolo:
        item_key = key(item)
        if item_key:
            groups.setdefault(item_key, {"yolo": []})["yolo"].append(item)
    for item in sam3:
        item_key = key(item)
        if item_key:
            groups.setdefault(item_key, {"sam3": []}).setdefault("sam3", []).append(item)

    comparable = [
        (group_key, value)
        for group_key, value in groups.items()
        if value.get("yolo") and value.get("sam3")
    ]
    if not comparable:
        return [
            "## 跨模型对照",
            "",
            "没有满足同口径条件（评价版本、数据哈希、候选门限一致）的 YOLO 与 SAM3 组合。",
            "",
        ]
    lines = ["## 跨模型对照", ""]
    for group_key, value in comparable:
        real_yolo = max(value["yolo"], key=lambda run: run.valid.map50_95 or 0.0)
        real_sam3 = max(value["sam3"], key=lambda run: run.valid.map50_95 or 0.0)
        delta = (real_sam3.valid.map50_95 or 0.0) - (real_yolo.valid.map50_95 or 0.0)
        lines.append(
            f"条件：{group_key[0]} · dataset={group_key[1][:8]} · conf={group_key[2]}"
        )
        lines.append("")
        lines.extend([
            "| 模型 | 实验 | valid mAP50–95 | 差值（SAM3−YOLO） |",
            "| --- | --- | ---: | ---: |",
            f"| YOLO | {real_yolo.experiment} | {_pct(real_yolo.valid.map50_95)} | — |",
            f"| SAM3 | {real_sam3.experiment} | {_pct(real_sam3.valid.map50_95)} "
            f"| {delta * 100:+.2f} pp |",
            "",
        ])
    return lines


def _write_csv(path: Path, runs: Sequence[RunResult]) -> None:
    columns = (
        "source", "experiment", "status", "dataset", "structure", "prompt",
        "epochs", "best_epoch", "mask_ratio", "imgsz", "batch", "optimizer", "lr0",
        "cos_lr", "conf", "image_count", "evaluator", "dataset_sha256",
        "native_precision", "native_recall", "native_map50", "native_map50_95",
        "valid_precision", "valid_recall", "valid_map50", "valid_map50_95",
        "test_precision", "test_recall", "test_map50", "test_map50_95",
        "weights", "plan",
    )
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            writer.writerow({
                "source": run.source, "experiment": run.experiment,
                "status": run.status, "dataset": run.dataset,
                "structure": run.structure, "prompt": run.prompt,
                "epochs": run.epochs, "best_epoch": run.best_epoch,
                "mask_ratio": run.mask_ratio, "imgsz": run.imgsz,
                "batch": run.batch, "optimizer": run.optimizer, "lr0": run.lr0,
                "cos_lr": run.cos_lr, "conf": run.conf,
                "image_count": run.image_count, "evaluator": run.evaluator,
                "dataset_sha256": run.dataset_sha256,
                **{f"native_{key}": getattr(run.native, key) for key in
                   ("precision", "recall", "map50", "map50_95")},
                **{f"valid_{key}": getattr(run.valid, key) for key in
                   ("precision", "recall", "map50", "map50_95")},
                **{f"test_{key}": getattr(run.test, key) for key in
                   ("precision", "recall", "map50", "map50_95")},
                "weights": run.weights,
                "plan": run.plan,
            })


def _read_entries(config: dict[str, object], key: str) -> list[str]:
    """读取清单中的实验 id 列表。

    YAML 会把 20261111_133043 这类串解析成整数，因此要求 id 使用引号。
    """
    entries = config.get(key) or []
    if not isinstance(entries, list):
        raise ValueError(f"清单的 {key} 必须为列表")
    ids: list[str] = []
    for entry in entries:
        run_id = entry.get("id") if isinstance(entry, dict) else entry
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(
                f"清单的 {key} 每项必须为实验 id 字符串并加引号，"
                f"例如 \"20261111_133043\"（当前值 {run_id!r}）"
            )
        ids.append(run_id)
    return ids


def build_experiment_report(
    config_path: Path = _REPORT_CONFIG,
    output_root: Path = _OUTPUT_ROOT,
) -> dict[str, object]:
    """按清单汇总 YOLO 与 SAM3 结果，写出报告并返回产物路径。

    Args:
        config_path: 实验清单 YAML。
        output_root: 报告根目录，本次结果写入其下时间戳子目录。

    Returns:
        输出目录、报告、CSV 路径与各来源的记录数。

    Raises:
        FileNotFoundError: 清单不存在或为空。
    """
    config = _read_yaml(config_path)
    yolo_ids = _read_entries(config, "yolo")
    sam3_ids = _read_entries(config, "sam3")
    if not yolo_ids and not sam3_ids:
        raise FileNotFoundError(f"清单为空或不存在 {config_path}")

    runs = [_collect_yolo(run_id) for run_id in yolo_ids]
    runs += [_collect_sam3(run_id) for run_id in sam3_ids]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    yolo_runs = [run for run in runs if run.source == "yolo"]
    sam3_runs = [run for run in runs if run.source == "sam3"]

    lines = [
        "# 钢轨实例分割实验汇总",
        "",
        f"生成 {datetime.now():%Y-%m-%d %H:%M} · 共 {len(runs)} 个实验",
        "",
        "## YOLO",
        "",
    ]
    lines.extend(_render_table(
        ("实验", "结构", "mask_ratio", "valid mAP50–95", "test mAP50–95",
         "权重", "状态"),
        [
            (run.experiment, run.structure, run.mask_ratio,
             _pct(run.valid.map50_95), _pct(run.test.map50_95),
             _cell(run.weights), run.status)
            for run in yolo_runs
        ],
        numeric_columns=(3, 4),
    ))
    lines.extend(["", "## SAM3", ""])
    lines.extend(_render_table(
        ("实验", "prompt", "valid mAP50–95", "test mAP50–95", "权重", "状态"),
        [
            (run.experiment, _cell(run.prompt), _pct(run.valid.map50_95),
             _pct(run.test.map50_95), _cell(run.weights), run.status)
            for run in sam3_runs
        ],
        numeric_columns=(2, 3),
    ))
    lines.append("")
    lines.extend(_cross_model_section(yolo_runs, sam3_runs))
    lines.extend(_missing_section(runs))
    lines.extend(_plan_section(runs))
    lines.extend(_unlisted_section(config))

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    _write_csv(output_dir / "experiments.csv", runs)
    (output_dir / "sources.json").write_text(
        json.dumps({
            "timestamp": timestamp,
            "config": str(config_path),
            "yolo_root": str(_YOLO_ROOT),
            "sam3_root": str(_SAM3_ROOT),
            "experiment_count": len(runs),
            "missing": [
                {"source": run.source, "experiment": run.experiment,
                 "status": run.status}
                for run in runs if run.status != "完整"
            ],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    for run in runs:
        if run.status != "完整":
            print(f"{run.source}/{run.experiment}: {run.status}")
    print(f"报告已保存至 {report_path}")
    return {
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "csv_path": str(output_dir / "experiments.csv"),
        "yolo_count": len(yolo_runs),
        "sam3_count": len(sam3_runs),
    }


def _main() -> None:
    build_experiment_report()


if __name__ == "__main__":
    _main()
