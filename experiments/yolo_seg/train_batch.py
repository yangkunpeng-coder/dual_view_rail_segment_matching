"""按清单顺序批量训练多个实验配置。

清单格式见 `config/experiments/batch.yaml`：

```yaml
runs:
  - file: p2_four_scale.yaml
    label: B1
```

每个实验配置是一份完整的训练配置；本入口只负责校验必填项、输出与参考配置的差异，
然后按顺序调用训练入口，把结果写进 `out/yolo_seg_batch/<时间戳>/index.csv`。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import yaml

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preprocess.common.config import PROJECT_ROOT, config_path
from experiments.yolo_seg.train_rail_seg import train_rail_seg

_BATCH_ROOT = PROJECT_ROOT / "out" / "yolo_seg_batch"
_MANIFEST = config_path("experiments/batch.yaml")
_REFERENCE_CONFIG = config_path("train_rail_seg.yaml")
_REQUIRED_KEYS = ("model", "data", "project", "epochs", "imgsz", "batch")
_REQUIRED_CHANGE_KEYS = ("plan", "reasoning", "problem")
_INDEX_COLUMNS = (
    "order", "label", "config", "run_dir",
    "valid_map50_95", "test_map50_95", "status",
)


def _read_yaml(path: Path) -> dict:
    """读取 YAML 映射，格式不符时抛错。"""
    with path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 必须是 YAML 映射")
    return data


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    """把嵌套映射摊平成点号键，便于逐项比较。"""
    items: dict[str, object] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            items.update(_flatten(child, name))
    else:
        items[prefix] = value
    return items


def _diff_lines(reference: dict, experiment: dict) -> list[str]:
    """返回两份配置的逐项差异，格式为 `键: 参考值 -> 实验值`。"""
    left, right = _flatten(reference), _flatten(experiment)
    lines: list[str] = []
    for key in sorted(set(left) | set(right)):
        before = left.get(key, "<缺失>")
        after = right.get(key, "<缺失>")
        if before != after:
            lines.append(f"{key}: {before!r} -> {after!r}")
    return lines


def _validate(config_path: Path, options: dict) -> None:
    """校验必填训练项与 change 说明。"""
    missing = [key for key in _REQUIRED_KEYS if key not in options]
    if missing:
        raise ValueError(f"{config_path} 缺少必填项 {missing}")
    change = options.get("change")
    if not isinstance(change, dict):
        raise ValueError(f"{config_path} 缺少 change 映射")
    empty = [
        key for key in _REQUIRED_CHANGE_KEYS
        if not str(change.get(key, "")).strip()
    ]
    if empty:
        raise ValueError(f"{config_path} 的 change 缺少 {empty}")


def _latest_metric(run_dir: Path, split: str) -> str:
    """读取实验最新一次独立评估的 mask mAP50-95（百分比字符串）。"""
    paths = sorted((run_dir / split).glob("*/metrics/evaluation.csv"))
    if not paths:
        return ""
    with paths[-1].open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return ""
    try:
        return f"{float(rows[-1]['mask_map50_95']) * 100:.2f}"
    except (KeyError, TypeError, ValueError):
        return ""


def _load_runs(manifest: Path) -> list[dict[str, str]]:
    """把清单解析成 `file` 与 `label` 两项的列表。"""
    data = _read_yaml(manifest)
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{manifest} 的 runs 必须是非空列表")
    items: list[dict[str, str]] = []
    for entry in runs:
        if isinstance(entry, str):
            items.append({"file": entry, "label": Path(entry).stem})
            continue
        if isinstance(entry, dict) and "file" in entry:
            file_name = str(entry["file"])
            items.append({
                "file": file_name,
                "label": str(entry.get("label", Path(file_name).stem)),
            })
            continue
        raise ValueError(f"{manifest} 的 runs 项必须是文件名或含 file 的映射")
    return items


def main() -> None:
    """命令行入口：按清单批量训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=_MANIFEST,
        help="批量清单路径，默认 config/experiments/batch.yaml",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只校验配置并输出差异，不启动训练",
    )
    parser.add_argument(
        "--keep-going", action="store_true",
        help="单个实验失败后继续执行后面的实验",
    )
    args = parser.parse_args()

    manifest = (
        args.manifest.resolve()
        if args.manifest.is_absolute()
        else config_path(args.manifest)
    )
    runs = _load_runs(manifest)
    output_dir = _BATCH_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    diff_dir = output_dir / "diffs"
    diff_dir.mkdir(parents=True)
    reference = _read_yaml(_REFERENCE_CONFIG)
    rows: list[dict[str, object]] = []
    for order, item in enumerate(runs, start=1):
        run_config = (manifest.parent / item["file"]).resolve()
        label = item["label"]
        status = "ok"
        run_dir = ""
        try:
            options = _read_yaml(run_config)
            _validate(run_config, options)
            lines = _diff_lines(reference, options)
            diff_path = diff_dir / f"{label}.diff"
            diff_path.write_text(
                "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
            )
            print(
                f"[{order}/{len(runs)}] {label}: {run_config.name}，"
                f"与参考配置差异 {len(lines)} 项"
            )
            if not args.dry_run:
                run_dir = str(train_rail_seg(run_config))
        except Exception as error:  # 记录后按需停止，避免批量中断丢失已跑结果
            status = f"failed: {error}"
            print(f"[{order}/{len(runs)}] {label} 失败：{error}")
        rows.append({
            "order": order,
            "label": label,
            "config": run_config.name,
            "run_dir": run_dir,
            "valid_map50_95": (
                _latest_metric(Path(run_dir), "valid") if run_dir else ""
            ),
            "test_map50_95": (
                _latest_metric(Path(run_dir), "test") if run_dir else ""
            ),
            "status": status,
        })
        if status != "ok" and not args.keep_going:
            break
    index_path = output_dir / "index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_INDEX_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"批量结果已写入 {index_path}")
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
