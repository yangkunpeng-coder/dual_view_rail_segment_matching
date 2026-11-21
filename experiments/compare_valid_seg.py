"""读取 SAM3 和 YOLO 已有的独立验证指标，导出 CSV 与 Markdown 对照。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preprocess.common.config import PROJECT_ROOT, load_yaml_mapping
from utils.segmentation_evaluation import METRIC_COLUMNS

_SUMMARY_COLUMNS = (
    "model", "experiment_id", "validation_id", "record_id",
    "image_count", *METRIC_COLUMNS,
    "evaluator", "dataset_sha256", "confidence_floor", "metrics_path",
)
_DIFFERENCE_COLUMNS = (
    "sam3_record", "yolo_record", "dataset_sha256", "confidence_floor",
    *(f"sam3_{key}" for key in METRIC_COLUMNS),
    *(f"yolo_{key}" for key in METRIC_COLUMNS),
    *(f"delta_{key}_pp" for key in METRIC_COLUMNS),
)


@dataclass(frozen=True)
class _ValidationResult:
    """一份已生成验证 CSV 的指标和可追溯信息。"""

    model: str
    experiment_id: str
    validation_id: str
    record_id: str
    metrics_path: Path
    metrics: dict[str, float]
    evaluator: str
    dataset_sha256: str
    confidence_floor: float | None
    image_count: int | None

    def as_row(self) -> dict[str, object]:
        """展开为 CSV 的一行。"""
        return {
            "model": self.model, "experiment_id": self.experiment_id,
            "validation_id": self.validation_id, "record_id": self.record_id,
            "metrics_path": str(self.metrics_path), **self.metrics,
            "evaluator": self.evaluator, "dataset_sha256": self.dataset_sha256,
            "confidence_floor": self.confidence_floor,
            "image_count": self.image_count,
        }


def compare_valid_seg(
    yolo_root: str | Path = "out/yolo_seg",
    sam3_root: str | Path = "out/sam3_seg",
) -> dict[str, object]:
    """汇总两种模型的独立 valid 产物，按已确认口径生成指标差值。

    Args:
        yolo_root: YOLO 实验根目录，相对路径以项目根目录为基准。
        sam3_root: SAM3 实验根目录，相对路径以项目根目录为基准。

    Returns:
        输出目录、CSV 和 Markdown 路径，以及各模型记录数和异常数。

    Raises:
        FileNotFoundError: 两个根目录均未发现可读取的 valid 指标。
        OSError: 无法创建或写入对比结果。
    """
    records: list[_ValidationResult] = []
    errors: list[dict[str, str]] = []
    roots = {
        "yolo26_seg": Path(PROJECT_ROOT, yolo_root).resolve(),
        "sam3": Path(PROJECT_ROOT, sam3_root).resolve(),
    }
    for model, root in roots.items():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("evaluation.csv")):
            if (
                path.parent.name != "metrics"
                or "valid" not in path.relative_to(root).parts
            ):
                continue
            try:
                records.extend(_read_results(path, model, root))
            except (OSError, ValueError, KeyError) as error:
                errors.append({"path": str(path), "error": str(error)})
    if not records:
        details = "; ".join(item["error"] for item in errors)
        raise FileNotFoundError(
            f"没有可用的 valid/metrics/evaluation.csv，先运行验证入口。{details}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = PROJECT_ROOT / "out" / "compare_valid_seg" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "metrics_comparison.csv"
    _write_csv(summary_path, _SUMMARY_COLUMNS, [item.as_row() for item in records])
    differences = _compare_records(records)
    difference_path = output_dir / "pairwise_comparison.csv"
    _write_csv(difference_path, _DIFFERENCE_COLUMNS, differences)
    report_path = output_dir / "compare_valid_seg.md"
    _write_report(report_path, records, differences, errors)
    (output_dir / "sources.json").write_text(
        json.dumps({
            "timestamp": timestamp,
            "roots": {name: str(path) for name, path in roots.items()},
            "source_metrics": [str(item.metrics_path) for item in records],
            "errors": errors,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    result: dict[str, object] = {
        "output_dir": str(output_dir), "summary_csv": str(summary_path),
        "pairwise_csv": str(difference_path), "report_path": str(report_path),
        "yolo_record_count": sum(item.model == "yolo26_seg" for item in records),
        "sam3_record_count": sum(item.model == "sam3" for item in records),
        "comparable_pair_count": len(differences), "error_count": len(errors),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _read_results(path: Path, model: str, root: Path) -> list[_ValidationResult]:
    """读取指标 CSV，兼容缺少输入清单的历史验证记录。"""
    parts = path.relative_to(root).parts
    experiment_id = parts[0]
    valid_index = parts.index("valid")
    validation_id = (
        parts[valid_index + 1]
        if parts[valid_index + 1] != "metrics" else experiment_id
    )
    manifest_path = path.parent / "evaluation_manifest.json"
    metadata: dict[str, object] = {}
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, dict):
            raise ValueError(f"评估清单必须为字典 {manifest_path}")
        metadata = loaded
    evaluator = metadata.get("evaluator", "legacy_unverified")
    signature = metadata.get("dataset_sha256", "")
    confidence = metadata.get("confidence_floor")
    image_count = metadata.get("image_count")
    if not isinstance(evaluator, str) or not isinstance(signature, str):
        raise ValueError(f"评估清单的口径或输入签名无效 {manifest_path}")
    if confidence is not None and (
        isinstance(confidence, bool) or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence) or not 0 <= confidence <= 1
    ):
        raise ValueError(f"评估清单的候选门限无效 {manifest_path}")
    if image_count is not None and (
        type(image_count) is not int or image_count < 0
    ):
        raise ValueError(f"评估清单的图片数量无效 {manifest_path}")
    legacy_data = path.parent / "evaluation_data.yaml"
    if image_count is None and legacy_data.is_file():
        images = load_yaml_mapping(legacy_data).get("val")
        if isinstance(images, list):
            image_count = len(images)

    records = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"split", *METRIC_COLUMNS}.issubset(reader.fieldnames or ()):
            raise ValueError(f"指标 CSV 缺少所需列 {path}")
        for index, row in enumerate(reader, start=1):
            split = row["split"]
            if not isinstance(split, str):
                raise ValueError(f"指标 CSV 第 {index + 1} 行缺少 split {path}")
            if split.lower() not in ("val", "valid"):
                continue
            try:
                metrics = {key: float(row[key]) for key in METRIC_COLUMNS}
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"指标 CSV 第 {index + 1} 行含有无效数值 {path}"
                ) from error
            if any(not math.isfinite(value) or not 0 <= value <= 1
                   for value in metrics.values()):
                raise ValueError(f"指标值必须在 [0, 1] 内 {path}")
            records.append(_ValidationResult(
                model, experiment_id, validation_id,
                f"{model}/{experiment_id}/{validation_id}/{index}",
                path, metrics, evaluator, signature, confidence, image_count,
            ))
    return records


def _compare_records(
    records: list[_ValidationResult],
) -> list[dict[str, object]]:
    """仅对相同输入哈希、评估口径及候选门限的跨模型结果计算差值。"""
    differences: list[dict[str, object]] = []
    for sam in (item for item in records if item.model == "sam3"):
        for yolo in (item for item in records if item.model == "yolo26_seg"):
            if (
                not sam.dataset_sha256 or not sam.evaluator
                or sam.confidence_floor is None
                or sam.dataset_sha256 != yolo.dataset_sha256
                or sam.evaluator != yolo.evaluator
                or sam.confidence_floor != yolo.confidence_floor
            ):
                continue
            row: dict[str, object] = {
                "sam3_record": sam.record_id, "yolo_record": yolo.record_id,
                "dataset_sha256": sam.dataset_sha256,
                "confidence_floor": sam.confidence_floor,
            }
            for key in METRIC_COLUMNS:
                row[f"sam3_{key}"] = sam.metrics[key]
                row[f"yolo_{key}"] = yolo.metrics[key]
                row[f"delta_{key}_pp"] = 100 * (sam.metrics[key] - yolo.metrics[key])
            differences.append(row)
    return differences


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    """按 UTF-8-SIG 编码保存对比表，空结果仍保留表头。"""
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path, records: list[_ValidationResult],
    differences: list[dict[str, object]], errors: list[dict[str, str]],
) -> None:
    """记录指标百分比、百分点差异和每条数据的可核对来源。"""
    lines = [
        "# 验证集实例分割对比", "",
        "所有数值来自各次独立验证的 evaluation.csv。四项指标越高越好。",
        "下表以百分比显示，CSV 中保留原始 0–1 数值。", "",
        "| 模型 | 实验 | 验证批次 | 图片数 | Mask P | Mask R | mAP50 | mAP50–95 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in records:
        cells = [
            item.model, item.experiment_id, item.validation_id,
            str(item.image_count) if item.image_count is not None else "待核对",
            *(f"{100 * item.metrics[key]:.2f}%" for key in METRIC_COLUMNS),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend((
        "", "## 同口径差值", "",
        "仅在图片与标签哈希、评价版本、候选分数门限相同时生成下表。",
        "差值为 SAM3 减 YOLO，单位为百分点。正值表示该项 SAM3 较高。", "",
    ))
    if differences:
        lines.extend((
            "| SAM3 记录 | YOLO 记录 | ΔP | ΔR | ΔmAP50 | ΔmAP50–95 |",
            "| --- | --- | --- | --- | --- | --- |",
        ))
        for row in differences:
            cells = [
                str(row["sam3_record"]), str(row["yolo_record"]),
                *(f"{row[f'delta_{key}_pp']:+.2f}" for key in METRIC_COLUMNS),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("当前尚无满足同口径条件的 SAM3 与 YOLO 验证组合。")
    lines.extend((
        "", "## 评价口径和来源", "",
        "新验证使用原图独立实例 Mask 和 Ultralytics AP 积分规则。",
        "Precision、Recall 取最优 F1 置信度位置。逐图漏检门限取候选与显示门限的较大值。",
        "SAM3 为文本提示预训练模型，YOLO 为项目训练权重。",
        "历史记录缺少输入哈希时标记 legacy_unverified，可通过源文件核对其配置。",
        "历史原生验证与新的原图独立 Mask 评估在栅格化尺寸、重叠区域处理上可能有差异。",
        "正式对比可运行新验证入口，生成带完整清单的两侧结果。", "",
    ))
    for item in records:
        lines.append(
            f"- {item.record_id}，评价版本 {item.evaluator}，"
            f"候选门限 {item.confidence_floor}。"
            f" [原始指标]({item.metrics_path.as_posix()})"
        )
    if errors:
        lines.extend(("", "## 读取异常", ""))
        for error in errors:
            message = error["error"].replace("\n", " ")
            lines.append(f"- {error['path']}，{message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    compare_valid_seg()
