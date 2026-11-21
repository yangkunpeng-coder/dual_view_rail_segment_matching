"""分割后处理配置、逐图统计与评测产物的读写。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from math import ceil
from pathlib import Path
import statistics

from core_algorithm.postprocess import PostprocessConfig
from data_preprocess.common.config import (
    PROJECT_ROOT,
    config_path,
    load_yaml_mapping,
)

POSTPROCESS_CONFIG = config_path("seg_postprocess.yaml")
_INSTANCE_COUNT_COLUMNS = (
    "split", "side", "image", "gt", "pred", "tp", "fp", "missed",
)


@dataclass(frozen=True)
class PostprocessImageStat:
    """单张图的后处理计数与耗时。"""

    image: str
    side: str
    raw_count: int
    kept_count: int
    suppressed_count: int
    postprocess_ms: float


def load_postprocess_config(
    config_path: Path = POSTPROCESS_CONFIG,
) -> PostprocessConfig:
    """读取并校验分割后处理配置。

    Args:
        config_path: YAML 配置文件路径，默认使用 config/eval/seg_postprocess.yaml。

    Returns:
        校验后的 PostprocessConfig。

    Raises:
        OSError: 配置文件不存在或无法读取。
        ValueError: 配置包含未知键或取值无效。
    """
    return parse_postprocess_config(load_yaml_mapping(config_path), config_path)


def parse_postprocess_config(
    mapping: dict[str, object], source: str | Path = "<mapping>"
) -> PostprocessConfig:
    """校验已解析的后处理配置映射。

    Args:
        mapping: YAML 解析后的顶层映射。
        source: 用于错误信息的来源说明。

    Returns:
        校验后的 PostprocessConfig。

    Raises:
        ValueError: 配置包含未知键或取值无效。
    """
    unknown = mapping.keys() - {"enabled", "mask_nms", "quality"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{source} 包含未知的后处理配置 {names}")

    enabled = _read_bool(mapping.get("enabled", True), "enabled")
    mask_nms = _read_mapping(mapping.get("mask_nms", {}), "mask_nms")
    unknown_mask = mask_nms.keys() - {"iou", "min_area_px"}
    if unknown_mask:
        names = ", ".join(sorted(unknown_mask))
        raise ValueError(f"{source} 的 mask_nms 包含未知配置 {names}")
    quality = _read_mapping(mapping.get("quality", {}), "quality")
    unknown_quality = quality.keys() - {"enabled"}
    if unknown_quality:
        names = ", ".join(sorted(unknown_quality))
        raise ValueError(f"{source} 的 quality 包含未知配置 {names}")

    iou = _read_float(mask_nms.get("iou", 0.60), "mask_nms.iou")
    if not 0 <= iou <= 1:
        raise ValueError(f"{source} 的 mask_nms.iou 必须位于 [0, 1]")
    min_area_px = _read_int(mask_nms.get("min_area_px", 8), "mask_nms.min_area_px")
    if min_area_px < 0:
        raise ValueError(f"{source} 的 mask_nms.min_area_px 不能为负数")
    quality_enabled = _read_bool(quality.get("enabled", True), "quality.enabled")
    return PostprocessConfig(
        enabled=enabled,
        mask_nms_iou=iou,
        min_area_px=min_area_px,
        quality_enabled=quality_enabled,
    )


def save_postprocess_report(
    metrics_dir: Path,
    config: PostprocessConfig,
    stats: list[PostprocessImageStat],
) -> Path:
    """写出后处理参数、逐图抑制计数和耗时。

    Args:
        metrics_dir: 本次预测的 metrics 目录。
        config: 实际使用的后处理配置。
        stats: 每张图的计数与耗时。

    Returns:
        metrics/postprocess.json 的路径。
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    latencies = sorted(stat.postprocess_ms for stat in stats)
    report = {
        "config": asdict(config),
        "image_count": len(stats),
        "raw_instances": sum(stat.raw_count for stat in stats),
        "kept_instances": sum(stat.kept_count for stat in stats),
        "suppressed_instances": sum(stat.suppressed_count for stat in stats),
        "postprocess_ms_median": (
            statistics.median(latencies) if latencies else 0.0
        ),
        "postprocess_ms_p95": _percentile(latencies, 95),
        "images": [asdict(stat) for stat in stats],
    }
    path = metrics_dir / "postprocess.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def save_instance_counts(
    metrics_dir: Path,
    split: str,
    rows: list[dict[str, object]],
) -> Path:
    """保存逐图实例计数，split 字段沿用评价口径（val/test）。

    Args:
        metrics_dir: 本次预测的 metrics 目录。
        split: 数据划分名称，val 或 test。
        rows: 每行包含 side、image、gt、pred、tp、fp、missed。

    Returns:
        metrics/instance_counts.csv 的路径。
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "instance_counts.csv"
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_INSTANCE_COUNT_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({"split": split, **row})
    return path


def image_side(image_name: str) -> str:
    """从图片文件名提取 north/south，无前缀时返回 unknown。"""
    side, _, _ = image_name.partition("_")
    return side if side in ("north", "south") else "unknown"


def _read_mapping(value: object, name: str) -> dict[str, object]:
    """读取嵌套映射并拒绝空值和非字符串键。"""
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{name} 必须为字符串键的映射")
    return value


def _read_bool(value: object, name: str) -> bool:
    """读取布尔配置，拒绝整数 0/1 冒充布尔值。"""
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须为布尔值")
    return value


def _read_float(value: object, name: str) -> float:
    """读取浮点配置，拒绝布尔值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须为数值")
    return float(value)


def _read_int(value: object, name: str) -> int:
    """读取整数配置，拒绝布尔值和小数。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须为整数")
    return value


def _percentile(sorted_values: list[float], percent: float) -> float:
    """对已排序列表取百分位，空列表返回 0。"""
    if not sorted_values:
        return 0.0
    position = ceil(len(sorted_values) * percent / 100.0) - 1
    return sorted_values[max(0, min(position, len(sorted_values) - 1))]
