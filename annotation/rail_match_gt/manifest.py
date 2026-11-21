"""生成南北钢轨匹配真值的图对清单。

配对与编号完全沿用项目既有逻辑：`match_pair_image()` 按采集时间给出南北图片，
`pair_match_json.py` 以该顺序的 1 起序号作为 `pair_id`（空标注跳过并留空号）。
本模块只在此基础上筛出两侧图像都落在同一分割划分内的图对。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from data_preprocess.common.config import (
    PROJECT_ROOT,
    config_path as resolve_config,
    load_dataset_relative_path,
    load_yaml_mapping,
)
from data_preprocess.match_pair_image import match_pair_image
from utils.match_gt import MANIFEST_COLUMNS, file_sha256

_DEFAULT_CONFIG = "config/eval/rail_match_gt.yaml"
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def build_manifest(config_path: str | Path = _DEFAULT_CONFIG) -> Path:
    """按既有配对逻辑筛选图对并写出清单。

    Args:
        config_path: 真值配置的项目相对路径或绝对路径。

    Returns:
        写出的清单 CSV 路径。

    Raises:
        OSError: 配置、图像或标注不可读。
        ValueError: 配置字段无效，或清单为空。
    """
    settings = _load_settings(config_path)
    splits = tuple(str(name) for name in settings["splits"])
    split_of_image = _scan_split_images(
        Path(str(settings["seg_dataset"])), str(settings["seg_image_subdir"]),
        splits, _split_dirs(settings),
    )
    dataset = load_dataset_relative_path()
    pairs = match_pair_image(dataset)
    rows: list[dict[str, object]] = []
    for index, (south, north) in enumerate(
        zip(pairs["south"], pairs["north"], strict=True), start=1
    ):
        south_split = split_of_image.get(f"south_{south}.png")
        north_split = split_of_image.get(f"north_{north}.png")
        if south_split is None or south_split != north_split:
            continue
        if south_split not in splits:
            continue
        rows.append(_manifest_row(
            f"P{index:04d}", south_split, dataset, south, north,
        ))
    if not rows:
        raise ValueError("没有找到两侧都在目标划分内的完整图对")
    rows.sort(key=lambda row: (splits.index(str(row["split"])), str(row["pair_id"])))
    destination = PROJECT_ROOT / str(settings["root"]) / "manifest.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=MANIFEST_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["split"])] = counts.get(str(row["split"]), 0) + 1
    summary = "，".join(f"{name} {count} 对" for name, count in counts.items())
    print(f"已写出 {destination}：{len(rows)} 对（{summary}）", flush=True)
    return destination


def _load_settings(config_path: str | Path) -> dict[str, object]:
    """读取并校验配置。"""
    settings = load_yaml_mapping(resolve_config(config_path))
    required = ("root", "splits", "seg_dataset")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(f"{absolute} 缺少字段 {', '.join(missing)}")
    settings.setdefault("seg_image_subdir", "images")
    splits = settings["splits"]
    if not isinstance(splits, list) or not splits:
        raise ValueError(f"{absolute} 的 splits 必须是非空列表")
    return settings


def _split_dirs(settings: dict[str, object]) -> dict[str, str]:
    """返回划分名到分割目录名的映射，缺省时同名。"""
    mapping = settings.get("seg_split_dirs") or {}
    if not isinstance(mapping, dict):
        raise ValueError("seg_split_dirs 必须是映射")
    return {str(key): str(value) for key, value in mapping.items()}


def _scan_split_images(
    dataset: Path, image_subdir: str, splits: tuple[str, ...],
    split_dirs: dict[str, str],
) -> dict[str, str]:
    """建立图像文件名到划分名的映射。"""
    mapping: dict[str, str] = {}
    for split in splits:
        directory = PROJECT_ROOT / dataset / image_subdir / split_dirs.get(split, split)
        if not directory.is_dir():
            raise ValueError(f"分割数据集缺少目录 {directory}")
        for image in sorted(directory.iterdir()):
            if image.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            mapping[image.name] = split
    return mapping


def _manifest_row(
    pair_id: str, split: str, dataset: Path,
    south_stem: str, north_stem: str,
) -> dict[str, object]:
    """组装一行清单记录，路径与哈希都指向原始数据。"""
    south_image = PROJECT_ROOT / dataset / "south" / f"{south_stem}.png"
    north_image = PROJECT_ROOT / dataset / "north" / f"{north_stem}.png"
    south_label = south_image.with_suffix(".json")
    north_label = north_image.with_suffix(".json")
    return {
        "pair_id": pair_id,
        "split": split,
        "south_image_path": _relative(south_image),
        "north_image_path": _relative(north_image),
        "south_label_sha256": file_sha256(south_label),
        "north_label_sha256": file_sha256(north_label),
        "south_image_sha256": file_sha256(south_image),
        "north_image_sha256": file_sha256(north_image),
        "south_instance_count": _rail_count(south_label),
        "north_instance_count": _rail_count(north_label),
    }


def _rail_count(label: Path) -> int:
    """统计标注文件中 `rail` 类多边形数量。"""
    payload = json.loads(label.read_text(encoding="utf-8-sig"))
    shapes = payload.get("shapes", [])
    return sum(1 for shape in shapes if shape.get("label") == "rail")


def _relative(path: Path) -> str:
    """优先返回项目相对路径。"""
    resolved = path.resolve()
    return (
        resolved.relative_to(PROJECT_ROOT).as_posix()
        if resolved.is_relative_to(PROJECT_ROOT) else str(resolved)
    )


def _main() -> None:
    """命令行入口：生成清单。"""
    build_manifest()


if __name__ == "__main__":
    _main()
