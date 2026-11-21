"""把带填充色的钢轨分割数据集导出为内容裁剪版。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preprocess.common.config import PROJECT_ROOT, config_path
from utils.bbox_crop import (
    IMAGE_SUFFIXES,
    BboxCropSettings,
    build_manifest,
    find_content_bbox,
    load_bbox_crop_settings,
    project_relative,
    write_manifest,
)

DEFAULT_TRAIN_CONFIG = config_path("train_rail_seg.yaml")
_SPLIT_KEYS = ("train", "val", "test")
# 标注与内容边界存在亚像素误差，允许的越界像素。
_LABEL_TOLERANCE_PX = 2.0


def export_bbox_dataset(
    settings: BboxCropSettings,
    source_config: Path,
    *,
    refresh: bool = False,
    limit: int | None = None,
) -> Path:
    """按内容外接框导出裁剪数据集，并生成数据配置与清单。

    Args:
        settings: 裁剪几何参数与导出位置。
        source_config: 原始数据集 YAML，需包含 path 与 train、val、test。
        refresh: 为 True 时删除已有导出目录后重建。
        limit: 每个划分仅导出前若干张，供调试使用，None 表示全部。

    Returns:
        生成的裁剪数据集配置路径。

    Raises:
        FileExistsError: 导出目录已存在且未要求刷新。
        FileNotFoundError: 缺少源数据集、图片或标签。
        ValueError: 标签在裁剪后越界、坐标数量无效或配置缺失。
    """
    source = _read_source_config(source_config)
    source_root = _resolve_source_root(source, source_config)
    if settings.dataset.exists():
        if not refresh:
            raise FileExistsError(
                f"导出目录已存在 {settings.dataset}，如需重建请加 --refresh"
            )
        shutil.rmtree(settings.dataset)
    staging = settings.dataset.parent / f".{settings.dataset.name}.exporting"
    if staging.exists():
        shutil.rmtree(staging)

    split_counts: dict[str, int] = {}
    entries: list[dict[str, object]] = []
    print(f"源数据集 {source_root}")
    print(f"导出目录 {settings.dataset}")
    for split in _SPLIT_KEYS:
        if split not in source:
            continue
        images_dir = source_root / str(source[split])
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        count = _export_split(
            settings, split, images_dir, labels_dir, staging,
            entries, limit=limit,
        )
        split_counts[split] = count
        print(f"  {split}: {count} 张")

    settings.data_config.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(
        staging, build_manifest(settings, source_config, split_counts, entries)
    )
    staging.rename(settings.dataset)
    _write_data_config(settings, source)
    print(f"已生成数据集配置 {settings.data_config}")
    return settings.data_config


def _export_split(
    settings: BboxCropSettings,
    split: str,
    images_dir: Path,
    labels_dir: Path,
    staging: Path,
    entries: list[dict[str, object]],
    limit: int | None,
) -> int:
    """导出单个划分，返回实际导出的图片数量。"""
    if not images_dir.is_dir():
        raise NotADirectoryError(f"缺少源图片目录 {images_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"缺少源标签目录 {labels_dir}")
    target_images = staging / "images" / split
    target_labels = staging / "labels" / split
    target_images.mkdir(parents=True)
    target_labels.mkdir(parents=True)
    images = sorted(
        item for item in images_dir.iterdir()
        if item.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit is not None:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"{images_dir} 下没有图片")
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"缺少标签 {label_path}")
        image = _read_image(image_path)
        bbox = find_content_bbox(image, settings)
        x0, y0, x1, y1 = bbox
        cropped = image[y0:y1, x0:x1]
        if not cropped.size:
            raise ValueError(f"{image_path} 的裁剪结果为空")
        shifted = _shift_labels(
            label_path.read_text(encoding="utf-8-sig"), bbox, image.shape[:2]
        )
        _write_image(target_images / f"{image_path.stem}.png", cropped)
        (target_labels / f"{image_path.stem}.txt").write_text(
            shifted, encoding="utf-8", newline="\n"
        )
        entries.append({
            "split": split,
            "name": f"{image_path.stem}.png",
            "bbox": [x0, y0, x1, y1],
        })
    return len(images)


def _shift_labels(
    text: str, bbox: tuple[int, int, int, int], image_shape: tuple[int, int]
) -> str:
    """把归一化标签平移到裁剪坐标系，并校验不越界。

    Args:
        text: 源标签内容，每行 ``类别 x1 y1 x2 y2 ...``，坐标为 0 到 1。
        bbox: 裁剪框 (x0, y0, x1, y1)，右下为开区间。
        image_shape: 源图 (高, 宽)。

    Returns:
        裁剪后的标签文本，坐标保留 6 位小数。

    Raises:
        ValueError: 坐标数量无效或平移后越出裁剪图。
    """
    x0, y0, x1, y1 = bbox
    old_height, old_width = image_shape
    new_width, new_height = x1 - x0, y1 - y0
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = raw.split()
        values = np.asarray(parts[1:], dtype=np.float64)
        if values.size < 6 or values.size % 2:
            raise ValueError("多边形坐标数量无效")
        points = values.reshape(-1, 2) * np.asarray((old_width, old_height))
        shifted = points - np.asarray((x0, y0))
        limit = np.asarray((new_width, new_height), dtype=np.float64)
        if (shifted < -_LABEL_TOLERANCE_PX).any() or (
            shifted > limit + _LABEL_TOLERANCE_PX
        ).any():
            raise ValueError("裁剪后标签越界，说明内容框没有覆盖全部标注")
        normalized = shifted.clip(0.0, limit) / limit
        body = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
        lines.append(f"{parts[0]} {body}")
    if not lines:
        raise ValueError("标签文件没有任何多边形")
    return "\n".join(lines) + "\n"


def _read_image(path: Path) -> np.ndarray:
    """读取图像，兼容中文路径。"""
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片 {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    """以 PNG 无损写出图像，兼容中文路径。"""
    success, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 6]
    )
    if not success:
        raise OSError(f"无法编码图片 {path}")
    encoded.tofile(path)


def _read_source_config(source_config: Path) -> dict[str, object]:
    """读取源数据集配置。"""
    if not source_config.is_file():
        raise FileNotFoundError(f"缺少源数据集配置 {source_config}")
    with source_config.open(encoding="utf-8-sig") as stream:
        source = yaml.safe_load(stream)
    if not isinstance(source, dict) or "path" not in source:
        raise ValueError(f"{source_config} 必须包含 path 字段")
    return source


def _resolve_source_root(source: dict[str, object], source_config: Path) -> Path:
    """解析源数据集根目录。"""
    value = source["path"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source_config} 的 path 必须为非空路径字符串")
    root = Path(value)
    return root if root.is_absolute() else (PROJECT_ROOT / root).resolve()


def _write_data_config(
    settings: BboxCropSettings, source: dict[str, object]
) -> None:
    """生成裁剪数据集配置，沿用源数据集的划分与类别名。"""
    payload: dict[str, object] = {"path": project_relative(settings.dataset)}
    for split in _SPLIT_KEYS:
        if split in source:
            payload[split] = source[split]
    payload["names"] = source.get("names") or {0: "rail"}
    text = (
        "# 由 data_preprocess/export_bbox_crop.py 生成，请勿手改。\n"
        + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    )
    settings.data_config.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    """命令行入口：按训练配置的 bbox_crop 块导出裁剪数据集。"""
    parser = argparse.ArgumentParser(description="导出内容裁剪版钢轨分割数据集")
    parser.add_argument(
        "--config", default=str(DEFAULT_TRAIN_CONFIG),
        help="训练配置路径，读取其中的 data 与 bbox_crop 块",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="删除已有导出目录后重建",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="每个划分只导出前若干张，调试用",
    )
    args = parser.parse_args(argv)

    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = (PROJECT_ROOT / config_file).resolve()
    with config_file.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"{config_file} 必须是以字符串为键的映射")
    block = config.get("bbox_crop")
    if block is None:
        raise ValueError(f"{config_file} 缺少 bbox_crop 配置块")
    settings = load_bbox_crop_settings(block, config_file)
    data_value = config.get("data")
    if not isinstance(data_value, str) or not data_value.strip():
        raise ValueError(f"{config_file} 的 data 必须为非空路径字符串")
    source_config = (PROJECT_ROOT / data_value).resolve()

    export_bbox_dataset(
        settings, source_config, refresh=args.refresh, limit=args.limit
    )
    print("导出完成，可在训练配置中设置 bbox_crop.enabled: true 后开始训练")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
