"""内容裁剪数据集的配置解析、内容外接框与缓存校验。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from data_preprocess.common.config import PROJECT_ROOT

BBOX_MANIFEST_NAME = "bbox_crop_manifest.json"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_MANIFEST_VERSION = 1
_FILL_VALUE_RANGE = (0, 255)
_MAX_MARGIN = 512
_MAX_MIN_SIZE = 4096
# 灰边边界存在抗锯齿与标注亚像素误差，裁剪框额外外扩的像素数。
_BOUNDARY_PAD = 2


@dataclass(frozen=True)
class BboxCropSettings:
    """内容裁剪的几何参数与导出位置。"""

    fill_value: int
    tolerance: int
    margin: int
    min_size: int
    dataset: Path
    data_config: Path

    def record(self) -> dict[str, int]:
        """返回用于清单比对的几何参数。"""
        return {
            "fill_value": self.fill_value,
            "tolerance": self.tolerance,
            "margin": self.margin,
            "min_size": self.min_size,
        }


def load_bbox_crop_settings(
    block: object, config_file: Path, project_root: Path = PROJECT_ROOT
) -> BboxCropSettings:
    """校验 bbox_crop 块并返回设置，忽略 enabled 字段。

    Args:
        block: 训练配置中的 bbox_crop 映射。
        config_file: 训练配置文件路径，仅用于错误信息。
        project_root: 相对路径的解析基准。

    Returns:
        校验通过的裁剪设置。

    Raises:
        ValueError: 配置块缺失、类型错误或取值越界。
    """
    if not isinstance(block, dict):
        raise ValueError(f"{config_file} 的 bbox_crop 必须为配置映射")
    return BboxCropSettings(
        fill_value=_read_int(block, "fill_value", config_file, 114, _FILL_VALUE_RANGE),
        tolerance=_read_int(block, "tolerance", config_file, 2, _FILL_VALUE_RANGE),
        margin=_read_int(block, "margin", config_file, 0, (0, _MAX_MARGIN)),
        min_size=_read_int(block, "min_size", config_file, 64, (32, _MAX_MIN_SIZE)),
        dataset=_read_path(block, "dataset", config_file, project_root),
        data_config=_read_path(block, "data_config", config_file, project_root),
    )


def parse_bbox_crop(
    block: object, config_file: Path, project_root: Path = PROJECT_ROOT
) -> BboxCropSettings | None:
    """解析 bbox_crop 块，关闭或缺少 enabled 时返回 None。"""
    if not isinstance(block, dict):
        raise ValueError(f"{config_file} 的 bbox_crop 必须为配置映射")
    enabled = block.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError(f"{config_file} 的 bbox_crop.enabled 必须为布尔值")
    if not enabled:
        return None
    return load_bbox_crop_settings(block, config_file, project_root)


def find_content_bbox(
    image: NDArray[np.uint8], settings: BboxCropSettings
) -> tuple[int, int, int, int]:
    """返回超过填充色的内容外接矩形，右下为开区间。

    Args:
        image: H×W×3 uint8 图像，BGR 或 RGB 均可。
        settings: 填充色、容差、外扩与最小边长。

    Returns:
        (x0, y0, x1, y1)，单位为原图像素。

    Raises:
        ValueError: 图像通道数不符或整幅等于填充色。
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"内容裁剪需要 H×W×3 图像，实际 {image.shape}")
    difference = np.abs(image.astype(np.int16) - settings.fill_value)
    content = (difference > settings.tolerance).any(axis=2)
    rows = np.flatnonzero(content.any(axis=1))
    columns = np.flatnonzero(content.any(axis=0))
    if not rows.size or not columns.size:
        raise ValueError("图片没有超出填充色的有效内容")
    height, width = image.shape[:2]
    pad = settings.margin + _BOUNDARY_PAD
    x0 = max(0, int(columns[0]) - pad)
    y0 = max(0, int(rows[0]) - pad)
    x1 = min(width, int(columns[-1]) + 1 + pad)
    y1 = min(height, int(rows[-1]) + 1 + pad)
    x0, x1 = _grow_to_size(x0, x1, width, settings.min_size)
    y0, y1 = _grow_to_size(y0, y1, height, settings.min_size)
    return x0, y0, x1, y1


def manifest_path(dataset: Path) -> Path:
    """返回裁剪数据集的清单路径。"""
    return dataset / BBOX_MANIFEST_NAME


def file_digest(path: Path) -> str:
    """返回文件内容的 sha256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    settings: BboxCropSettings,
    source_config: Path,
    splits: dict[str, int],
    images: list[dict[str, object]],
) -> dict[str, object]:
    """构造导出清单，用于训练前的缓存一致性校验。"""
    return {
        "version": _MANIFEST_VERSION,
        "source_config": project_relative(source_config),
        "source_config_sha256": file_digest(source_config),
        "settings": settings.record(),
        "splits": splits,
        "images": images,
    }


def write_manifest(dataset: Path, payload: dict[str, object]) -> None:
    """把导出清单写入数据集目录。"""
    manifest_path(dataset).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def verify_bbox_dataset(settings: BboxCropSettings, source_config: Path) -> Path:
    """校验裁剪数据集与当前配置一致，返回数据集配置文件路径。

    Args:
        settings: 训练配置中的裁剪设置。
        source_config: 原始数据集配置，用于比对 sha256。

    Returns:
        可直接写入训练选项的裁剪数据集配置路径。

    Raises:
        FileNotFoundError: 缺少数据集配置、清单或图片目录。
        ValueError: 清单与当前配置或实际文件数量不一致。
    """
    if not settings.data_config.is_file():
        raise FileNotFoundError(
            f"缺少裁剪数据集配置 {settings.data_config}，"
            "请先运行 data_preprocess/export_bbox_crop.py"
        )
    manifest_file = manifest_path(settings.dataset)
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"裁剪数据集缺少清单 {manifest_file}，"
            "请先运行 data_preprocess/export_bbox_crop.py"
        )
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_file} 的格式无效")
    if payload.get("source_config_sha256") != file_digest(source_config):
        raise ValueError(
            f"裁剪数据集与 {source_config} 不一致，请重新导出并加 --refresh"
        )
    if payload.get("settings") != settings.record():
        raise ValueError("裁剪数据集的参数与当前配置不一致，请加 --refresh 重新导出")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"{manifest_file} 缺少 splits 字段")
    for split, expected in splits.items():
        images_dir = settings.dataset / "images" / str(split)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"裁剪数据集缺少划分目录 {images_dir}")
        actual = sum(
            1 for item in images_dir.iterdir()
            if item.suffix.lower() in IMAGE_SUFFIXES
        )
        if actual != int(expected):
            raise ValueError(
                f"裁剪数据集 {split} 数量不一致：清单 {expected}，实际 {actual}"
            )
    return settings.data_config


def project_relative(path: Path) -> str:
    """返回项目相对路径，项目外路径保留绝对形式。"""
    resolved = path.resolve()
    return (
        resolved.relative_to(PROJECT_ROOT).as_posix()
        if resolved.is_relative_to(PROJECT_ROOT) else str(resolved)
    )


def _read_int(
    block: dict[str, object],
    key: str,
    config_file: Path,
    default: int,
    bounds: tuple[int, int],
) -> int:
    """读取整数配置并校验取值范围。"""
    value = block.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{config_file} 的 bbox_crop.{key} 必须为整数")
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(
            f"{config_file} 的 bbox_crop.{key} 必须在 {bounds[0]} 到 {bounds[1]} 之间"
        )
    return value


def _read_path(
    block: dict[str, object], key: str, config_file: Path, project_root: Path
) -> Path:
    """读取路径配置并解析为绝对路径。"""
    value = block.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{config_file} 的 bbox_crop.{key} 必须为非空路径字符串")
    return (project_root / value).resolve()


def _grow_to_size(start: int, end: int, limit: int, size: int) -> tuple[int, int]:
    """把区间居中扩展到至少 size，越界时整体平移。"""
    if end - start >= size:
        return start, end
    start = max(0, start - (size - (end - start)) // 2)
    end = min(limit, start + size)
    return max(0, end - size), end
