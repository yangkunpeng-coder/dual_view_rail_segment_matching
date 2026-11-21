"""预览辅助监督目标：把中心线、方向与宽度叠加到训练图上。

默认从 ``config/data/clop_seg.yaml`` 读取数据划分，抽样若干训练图生成目标，输出叠加图与逐图
统计 CSV。只读数据，不写任何训练产物。

用法：
    python -m experiments.yolo_seg.gas_target_preview --count 6
    python tools/run_pipeline.py gas-preview
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from math import atan2, cos, sin
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import yaml

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preprocess.common.config import PROJECT_ROOT, config_path
from data_preprocess.gas_targets import GasTargetSpec, build_gas_targets

_DATA_CONFIG = config_path("clop_seg.yaml")
_OUTPUT_ROOT = PROJECT_ROOT / "out" / "gas_target_preview"
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_PREVIEW_MAX_WIDTH = 1400
_DIRECTION_COLOR = (0, 220, 255)
_CENTER_COLOR = (80, 255, 120)
_STAT_COLUMNS = (
    "split", "image", "instances", "center_cells", "center_valid_cells",
    "direction_cells", "width_median_px", "elapsed_ms",
)


@dataclass(frozen=True)
class PreviewOptions:
    """预览参数。

    Attributes:
        data_config: 数据集 YAML 路径。
        split: 数据划分名，取 ``train``/``val``/``test``。
        image_count: 抽样图片数。
        seed: 抽样随机种子。
        target_spec: 目标生成参数。
        imgsz: 训练画布长边，与训练配置一致。
        stride: 画布补边对齐步长，与训练一致。
    """

    data_config: Path
    split: str
    image_count: int
    seed: int
    target_spec: GasTargetSpec
    imgsz: int = 1280
    stride: int = 32


def _parse_args() -> argparse.Namespace:
    """解析命令行参数，默认值指向项目内现有配置。"""
    parser = argparse.ArgumentParser(description="预览辅助监督目标")
    parser.add_argument("--data", default=str(_DATA_CONFIG), help="数据集 YAML 路径")
    parser.add_argument("--split", default="train", help="数据划分 train/val/test")
    parser.add_argument("--count", type=int, default=6, help="抽样图片数")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--mask-ratio", type=int, default=4, help="目标网格比例")
    parser.add_argument("--imgsz", type=int, default=1280, help="训练画布长边")
    parser.add_argument("--stride", type=int, default=32, help="画布补边对齐步长")
    parser.add_argument("--output", default="", help="输出目录，默认 out 下时间戳目录")
    return parser.parse_args()


def _letterbox(
    image: np.ndarray, imgsz: int, stride: int
) -> tuple[np.ndarray, float, int, int]:
    """按训练 LetterBox 规则缩放并补边，返回画布、缩放比与左上补边量。

    与训练一致：只缩小不放大，长边对齐 ``imgsz``，再补到 ``stride`` 的整数倍。
    """
    height, width = image.shape[:2]
    scale = min(imgsz / max(height, width), 1.0)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height))
    pad_width = stride - new_width % stride
    pad_height = stride - new_height % stride
    left, top = pad_width // 2, pad_height // 2
    canvas = np.full(
        (new_height + pad_height, new_width + pad_width, 3), 114, dtype=np.uint8
    )
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas, scale, left, top


def _load_dataset_config(config_path: Path) -> dict:
    """读取数据集 YAML，返回 path/划分路径/类别名。"""
    with config_path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"数据集配置格式无效 {config_path}")
    return config


def _split_directory(config: dict, split: str) -> Path:
    """解析某个划分的图片目录绝对路径。"""
    if split not in config:
        raise KeyError(f"数据集配置缺少 {split} 划分")
    root = PROJECT_ROOT / str(config["path"])
    return (root / str(config[split])).resolve()


def _label_path(image_path: Path) -> Path:
    """由图片路径推出同名 YOLO 标签路径（把 images 目录换成 labels）。"""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"图片路径里没有 images 目录 {image_path}")


def _read_segments(
    label_path: Path, width: int, height: int
) -> list[np.ndarray]:
    """读取 YOLO 多边形标签并换算成画布像素坐标。"""
    if not label_path.is_file():
        raise FileNotFoundError(f"缺少标签文件 {label_path}")
    segments = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = np.array(line.split(), dtype=np.float64)
        if values.size < 7:
            continue
        points = values[1:].reshape(-1, 2) * np.array((width, height))
        segments.append(points.astype(np.float32))
    return segments


def _direction_hint(
    targets: dict[str, np.ndarray], ratio: int, image: np.ndarray
) -> None:
    """在有效方向上画短线段，直观检查每处的延伸方向。"""
    direction = targets["gas_direction"]
    valid = targets["gas_direction_valid"][0] > 0
    rows, columns = np.nonzero(valid)
    for row, column in zip(rows[::3], columns[::3]):
        angle = 0.5 * atan2(float(direction[1, row, column]),
                            float(direction[0, row, column]))
        center = (
            int((column + 0.5) * ratio),
            int((row + 0.5) * ratio),
        )
        delta = np.array((cos(angle), sin(angle))) * ratio * 1.6
        start = (int(center[0] - delta[0]), int(center[1] - delta[1]))
        end = (int(center[0] + delta[0]), int(center[1] + delta[1]))
        cv2.line(image, start, end, _DIRECTION_COLOR, 1, cv2.LINE_AA)


def _render(
    image: np.ndarray, targets: dict[str, np.ndarray], ratio: int
) -> np.ndarray:
    """把中心线热力图与方向提示叠加到原图。"""
    overlay = image.copy()
    height, width = image.shape[:2]
    heat = targets["gas_center"][0]
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = heat > 0.15
    tint = np.zeros_like(overlay)
    tint[mask] = _CENTER_COLOR
    overlay = cv2.addWeighted(overlay, 1.0, tint, 0.35, 0)
    _direction_hint(targets, ratio, overlay)
    if overlay.shape[1] > _PREVIEW_MAX_WIDTH:
        scale = _PREVIEW_MAX_WIDTH / overlay.shape[1]
        overlay = cv2.resize(overlay, None, fx=scale, fy=scale)
    return overlay


def _image_statistics(
    targets: dict[str, np.ndarray], ratio: int
) -> tuple[int, int, int, float]:
    """统计中心线格数、方向格数与中位视宽（像素）。"""
    center = targets["gas_center"][0]
    center_valid = targets["gas_center_valid"][0]
    direction_valid = targets["gas_direction_valid"][0]
    width = targets["gas_width"][0]
    valid = center_valid > 0
    width_px = float(np.exp(np.median(width[valid])) * ratio) if valid.any() else 0.0
    return (
        int((center > 0.5).sum()),
        int((center_valid > 0).sum()),
        int((direction_valid > 0).sum()),
        width_px,
    )


def run_preview(options: PreviewOptions, output_dir: Path | None = None) -> Path:
    """抽样生成叠加图与统计 CSV，返回输出目录。

    Args:
        options: 预览参数。
        output_dir: 显式输出目录；为 None 时在 ``out/gas_target_preview`` 下建时间戳目录。

    Returns:
        本次输出目录（含叠加图与 ``targets_preview.csv``）。
    """
    config = _load_dataset_config(options.data_config)
    image_dir = _split_directory(config, options.split)
    images = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        raise FileNotFoundError(f"划分目录没有图片 {image_dir}")
    random_source = random.Random(options.seed)
    selected = random_source.sample(images, min(options.image_count, len(images)))
    if output_dir is None:
        output_dir = _OUTPUT_ROOT / datetime.now().strftime(_TIMESTAMP_FORMAT)
    output_dir.mkdir(parents=True)
    rows = []
    for image_path in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError(f"无法读取图片 {image_path}")
        source_height, source_width = image.shape[:2]
        canvas, scale, left, top = _letterbox(image, options.imgsz, options.stride)
        height, width = canvas.shape[:2]
        segments = _read_segments(
            _label_path(image_path), source_width, source_height
        )
        segments = [
            segment * scale + np.array((left, top), dtype=np.float32)
            for segment in segments
        ]
        started = time.perf_counter()
        targets = {
            key: value.numpy()
            for key, value in build_gas_targets(
                segments, height, width, options.target_spec
            ).items()
        }
        elapsed_ms = (time.perf_counter() - started) * 1000
        preview = _render(canvas, targets, options.target_spec.mask_ratio)
        cv2.imwrite(str(output_dir / image_path.name), preview)
        center_cells, valid_cells, direction_cells, width_median = (
            _image_statistics(targets, options.target_spec.mask_ratio)
        )
        rows.append(
            {
                "split": options.split,
                "image": image_path.name,
                "instances": len(segments),
                "center_cells": center_cells,
                "center_valid_cells": valid_cells,
                "direction_cells": direction_cells,
                "width_median_px": f"{width_median:.2f}",
                "elapsed_ms": f"{elapsed_ms:.1f}",
            }
        )
        print(
            f"{image_path.name} | 实例 {len(segments)} | 中心格 {center_cells} | "
            f"方向格 {direction_cells} | 中位视宽 {width_median:.1f}px | "
            f"{elapsed_ms:.0f} ms"
        )
    with (output_dir / "targets_preview.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_STAT_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"预览图与统计已写入 {output_dir}")
    return output_dir


def main() -> None:
    """命令行入口。"""
    args = _parse_args()
    output_dir = Path(args.output).resolve() if args.output else None
    run_preview(
        PreviewOptions(
            data_config=(PROJECT_ROOT / args.data).resolve(),
            split=args.split,
            image_count=max(args.count, 1),
            seed=args.seed,
            target_spec=GasTargetSpec(mask_ratio=args.mask_ratio),
            imgsz=args.imgsz,
            stride=args.stride,
        ),
        output_dir,
    )


if __name__ == "__main__":
    main()
