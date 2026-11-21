"""用已训练的钢轨分割权重预测并评估验证集。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preprocess.common.config import PROJECT_ROOT
from utils.segmentation_predict import (
    PREDICT_CONFIG,
    load_split_image_dir,
    predict_split,
)


def predict_rail_seg(
    weights_path: str | Path,
    validation_dir: str | Path,
    config_path: str | Path,
    *,
    save_overlays: bool | None = None,
) -> None:
    """将验证集掩码、叠加图和指标 CSV 保存到实验的 valid 目录。

    相对路径以项目根目录为基准。

    Args:
        weights_path: 实例分割 .pt 权重文件路径。
        validation_dir: 验证集图片目录，需有对应的 YOLO 分割标签。
        config_path: 预测 YAML 文件路径。
        save_overlays: 为 True 时保存逐图叠加图；None 时读取预测配置，
            缺省不保存。
    """
    predict_split(
        weights_path, validation_dir, config_path, "valid",
        save_overlays=save_overlays,
    )


def latest_experiment(root: Path | None = None) -> Path:
    """返回 out/yolo_seg 下最新的、含权重目录的实验。"""
    base = root if root is not None else PROJECT_ROOT / "out" / "yolo_seg"
    runs = sorted(
        path for path in base.iterdir()
        if path.is_dir() and (path / "weights").is_dir()
    ) if base.is_dir() else []
    if not runs:
        raise SystemExit(f"{base} 下没有含 weights 的实验目录，请先训练或指定 --run")
    return runs[-1]


def best_weights(run_dir: Path) -> Path:
    """在实验目录里优先取归档最优权重，其次 best.pt、last.pt。"""
    weights = run_dir / "weights"
    for name in ("rail_yolo26m-seg.pt", "best.pt", "last.pt"):
        candidate = weights / name
        if candidate.is_file():
            return candidate
    candidates = sorted(weights.glob("*.pt"))
    if not candidates:
        raise SystemExit(f"{weights} 下没有 .pt 权重")
    return candidates[-1]


def dataset_config(run_dir: Path) -> Path:
    """返回实验归档的数据集配置，优先裁边版本。"""
    configs = run_dir / "configs"
    for name in ("clop_seg_bbox.yaml", "clop_seg.yaml"):
        candidate = configs / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"{configs} 下没有 clop_seg.yaml 或 clop_seg_bbox.yaml")


def predict_config(path: Path | None) -> Path:
    """返回预测配置路径，None 表示使用项目默认配置。

    Args:
        path: 命令行传入的配置路径；相对路径以项目根目录为基准。

    Returns:
        可直接传给预测函数的配置路径。
    """
    if path is None:
        return PREDICT_CONFIG
    return path if path.is_absolute() else Path(PROJECT_ROOT, path)


def main() -> None:
    """命令行入口：默认使用最新一次 YOLO 训练实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path,
        help="实验目录（相对项目根目录）；省略时取 out/yolo_seg 下最新一次",
    )
    parser.add_argument(
        "--config", type=Path,
        help="预测配置；省略时用 config/predict/rail_predict.yaml，"
        "免 NMS 对照用 config/predict/rail_predict_nmsfree.yaml",
    )
    parser.add_argument(
        "--overlays", action="store_true", help="保存逐图叠加图（默认不保存）"
    )
    args = parser.parse_args()
    run_dir = (
        Path(PROJECT_ROOT, args.run) if args.run is not None else latest_experiment()
    )
    print(f"实验目录：{run_dir}")
    validation_dir = load_split_image_dir(dataset_config(run_dir), "val")
    predict_rail_seg(
        best_weights(run_dir), validation_dir, predict_config(args.config),
        save_overlays=True if args.overlays else None,
    )


if __name__ == "__main__":
    main()
