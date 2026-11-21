"""用已训练的钢轨分割权重预测并评估训练集。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preprocess.common.config import PROJECT_ROOT
from experiments.yolo_seg.valid_rail_seg import best_weights, latest_experiment
from utils.segmentation_predict import (
    PREDICT_CONFIG,
    load_split_image_dir,
    predict_split,
)


def predict_train_rail_seg(
    weights_path: str | Path,
    train_dir: str | Path,
    config_path: str | Path,
) -> None:
    """将训练集掩码和指标 CSV 保存到实验的 train 目录。

    训练集图片较多，默认不生成全尺寸叠加图和漏检样本对照，只保留
    原图尺寸的实例掩码 NPZ 与 Mask 评价指标。

    相对路径以项目根目录为基准。

    Args:
        weights_path: 实例分割 .pt 权重文件路径。
        train_dir: 训练集图片目录，需有对应的 YOLO 分割标签。
        config_path: 预测 YAML 文件路径。
    """
    predict_split(
        weights_path, train_dir, config_path, "train", save_overlays=False
    )


def main() -> None:
    """命令行入口：默认使用最新一次 YOLO 训练实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path,
        help="实验目录（相对项目根目录）；省略时取 out/yolo_seg 下最新一次",
    )
    args = parser.parse_args()
    run_dir = (
        Path(PROJECT_ROOT, args.run) if args.run is not None else latest_experiment()
    )
    print(f"实验目录：{run_dir}")
    train_dir = load_split_image_dir(run_dir / "configs" / "clop_seg.yaml", "train")
    predict_train_rail_seg(best_weights(run_dir), train_dir, PREDICT_CONFIG)


if __name__ == "__main__":
    main()
