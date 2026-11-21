"""用已训练的钢轨分割权重预测并评估测试集。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preprocess.common.config import PROJECT_ROOT
from experiments.yolo_seg.valid_rail_seg import (
    best_weights,
    dataset_config,
    latest_experiment,
    predict_config,
)
from utils.segmentation_predict import load_split_image_dir, predict_split


def predict_rail_seg(
    weights_path: str | Path,
    test_dir: str | Path,
    config_path: str | Path,
    *,
    save_overlays: bool | None = None,
) -> None:
    """将测试集掩码、叠加图和指标 CSV 保存到实验的 test 目录。

    相对路径以项目根目录为基准。

    Args:
        weights_path: 实例分割 .pt 权重文件路径。
        test_dir: 测试集图片目录，需有对应的 YOLO 分割标签。
        config_path: 预测 YAML 文件路径。
        save_overlays: 为 True 时保存逐图叠加图；None 时读取预测配置，
            缺省不保存。
    """
    predict_split(
        weights_path, test_dir, config_path, "test",
        save_overlays=save_overlays,
    )


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
    test_dir = load_split_image_dir(dataset_config(run_dir), "test")
    predict_rail_seg(
        best_weights(run_dir), test_dir, predict_config(args.config),
        save_overlays=True if args.overlays else None,
    )


if __name__ == "__main__":
    main()
