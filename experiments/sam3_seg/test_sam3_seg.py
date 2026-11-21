"""用本地 SAM3 预训练权重在项目测试集上生成 Mask 和评价指标。"""

import argparse
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preprocess.common.config import PROJECT_ROOT, config_path
from utils.segmentation_predict import load_split_image_dir

_PREDICT_CONFIG = config_path("sam3_predict.yaml")


def predict_sam3_seg(
    weights_path: str | Path,
    test_dir: str | Path,
    config_path: str | Path,
    *,
    save_overlays: bool | None = None,
) -> None:
    """评价测试集，输出到 out/sam3_seg/时间戳/test。

    Args:
        weights_path: 本地 SAM3 .pt 权重，路径相对于项目根目录。
        test_dir: 测试图片目录，图片须有对应的 YOLO Polygon 标签。
        config_path: SAM3 预测 YAML 路径。
        save_overlays: 为 True 时保存逐图叠加图；None 时读取预测配置，
            缺省不保存。
    """
    from utils.sam3_predict import predict_sam3_split

    predict_sam3_split(
        weights_path, test_dir, config_path, "test",
        save_overlays=save_overlays,
    )


def main() -> None:
    """命令行入口：默认使用预训练权重评价测试集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlays", action="store_true", help="保存逐图叠加图（默认不保存）"
    )
    args = parser.parse_args()
    weights_path = "pretrain_pt/sam3.pt"
    test_dir = load_split_image_dir(
        config_path("clop_seg.yaml"), "test"
    )
    predict_sam3_seg(
        weights_path, test_dir, _PREDICT_CONFIG,
        save_overlays=True if args.overlays else None,
    )


if __name__ == "__main__":
    main()
