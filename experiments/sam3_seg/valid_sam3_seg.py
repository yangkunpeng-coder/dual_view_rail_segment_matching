"""用本地 SAM3 预训练权重在项目验证集上生成 Mask 和评价指标。"""

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
    validation_dir: str | Path,
    config_path: str | Path,
    *,
    save_overlays: bool | None = None,
) -> None:
    """验证一份 YOLO 格式数据，将产物归档到 out/sam3_seg/时间戳。

    Args:
        weights_path: 本地 SAM3 .pt 权重，相对路径以项目根目录为基准。
        validation_dir: 验证图片目录，图片须有对应的 YOLO Polygon 标签。
        config_path: SAM3 预测 YAML 路径。
        save_overlays: 为 True 时保存逐图叠加图；None 时读取预测配置，
            缺省不保存。

    Raises:
        OSError: 缺少权重、图片、标签或无法写入产物。
        ValueError: 预测配置或数据格式无效。
    """
    from utils.sam3_predict import predict_sam3_split

    predict_sam3_split(
        weights_path, validation_dir, config_path, "valid",
        save_overlays=save_overlays,
    )


def main() -> None:
    """命令行入口：默认使用预训练权重评价验证集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlays", action="store_true", help="保存逐图叠加图（默认不保存）"
    )
    args = parser.parse_args()
    weights_path = "pretrain_pt/sam3.pt"
    validation_dir = load_split_image_dir(config_path("clop_seg.yaml"), "val")
    predict_sam3_seg(
        weights_path, validation_dir, _PREDICT_CONFIG,
        save_overlays=True if args.overlays else None,
    )


if __name__ == "__main__":
    main()
