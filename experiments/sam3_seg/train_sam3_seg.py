"""用本地 SAM3 预训练权重在项目训练集上生成 Mask 和评价指标。"""

import argparse
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preprocess.common.config import PROJECT_ROOT, config_path, load_yaml_mapping

# SAM3 在项目内没有训练流程，训练划分同样按独立预测方式评价并归档。
_PREDICT_CONFIG = config_path("sam3_predict.yaml")
_DATA_CONFIG = config_path("clop_seg.yaml")


def _train_image_dir(data_config: Path) -> Path:
    """从数据集配置解析训练图片目录。

    `load_split_image_dir` 只登记验证与测试划分，训练划分在这里单独解析，
    避免为一次训练集评价改动 YOLO 预测共用的路径函数。

    Args:
        data_config: 数据集 YAML 文件路径。

    Returns:
        训练图片目录的绝对路径。

    Raises:
        OSError: 无法读取数据集配置。
        ValueError: 数据集根目录或训练划分缺少路径字符串。
    """
    options = load_yaml_mapping(data_config)
    dataset_dir = options.get("path")
    split_dir = options.get("train")
    if not isinstance(dataset_dir, str) or not dataset_dir.strip():
        raise ValueError(f"{data_config} 的 path 必须为非空路径字符串")
    if not isinstance(split_dir, str) or not split_dir.strip():
        raise ValueError(f"{data_config} 的 train 必须为非空路径字符串")
    return (PROJECT_ROOT / dataset_dir / split_dir).resolve()


def predict_sam3_seg(
    weights_path: str | Path,
    train_dir: str | Path,
    config_path: str | Path,
    *,
    save_overlays: bool | None = None,
) -> None:
    """在训练集上评价 SAM3，将产物归档到 out/sam3_seg/时间戳/train。

    指标 CSV 的 split 记为 train，其余列与验证、测试归档保持一致。

    Args:
        weights_path: 本地 SAM3 .pt 权重，相对路径以项目根目录为基准。
        train_dir: 训练图片目录，图片须有对应的 YOLO Polygon 标签。
        config_path: SAM3 预测 YAML 路径。
        save_overlays: 为 True 时保存逐图叠加图；None 时读取预测配置，
            缺省不保存。

    Raises:
        OSError: 缺少权重、图片、标签或无法写入产物。
        ValueError: 预测配置或数据格式无效。
    """
    from utils.sam3_predict import predict_sam3_split

    predict_sam3_split(
        weights_path, train_dir, config_path, "train",
        save_overlays=save_overlays,
    )


def main() -> None:
    """命令行入口：默认使用预训练权重评价训练集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlays", action="store_true", help="保存逐图叠加图（默认不保存）"
    )
    args = parser.parse_args()
    predict_sam3_seg(
        "pretrain_pt/sam3.pt", _train_image_dir(_DATA_CONFIG), _PREDICT_CONFIG,
        save_overlays=True if args.overlays else None,
    )


if __name__ == "__main__":
    main()
