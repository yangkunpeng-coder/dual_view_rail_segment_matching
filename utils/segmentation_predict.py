"""保存钢轨实例掩码叠加图和独立评估指标。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
import os
from pathlib import Path
from shutil import copy2
from time import perf_counter
from typing import TYPE_CHECKING, Literal

from data_preprocess.common.config import (
    PROJECT_ROOT,
    config_path,
    load_yaml_mapping,
)

if TYPE_CHECKING:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results

PREDICT_CONFIG = config_path("rail_predict.yaml")
_PREDICT_KEYS = {
    "imgsz", "device", "batch", "conf", "iou", "nms", "max_det",
    "agnostic_nms", "retina_masks", "augment", "quantize",
}
# 项目自己的预测开关，不传给 YOLO.predict。
_PROJECT_KEYS = {"save_overlays"}
_DATA_SPLIT_NAMES: dict[str, Literal["train", "val", "test"]] = {
    "train": "train",
    "valid": "val",
    "test": "test",
}
_PREDICTION_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"
# 叠加图只供人工复核，存 JPG 足够；单张体积约为原图 PNG 的十分之一。
OVERLAY_SUFFIX = ".jpg"
OVERLAY_JPEG_QUALITY = 92


def load_predict_options(config_path: Path) -> dict[str, object]:
    """读取可调的 YOLO 预测参数并补全默认置信度门限。

    Args:
        config_path: 预测 YAML 文件路径。

    Returns:
        可传给 YOLO.predict 的选项，conf 为浮点数。

    Raises:
        OSError: 无法读取配置文件。
        ValueError: 预测参数未知或取值无效。
    """
    from ultralytics.cfg import get_cfg

    options = load_yaml_mapping(config_path)
    unknown_keys = options.keys() - _PREDICT_KEYS - _PROJECT_KEYS
    if unknown_keys:
        names = ", ".join(sorted(unknown_keys))
        raise ValueError(f"{config_path} 包含未开放的预测参数 {names}")
    options.pop("save_overlays", None)
    try:
        get_cfg(overrides=options)
    except (TypeError, ValueError) as error:
        raise ValueError(f"预测配置无效 {config_path}\n{error}") from error
    confidence = options.get("conf")
    options["conf"] = (
        float(confidence) if isinstance(confidence, (int, float)) else 0.25
    )
    return options


def load_overlay_setting(config_path: Path) -> bool:
    """读取是否保存逐图叠加图，缺省关闭。

    Args:
        config_path: 预测 YAML 文件路径。

    Returns:
        需要保存全尺寸叠加图时为 True。

    Raises:
        OSError: 无法读取配置文件。
        ValueError: save_overlays 不是布尔值。
    """
    value = load_yaml_mapping(config_path).get("save_overlays", False)
    if not isinstance(value, bool):
        raise ValueError(f"{config_path} 的 save_overlays 必须为布尔值")
    return value


def load_split_image_dir(
    data_config: Path, split: Literal["train", "val", "test"]
) -> Path:
    """从本项目的 YOLO 数据集 YAML 定位训练、验证或测试图片目录。

    Args:
        data_config: 数据集 YAML 文件路径。
        split: 数据集配置中的划分名称。

    Returns:
        图片目录的绝对路径。

    Raises:
        OSError: 无法读取数据集配置。
        ValueError: 数据集根目录或指定划分缺少路径字符串。
    """
    options = load_yaml_mapping(data_config)
    dataset_dir = options.get("path")
    split_dir = options.get(split)
    if not isinstance(dataset_dir, str) or not dataset_dir.strip():
        raise ValueError(f"{data_config} 的 path 必须为非空路径字符串")
    if not isinstance(split_dir, str) or not split_dir.strip():
        raise ValueError(f"{data_config} 的 {split} 必须为非空路径字符串")
    return (PROJECT_ROOT / dataset_dir / split_dir).resolve()


def prediction_image_path(output_dir: Path, image_name: str) -> Path:
    """根据图片的相机前缀生成预测图归档路径。

    叠加图统一使用 JPG 扩展名以控制体积；调用方需要掩码路径时，
    可继续对返回值调用 ``with_suffix(".npz")``。

    Args:
        output_dir: 本次预测图的根目录。
        image_name: 源图片文件名。

    Returns:
        按南北相机归档的 JPG 路径，无相机前缀的图片直接归入根目录。
    """
    side, _, original_name = image_name.partition("_")
    if side in ("north", "south") and original_name:
        path = output_dir / side / original_name
    else:
        path = output_dir / image_name
    return path.with_suffix(OVERLAY_SUFFIX)


def iter_segmentation_predictions(
    model: YOLO, image_dir: Path, options: dict[str, object]
) -> Iterator[Results]:
    """逐张返回 Ultralytics 的图像预测结果。

    Args:
        model: 已加载权重的 YOLO 实例分割模型。
        image_dir: 包含待预测图片的目录。
        options: 从预测配置读取的 YOLO 参数。

    Yields:
        含图像路径及可选实例 Mask 的预测结果。

    Raises:
        TypeError: 模型返回了特征张量等非图像预测结果。
    """
    from ultralytics.engine.results import Results

    predictions = model.predict(
        source=str(image_dir), stream=True, save=False, verbose=False, **options
    )
    for result in predictions:
        if not isinstance(result, Results):
            raise TypeError("预测未返回图像结果")
        yield result


def save_segmentation_predictions(
    model: YOLO,
    image_dir: Path,
    output_dir: Path,
    options: dict[str, object],
) -> dict[str, Path]:
    """逐图保存实例掩码叠加图，沿用原图尺寸和文件名。

    Args:
        model: 已加载权重的 YOLO 实例分割模型。
        image_dir: 包含待预测图片的目录。
        output_dir: 预测图归档目录。
        options: 从预测配置读取的 YOLO 参数。

    Returns:
        数据集文件名到源图片路径的映射。

    Raises:
        OSError: 无法读取图片目录或写入预测图。
    """
    image_paths: dict[str, Path] = {}
    for result in iter_segmentation_predictions(model, image_dir, options):
        image_path = Path(result.path)
        destination = prediction_image_path(output_dir, image_path.name)
        result.save(filename=str(destination), boxes=False, color_mode="instance")
        if not destination.is_file():
            raise OSError(f"保存预测图失败 {destination}")
        image_paths[image_path.name] = image_path
    return image_paths


def predict_split(
    weights_path: str | Path,
    image_dir: str | Path,
    config_path: str | Path,
    split: Literal["train", "valid", "test"],
    *,
    save_overlays: bool | None = None,
) -> None:
    """用本地权重预测并评估一个划分，按时间归档结果。

    相对路径以项目根目录为基准。

    Args:
        weights_path: 实例分割权重路径。
        image_dir: 待预测的 YOLO 图片目录，需有对应分割标签。
        config_path: 预测 YAML 文件路径。
        split: 实验目录下用于归档的子目录名，支持 train、valid、test。
        save_overlays: 为 False 时只保存掩码和指标；None 时读取预测配置
            的 `save_overlays`，缺省关闭。

    Raises:
        OSError: 缺少输入文件、图片目录或分割标签，或无法保存结果。
        ValueError: 权重类型或预测配置无效。
    """
    weights_file = Path(PROJECT_ROOT, weights_path).resolve()
    source_dir = Path(PROJECT_ROOT, image_dir).resolve()
    config_file = Path(PROJECT_ROOT, config_path).resolve()
    if weights_file.suffix.lower() != ".pt":
        raise ValueError(f"预测需要本地 .pt 权重 {weights_file}")
    if not weights_file.is_file():
        raise FileNotFoundError(f"缺少预测权重 {weights_file}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"缺少预测图片目录 {source_dir}")
    if not config_file.is_file():
        raise FileNotFoundError(f"缺少预测配置 {config_file}")

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    model = YOLO(str(weights_file))
    if model.task != "segment":
        raise ValueError(f"预测需要实例分割权重 {weights_file}")

    run_dir = weights_file.parent
    if run_dir.name.lower() == "weights":
        run_dir = run_dir.parent
    timestamp = datetime.now().strftime(_PREDICTION_TIMESTAMP_FORMAT)
    output_dir = run_dir / split / timestamp
    output_dir.mkdir(parents=True)
    config_snapshot = output_dir / config_file.name
    copy2(config_file, config_snapshot)
    options = load_predict_options(config_snapshot)
    if save_overlays is None:
        save_overlays = load_overlay_setting(config_snapshot)
    import numpy as np

    from core_algorithm.postprocess import run_postprocess
    from utils.instance_masks import as_mask_stack
    from utils.segmentation_evaluation import (
        MaskEvaluation,
        MaskPrediction,
        save_mask_prediction,
    )
    from utils.segmentation_postprocess import (
        POSTPROCESS_CONFIG,
        PostprocessImageStat,
        image_side,
        load_postprocess_config,
        save_instance_counts,
        save_postprocess_report,
    )
    from utils.segmentation_review import save_missed_samples

    postprocess_source = run_dir / "configs" / POSTPROCESS_CONFIG.name
    if not postprocess_source.is_file():
        postprocess_source = POSTPROCESS_CONFIG
    postprocess_config = load_postprocess_config(postprocess_source)
    copy2(postprocess_source, output_dir / POSTPROCESS_CONFIG.name)
    confidence = options["conf"]
    if not isinstance(confidence, float):
        raise TypeError("已解析的 YOLO 预测置信度必须为浮点数")
    evaluation = MaskEvaluation(confidence)
    image_paths: dict[str, Path] = {}
    postprocess_stats: list[PostprocessImageStat] = []
    for result in iter_segmentation_predictions(model, source_dir, options):
        image_path = Path(result.path)
        started = perf_counter()
        raw_masks = as_mask_stack(result.masks, result.orig_shape)
        raw_scores = (
            result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            if result.boxes is not None else np.empty(0, dtype=np.float32)
        )
        if postprocess_config.enabled:
            raw_path = prediction_image_path(
                output_dir / "masks_raw", image_path.name
            ).with_suffix(".npz")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                raw_path, masks=raw_masks, scores=raw_scores
            )
        boxes = None
        if result.boxes is not None and len(result.boxes) == len(raw_masks):
            boxes = result.boxes.xyxy.detach().cpu().numpy()
        runtime = sum(value for value in result.speed.values() if value is not None)
        runtime += (perf_counter() - started) * 1000
        postprocessed = run_postprocess(
            raw_masks, raw_scores, postprocess_config, boxes=boxes
        )
        masks, scores = postprocessed.masks, postprocessed.scores
        prediction = MaskPrediction(masks, scores, runtime)
        evaluation.update(image_path, prediction)
        save_mask_prediction(
            output_dir, image_path, result.orig_img, prediction, confidence,
            save_overlay=save_overlays,
        )
        postprocess_stats.append(PostprocessImageStat(
            image=image_path.name,
            side=image_side(image_path.name),
            raw_count=len(raw_masks),
            kept_count=len(masks),
            suppressed_count=postprocessed.suppressed_count,
            postprocess_ms=postprocessed.latency_ms,
        ))
        image_paths[image_path.name] = image_path
        print(f"已处理 {image_path.name}")
    data_split = _DATA_SPLIT_NAMES[split]
    evaluation.save(
        output_dir, data_split, "yolo26_seg", weights_file, options
    )
    count_rows: list[dict[str, object]] = []
    for stat in postprocess_stats:
        counts = evaluation.image_counts.get(stat.image)
        if counts is None:
            continue
        count_rows.append({
            "side": stat.side,
            "image": stat.image,
            "gt": counts.target_count,
            "pred": stat.kept_count,
            "tp": counts.matched_count,
            "fp": stat.kept_count - counts.matched_count,
            "missed": counts.missed_count,
        })
    save_instance_counts(output_dir / "metrics", data_split, count_rows)
    save_postprocess_report(
        output_dir / "metrics", postprocess_config, postprocess_stats
    )
    if save_overlays:
        save_missed_samples(
            output_dir, image_paths, evaluation.image_counts, confidence,
            prediction_dir=output_dir,
        )
    print(f"已保存 {len(image_paths)} 张图片的 Mask 和评价指标至 {output_dir}")
