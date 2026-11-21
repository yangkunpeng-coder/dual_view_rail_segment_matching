"""封装本地 SAM3 图片模型，按块在 CPU 上恢复高分辨率实例 Mask。"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from shutil import copy2
from time import perf_counter
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional as torch_functional
from sam3.model.geometry_encoders import Prompt
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from data_preprocess.common.config import PROJECT_ROOT, load_yaml_mapping
from utils.segmentation_evaluation import MaskPrediction

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sam3PredictOptions:
    """SAM3 推理参数，conf 为实例门限，mask_threshold 为像素门限。"""

    prompt: str = "steel rail"
    device: str = "cuda:0"
    resolution: int = 1008
    precision: str = "bfloat16"
    conf: float = 0.25
    display_conf: float = 0.25
    mask_threshold: float = 0.5
    max_det: int = 300
    mask_chunk_size: int = 4
    # 是否保存逐图全尺寸叠加图；默认关闭以控制体积。
    save_overlays: bool = False


class _ChunkedSam3Processor(Sam3Processor):
    """复用官方图片和文本处理，仅将 Mask 放回 CPU 分块恢复。"""

    def __init__(self, model: Sam3Image, options: Sam3PredictOptions) -> None:
        super().__init__(
            model, resolution=options.resolution, device=options.device,
            confidence_threshold=options.conf,
        )
        self._options = options

    @torch.inference_mode()
    def _forward_grounding(self, state: dict[str, object]) -> MaskPrediction:
        """保持官方打分和插值规则，避免同时在 GPU 展开全部原图 Mask。"""
        prompt = state["geometric_prompt"]
        height, width = state["original_height"], state["original_width"]
        if not isinstance(prompt, Prompt):
            raise TypeError("SAM3 几何提示状态格式无效")
        if not isinstance(height, int) or not isinstance(width, int):
            raise TypeError("SAM3 图片状态缺少原始图像尺寸")
        outputs: dict[str, Tensor] = self.model.forward_grounding(
            backbone_out=state["backbone_out"], find_input=self.find_stage,
            geometric_prompt=prompt, find_target=None,
        )
        probabilities = (
            outputs["pred_logits"].sigmoid()
            * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        ).squeeze(-1)
        keep = probabilities > self.confidence_threshold
        scores = probabilities[keep]
        order = scores.argsort(descending=True, stable=True)[:self._options.max_det]
        scores = scores[order].float().cpu().numpy()
        logits = outputs["pred_masks"][keep][order].float().cpu()
        masks = np.empty((len(scores), height, width), dtype=np.bool_)
        step = self._options.mask_chunk_size
        for start in range(0, len(scores), step):
            restored = torch_functional.interpolate(
                logits[start:start + step, None], (height, width),
                mode="bilinear", align_corners=False,
            )
            masks[start:start + step] = (
                restored[:, 0].sigmoid() > self._options.mask_threshold
            ).numpy()
        return MaskPrediction(masks, scores, 0.0)


class Sam3RailPredictor:
    """一次加载本地权重，对独立的南北图片逐张进行文本提示分割。"""

    def __init__(self, weights_path: Path, options: Sam3PredictOptions) -> None:
        """加载官方图片模型并使用显式设备。

        Args:
            weights_path: 本地 sam3.pt 文件。
            options: 已验证的预测配置。
        """
        self.options = options
        self.device = torch.device(options.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SAM3 配置选择了 CUDA，当前环境无法使用 CUDA")
        if self.device.type == "cpu" and options.precision != "float32":
            raise ValueError("CPU 推理请将 precision 设置为 float32")
        model = build_sam3_image_model(
            checkpoint_path=str(weights_path), device="cpu", load_from_HF=False,
            enable_inst_interactivity=False, compile=False,
        )
        self._model = model.to(self.device).eval()
        self._processor = _ChunkedSam3Processor(self._model, options)

    def predict(self, image: NDArray[np.uint8]) -> MaskPrediction:
        """预测一张 BGR 原图，返回 CPU 上原图尺寸的 Mask、分数和耗时。

        Args:
            image: H×W×3、uint8、BGR 图像。

        Returns:
            原图实例掩码、实例分数及本次推理耗时。
        """
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("SAM3 输入必须为 uint8 BGR 三通道图像")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        dtypes = {
            "float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        with torch.inference_mode(), torch.autocast(
            self.device.type, dtype=dtypes[self.options.precision],
            enabled=self.options.precision != "float32",
        ):
            source = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            state = self._processor.set_image(source)
            prediction = self._processor.set_text_prompt(
                prompt=self.options.prompt, state=state
            )
        if not isinstance(prediction, MaskPrediction):
            raise TypeError("SAM3 图片处理器没有返回项目实例掩码格式")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return MaskPrediction(
            prediction.masks, prediction.scores,
            (perf_counter() - started) * 1000,
        )


def load_sam3_options(config_path: Path) -> Sam3PredictOptions:
    """读取独立预测配置并检查文本、门限、精度和正整数参数。

    Args:
        config_path: SAM3 预测 YAML 文件。

    Returns:
        可直接传给项目图片预测器的参数。
    """
    values = asdict(Sam3PredictOptions())
    configured = load_yaml_mapping(config_path)
    unknown = configured.keys() - values.keys()
    if unknown:
        raise ValueError(f"SAM3 预测配置有未知参数 {', '.join(sorted(unknown))}")
    values.update(configured)
    for key in ("prompt", "device", "precision"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"SAM3 的 {key} 必须是非空字符串")
    if values["precision"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("precision 只支持 float32、float16、bfloat16")
    for key in ("resolution", "max_det", "mask_chunk_size"):
        if type(values[key]) is not int or values[key] <= 0:
            raise ValueError(f"SAM3 的 {key} 必须为正整数")
    for key in ("conf", "display_conf", "mask_threshold"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"SAM3 的 {key} 必须为数值")
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"SAM3 的 {key} 必须在 [0, 1] 内")
        values[key] = float(value)
    if not 0 < values["mask_threshold"] < 1:
        raise ValueError("mask_threshold 必须大于 0 且小于 1")
    if not isinstance(values["save_overlays"], bool):
        raise ValueError("save_overlays 必须为布尔值")
    return Sam3PredictOptions(**values)


def predict_sam3_split(
    weights_path: str | Path,
    image_dir: str | Path,
    config_path: str | Path,
    split: Literal["valid", "train", "test"],
    *,
    save_overlays: bool | None = None,
) -> None:
    """预测 SAM3 验证、训练或测试划分，统一保存实例掩码和指标。

    Args:
        weights_path: 本地 SAM3 .pt 权重。
        image_dir: 待评价的 YOLO 图片目录，需有对应标签。
        config_path: SAM3 预测配置路径。
        split: 本次产物子目录名，valid、train 或 test。
        save_overlays: 为 True 时保存逐图全尺寸叠加图；None 时读取预测
            配置的 `save_overlays`，缺省不保存。关闭时同时跳过
            `missed_samples` 对照图。

    Raises:
        OSError: 缺少输入文件，或无法创建、保存产物。
        ValueError: 划分、配置或图像数据无效。
    """
    from utils.experiment_artifacts import save_environment
    from utils.segmentation_evaluation import MaskEvaluation, save_mask_prediction
    from utils.segmentation_review import save_missed_samples

    if split not in ("valid", "train", "test"):
        raise ValueError("SAM3 划分必须为 valid、train 或 test")
    weights_file = Path(PROJECT_ROOT, weights_path).resolve()
    source_dir = Path(PROJECT_ROOT, image_dir).resolve()
    config_file = Path(PROJECT_ROOT, config_path).resolve()
    if not weights_file.is_file() or weights_file.suffix.lower() != ".pt":
        raise FileNotFoundError(f"缺少 SAM3 .pt 权重 {weights_file}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"缺少待评价图片目录 {source_dir}")
    images = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"数据目录没有支持的图片 {source_dir}")
    load_sam3_options(config_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = PROJECT_ROOT / "out" / "sam3_seg" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    configs_dir = run_dir / "configs"
    configs_dir.mkdir()
    snapshot = Path(copy2(config_file, configs_dir))
    # 预测与归档都使用同一份配置副本。
    options = load_sam3_options(snapshot)
    if save_overlays is None:
        save_overlays = options.save_overlays
    save_environment(run_dir)
    predictor = Sam3RailPredictor(weights_file, options)
    output_dir = run_dir / split
    output_dir.mkdir()
    if save_overlays:
        for side in ("north", "south"):
            (output_dir / side).mkdir()
    confidence = max(options.conf, options.display_conf)
    evaluation = MaskEvaluation(confidence)
    image_paths: dict[str, Path] = {}
    for image_path in images:
        image = cv2.imdecode(
            np.fromfile(image_path, dtype=np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if image is None:
            raise ValueError(f"无法读取图片 {image_path}")
        prediction = predictor.predict(image)
        evaluation.update(image_path, prediction)
        save_mask_prediction(
            output_dir, image_path, image, prediction, confidence,
            save_overlay=save_overlays,
        )
        image_paths[image_path.name] = image_path
        print(f"已评价 {image_path.name}，预测 {len(prediction.scores)} 个实例")
    data_split = "val" if split == "valid" else split
    evaluation.save(output_dir, data_split, "sam3", weights_file, asdict(options))
    if save_overlays:
        save_missed_samples(
            output_dir, image_paths, evaluation.image_counts, confidence,
            prediction_dir=output_dir,
        )
    source_key = "validation_dir" if split == "valid" else f"{split}_dir"
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "timestamp": timestamp, "weights_path": str(weights_file),
            source_key: str(source_dir), "split": data_split,
            "image_count": len(images), "config_source": str(config_file),
            "config_snapshot": snapshot.relative_to(run_dir).as_posix(),
            "actual_options": asdict(options),
            "metrics": f"{split}/metrics/evaluation.csv",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"SAM3 {split} 结果已保存至 {run_dir}")
