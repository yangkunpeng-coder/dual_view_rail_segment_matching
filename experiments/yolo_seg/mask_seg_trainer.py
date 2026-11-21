"""YOLO 钢轨实例分割训练器，支持掩码评分、可选在线增强与辅助监督。"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import get_split_fraction
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.utils import colorstr
from ultralytics.utils.metrics import SegmentMetrics
from ultralytics.utils.torch_utils import unwrap_model

from core_algorithm.gas_rail import (
    GasCriterion,
    GasLossWeights,
    build_gas_criterion,
)
from data_preprocess.common.config import PROJECT_ROOT, config_path
from data_preprocess.gas_targets import (
    GAS_TARGET_KEYS,
    GasTargetSpec,
    GasDataset,
)
from utils.segmentation_predict import PREDICT_CONFIG, load_predict_options
from utils.segmentation_review import ImageMaskCounts

ONLINE_AUGMENTATION_SNAPSHOT = Path(
    "online_augmentation", "rail_online_augmentation.yaml"
)
GAS_CONFIG_SNAPSHOT = "gas_rail.yaml"
GAS_CONFIG_SOURCE = config_path(GAS_CONFIG_SNAPSHOT)


@dataclass(frozen=True)
class GasSettings:
    """辅助监督的生效设置。

    Attributes:
        weights: 四项辅助损失与 Dice 的权重。
        target_spec: 目标生成参数。
        smooth_delta: 平滑项的采样步长，单位原型格。
        hidden_channels: 辅助头中间通道数；None 表示按通道数取默认值。
        centerline_prior: 中心线偏置先验。
        width_prior: 宽度偏置先验，单位原型格。
    """

    weights: GasLossWeights
    target_spec: GasTargetSpec
    smooth_delta: float
    hidden_channels: int | None
    centerline_prior: float
    width_prior: float


def _require_number(
    mapping: dict, key: str, default: float, config_path: Path
) -> float:
    """读取浮点配置项并校验有限、非负。"""
    value = mapping.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{config_path} 的 {key} 必须为数值")
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{config_path} 的 {key} 须为有限非负数")
    return number


def _require_positive_int(
    mapping: dict, key: str, default: int, config_path: Path
) -> int:
    """读取正整数配置项。"""
    value = mapping.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{config_path} 的 {key} 必须为正整数")
    return value


def load_gas_settings(config_path: Path) -> GasSettings:
    """读取辅助监督配置，返回生效设置。

    Args:
        config_path: ``gas_rail.yaml`` 路径。

    Returns:
        权重、目标参数与头部先验。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: 字段类型或取值无效。
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少辅助监督配置 {config_path}")
    with config_path.open(encoding="utf-8-sig") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"辅助监督配置必须为映射 {config_path}")
    target = raw.get("target") or {}
    head = raw.get("head") or {}
    if not isinstance(target, dict) or not isinstance(head, dict):
        raise ValueError(f"{config_path} 的 target/head 必须为映射")
    weights = GasLossWeights(
        center=_require_number(raw, "center_weight", 0.0, config_path),
        direction=_require_number(raw, "direction_weight", 0.0, config_path),
        smooth=_require_number(raw, "smooth_weight", 0.0, config_path),
        width=_require_number(raw, "width_weight", 0.0, config_path),
        dice=_require_number(raw, "dice_weight", 0.0, config_path),
        warmup_epochs=_require_number(raw, "warmup_epochs", 3.0, config_path),
    )
    spec = GasTargetSpec(
        mask_ratio=_require_positive_int(target, "mask_ratio", 4, config_path),
        sigma_cells=_require_number(target, "sigma_cells", 0.75, config_path),
        direction_band_cells=_require_number(
            target, "direction_band_cells", 2.0, config_path
        ),
        tangent_window=_require_positive_int(
            target, "tangent_window", 5, config_path
        ),
        tangent_max_residual_px=_require_number(
            target, "tangent_max_residual_px", 1.5, config_path
        ),
    )
    hidden = head.get("hidden_channels")
    if hidden is not None:
        hidden = _require_positive_int(head, "hidden_channels", 32, config_path)
    return GasSettings(
        weights=weights,
        target_spec=spec,
        smooth_delta=_require_number(raw, "smooth_delta", 1.0, config_path),
        hidden_channels=hidden,
        centerline_prior=_require_number(
            head, "centerline_prior", 0.02, config_path
        ),
        width_prior=_require_number(head, "width_prior", 1.9, config_path),
    )


class MaskCheckpointMetrics(SegmentMetrics):
    """使用 Mask mAP50–95 评分，并记录逐图漏检统计。"""

    def __init__(self, preview_confidence: float) -> None:
        """初始化从验证预测中筛选漏检计数的置信度门限。

        Args:
            preview_confidence: 逐图诊断使用的预测置信度门限。
        """
        super().__init__()
        self.preview_confidence = preview_confidence
        self.image_mask_counts: dict[str, ImageMaskCounts] = {}

    @property
    def fitness(self) -> float:
        """返回供最优权重选择和早停使用的验证集掩码评分。"""
        return float(self.seg.map)

    def update_stats(self, stat: dict[str, NDArray[np.generic] | str]) -> None:
        """更新常规指标，并统计置信度门限内的 Mask 匹配数。

        Args:
            stat: Ultralytics 当前图片的预测与标注统计。
        """
        image_name = stat["im_name"]
        if not isinstance(image_name, str):
            raise TypeError("验证统计的 im_name 必须为字符串")
        super().update_stats(stat)
        is_selected = np.asarray(stat["conf"]) >= self.preview_confidence
        # 验证器按置信度分配实例，tp_m 的首列对应 Mask IoU 0.50。
        matched_count = int(np.asarray(stat["tp_m"])[is_selected, 0].sum())
        target_count = len(np.asarray(stat["target_cls"]))
        self.image_mask_counts[image_name] = ImageMaskCounts(
            target_count, matched_count
        )

    def clear_image_metrics(self) -> None:
        """在开始新一轮验证时清除上一轮逐图统计。"""
        super().clear_image_metrics()
        self.image_mask_counts.clear()


class MaskSegmentationTrainer(SegmentationTrainer):
    """使用掩码评分的实例分割训练器。"""

    def get_validator(self) -> SegmentationValidator:
        """构建用于本次训练的实例分割验证器。

        Returns:
            按掩码指标计算评分的验证器。
        """
        validator = super().get_validator()
        predict_config = Path(self.save_dir) / "configs" / PREDICT_CONFIG.name
        if not predict_config.is_file():
            predict_config = PREDICT_CONFIG
        preview_confidence = cast(float, load_predict_options(predict_config)["conf"])
        # 验证器已按自身门限过滤候选，逐图计数只能使用保留的预测。
        confidence = max(preview_confidence, float(validator.args.conf))
        validator.metrics = MaskCheckpointMetrics(confidence)
        return validator


class OnlineAugmentationTrainer(MaskSegmentationTrainer):
    """仅在训练数据集中接入裁剪 ROI 光度增强。"""

    def build_dataset(
        self, img_path: str | list[str], mode: str = "train", batch: int | None = None
    ) -> YOLODataset:
        """为训练集构建项目数据集，其他数据集沿用原训练器。

        Args:
            img_path: YOLO 数据集的图片目录或路径列表。
            mode: 当前数据集模式。
            batch: 批大小，用于矩形批处理。

        Returns:
            当前模式对应的 YOLO 实例分割数据集。
        """
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)

        from data_preprocess.rail_online_augmentation import (
            RailAugmentedDataset,
            RailOnlineAugmentation,
        )

        config_path = Path(self.save_dir) / "configs" / ONLINE_AUGMENTATION_SNAPSHOT
        augmentation = RailOnlineAugmentation(config_path)
        args = self.args
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        # 与 build_yolo_dataset 的实例分割训练分支使用相同的数据参数。
        fraction = (
            1.0
            if self.data.get("complete")
            else get_split_fraction(args.fraction, mode)
        )
        return RailAugmentedDataset(
            augmentation,
            img_path=img_path,
            imgsz=args.imgsz,
            batch_size=batch if batch is not None else max(1, int(self.batch_size)),
            augment=True,
            hyp=args,
            rect=args.rect,
            cache=args.cache or None,
            single_cls=args.single_cls or False,
            stride=stride,
            pad=0.0,
            prefix=colorstr("train: "),
            task=args.task,
            classes=args.classes,
            data=self.data,
            fraction=fraction,
            tile=args.tile,
        )


class GasTrainer(MaskSegmentationTrainer):
    """在原生训练流程上接入几何辅助监督。

    辅助头在 ``setup_model`` 挂载：该位置早于 optimizer 与 EMA 构建，新参数才会进入两者。
    criterion 也在那里安装一次（让 EMA 带上），再在 ``_setup_train`` 末尾补装一次，因为
    resume 时上游会用 ``init_criterion()`` 覆盖它。
    """

    def setup_model(self) -> None:
        """挂辅助头并安装包装后的 criterion。

        上游把 ``nc``/``names``/``args`` 挂在模型上是在 ``_setup_train`` 的
        ``set_model_attributes``（trainer.py:360），晚于 ``setup_model``（347）。而
        ``E2ELoss`` 构造时要读模型上的 ``args``、``nc`` 与设备，所以这里先补属性并把模型
        移到训练设备再建 criterion；否则 loss 会把目标搬到 CPU，与 GPU 上的 anchor 混用。
        """
        super().setup_model()
        self._gas_settings = self._load_gas_settings()
        self.set_model_attributes()
        model = unwrap_model(self.model).to(self.device)
        model.criterion = self._build_gas_criterion(model, self._gas_settings)

    def _setup_train(self) -> None:
        """resume 会重置 criterion，这里按快照补装并同步调度状态。"""
        super()._setup_train()
        model = unwrap_model(self.model)
        if isinstance(model.criterion, GasCriterion):
            return
        criterion = self._build_gas_criterion(model, self._gas_settings)
        criterion.base.updates = self.start_epoch - 1
        criterion.epoch_index = self.start_epoch - 1
        criterion.update()
        model.criterion = criterion

    def build_dataset(
        self, img_path: str | list[str], mode: str = "train", batch: int | None = None
    ) -> YOLODataset:
        """训练集换成带辅助目标的数据集，其他划分沿用原训练器。"""
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        args = self.args
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        # 与 build_yolo_dataset 的实例分割训练分支使用相同的数据参数。
        fraction = (
            1.0
            if self.data.get("complete")
            else get_split_fraction(args.fraction, mode)
        )
        return GasDataset(
            img_path=img_path,
            imgsz=args.imgsz,
            batch_size=batch if batch is not None else max(1, int(self.batch_size)),
            augment=True,
            hyp=args,
            rect=args.rect,
            cache=args.cache or None,
            single_cls=args.single_cls or False,
            stride=stride,
            pad=0.0,
            prefix=colorstr("train: "),
            task=args.task,
            classes=args.classes,
            data=self.data,
            fraction=fraction,
            tile=args.tile,
            target_spec=self._gas_settings.target_spec,
        )

    def _load_gas_settings(self) -> GasSettings:
        """优先读实验目录里的配置快照，缺失时回退项目配置。"""
        snapshot = Path(self.save_dir) / "configs" / GAS_CONFIG_SNAPSHOT
        source = snapshot if snapshot.is_file() else GAS_CONFIG_SOURCE
        return load_gas_settings(source)

    @staticmethod
    def _build_gas_criterion(model, settings: GasSettings) -> GasCriterion:
        """按生效设置挂头并构建 criterion。"""
        return build_gas_criterion(
            model,
            settings.weights,
            GAS_TARGET_KEYS,
            smooth_delta=settings.smooth_delta,
            hidden_channels=settings.hidden_channels,
            centerline_prior=settings.centerline_prior,
            width_prior=settings.width_prior,
        )
