"""为训练图像提供带裁剪 ROI 保护的在线光度增强。"""

from collections import OrderedDict
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics.data.augment import Compose
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.patches import imread

from data_preprocess.common.config import PROJECT_ROOT, load_yaml_mapping

_PIXEL_MAX = 255
_PIXEL_VALUES = np.arange(_PIXEL_MAX + 1, dtype=np.float32)
_HUE_PERIOD = 180


class _TileMetadata(TypedDict, total=False):
    """Ultralytics 在启用固定图像分块时额外传入的坐标。"""

    tile: tuple[int, int, int, int]
    source_shape: tuple[int, int]


class _TrainingSample(_TileMetadata):
    """增强所需的训练样本字段，其他标注字段由 YOLODataset 管理。"""

    img: NDArray[np.uint8]
    im_file: str
    ori_shape: tuple[int, int]


@dataclass(frozen=True)
class _Operation:
    """单项增强的启用状态、概率和取值区间。"""

    is_enabled: bool
    probability: float
    ranges: dict[str, tuple[float, float]]

    def should_apply(self) -> bool:
        """按本项独立概率抽样。"""
        return self.is_enabled and np.random.random() < self.probability

    def sample(self, key: str) -> float:
        """均匀抽取指定参数。"""
        return float(np.random.uniform(*self.ranges[key]))

    def sample_integer(self, key: str) -> int:
        """均匀抽取整数，含区间两端。"""
        lower, upper = self.ranges[key]
        return int(np.random.randint(int(lower), int(upper) + 1))


class RailOnlineAugmentation:
    """在 YOLO 的 LetterBox 前增强 BGR 图像并保留 ROI 外原像素。"""

    def __init__(self, config_path: Path) -> None:
        """读取增强配置和裁剪导出的逐图 ROI 路径。

        Args:
            config_path: 本次训练使用的独立增强配置快照。
        """
        try:
            config = load_yaml_mapping(config_path)
            self._load_operations(config)
            roi = _read_section(config, "roi")
            manifest_path = roi.get("manifest")
            cache_size = roi.get("cache_size")
            if not isinstance(manifest_path, str) or not manifest_path.strip():
                raise ValueError("roi.manifest 必须为非空路径字符串")
            if type(cache_size) is not int or cache_size < 0:
                raise ValueError("roi.cache_size 必须为非负整数")
        except ValueError as error:
            raise ValueError(f"在线增强配置 {config_path} 无效，{error}") from error

        self._manifest_path = (PROJECT_ROOT / manifest_path).resolve()
        self._roi_paths = _read_roi_paths(self._manifest_path)
        self._cache_size = cache_size
        self._roi_cache: OrderedDict[
            tuple[str, tuple[int, int, int, int] | None, tuple[int, int]],
            NDArray[np.bool_],
        ] = OrderedDict()

    def _load_operations(self, config: dict[str, object]) -> None:
        """验证光度增强参数和互斥退化的概率分配。"""
        self._brightness = _read_operation(
            config, "brightness_contrast",
            alpha=(0, math.inf), beta=(-_PIXEL_MAX, _PIXEL_MAX),
        )
        self._gamma = _read_operation(config, "gamma", exponent=(1e-6, math.inf))
        self._white_balance = _read_operation(
            config, "white_balance", channel_gain=(0, math.inf)
        )
        self._saturation = _read_operation(config, "saturation", gain=(0, math.inf))
        self._hue = _read_operation(
            config, "hue", delta=(1 - _HUE_PERIOD, _HUE_PERIOD - 1)
        )
        degradation = _read_section(config, "degradation")
        self._none_probability = _read_number(
            degradation.get("none_probability"), "degradation.none_probability", 0, 1
        )
        self._noise = _read_operation(
            degradation, "gaussian_noise", sigma=(0, math.inf)
        )
        self._blur = _read_operation(
            degradation, "gaussian_blur", sigma=(0, math.inf)
        )
        kernel_size = _read_section(degradation, "gaussian_blur").get("kernel_size")
        if type(kernel_size) is not int or kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("gaussian_blur.kernel_size 必须为不小于 3 的奇数")
        self._blur_kernel = (kernel_size, kernel_size)
        self._jpeg = _read_operation(degradation, "jpeg_compression", quality=(0, 100))
        for operation, key in ((self._hue, "delta"), (self._jpeg, "quality")):
            if not all(value.is_integer() for value in operation.ranges[key]):
                raise ValueError(f"{key} 的范围端点必须为整数")
        total_probability = (
            self._none_probability + self._noise.probability
            + self._blur.probability + self._jpeg.probability
        )
        if not math.isclose(total_probability, 1.0):
            raise ValueError("无退化、噪声、模糊与 JPEG 的配置概率之和必须为 1")

    def check_training_images(self, image_paths: list[str]) -> None:
        """在训练开始前确认每张训练图都能找到对应的 ROI 文件。

        Args:
            image_paths: YOLODataset 实际加载的图像路径。

        Raises:
            FileNotFoundError: 训练图像缺少清单记录或 ROI 掩码文件。
        """
        for image_path in set(image_paths):
            mask_path = self._get_roi_path(image_path)
            if not mask_path.is_file():
                raise FileNotFoundError(f"训练图像 {image_path} 缺少 ROI 掩码 {mask_path}")

    def _get_roi_path(self, image_path: str) -> Path:
        """按训练图像的完整路径查找裁剪时导出的掩码。"""
        mask_path = self._roi_paths.get((PROJECT_ROOT / image_path).resolve())
        if mask_path is None:
            raise FileNotFoundError(
                f"裁剪清单 {self._manifest_path} 未登记训练图像 {image_path} 的 ROI"
            )
        return mask_path

    def _get_roi(self, sample: _TrainingSample) -> NDArray[np.bool_]:
        """用最近邻将有效区域对齐至当前图像大小，并限制缓存数量。"""
        height, width = sample["img"].shape[:2]
        tile = sample.get("tile")
        cache_key = (sample["im_file"], tile, (height, width))
        cached = self._roi_cache.get(cache_key)
        if cached is not None:
            self._roi_cache.move_to_end(cache_key)
            return cached

        mask_path = self._get_roi_path(sample["im_file"])
        decoded = imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError(f"无法读取 ROI 掩码 {mask_path}")
        mask = decoded[:, :, 0]
        source_shape = sample.get("source_shape", sample["ori_shape"])
        if mask.shape != source_shape:
            raise ValueError(
                f"ROI 掩码 {mask_path} 尺寸 {mask.shape} 与图像 {source_shape} 不符"
            )
        if tile is not None:
            x_min, y_min, x_max, y_max = tile
            mask = mask[y_min:y_max, x_min:x_max]
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        roi = mask != 0
        if self._cache_size:
            self._roi_cache[cache_key] = roi
            if len(self._roi_cache) > self._cache_size:
                self._roi_cache.popitem(last=False)
        return roi

    def __call__(self, sample: _TrainingSample) -> _TrainingSample:
        """仅替换样本中的 H×W×3、uint8、BGR 图像。

        Args:
            sample: 已经读入和缩放的训练样本。

        Returns:
            像素增强后的样本，标签和图像空间信息保留。
        """
        image = sample["img"]
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"在线增强需要 uint8 BGR 图像 {sample['im_file']}")
        roi = self._get_roi(sample)
        enhanced = self._apply_photometric(image)
        enhanced = self._apply_degradation(enhanced)
        if enhanced is not image:
            np.copyto(enhanced, image, where=~roi[..., None])
            sample["img"] = enhanced
        return sample

    def _apply_photometric(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """按配置依次执行亮度对比度、Gamma、白平衡和 HSV 微调。"""
        if self._brightness.should_apply():
            values = (
                _PIXEL_VALUES * self._brightness.sample("alpha")
                + self._brightness.sample("beta")
            )
            image = cv2.LUT(image, _to_uint8(values))
        if self._gamma.should_apply():
            exponent = self._gamma.sample("exponent")
            values = (_PIXEL_VALUES / _PIXEL_MAX) ** exponent * _PIXEL_MAX
            image = cv2.LUT(image, _to_uint8(values))
        if self._white_balance.should_apply():
            gains = np.random.uniform(
                *self._white_balance.ranges["channel_gain"], size=3
            ).astype(np.float32)
            image = _to_uint8(image * gains)

        adjusts_saturation = self._saturation.should_apply()
        adjusts_hue = self._hue.should_apply()
        if not (adjusts_saturation or adjusts_hue):
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if adjusts_saturation:
            hsv[:, :, 1] = _to_uint8(hsv[:, :, 1] * self._saturation.sample("gain"))
        if adjusts_hue:
            hue = hsv[:, :, 0].astype(np.int16) + self._hue.sample_integer("delta")
            hsv[:, :, 0] = hue % _HUE_PERIOD
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_degradation(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """保留各项原有概率区间，抽中关闭的项目时直接跳过退化。"""
        draw = np.random.random()
        noise_limit = self._none_probability + self._noise.probability
        blur_limit = noise_limit + self._blur.probability
        if draw < self._none_probability:
            return image
        if draw < noise_limit:
            if not self._noise.is_enabled:
                return image
            sigma = self._noise.sample("sigma")
            noise = np.random.normal(0, sigma, size=image.shape).astype(np.float32)
            return _to_uint8(image + noise)
        if draw < blur_limit:
            if not self._blur.is_enabled:
                return image
            sigma = self._blur.sample("sigma")
            return cv2.GaussianBlur(image, self._blur_kernel, sigmaX=sigma)
        if not self._jpeg.is_enabled or draw >= blur_limit + self._jpeg.probability:
            return image

        quality = self._jpeg.sample_integer("quality")
        success, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not success:
            raise ValueError("在线增强的 JPEG 编码失败")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("在线增强的 JPEG 解码失败")
        return decoded


class RailAugmentedDataset(YOLODataset):
    """保留 YOLO 训练集的样本和标签处理，替换默认随机增强。"""

    def __init__(
        self,
        augmentation: RailOnlineAugmentation,
        *,
        data: dict[str, object],
        task: str,
        **dataset_options: object,
    ) -> None:
        """在构建数据集变换前设置 ROI 光度增强。

        Args:
            augmentation: 本次训练的图像变换。
            data: Ultralytics 已解析的数据集配置。
            task: Ultralytics 任务类型。
            **dataset_options: YOLODataset 的加载参数。
        """
        self._online_augmentation = augmentation
        super().__init__(data=data, task=task, **dataset_options)
        augmentation.check_training_images(self.im_files)

    def build_transforms(self, hyp: IterableSimpleNamespace) -> Compose:
        """复用 YOLO 的确定性 LetterBox 和 Format，并在之前增强图像。"""
        # 只在组装变换时跳过默认增强，保留训练集加载与标签检查行为。
        was_augmenting = self.augment
        self.augment = False
        try:
            transforms = super().build_transforms(hyp)
        finally:
            self.augment = was_augmenting
        transforms.insert(0, self._online_augmentation)
        return transforms


def _read_section(config: dict[str, object], key: str) -> dict[str, object]:
    """读取具有字符串键的配置小节。"""
    section = config.get(key)
    if not isinstance(section, dict) or not all(
        isinstance(name, str) for name in section
    ):
        raise ValueError(f"{key} 必须为配置映射")
    return section


def _read_number(value: object, key: str, lower: float, upper: float) -> float:
    """读取指定区间内的有限数值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} 必须为数值")
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{key} 必须在 [{lower}, {upper}] 内")
    return float(value)


def _read_operation(
    config: dict[str, object], key: str, **limits: tuple[float, float]
) -> _Operation:
    """读取单项增强配置并校验参数区间。"""
    section = _read_section(config, key)
    is_enabled = section.get("enabled")
    if not isinstance(is_enabled, bool):
        raise ValueError(f"{key}.enabled 必须为布尔值")
    probability = _read_number(section.get("probability"), f"{key}.probability", 0, 1)
    ranges: dict[str, tuple[float, float]] = {}
    for parameter, (lower, upper) in limits.items():
        endpoints = section.get(parameter)
        label = f"{key}.{parameter}"
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            raise ValueError(f"{label} 必须为包含最小值和最大值的列表")
        minimum = _read_number(endpoints[0], label, lower, upper)
        maximum = _read_number(endpoints[1], label, minimum, upper)
        ranges[parameter] = (minimum, maximum)
    return _Operation(is_enabled, probability, ranges)


def _read_roi_paths(manifest_path: Path) -> dict[Path, Path]:
    """读取导出清单中的训练图像与 ROI 掩码路径映射。"""
    roi_paths: dict[Path, Path] = {}
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        required_fields = {"split", "yolo_image", "valid_mask"}
        if not required_fields.issubset(rows.fieldnames or ()):
            raise ValueError(f"裁剪清单 {manifest_path} 缺少 ROI 映射字段")
        for row in rows:
            if row["split"] != "train":
                continue
            if not row["yolo_image"] or not row["valid_mask"]:
                raise ValueError(f"裁剪清单 {manifest_path} 第 {rows.line_num} 行缺少路径")
            image_path = (PROJECT_ROOT / row["yolo_image"]).resolve()
            if image_path in roi_paths:
                raise ValueError(f"裁剪清单 {manifest_path} 重复登记 {image_path}")
            roi_paths[image_path] = (PROJECT_ROOT / row["valid_mask"]).resolve()
    return roi_paths


def _to_uint8(values: NDArray[np.floating]) -> NDArray[np.uint8]:
    """将计算后的像素四舍五入并限制为 8-bit 范围。"""
    return np.rint(np.clip(values, 0, _PIXEL_MAX)).astype(np.uint8)
