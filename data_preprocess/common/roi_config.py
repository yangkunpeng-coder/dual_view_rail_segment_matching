"""读取钢轨非矩形有效区域及诊断产物配置。"""

from dataclasses import dataclass
from pathlib import Path

from data_preprocess.common.config import PROJECT_ROOT, config_path, load_yaml_mapping

_ROI_CONFIG = config_path("rail_roi.yaml")
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class CorridorSettings:
    """有效区域算法参数，边距和空行阈值使用原图像素。"""

    target_label: str
    margins: tuple[int, int]
    margin_overrides: dict[str, tuple[int, int]]
    smoothing_kernel: int
    max_interp_gap: int
    outside_fill_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class RoiSettings:
    """数据输入目录、诊断产物选项和有效区域参数。"""

    image_dir: Path
    json_dir: Path
    image_extensions: tuple[str, ...]
    split_config: Path | None
    output_dir: Path
    save_valid_mask: bool
    save_preview: bool
    preserve_original_json: bool
    corridor: CorridorSettings


def load_roi_settings(config_path: Path = _ROI_CONFIG) -> RoiSettings:
    """读取 ROI 配置并检查必需的 Mask 完整性约束。

    Args:
        config_path: ROI YAML 文件路径。

    Returns:
        已校验的数据导出与有效区域配置。

    Raises:
        OSError: 无法读取配置。
        ValueError: 配置值无效或请求了未启用的裁剪方式。
    """
    config = load_yaml_mapping(config_path)
    input_settings = _mapping(config.get("input"), "input")
    annotation = _mapping(config.get("annotation"), "annotation")
    corridor = _mapping(config.get("corridor"), "corridor")
    output = _mapping(config.get("output"), "output")
    quality = _mapping(config.get("quality"), "quality")

    target_label = annotation.get("target_label")
    if not isinstance(target_label, str) or not target_label.strip():
        raise ValueError("annotation.target_label 必须是非空字符串")
    if annotation.get("polygon_only") is not True:
        raise ValueError("annotation.polygon_only 必须为 true")
    if corridor.get("tight_bbox_after_mask") is not False:
        raise ValueError("当前版本仅支持保留原画布，请关闭 tight_bbox_after_mask")
    if quality.get("require_all_mask_pixels_inside") is not True:
        raise ValueError("必须开启 Mask 完整性检查")

    extensions = input_settings.get("image_extensions")
    if not isinstance(extensions, list) or not extensions or not all(
        isinstance(extension, str) and extension in _IMAGE_EXTENSIONS
        for extension in extensions
    ):
        raise ValueError("input.image_extensions 须从 .png、.jpg、.jpeg 中选择")

    split_config = input_settings.get("split_config")
    reference_config = (
        None if split_config is None else _relative_path(split_config, "split_config")
    )
    return RoiSettings(
        image_dir=_relative_path(input_settings.get("image_dir"), "input.image_dir"),
        json_dir=_relative_path(input_settings.get("json_dir"), "input.json_dir"),
        image_extensions=tuple(extensions),
        split_config=reference_config,
        output_dir=_relative_path(output.get("output_dir"), "output.output_dir"),
        save_valid_mask=_boolean(output.get("save_valid_mask"), "save_valid_mask"),
        save_preview=_boolean(output.get("save_preview"), "save_preview"),
        preserve_original_json=_boolean(
            output.get("preserve_original_json"), "preserve_original_json"
        ),
        corridor=_load_corridor(corridor, target_label),
    )


def _load_corridor(values: dict[str, object], target_label: str) -> CorridorSettings:
    """校验边界、平滑、填色和逐图覆盖参数。"""
    margin = _pixel_count(values.get("margin_px"), "margin_px")
    margins = _read_margins(values, (margin, margin))
    overrides = _mapping(values.get("margin_overrides", {}), "margin_overrides")
    margin_overrides = {
        name: _read_margins(_mapping(override, name), margins)
        for name, override in overrides.items()
    }

    smoothing = values.get("smoothing")
    kernel = _pixel_count(values.get("smoothing_kernel"), "smoothing_kernel")
    if smoothing not in ("median", "none") or kernel % 2 != 1:
        raise ValueError("smoothing 须为 median 或 none，smoothing_kernel 须为正奇数")

    fill = values.get("outside_fill_bgr")
    if not isinstance(fill, list) or len(fill) != 3 or not all(
        type(channel) is int and 0 <= channel <= 255 for channel in fill
    ):
        raise ValueError("outside_fill_bgr 须包含三个 0 到 255 的整数")
    blue, green, red = fill

    return CorridorSettings(
        target_label=target_label,
        margins=margins,
        margin_overrides=margin_overrides,
        smoothing_kernel=kernel if smoothing == "median" else 1,
        max_interp_gap=_pixel_count(values.get("max_interp_gap"), "max_interp_gap"),
        outside_fill_rgb=(red, green, blue),
    )


def _read_margins(
    values: dict[str, object], defaults: tuple[int, int]
) -> tuple[int, int]:
    """读取可独立覆盖的左右边距。"""
    left = values.get("margin_left_px")
    right = values.get("margin_right_px")
    return (
        defaults[0] if left is None else _pixel_count(left, "margin_left_px"),
        defaults[1] if right is None else _pixel_count(right, "margin_right_px"),
    )


def _mapping(value: object, key: str) -> dict[str, object]:
    """校验配置段为字符串键的映射。"""
    if not isinstance(value, dict) or not all(isinstance(name, str) for name in value):
        raise ValueError(f"{key} 必须为映射")
    return value


def _pixel_count(value: object, key: str) -> int:
    """校验像素配置为非负整数。"""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} 必须为非负整数")
    return value


def _relative_path(value: object, key: str) -> Path:
    """校验配置路径以规定的基准目录为起点。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须为非空相对路径")
    path = Path(value)
    if path.anchor or ".." in path.parts:
        raise ValueError(f"{key} 必须为目录内的相对路径")
    return path


def _boolean(value: object, key: str) -> bool:
    """校验输出选项为 YAML 布尔值。"""
    if not isinstance(value, bool):
        raise ValueError(f"{key} 必须为布尔值")
    return value
