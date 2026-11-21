"""使用本地 YOLO26 分割权重训练钢轨实例分割模型。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import os
from pathlib import Path
from shutil import copy2
import sys
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import yaml

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preprocess.common.config import PROJECT_ROOT, config_path
from utils.bbox_crop import (
    BboxCropSettings,
    parse_bbox_crop,
    project_relative,
    verify_bbox_dataset,
)
from utils.experiment_artifacts import (
    append_mask_metrics,
    archive_training_outputs,
    save_environment,
    save_training_change,
    snapshot_configs,
)
from utils.segmentation_postprocess import POSTPROCESS_CONFIG
from utils.segmentation_predict import (
    PREDICT_CONFIG,
    load_predict_options,
    load_split_image_dir,
    save_segmentation_predictions,
)
from utils.segmentation_review import save_missed_samples

if TYPE_CHECKING:
    from ultralytics import YOLO

    from core_algorithm.p2_assignment_warmup import P2AssignmentWarmupSettings

_TRAIN_CONFIG = config_path("train_rail_seg.yaml")
_RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_BEST_WEIGHT_NAME = "rail_yolo26m-seg.pt"
_DISABLED_YOLO_AUGMENTATIONS = {
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cutmix": 0.0,
    "close_mosaic": 0,
    "multi_scale": 0.0,
    "erasing": 0.0,
    "auto_augment": None,
    "augmentations": None,
}


@dataclass(frozen=True)
class P2Settings:
    """P2 四尺度开关的生效设置，用于写入实验记录。"""

    model: str
    pretrained: str
    mask_ratio: int
    assignment_warmup: P2AssignmentWarmupSettings | None = None


@dataclass(frozen=True)
class GasLaunch:
    """辅助监督开关的生效设置，用于选择训练器与写实验记录。"""

    config: str


def train_rail_seg(config_source: str | Path = _TRAIN_CONFIG) -> Path:
    """按模型配置训练钢轨分割模型，完成后自动验证和测试。

    Args:
        config_source: 训练配置路径，相对路径以项目根目录为基准。

    Raises:
        FileExistsError: 同名时间戳实验目录已存在。
        FileNotFoundError: 缺少配置文件或权重。
        ValueError: 配置格式无效或模型的任务类型不是实例分割。

    Returns:
        本次实验的归档目录。

    """
    config_file = Path(PROJECT_ROOT, config_source).resolve()
    options = _read_train_options(config_file)
    # P2 覆盖必须早于 model 路径解析，否则训练仍使用基线结构。
    p2_settings = _pop_p2(options, config_file)
    p2_warmup = p2_settings.assignment_warmup if p2_settings else None
    bbox_crop = _pop_bbox_crop(options, config_file)
    model_path = _resolve_config_path(options, "model", config_file)
    data_config = _resolve_config_path(options, "data", config_file)
    if bbox_crop is not None:
        # 只切换数据集；预测与评估仍走原流程，指标按裁剪坐标计算。
        data_config = verify_bbox_dataset(bbox_crop, data_config)
        options["data"] = str(data_config)
        print(f"bbox_crop 已启用：数据集切换为 {data_config}")
    output_dir = _resolve_config_path(options, "project", config_file)
    if output_dir == PROJECT_ROOT / "out":
        output_dir = output_dir / "yolo_seg"
    use_online_augmentation, augmentation_file = _pop_online_augmentation(
        options, config_file
    )
    gas_launch = _pop_gas(options, config_file)
    if p2_warmup is not None and (gas_launch is not None or use_online_augmentation):
        raise ValueError("P2 分配热身实验要求关闭 GAS 辅助监督和项目在线增强")
    if gas_launch is not None and use_online_augmentation:
        raise ValueError("辅助监督与在线增强不能同时开启，两条数据管线不叠加")
    if gas_launch is not None and float(options.get("multi_scale", 0.0)) > 0:
        raise ValueError("开启辅助监督时 multi_scale 必须为 0，否则目标网格与输入不匹配")
    gas_source = (
        config_path(gas_launch.config) if gas_launch else None
    )
    is_tensorboard = _pop_tensorboard(options, config_file)
    uses_model_yaml = model_path.suffix.lower() in (".yaml", ".yml")
    required_files = [model_path, data_config, PREDICT_CONFIG]
    if gas_source is not None:
        required_files.append(gas_source)
    if uses_model_yaml:
        pretrained_path = _resolve_config_path(options, "pretrained", config_file)
        options["pretrained"] = str(pretrained_path)
        required_files.append(pretrained_path)
    for required_file in required_files:
        if not required_file.is_file():
            raise FileNotFoundError(f"缺少训练输入文件 {required_file}")

    # 兼容旧配置，新流程在每次训练后自动运行独立 valid 和 test。
    options.pop("evaluate_test", None)
    options.pop("model")
    options.pop("change", None)

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO
    from ultralytics.utils import YAML

    from experiments.yolo_seg.mask_seg_trainer import (
        GAS_CONFIG_SNAPSHOT,
        ONLINE_AUGMENTATION_SNAPSHOT,
        GasTrainer,
        MaskSegmentationTrainer,
        OnlineAugmentationTrainer,
    )

    trainer_class: type[MaskSegmentationTrainer] = MaskSegmentationTrainer
    if p2_warmup is not None:
        from experiments.yolo_seg.p2_assignment_trainer import (
            P2_ASSIGNMENT_WARMUP_SNAPSHOT,
            P2AssignmentWarmupTrainer,
        )

        trainer_class = P2AssignmentWarmupTrainer
    elif use_online_augmentation:
        trainer_class = OnlineAugmentationTrainer
        options.update(_DISABLED_YOLO_AUGMENTATIONS)
    elif gas_launch is not None:
        trainer_class = GasTrainer
    if augmentation_file is not None:
        archived_file = config_file.parent / ONLINE_AUGMENTATION_SNAPSHOT
        if config_file.parent.name == "configs" and archived_file.is_file():
            augmentation_file = archived_file
        if not augmentation_file.is_file():
            raise FileNotFoundError(f"缺少在线增强配置文件 {augmentation_file}")
    if gas_source is not None:
        archived_gas = config_file.parent / GAS_CONFIG_SNAPSHOT
        if config_file.parent.name == "configs" and archived_gas.is_file():
            gas_source = archived_gas

    original_dir = Path.cwd()
    try:
        # 数据集 YAML 的 path 相对于项目根目录。
        os.chdir(PROJECT_ROOT)
        run_dir = _create_run_dir(output_dir)
        config_dir = snapshot_configs(run_dir, config_file, data_config)
        if p2_warmup is not None:
            YAML.save(
                config_dir / P2_ASSIGNMENT_WARMUP_SNAPSHOT,
                {"enabled": True, **asdict(p2_warmup)},
            )
        augmentation_snapshot = None
        if augmentation_file is not None:
            augmentation_snapshot = config_dir / ONLINE_AUGMENTATION_SNAPSHOT
            augmentation_snapshot.parent.mkdir()
            copy2(augmentation_file, augmentation_snapshot)
        gas_snapshot = None
        if gas_source is not None:
            gas_snapshot = config_dir / GAS_CONFIG_SNAPSHOT
            copy2(gas_source, gas_snapshot)
        if uses_model_yaml:
            # 从归档副本构建，确保两次读入模型结构都使用同一份 YAML。
            model_config_dir = config_dir / "model_source"
            model_config_dir.mkdir()
            model_path = Path(copy2(model_path, model_config_dir))
        model = YOLO(str(model_path))
        if model.task != "segment":
            raise ValueError(f"训练需要实例分割模型 {model_path}")

        predict_config = config_dir / PREDICT_CONFIG.name
        copy2(PREDICT_CONFIG, predict_config)
        # 后处理参数同样影响评价指标，必须随实验归档才能复现。
        copy2(POSTPROCESS_CONFIG, config_dir / POSTPROCESS_CONFIG.name)
        predict_options = load_predict_options(predict_config)
        save_training_change(
            run_dir, config_dir / config_file.name, config_dir / data_config.name,
            predict_config,
        )
        _record_online_augmentation(
            run_dir, use_online_augmentation, augmentation_file, augmentation_snapshot
        )
        _record_p2(run_dir, p2_settings)
        _record_bbox_crop(run_dir, bbox_crop, data_config)
        _record_gas(run_dir, gas_launch, gas_snapshot)
        _record_tensorboard(run_dir, is_tensorboard)
        save_environment(run_dir)
        model.add_callback(
            "on_train_start",
            lambda trainer: YAML.save(config_dir / "model.yaml", trainer.model.yaml),
        )
        options.update(
            data=str(data_config),
            project=str(output_dir),
            name=run_dir.name,
            save_dir=str(run_dir),
        )
        if is_tensorboard:
            # 回调在 trainer 构造时读取 SETTINGS，所以要在 model.train 之前打开。
            from ultralytics.utils import SETTINGS as ULTRALYTICS_SETTINGS

            ULTRALYTICS_SETTINGS.update({"tensorboard": True})
            _disable_tensorboard_graph()
        model.train(trainer=trainer_class, **options)
        best_weight = _archive_results(model, run_dir, predict_options)
        _run_post_training_evaluations(
            best_weight, config_dir / data_config.name, predict_config
        )
        return run_dir
    finally:
        os.chdir(original_dir)


def _read_train_options(config_file: Path) -> dict[str, object]:
    """读取 YAML 中的训练选项。"""
    with config_file.open(encoding="utf-8-sig") as config_stream:
        try:
            options = yaml.safe_load(config_stream)
        except yaml.YAMLError as error:
            raise ValueError(f"无法解析训练配置 {config_file}") from error
    if not isinstance(options, dict) or not all(
        isinstance(key, str) for key in options
    ):
        raise ValueError(f"训练配置必须是以字符串为键的映射 {config_file}")
    return options


def _resolve_config_path(
    options: dict[str, object], key: str, config_file: Path
) -> Path:
    """将配置中的路径解析为绝对路径。"""
    value = options.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{config_file} 的 {key} 必须为非空路径字符串")
    if key in ("model", "data"):
        return config_path(value)
    return (PROJECT_ROOT / value).resolve()


def _pop_p2(
    options: dict[str, object], config_file: Path
) -> P2Settings | None:
    """取出 P2 四尺度开关；开启时覆盖模型、权重与 mask_ratio。

    必须在解析 model 路径之前调用，否则训练仍使用基线结构。

    Args:
        options: 训练配置字典，原字典会被更新。
        config_file: 训练配置文件路径，仅用于错误信息。

    Returns:
        开启时返回生效设置，关闭或未配置时返回 None。

    Raises:
        ValueError: 配置块格式、类型或取值无效。
    """
    if "p2" not in options:
        return None
    settings = options.pop("p2")
    if not isinstance(settings, dict):
        raise ValueError(f"{config_file} 的 p2 必须为配置映射")
    is_enabled = settings.get("enabled")
    if not isinstance(is_enabled, bool):
        raise ValueError(f"{config_file} 的 p2.enabled 必须为布尔值")
    assignment_warmup = None
    if "assignment_warmup" in settings:
        from core_algorithm.p2_assignment_warmup import parse_p2_assignment_warmup

        assignment_warmup = parse_p2_assignment_warmup(
            settings["assignment_warmup"], config_file
        )
    if not is_enabled:
        if assignment_warmup is not None:
            raise ValueError(f"{config_file} 的 P2 分配热身要求 p2.enabled 为 true")
        return None
    model = settings.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{config_file} 的 p2.model 必须为非空路径字符串")
    pretrained = settings.get("pretrained")
    if not isinstance(pretrained, str) or not pretrained.strip():
        raise ValueError(f"{config_file} 的 p2.pretrained 必须为非空路径字符串")
    mask_ratio = settings.get("mask_ratio")
    if (
        not isinstance(mask_ratio, int)
        or isinstance(mask_ratio, bool)
        or mask_ratio <= 0
    ):
        raise ValueError(f"{config_file} 的 p2.mask_ratio 必须为正整数")
    previous = options.get("mask_ratio")
    if previous is not None and previous != mask_ratio:
        print(f"P2 已启用：mask_ratio 由 {previous} 覆盖为 {mask_ratio}")
    options["model"] = model
    options["pretrained"] = pretrained
    options["mask_ratio"] = mask_ratio
    return P2Settings(
        model=model,
        pretrained=pretrained,
        mask_ratio=mask_ratio,
        assignment_warmup=assignment_warmup,
    )


def _pop_online_augmentation(
    options: dict[str, object], config_file: Path
) -> tuple[bool, Path | None]:
    """取出项目专用增强配置，旧训练配置仍按基线运行。"""
    if "online_augmentation" not in options:
        return False, None
    settings = options.pop("online_augmentation")
    if not isinstance(settings, dict):
        raise ValueError(f"{config_file} 的 online_augmentation 必须为配置映射")
    is_enabled = settings.get("enabled")
    if not isinstance(is_enabled, bool):
        raise ValueError(f"{config_file} 的 online_augmentation.enabled 必须为布尔值")
    source = settings.get("config")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{config_file} 的 online_augmentation.config 必须为非空路径")
    return is_enabled, config_path(source)


def _record_online_augmentation(
    run_dir: Path,
    is_enabled: bool,
    source_path: Path | None,
    snapshot_path: Path | None,
) -> None:
    """把本次在线增强的状态和配置路径写入现有实验记录。"""
    state = "开启" if is_enabled else "关闭"
    with (run_dir / "change.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"\n## 在线增强\n\n- 实际启用状态 **{state}**\n")
        if source_path is not None and snapshot_path is not None:
            stream.write(f"- 配置来源 `{source_path}`\n")
            stream.write(f"- 配置快照 `{snapshot_path.relative_to(run_dir).as_posix()}`\n")


def _pop_gas(options: dict[str, object], config_file: Path) -> GasLaunch | None:
    """取出辅助监督开关；关闭或未配置时返回 None。

    Args:
        options: 训练配置字典，原字典会被更新。
        config_file: 训练配置文件路径，仅用于错误信息。

    Returns:
        开启时返回配置路径，关闭或未配置时返回 None。

    Raises:
        ValueError: 配置块格式、类型或取值无效。
    """
    if "gas" not in options:
        return None
    settings = options.pop("gas")
    if not isinstance(settings, dict):
        raise ValueError(f"{config_file} 的 gas 必须为配置映射")
    is_enabled = settings.get("enabled")
    if not isinstance(is_enabled, bool):
        raise ValueError(f"{config_file} 的 gas.enabled 必须为布尔值")
    if not is_enabled:
        return None
    source = settings.get("config")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{config_file} 的 gas.config 必须为非空路径")
    return GasLaunch(config=source)


def _record_gas(
    run_dir: Path, gas_launch: GasLaunch | None, snapshot_path: Path | None
) -> None:
    """把辅助监督的开关、权重与配置路径写入实验记录。"""
    state = "开启" if gas_launch is not None else "关闭"
    with (run_dir / "change.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"\n## GAS-Rail 辅助监督\n\n- 实际启用状态 **{state}**\n")
        if gas_launch is None or snapshot_path is None:
            return
        stream.write(f"- 配置来源 `{gas_launch.config}`\n")
        stream.write(
            f"- 配置快照 `{snapshot_path.relative_to(run_dir).as_posix()}`\n"
        )
        with snapshot_path.open(encoding="utf-8-sig") as config_stream:
            weights = yaml.safe_load(config_stream)
        if isinstance(weights, dict):
            summary = " · ".join(
                f"{key} {weights.get(key, 0)}"
                for key in (
                    "center_weight",
                    "direction_weight",
                    "smooth_weight",
                    "width_weight",
                    "dice_weight",
                    "warmup_epochs",
                )
            )
            stream.write(f"- 生效权重 {summary}\n")


def _pop_tensorboard(options: dict[str, object], config_file: Path) -> bool:
    """取出 TensorBoard 开关；关闭或未配置时返回 False。

    Args:
        options: 训练配置字典，原字典会被更新。
        config_file: 训练配置文件路径，仅用于错误信息。

    Returns:
        是否在本次训练中开启 TensorBoard 集成。

    Raises:
        ValueError: 配置块格式或类型无效。
    """
    if "tensorboard" not in options:
        return False
    settings = options.pop("tensorboard")
    if not isinstance(settings, dict):
        raise ValueError(f"{config_file} 的 tensorboard 必须为配置映射")
    is_enabled = settings.get("enabled")
    if not isinstance(is_enabled, bool):
        raise ValueError(f"{config_file} 的 tensorboard.enabled 必须为布尔值")
    return is_enabled


def _disable_tensorboard_graph() -> None:
    """关闭 TensorBoard 的模型结构图记录。

    Ultralytics 会在训练开始前 trace 一次模型并调用 ``add_graph``，这一步会额外占用一份
    显存；训练曲线不需要结构图，因此把该回调置空，避免在 16 GB 卡上产生额外峰值。
    """
    from ultralytics.utils.callbacks import tensorboard as tensorboard_callbacks

    tensorboard_callbacks._log_tensorboard_graph = lambda trainer: None


def _record_tensorboard(run_dir: Path, is_enabled: bool) -> None:
    """把 TensorBoard 开关与看板命令写入实验记录。"""
    state = "开启" if is_enabled else "关闭"
    with (run_dir / "change.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"\n## TensorBoard\n\n- 实际启用状态 **{state}**\n")
        if not is_enabled:
            return
        stream.write(f"- 事件文件位于 `{run_dir.name}/events.out.tfevents.*`\n")
        stream.write("- 看板 `python tools\\run_pipeline.py yolo-board`（端口 6006）\n")


def _record_p2(run_dir: Path, settings: P2Settings | None) -> None:
    """把 P2 开关状态与生效路径写入实验记录。"""
    state = "开启" if settings is not None else "关闭"
    with (run_dir / "change.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"\n## P2 四尺度\n\n- 实际启用状态 **{state}**\n")
        if settings is None:
            return
        stream.write(f"- 模型结构 `{settings.model}`\n")
        stream.write(f"- 初始化权重 `{settings.pretrained}`\n")
        stream.write(f"- mask_ratio 覆盖 **{settings.mask_ratio}**\n")
        if (warmup := settings.assignment_warmup) is not None:
            stream.write(
                f"- P2 分配热身 **{warmup.epochs} 轮**，"
                f"每个 GT 在各头内补到至多 **{warmup.max_positives_per_gt}** 个 P2 正样本，"
                f"补充分数 **{warmup.target_score}**\n"
            )


def _pop_bbox_crop(
    options: dict[str, object], config_file: Path
) -> BboxCropSettings | None:
    """取出内容裁剪开关；缺少配置块或关闭时返回 None。

    Args:
        options: 训练配置字典，原字典会被更新。
        config_file: 训练配置文件路径，仅用于错误信息。

    Returns:
        开启时返回生效设置，关闭或未配置时返回 None。

    Raises:
        ValueError: 配置块格式、类型或取值无效。
    """
    if "bbox_crop" not in options:
        return None
    block = options.pop("bbox_crop")
    return parse_bbox_crop(block, config_file)


def _record_bbox_crop(
    run_dir: Path, settings: BboxCropSettings | None, data_config: Path
) -> None:
    """把内容裁剪状态与生效数据集写入实验记录。"""
    state = "开启" if settings is not None else "关闭"
    with (run_dir / "change.md").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"\n## 内容裁剪\n\n- 实际启用状态 **{state}**\n")
        if settings is None:
            return
        stream.write(
            f"- 裁剪参数 fill={settings.fill_value} "
            f"tolerance={settings.tolerance} margin={settings.margin} "
            f"min_size={settings.min_size}\n"
        )
        stream.write(f"- 数据集 `{project_relative(settings.dataset)}`\n")
        stream.write(f"- 数据集配置 `{project_relative(data_config)}`\n")


def _create_run_dir(output_dir: Path) -> Path:
    """为本次实验独占一个时间戳目录。"""
    run_dir = output_dir / datetime.now().strftime(_RUN_TIMESTAMP_FORMAT)
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError as error:
        raise FileExistsError(f"实验目录已存在 {run_dir}") from error
    return run_dir


def _archive_results(
    model: YOLO,
    run_dir: Path,
    predict_options: dict[str, object],
) -> Path:
    """归档训练结果和验证预测，返回已归档的最优权重路径。"""
    from ultralytics import YOLO

    from experiments.yolo_seg.mask_seg_trainer import MaskCheckpointMetrics

    trainer = model.trainer
    if Path(trainer.save_dir) != run_dir:
        raise RuntimeError(f"实际训练目录与实验目录不一致 {trainer.save_dir}")

    best_weight = archive_training_outputs(run_dir, _BEST_WEIGHT_NAME)
    val_metrics = model.metrics
    if not isinstance(val_metrics, MaskCheckpointMetrics):
        raise RuntimeError("训练未返回最终验证指标")
    append_mask_metrics(run_dir, "val", val_metrics)

    best_model = YOLO(str(best_weight))
    # 叠加图只用于绘制漏检对照，写到临时目录并在使用后删除，不归档到实验目录。
    with TemporaryDirectory(prefix="_overlays_", dir=run_dir) as staged_name:
        staged_dir = Path(staged_name)
        image_paths = save_segmentation_predictions(
            best_model, Path(trainer.data["val"]), staged_dir, predict_options
        )
        save_missed_samples(
            run_dir, image_paths, val_metrics.image_mask_counts,
            val_metrics.preview_confidence, prediction_dir=staged_dir,
        )
    print(f"实验产物已归档到 {run_dir}")
    return best_weight


def _run_post_training_evaluations(
    best_weight: Path, data_config: Path, predict_config: Path
) -> None:
    """用本次最优权重和配置快照依次调用独立验证、测试入口。"""
    from experiments.yolo_seg.test_rail_seg import predict_rail_seg as predict_test
    from experiments.yolo_seg.valid_rail_seg import predict_rail_seg as predict_valid

    validation_dir = load_split_image_dir(data_config, "val")
    test_dir = load_split_image_dir(data_config, "test")
    predict_valid(best_weight, validation_dir, predict_config)
    predict_test(best_weight, test_dir, predict_config)


def main() -> None:
    """命令行入口：按指定训练配置启动一次实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=_TRAIN_CONFIG,
        help="训练配置路径（相对项目根目录），默认 config/train/train_rail_seg.yaml",
    )
    args = parser.parse_args()
    train_rail_seg(args.config)


if __name__ == "__main__":
    main()
