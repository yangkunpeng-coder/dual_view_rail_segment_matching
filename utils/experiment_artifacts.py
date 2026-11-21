"""保存实验配置、环境、训练文件和评价指标。"""

from __future__ import annotations

import csv
from pathlib import Path
import platform
import re
from shutil import copy2
from subprocess import CalledProcessError, run
import sys
from typing import TYPE_CHECKING

from data_preprocess.common.config import load_yaml_mapping

if TYPE_CHECKING:
    from ultralytics.utils.metrics import SegmentMetrics

_FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg")
_EVALUATION_COLUMNS = ("split", "precision", "recall", "mask_map50", "mask_map50_95")
_TRAIN_RUN_NAME = re.compile(r"\d{8}_\d{6}")
# 已归档的实验只保留最优权重；best.pt 是旧实验的兼容文件名。
_ARCHIVED_WEIGHTS = ("rail_yolo26m-seg.pt", "best.pt")
# Ultralytics 的批样本预览图，信息已由训练曲线覆盖，不再归档。
_BATCH_PREVIEW = re.compile(r"(?:train|val)_batch\d+(?:_[a-z]+)*", re.IGNORECASE)
_CHANGE_HEADINGS = {
    "plan": "本次方案",
    "reasoning": "实验思路",
    "problem": "要解决的问题",
}


def snapshot_configs(run_dir: Path, train_config: Path, data_config: Path) -> Path:
    """复制训练与数据集配置，保留配置文件的原名称。

    Args:
        run_dir: 本次实验目录。
        train_config: 训练配置文件。
        data_config: 数据集配置文件。

    Returns:
        配置快照目录。
    """
    config_dir = run_dir / "configs"
    config_dir.mkdir()
    for config_file in (train_config, data_config):
        copy2(config_file, config_dir / config_file.name)
    return config_dir


def save_training_change(
    run_dir: Path,
    train_config: Path,
    data_config: Path,
    predict_config: Path | None,
    notes: dict[str, str] | None = None,
) -> None:
    """记录本次意图和相对上一轮已归档训练的配置差异。

    Args:
        run_dir: 当前实验目录。
        train_config: 当前训练配置快照。
        data_config: 当前数据配置快照。
        predict_config: 当前预测配置快照。旧实验可缺省。
        notes: 手动补记时提供的说明。默认读取训练配置的 change 字段。

    Raises:
        OSError: 无法读取配置或新建 change.md。
        ValueError: change 字段格式无效。
    """
    current_train = load_yaml_mapping(train_config)
    configured_notes = current_train.pop("change", None)
    change = notes if notes is not None else configured_notes
    if change is None:
        change = {}
    if not isinstance(change, dict) or any(
        key not in _CHANGE_HEADINGS or not isinstance(value, str)
        for key, value in change.items()
    ):
        raise ValueError(f"{train_config} 的 change 只接受 plan、reasoning、problem 文本")

    previous_dir = max(
        (
            path for path in run_dir.parent.iterdir()
            if _TRAIN_RUN_NAME.fullmatch(path.name) and path.name < run_dir.name
            and (path / "logs" / "results.csv").is_file()
            and _has_archived_weight(path / "weights")
        ),
        default=None,
    )
    previous_name = previous_dir.name if previous_dir else "首轮训练"
    lines = ["# 训练变更记录", "", f"本次实验 `{run_dir.name}`", "",
             f"对照实验 `{previous_name}`", ""]
    for key, heading in _CHANGE_HEADINGS.items():
        lines.extend((f"## {heading}", "", change.get(key, "").strip() or "待填写", ""))

    lines.extend(("## 配置差异", ""))
    if previous_dir is not None:
        previous_configs = previous_dir / "configs"
        previous_train = _archived_train_config(previous_configs)
        previous_data_value = (previous_train or {}).get("data")
        previous_data = (
            previous_configs / Path(previous_data_value).name
            if isinstance(previous_data_value, str)
            else None
        )
        previous_predict = previous_configs / "rail_predict.yaml"
        current_data = load_yaml_mapping(data_config)
        current = (
            ("训练", current_train, previous_train),
            ("数据集", current_data, _read_config_if_exists(previous_data)),
            ("预测", _read_config_if_exists(predict_config),
             _read_config_if_exists(previous_predict)),
        )
        lines.extend(("| 配置 | 参数 | 上次 | 本次 |", "| --- | --- | --- | --- |"))
        for label, after, before in current:
            if before is None or after is None:
                old = "有快照" if before is not None else "无快照"
                new = "有快照" if after is not None else "无快照"
                lines.append(f"| {label} | 配置快照 | {old} | {new} |")
                continue
            keys = list(before.keys() | after.keys())
            keys.sort()
            for key in keys:
                old = repr(before[key]) if key in before else "未设置"
                new = repr(after[key]) if key in after else "未设置"
                if old != new:
                    cells = (label, key, old, new)
                    escaped = [cell.replace("|", "\\|") for cell in cells]
                    lines.append("| " + " | ".join(escaped) + " |")
        if lines[-1].startswith("| ---"):
            lines.append("| 全部 | 差异项 | 0 | 0 |")
    else:
        lines.append("首轮配置以本次 `configs/` 快照为基准。")

    with (run_dir / "change.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")


def _has_archived_weight(weights_dir: Path) -> bool:
    """判断实验目录是否已归档最优权重。

    Args:
        weights_dir: 实验目录下的 weights 子目录。

    Returns:
        存在任一归档权重文件名时为 True。
    """
    return any((weights_dir / name).is_file() for name in _ARCHIVED_WEIGHTS)


def _read_config_if_exists(path: Path | None) -> dict[str, object] | None:
    """读取存在的配置快照。"""
    return load_yaml_mapping(path) if path is not None and path.is_file() else None


def _archived_train_config(config_dir: Path) -> dict[str, object] | None:
    """识别旧实验的训练快照，兼容自定义文件名。"""
    for path in sorted((*config_dir.glob("*.yaml"), *config_dir.glob("*.yml"))):
        options = load_yaml_mapping(path)
        if "model" in options and "data" in options:
            options.pop("change", None)
            return options
    return None


def save_environment(run_dir: Path) -> None:
    """记录实验使用的软件版本、GPU 和源码状态。

    Args:
        run_dir: 本次实验目录。
    """
    import torch
    import ultralytics

    module_path = ultralytics.__file__
    if module_path is None:
        raise RuntimeError("无法确定 Ultralytics 源码文件位置")
    source_path = Path(module_path).resolve()
    commit, source_status = _read_git_state(source_path.parents[1])
    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    lines = (
        f"Python {platform.python_version()}",
        f"Python executable {sys.executable}",
        f"PyTorch {torch.__version__}",
        f"CUDA {torch.version.cuda}",
        f"Ultralytics {ultralytics.__version__}",
        f"GPU {', '.join(gpu_names) or 'none'}",
        f"Ultralytics source {source_path}",
        f"Ultralytics Git commit {commit}",
        f"Ultralytics Git status {source_status}",
    )
    (run_dir / "environment.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _read_git_state(source_dir: Path) -> tuple[str, str]:
    """读取源码提交号，并标明是否有未提交的修改。"""
    git = ["git", "--no-optional-locks", "-C", str(source_dir)]
    try:
        commit = run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = run(
            [*git, "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, CalledProcessError) as error:
        return f"unavailable ({type(error).__name__})", "unknown"
    return commit, "dirty" if status.strip() else "clean"


def archive_training_outputs(run_dir: Path, best_weight_name: str) -> Path:
    """整理训练原始记录、最优权重和实际生成的图表。

    只保留最优权重，末轮权重在归档时删除；Ultralytics 的批样本预览图
    （`train_batch*`、`val_batch*`）信息已由训练曲线覆盖，同样不归档。

    Args:
        run_dir: Ultralytics 本次训练的输出目录。
        best_weight_name: 归档后的最优权重文件名。

    Returns:
        归档后的最优权重路径。

    Raises:
        FileNotFoundError: 缺少需要归档的训练记录或最优权重。
    """
    args_file = run_dir / "args.yaml"
    results_csv = run_dir / "results.csv"
    weights_dir = run_dir / "weights"
    best_weight = weights_dir / "best.pt"
    for required_file in (args_file, results_csv, best_weight):
        if not required_file.is_file():
            raise FileNotFoundError(f"缺少训练产物 {required_file}")

    if not (run_dir / "results.png").is_file():
        from ultralytics.utils.plotting import plot_results

        plot_results(file=results_csv)

    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    for log_file in (args_file, results_csv):
        log_file.rename(logs_dir / log_file.name)

    figures: list[Path] = []
    for path in run_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in _FIGURE_SUFFIXES:
            continue
        if _BATCH_PREVIEW.fullmatch(path.stem):
            path.unlink()
        else:
            figures.append(path)
    if figures:
        figures_dir = run_dir / "figures"
        figures_dir.mkdir()
        for figure in figures:
            figure.rename(figures_dir / figure.name)

    archived_weight = best_weight.rename(weights_dir / best_weight_name)
    (weights_dir / "last.pt").unlink(missing_ok=True)
    return archived_weight


def append_mask_metrics(run_dir: Path, split: str, metrics: SegmentMetrics) -> None:
    """追加一个已执行集合的最终 Mask 评价指标。

    Args:
        run_dir: 本次实验目录。
        split: 已完成评价的数据集划分。
        metrics: Ultralytics 返回的实例分割指标。
    """
    metrics_path = run_dir / "metrics" / "evaluation.csv"
    metrics_path.parent.mkdir(exist_ok=True)
    mask = metrics.seg
    is_new_file = not metrics_path.exists()
    with metrics_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        if is_new_file:
            writer.writerow(_EVALUATION_COLUMNS)
        writer.writerow(
            (split, float(mask.mp), float(mask.mr), float(mask.map50), float(mask.map))
        )
