"""实时查看 YOLO 训练曲线：只读实验目录的 ``results.csv``。

不参与训练、不写实验目录，可以在训练运行期间随时打开。默认跟随 ``out/yolo_seg`` 下最新
的实验；训练结束或窗口关闭后退出。

用法：
    python -m experiments.yolo_seg.train_monitor
    python -m experiments.yolo_seg.train_monitor --run out/yolo_seg/20261112_023205
    python -m experiments.yolo_seg.train_monitor --snapshot
    python -m experiments.yolo_seg.train_monitor --text
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from data_preprocess.common.config import PROJECT_ROOT

_RUN_ROOT = PROJECT_ROOT / "out" / "yolo_seg"
_SNAPSHOT_ROOT = PROJECT_ROOT / "out" / "train_monitor"
# 训练期写在实验目录根部，归档后移到 logs/ 下，两处都兼容。
_RESULTS_CANDIDATES = (Path("logs", "results.csv"), Path("results.csv"))
_ARGS_CANDIDATES = (Path("logs", "args.yaml"), Path("args.yaml"))
_AUX_CONFIG = Path("configs", "gas_rail.yaml")

_EPOCH_COLUMN = "epoch"
_TIME_COLUMN = "time"
_LR_COLUMN = "lr/pg0"
_METRIC_COLUMN = "metrics/mAP50-95(M)"
_AUX_WEIGHT_KEYS = (
    "center_weight",
    "direction_weight",
    "smooth_weight",
    "width_weight",
    "dice_weight",
)

_LOSS_SERIES = (
    ("train/box_loss", "box", "#4c72b0"),
    ("train/seg_loss", "seg", "#dd8452"),
    ("train/cls_loss", "cls", "#55a868"),
    ("train/sem_loss", "sem", "#c44e52"),
)
_AUX_SERIES = (
    ("train/center_loss", "center", "#8172b3"),
    ("train/direction_loss", "direction", "#937860"),
    ("train/smooth_loss", "smooth", "#da8bc3"),
    ("train/width_loss", "width", "#8c8c8c"),
)
_METRIC_SERIES = (
    ("metrics/mAP50(M)", "Mask mAP50", "#4c72b0"),
    ("metrics/mAP50-95(M)", "Mask mAP50-95", "#dd8452"),
    ("metrics/mAP50(B)", "Box mAP50", "#55a868"),
)

_BACKGROUND = "#ffffff"
_PANEL_BACKGROUND = "#fbfbfd"
_GRID_COLOR = "#e4e6eb"
_MUTED_TEXT = "#70757d"
_CURRENT_EPOCH_COLOR = "#b9bdc6"
_BEST_COLOR = "#d1495b"
_PROGRESS_TRACK = "#eceef2"
_PROGRESS_FILL = "#4c72b0"

_FONT_CANDIDATES = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans")


@dataclass(frozen=True)
class MonitorOptions:
    """监视参数。

    Attributes:
        run_dir: 实验目录；为 None 时跟随最新实验。
        interval: 刷新间隔（秒）。
        is_snapshot: 只渲染一张静态图，不进入刷新循环。
        output_dir: 静态图输出目录；为 None 时写到 ``out/train_monitor`` 下的时间戳目录。
        is_text: 用控制台表格代替图形窗口。
        stop_when_done: 训练跑满后自动退出，而不是继续显示。
        smooth: 损失曲线的指数平滑系数，0 表示只画原始值。
    """

    run_dir: Path | None
    interval: float
    is_snapshot: bool
    output_dir: Path | None
    is_text: bool
    stop_when_done: bool
    smooth: float


@dataclass(frozen=True)
class RunSnapshot:
    """某一时刻的实验状态。

    Attributes:
        run_dir: 实验目录。
        rows: ``results.csv`` 的数据行，按 epoch 升序。
        total_epochs: 训练计划轮数，未知时为 None。
        aux_weights: GAS-Rail 生效权重，未启用时为 None。
        done_count: 已完成的 epoch 数。
        best_epoch: 目前验证集 mAP50-95 最高的 epoch。
        seconds_per_epoch: 最近几轮的平均单轮耗时，未知时为 None。
    """

    run_dir: Path
    rows: list[dict[str, float]]
    total_epochs: int | None
    aux_weights: dict[str, float] | None
    done_count: int
    best_epoch: int | None
    seconds_per_epoch: float | None


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="实时查看 YOLO 训练曲线")
    parser.add_argument("--run", default="", help="实验目录；默认 out/yolo_seg 下最新")
    parser.add_argument("--interval", type=float, default=15.0, help="刷新间隔秒")
    parser.add_argument("--snapshot", action="store_true", help="只出静态图")
    parser.add_argument("--output", default="", help="静态图输出目录")
    parser.add_argument("--text", action="store_true", help="用控制台表格代替窗口")
    parser.add_argument("--stop-when-done", action="store_true", help="训练结束后退出")
    parser.add_argument(
        "--smooth", type=float, default=0.55, help="损失曲线平滑系数，0 关闭"
    )
    return parser.parse_args()


def _first_existing(run_dir: Path, candidates: tuple[Path, ...]) -> Path | None:
    """返回候选相对路径中第一个存在的文件。"""
    for relative in candidates:
        path = run_dir / relative
        if path.is_file():
            return path
    return None


def _results_path(run_dir: Path) -> Path | None:
    """返回 results.csv 的实际路径（训练期在根部，归档后在 logs/）。"""
    return _first_existing(run_dir, _RESULTS_CANDIDATES)


def _latest_run() -> Path | None:
    """返回 ``out/yolo_seg`` 下最新、且已产生 results.csv 的实验目录。"""
    if not _RUN_ROOT.is_dir():
        return None
    candidates = sorted(
        (path for path in _RUN_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidates:
        if _results_path(candidate) is not None:
            return candidate
    return candidates[0] if candidates else None


def _resolve_run_dir(raw: str) -> Path:
    """解析 ``--run``：空串跟随最新实验，相对路径以项目根为基准。"""
    if not raw:
        latest = _latest_run()
        if latest is None:
            raise FileNotFoundError(f"{_RUN_ROOT} 下没有实验目录")
        return latest
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_dir():
        raise NotADirectoryError(f"实验目录不存在 {path}")
    return path.resolve()


def _read_rows(results_path: Path) -> list[dict[str, float]]:
    """读取 results.csv，跳过字段数不符的残缺行。"""
    if not results_path.is_file():
        return []
    with results_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            return []
        rows = []
        for values in reader:
            if len(values) != len(header):
                continue
            row = {}
            for name, value in zip(header, values):
                try:
                    row[name] = float(value)
                except ValueError:
                    continue
            if _EPOCH_COLUMN in row:
                rows.append(row)
    rows.sort(key=lambda item: item[_EPOCH_COLUMN])
    return rows


def _read_total_epochs(run_dir: Path) -> int | None:
    """从 ``args.yaml`` 读取训练计划轮数。"""
    path = _first_existing(run_dir, _ARGS_CANDIDATES)
    if path is None:
        return None
    with path.open(encoding="utf-8-sig") as stream:
        args = yaml.safe_load(stream)
    if not isinstance(args, dict):
        return None
    epochs = args.get("epochs")
    return int(epochs) if isinstance(epochs, (int, float)) else None


def _read_aux_weights(run_dir: Path) -> dict[str, float] | None:
    """读取 GAS-Rail 配置快照；实验未启用辅助监督时返回 None。"""
    path = run_dir / _AUX_CONFIG
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        return None
    return {
        key: float(config[key])
        for key in _AUX_WEIGHT_KEYS
        if isinstance(config.get(key), (int, float))
    }


def _build_snapshot(run_dir: Path) -> RunSnapshot:
    """汇总当前实验状态。"""
    results_path = _results_path(run_dir)
    rows = _read_rows(results_path) if results_path is not None else []
    best_epoch, _ = _best_epoch(rows)
    seconds_per_epoch = None
    if len(rows) >= 2:
        times = [row[_TIME_COLUMN] for row in rows if _TIME_COLUMN in row]
        if len(times) >= 2:
            deltas = [b - a for a, b in zip(times, times[1:]) if b >= a]
            if deltas:
                seconds_per_epoch = sum(deltas[-5:]) / len(deltas[-5:])
    return RunSnapshot(
        run_dir=run_dir,
        rows=rows,
        total_epochs=_read_total_epochs(run_dir),
        aux_weights=_read_aux_weights(run_dir),
        done_count=len(rows),
        best_epoch=best_epoch,
        seconds_per_epoch=seconds_per_epoch,
    )


def _best_epoch(rows: list[dict[str, float]]) -> tuple[int | None, float | None]:
    """返回验证集 mAP50-95 最高的 epoch 与对应值。"""
    best_epoch, best_value = None, None
    for row in rows:
        value = row.get(_METRIC_COLUMN)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_epoch, best_value = int(row[_EPOCH_COLUMN]), value
    return best_epoch, best_value


def _format_summary(snapshot: RunSnapshot) -> str:
    """生成一行状态摘要。"""
    total = snapshot.total_epochs or "?"
    parts = [f"epoch {snapshot.done_count}/{total}"]
    best_epoch, best_value = _best_epoch(snapshot.rows)
    if best_epoch is not None and best_value is not None:
        parts.append(f"最佳 mAP50-95 {best_value * 100:.1f}%（第 {best_epoch} 轮）")
    if snapshot.seconds_per_epoch:
        parts.append(f"单轮 {snapshot.seconds_per_epoch:.0f} s")
        if snapshot.total_epochs:
            remaining = max(snapshot.total_epochs - snapshot.done_count, 0)
            parts.append(f"预计剩余 {remaining * snapshot.seconds_per_epoch / 60:.0f} 分钟")
    if snapshot.aux_weights is not None:
        weights = " ".join(
            f"{key.replace('_weight', '')}={value:g}"
            for key, value in snapshot.aux_weights.items()
            if value > 0
        )
        parts.append(f"GAS-Rail {weights or '关闭'}")
    return " · ".join(parts)


def _series(
    rows: list[dict[str, float]], column: str
) -> tuple[list[float], list[float]]:
    """抽取某个列的 (epoch, value) 序列。"""
    xs, ys = [], []
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        xs.append(row[_EPOCH_COLUMN])
        ys.append(value)
    return xs, ys


def _smooth(values: list[float], factor: float) -> list[float]:
    """指数滑动平均；factor 为 0 时原样返回。"""
    if factor <= 0 or len(values) < 2:
        return values
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(factor * smoothed[-1] + (1 - factor) * value)
    return smoothed


def _style_axes(axes: Any) -> None:
    """统一的坐标轴外观：白底、浅网格、去掉上右框。"""
    for axis in axes:
        axis.set_facecolor(_PANEL_BACKGROUND)
        axis.grid(True, color=_GRID_COLOR, linewidth=0.8, alpha=0.9)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(_GRID_COLOR)
        axis.tick_params(colors=_MUTED_TEXT, labelsize=8, length=3)
        axis.xaxis.set_major_locator(_integer_locator())
        axis.title.set_color("#3d4248")
        axis.title.set_fontsize(10)


def _integer_locator() -> Any:
    """整数 epoch 刻度定位器。"""
    from matplotlib.ticker import MaxNLocator

    return MaxNLocator(integer=True)


def _set_x_limits(axes: Any, snapshot: RunSnapshot) -> None:
    """四个面板共用相同横轴，便于对照。"""
    if snapshot.rows:
        first = snapshot.rows[0][_EPOCH_COLUMN]
        last = snapshot.rows[-1][_EPOCH_COLUMN]
    else:
        first, last = 1.0, 1.0
    upper = float(snapshot.total_epochs or last)
    lower = max(first - 1.0, 0.0)
    for axis in axes:
        axis.set_xlim(lower, upper + 1.0)


def _mark_epochs(axes: Any, snapshot: RunSnapshot) -> None:
    """标出最佳轮（红虚线）与当前轮（灰虚线）。"""
    if not snapshot.rows:
        return
    best_epoch, _ = _best_epoch(snapshot.rows)
    current = snapshot.rows[-1][_EPOCH_COLUMN]
    for axis in axes:
        if best_epoch is not None:
            axis.axvline(
                best_epoch, color=_BEST_COLOR, linewidth=1.0,
                linestyle=(0, (4, 3)), alpha=0.55, zorder=1,
            )
        if current != best_epoch:
            axis.axvline(
                current, color=_CURRENT_EPOCH_COLOR, linewidth=1.0,
                linestyle=(0, (2, 2)), alpha=0.8, zorder=1,
            )


def _plot_loss_panel(axis: Any, snapshot: RunSnapshot, smooth: float) -> None:
    """原生损失：原始细线 + 平滑粗线，并在右端标注当前值。"""
    plotted = False
    for column, label, color in _LOSS_SERIES:
        xs, ys = _series(snapshot.rows, column)
        if not xs:
            continue
        plotted = True
        axis.plot(xs, ys, color=color, linewidth=0.9, alpha=0.28)
        axis.plot(
            xs, _smooth(ys, smooth), color=color, linewidth=1.9, label=label
        )
        axis.annotate(
            f"{ys[-1]:.3f}", (xs[-1], _smooth(ys, smooth)[-1]),
            textcoords="offset points", xytext=(4, 2),
            fontsize=7, color=color,
        )
    axis.set_title("原生损失（细线为原始值，粗线为平滑值）")
    if plotted:
        axis.legend(fontsize=8, ncol=4, frameon=False, loc="upper right")
    else:
        axis.text(
            0.5, 0.5, "等待第一轮结果…", ha="center", va="center",
            fontsize=9, color=_MUTED_TEXT, transform=axis.transAxes,
        )


def _plot_metric_panel(axis: Any, snapshot: RunSnapshot) -> None:
    """验证集 mAP：百分比坐标，标注最佳轮与最新值。"""
    from matplotlib.ticker import FuncFormatter

    plotted = False
    last_points = []
    for column, label, color in _METRIC_SERIES:
        xs, ys = _series(snapshot.rows, column)
        if not xs:
            continue
        plotted = True
        axis.plot(xs, ys, color=color, linewidth=1.9, label=label)
        last_points.append((xs[-1], ys[-1], color, label))
    if plotted:
        axis.set_ylim(0.0, max(1.0, max(
            max(_series(snapshot.rows, column)[1] or [0.0])
            for column, _, _ in _METRIC_SERIES
        ) * 1.08))
        best_epoch, best_value = _best_epoch(snapshot.rows)
        if best_epoch is not None and best_value is not None:
            axis.scatter([best_epoch], [best_value], color=_BEST_COLOR, s=26,
                         zorder=5, edgecolor="white", linewidth=0.8)
            axis.annotate(
                f"最佳 {best_value * 100:.1f}%（第 {best_epoch} 轮）",
                (best_epoch, best_value), textcoords="offset points",
                xytext=(6, -12), fontsize=8, color=_BEST_COLOR,
            )
        for x_value, y_value, color, label in last_points:
            axis.annotate(
                f"{label} {y_value * 100:.1f}%", (x_value, y_value),
                textcoords="offset points", xytext=(4, 3),
                fontsize=7, color=color,
            )
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    axis.set_title("验证集 mAP")
    if plotted:
        axis.legend(fontsize=8, ncol=3, frameon=False, loc="lower right")


def _plot_aux_panel(axis: Any, snapshot: RunSnapshot, smooth: float) -> None:
    """GAS-Rail 辅助损失：只画实际启用的项。"""
    enabled = []
    for column, label, color in _AUX_SERIES:
        xs, ys = _series(snapshot.rows, column)
        if not xs or max(ys) <= 0:
            continue
        enabled.append((xs, ys, label, color))
        axis.plot(xs, ys, color=color, linewidth=0.9, alpha=0.25)
        axis.plot(xs, _smooth(ys, smooth), color=color, linewidth=1.9, label=label)
        axis.annotate(
            f"{ys[-1]:.4f}", (xs[-1], _smooth(ys, smooth)[-1]),
            textcoords="offset points", xytext=(4, 2), fontsize=7, color=color,
        )
    axis.set_title("GAS-Rail 辅助损失")
    if enabled:
        axis.legend(fontsize=8, ncol=2, frameon=False, loc="upper right")
    else:
        axis.text(
            0.5, 0.5, "本次实验未启用辅助监督", ha="center", va="center",
            fontsize=9, color=_MUTED_TEXT, transform=axis.transAxes,
        )
        axis.set_yticks([])


def _plot_lr_panel(axis: Any, snapshot: RunSnapshot) -> None:
    """学习率曲线，带浅色填充。"""
    xs, ys = _series(snapshot.rows, _LR_COLUMN)
    if xs:
        axis.plot(xs, ys, color="#6c757d", linewidth=1.8)
        axis.fill_between(xs, 0.0, ys, color="#6c757d", alpha=0.12)
        axis.annotate(
            f"{ys[-1]:.5f}", (xs[-1], ys[-1]), textcoords="offset points",
            xytext=(4, 3), fontsize=7, color="#6c757d",
        )
    axis.set_title("学习率 lr/pg0")


def _configure_fonts() -> None:
    """优先使用系统中文字体，缺失时退回默认字体。"""
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = list(_FONT_CANDIDATES)
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.facecolor"] = _BACKGROUND


def _print_text(snapshot: RunSnapshot) -> None:
    """控制台模式：打印最近几轮的损失与指标。"""
    print(f"{snapshot.run_dir.name} | {_format_summary(snapshot)}")
    print(
        f"{'ep':>4} {'box':>8} {'seg':>8} {'cls':>8} {'center':>8} "
        f"{'mAP50':>8} {'mAP50-95':>9} {'lr':>9}"
    )
    for row in snapshot.rows[-10:]:
        values = [row.get(column) for column in (
            "train/box_loss", "train/seg_loss", "train/cls_loss",
            "train/center_loss", "metrics/mAP50(M)", "metrics/mAP50-95(M)",
            _LR_COLUMN,
        )]
        formatted = " ".join(
            f"{value:8.4f}" if value is not None else f"{'-':>8}" for value in values
        )
        print(f"{int(row[_EPOCH_COLUMN]):>4} {formatted}")


def _draw_figure(
    figure: Any, axes: Any, fill: Any, snapshot: RunSnapshot, smooth: float
) -> None:
    """更新整张图：标题、状态行、进度条与四个面板。"""
    axes = list(axes)
    figure.text(
        0.02, 0.975, snapshot.run_dir.name, fontsize=13, fontweight="bold",
        color="#2f3439", va="top",
    )
    figure.text(
        0.02, 0.945, _format_summary(snapshot), fontsize=9, color=_MUTED_TEXT,
        va="top",
    )
    for axis in axes:
        axis.clear()
    _style_axes(axes)
    if snapshot.rows:
        _plot_loss_panel(axes[0], snapshot, smooth)
        _plot_metric_panel(axes[1], snapshot)
        _plot_aux_panel(axes[2], snapshot, smooth)
        _plot_lr_panel(axes[3], snapshot)
        _set_x_limits(axes, snapshot)
        _mark_epochs(axes, snapshot)
    else:
        axes[0].text(
            0.5, 0.5, "等待第一轮结果…", ha="center", va="center",
            fontsize=10, color=_MUTED_TEXT, transform=axes[0].transAxes,
        )
        for axis in axes[1:]:
            axis.set_yticks([])
    total = snapshot.total_epochs or max(snapshot.done_count, 1)
    fill.set_width(min(snapshot.done_count / max(total, 1), 1.0))


def _create_figure() -> tuple[Any, Any, Any]:
    """创建图、四个面板与顶部进度条，返回 (figure, axes, 进度填充)。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    figure.subplots_adjust(
        top=0.865, bottom=0.06, left=0.055, right=0.985, hspace=0.34, wspace=0.16
    )
    track = figure.add_axes((0.06, 0.910, 0.88, 0.007))
    track.set_axis_off()
    track.set_xlim(0.0, 1.0)
    track.set_ylim(0.0, 1.0)
    track.add_patch(
        Rectangle((0.0, 0.0), 1.0, 1.0, facecolor=_PROGRESS_TRACK, edgecolor="none")
    )
    fill = Rectangle(
        (0.0, 0.0), 0.0, 1.0, facecolor=_PROGRESS_FILL, edgecolor="none"
    )
    track.add_patch(fill)
    return figure, axes.ravel(), fill


def _render_snapshot(
    snapshot: RunSnapshot, output_dir: Path | None, smooth: float
) -> Path:
    """渲染一张静态图并返回文件路径。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_fonts()
    figure, axes, fill = _create_figure()
    _draw_figure(figure, axes, fill, snapshot, smooth)
    if output_dir is None:
        output_dir = _SNAPSHOT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{snapshot.run_dir.name}_monitor.png"
    figure.savefig(path, dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)
    return path


def _run_live(options: MonitorOptions, run_dir: Path, smooth: float) -> None:
    """打开窗口持续刷新；训练完成且 ``stop_when_done`` 时退出。"""
    import matplotlib.pyplot as plt

    _configure_fonts()
    figure, axes, fill = _create_figure()
    try:
        while plt.fignum_exists(figure.number):
            snapshot = _build_snapshot(run_dir)
            _draw_figure(figure, axes, fill, snapshot, smooth)
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(options.interval)
            if options.stop_when_done and snapshot.rows:
                total = snapshot.total_epochs
                if total is not None and snapshot.done_count >= total:
                    print(f"训练已完成：{snapshot.run_dir.name} | {_format_summary(snapshot)}")
                    break
    except KeyboardInterrupt:
        print("\n已停止监视")
    finally:
        plt.close("all")


def run_monitor(options: MonitorOptions) -> Path | None:
    """按参数选择静态、控制台或窗口模式。

    Args:
        options: 监视参数。

    Returns:
        静态图路径；窗口与控制台模式返回 None。

    Raises:
        FileNotFoundError: 目标实验还没有 ``results.csv``。
        NotADirectoryError: ``--run`` 指向的目录不存在。
    """
    run_dir = options.run_dir or _resolve_run_dir("")
    if options.is_snapshot or options.is_text:
        result_path = _results_path(run_dir)
        if result_path is None:
            if options.is_text:
                print(f"{run_dir.name} 还没有训练结果，等待第一轮写入")
                return None
            raise FileNotFoundError(f"实验还没有训练结果：{run_dir}")
        snapshot = _build_snapshot(run_dir)
        if options.is_text:
            _print_text(snapshot)
            return None
        path = _render_snapshot(snapshot, options.output_dir, options.smooth)
        print(f"监视图已写入 {path}")
        return path
    _run_live(options, run_dir, options.smooth)
    return None


def main() -> None:
    """命令行入口。"""
    args = _parse_args()
    run_dir = Path(args.run) if args.run else None
    resolved = _resolve_run_dir(str(run_dir)) if run_dir else None
    output_dir = Path(args.output).resolve() if args.output else None
    run_monitor(
        MonitorOptions(
            run_dir=resolved,
            interval=max(args.interval, 1.0),
            is_snapshot=args.snapshot,
            output_dir=output_dir,
            is_text=args.text,
            stop_when_done=args.stop_when_done,
            smooth=min(max(args.smooth, 0.0), 0.95),
        )
    )


if __name__ == "__main__":
    main()
