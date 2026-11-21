"""项目统一运行入口：PyCharm 直接运行弹菜单，命令行可指定阶段直跑。

设计目标：同一套命令既能在 PyCharm 里"直接 Run"，也能在 CMD 里带参数执行，
不再出现"有的脚本在 PyCharm 改路径、有的在命令行敲参数"的情况。

用法：
    python tools/run_pipeline.py                  # 交互菜单
    python tools/run_pipeline.py --list           # 只列出阶段
    python tools/run_pipeline.py gt-eval-latest   # 直接跑某个阶段
    python tools/run_pipeline.py gt-eval-latest --run out/evaluate_rail_match/<时间戳>
    python tools/run_pipeline.py match-eval --dry-run

阶段定义在 `_STAGES`；新增入口只需加一条 `Stage`。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MATCH_ROOT = _PROJECT_ROOT / "out" / "evaluate_rail_match"
_GT_ROOT = _PROJECT_ROOT / "out" / "rail_match_label" / "gt"
_YOLO_ROOT = _PROJECT_ROOT / "out" / "yolo_seg"
_YOLO_BATCH_ROOT = _PROJECT_ROOT / "out" / "yolo_seg_batch"
_SAM3_ROOT = _PROJECT_ROOT / "out" / "sam3_seg"
_GAS_PREVIEW_ROOT = _PROJECT_ROOT / "out" / "gas_target_preview"
_TRAIN_MONITOR_ROOT = _PROJECT_ROOT / "out" / "train_monitor"
_SEG_POSTPROCESS_ROOT = _PROJECT_ROOT / "out" / "seg_postprocess"
_POSTPROCESS_MATCHING_ROOT = _PROJECT_ROOT / "out" / "postprocess_matching"
_RAIL_MATCH_DEMO_ROOT = _PROJECT_ROOT / "out" / "rail_match_demo"
_COMPARE_MASK_ROOT = _PROJECT_ROOT / "out" / "compare_mask"

# ANSI 颜色；Windows 控制台需要先打开 VT 处理，重定向或设置 NO_COLOR 时自动关闭。
_COLOR_ARTIFACT = "1;32"
_COLOR_ERROR = "1;31"
_COLOR_COMMAND = "36"
_COLOR_ENABLED: bool | None = None


def _use_color() -> bool:
    """判断是否输出 ANSI 颜色，并在 Windows 控制台里打开 VT 模式。"""
    global _COLOR_ENABLED
    if _COLOR_ENABLED is not None:
        return _COLOR_ENABLED
    if os.environ.get("NO_COLOR"):
        _COLOR_ENABLED = False
        return False
    if os.environ.get("FORCE_COLOR"):
        _COLOR_ENABLED = True
        return True
    if not sys.stdout.isatty():
        _COLOR_ENABLED = False
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                _COLOR_ENABLED = False
                return False
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except (AttributeError, OSError):
            _COLOR_ENABLED = False
            return False
    _COLOR_ENABLED = True
    return True


def _paint(text: str, code: str) -> str:
    """按需给文本加 ANSI 颜色。"""
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"

_DESCRIPTION = (
    "项目统一运行入口：无参数进入两级菜单，也可直接指定阶段名、别名或编号。"
)
_EPILOG = """\
常用写法：
  python tools/run_pipeline.py                  两步菜单
  python tools/run_pipeline.py train            直接跑（短别名）
  python tools/run_pipeline.py yolo-valid --run 20261112_023205 --overlays
  python tools/run_pipeline.py yolo-valid -h    查看该阶段自己的参数

阶段参数：
  阶段名后面的参数会原样转发给阶段脚本，例如 --run、--overlays、--image、
  --limit、--algorithms。每个阶段的完整参数用 `<阶段> -h` 查看。
"""


@dataclass(frozen=True)
class Stage:
    """一个可执行阶段。

    Attributes:
        key: 命令行使用的阶段名。
        group: 菜单分组标题。
        title: 菜单里显示的中文说明。
        module: 以 `-m` 运行的模块名。
        arguments: 固定参数。
        latest_under: 成功后在哪个根目录下找最新时间戳产物并打印。
        run_root: 产物写在实验目录内时，实验目录所在的根目录。
        latest_in_run: 实验目录内定位产物的相对路径，例如 ("valid",)。
        use_latest_run: 是否自动把 `--run` 指到最新一次匹配评测运行。
    """

    key: str
    group: str
    title: str
    module: str
    arguments: tuple[str, ...] = ()
    latest_under: Path | None = None
    run_root: Path | None = None
    latest_in_run: tuple[str, ...] = ()
    use_latest_run: bool = False


_STAGES: tuple[Stage, ...] = (
    Stage(
        "yolo-train", "训练",
        "YOLO 训练（可用 --config 指定实验配置）",
        "experiments.yolo_seg.train_rail_seg",
        latest_under=_YOLO_ROOT,
    ),
    Stage(
        "yolo-batch", "训练",
        "按清单批量训练（默认 config/experiments/batch.yaml）",
        "experiments.yolo_seg.train_batch",
        latest_under=_YOLO_BATCH_ROOT,
    ),
    Stage(
        "gas-preview", "训练",
        "GAS-Rail 目标预览：中心线/方向/宽度叠加图",
        "experiments.yolo_seg.gas_target_preview",
        latest_under=_GAS_PREVIEW_ROOT,
    ),
    Stage(
        "p2-init", "训练",
        "用 YOLO26m-seg 预训练权重生成 P2 四尺度初始权重",
        "data_preprocess.transfer_yolo26m_seg_p2",
    ),
    Stage(
        "yolo-valid", "评测与监控",
        "YOLO 独立验证（默认最新实验）",
        "experiments.yolo_seg.valid_rail_seg",
        run_root=_YOLO_ROOT,
        latest_in_run=("valid",),
    ),
    Stage(
        "yolo-test", "评测与监控",
        "YOLO 独立测试（默认最新实验）",
        "experiments.yolo_seg.test_rail_seg",
        run_root=_YOLO_ROOT,
        latest_in_run=("test",),
    ),
    Stage(
        "seg-report", "评测与监控",
        "实验汇总报告",
        "experiments.yolo_seg.experiment_report",
        latest_under=_PROJECT_ROOT / "out" / "seg_report",
    ),
    Stage(
        "yolo-watch", "评测与监控",
        "训练曲线：实时看最新实验（只读）",
        "experiments.yolo_seg.train_monitor",
        latest_under=_TRAIN_MONITOR_ROOT,
    ),
    Stage(
        "seg-compare", "评测与监控",
        "跨模型独立验证对比",
        "experiments.compare_valid_seg",
        latest_under=_PROJECT_ROOT / "out" / "compare_valid_seg",
    ),
    Stage(
        "yolo-train-predict", "评测与监控",
        "YOLO 训练集预测（默认最新实验）",
        "experiments.yolo_seg.predict_train_rail_seg",
        run_root=_YOLO_ROOT,
        latest_in_run=("train",),
    ),
    Stage(
        "sam3-valid", "评测与监控",
        "SAM3 验证",
        "experiments.sam3_seg.valid_sam3_seg",
        latest_under=_SAM3_ROOT,
    ),
    Stage(
        "sam3-test", "评测与监控",
        "SAM3 测试",
        "experiments.sam3_seg.test_sam3_seg",
        latest_under=_SAM3_ROOT,
    ),
    Stage(
        "sam3-train-predict", "评测与监控",
        "SAM3 训练集预测",
        "experiments.sam3_seg.train_sam3_seg",
        latest_under=_SAM3_ROOT,
    ),
    Stage(
        "yolo-board", "评测与监控",
        "TensorBoard 看板（全部训练实验，端口 6006）",
        "tensorboard.main",
        ("--logdir", "out/yolo_seg", "--port", "6006"),
    ),
    Stage(
        "match-eval", "匹配与真值",
        "匹配评测：117 对、五算法覆盖率口径",
        "experiments.rail_match_eval",
        ("--algorithms", "rail_matching", "rail_matching_mslc",
         "rail_matching_gmsa", "rail_matching_isa", "rail_matching_cgm",
         "--review-limit", "0"),
        latest_under=_MATCH_ROOT,
    ),
    Stage(
        "gt-eval-latest", "匹配与真值",
        "真值评测：评最近一次匹配评测运行",
        "experiments.rail_match_gt_eval",
        use_latest_run=True,
        latest_under=_GT_ROOT,
    ),
    Stage(
        "gt-eval-algo", "匹配与真值",
        "真值评测：直接跑五算法（改动前后 A/B）",
        "experiments.rail_match_gt_eval",
        ("--algorithms", "rail_matching", "rail_matching_mslc",
         "rail_matching_gmsa", "rail_matching_isa", "rail_matching_cgm"),
        latest_under=_GT_ROOT,
    ),
    Stage(
        "match-demo", "匹配与真值",
        "匹配预览：MSLC 双视角 Demo",
        "core_algorithm.matching.rail_matching_mslc",
    ),
    Stage(
        "rail-match-demo", "匹配与真值",
        "南北相机原图分割与匹配 Demo",
        "experiments.rail_match_demo",
        ("--limit", "5"),
        latest_under=_RAIL_MATCH_DEMO_ROOT,
    ),
    Stage(
        "mask-analyze", "掩码与预览",
        "GT 与预测掩码对比分析（默认最新实验）",
        "experiments.yolo_seg.mask_compare",
        latest_under=_COMPARE_MASK_ROOT,
    ),
    Stage(
        "mask-bundle", "掩码与预览",
        "导出实验掩码包（默认最新实验）",
        "experiments.yolo_seg.mask_bundle",
        latest_under=_PROJECT_ROOT / "out" / "mask_bundle",
    ),
    Stage(
        "postprocess-ablation", "后处理",
        "分割后处理固定预测消融（raw / Mask NMS 阈值）",
        "experiments.yolo_seg.postprocess_ablation",
        latest_under=_SEG_POSTPROCESS_ROOT,
    ),
    Stage(
        "postprocess-matching", "后处理",
        "关系级匹配评测：raw vs 去重后预测",
        "experiments.postprocess_matching_eval",
        latest_under=_POSTPROCESS_MATCHING_ROOT,
    ),
    Stage(
        "check-data", "数据",
        "检查原始 LabelMe 标注，生成南北空标注清单",
        "data_preprocess.check_data",
    ),
    Stage(
        "yolo-data-export", "数据",
        "按采集组划分并导出 YOLO 分割数据集",
        "data_preprocess.create_YOLO_dataset",
    ),
    Stage(
        "view-data", "数据",
        "预览南北配对 Mask 叠加图",
        "data_preprocess.view_data",
    ),
    Stage(
        "pair-export", "数据",
        "导出成对 LabelMe 标注 ZIP",
        "data_preprocess.pair_match_json",
    ),
    Stage(
        "bbox-export", "数据",
        "导出内容裁剪数据集（bbox_crop 开关）",
        "data_preprocess.export_bbox_crop",
    ),
    Stage(
        "geometry", "数据",
        "GT 几何分析（消失点/宽度/曲率）",
        "data_preprocess.geometry_analysis",
        latest_under=_PROJECT_ROOT / "out" / "rail_geometry",
    ),
    Stage(
        "gt-manifest", "真值标注",
        "刷新真值清单",
        "annotation.rail_match_gt.manifest",
    ),
    Stage(
        "gt-export", "真值标注",
        "导出真值 CSV",
        "annotation.rail_match_gt.export",
    ),
    Stage(
        "gt-annotate", "真值标注",
        "打开标注工具",
        "annotation.rail_match_gt.annotate",
    ),
)


_ALIASES: dict[str, str] = {
    "train": "yolo-train",
    "batch": "yolo-batch",
    "valid": "yolo-valid",
    "test": "yolo-test",
    "predict": "yolo-train-predict",
    "watch": "yolo-watch",
    "board": "yolo-board",
    "report": "seg-report",
    "compare": "seg-compare",
    "mask": "mask-analyze",
    "match": "match-eval",
    "gt": "gt-eval-latest",
    # 旧阶段名保留兼容：sam3-train 原为“SAM3 训练集预测”。
    "sam3-train": "sam3-train-predict",
}


def _latest(directory: Path) -> Path | None:
    """取目录下最新的时间戳子目录。"""
    if not directory.is_dir():
        return None
    candidates = [path for path in directory.iterdir() if path.is_dir()]
    return max(candidates, key=lambda path: path.name) if candidates else None


def _run_dir_from_args(run_root: Path, extra: list[str]) -> Path | None:
    """按 `--run` 参数或最新实验定位本次实验目录。"""
    if "--run" in extra:
        index = extra.index("--run")
        if index + 1 >= len(extra):
            return None
        value = Path(extra[index + 1])
        candidates = (
            [value] if value.is_absolute()
            else [_PROJECT_ROOT / value, run_root / value]
        )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        return None
    return _latest(run_root)


def _resolve_artifact(stage: Stage, extra: list[str]) -> Path | None:
    """解析本次运行的产物目录，无法确定时返回 None。"""
    if stage.latest_under is not None:
        return _latest(stage.latest_under)
    if stage.run_root is None or not stage.latest_in_run:
        return None
    run_dir = _run_dir_from_args(stage.run_root, extra)
    if run_dir is None:
        return None
    return _latest(run_dir.joinpath(*stage.latest_in_run))


def _resolve(token: str) -> Stage | None:
    """按阶段名、短别名或菜单编号找到阶段。"""
    token = token.strip()
    token = _ALIASES.get(token, token)
    for stage in _STAGES:
        if stage.key == token:
            return stage
    if token.isdigit():
        index = int(token) - 1
        if 0 <= index < len(_STAGES):
            return _STAGES[index]
    return None


def _print_stages() -> None:
    """打印分组菜单和常用别名。"""
    width = max(len(stage.key) for stage in _STAGES)
    group = None
    for index, stage in enumerate(_STAGES, start=1):
        if stage.group != group:
            group = stage.group
            print(f"\n【{group}】")
        print(f"  {index:>2}. {stage.key:<{width}} {stage.title}")
    aliases = "  ".join(f"{alias}={key}" for alias, key in _ALIASES.items())
    print(f"\n常用别名：{aliases}")
    print("不带参数运行可进入编号菜单。")


def _groups() -> list[tuple[str, list[Stage]]]:
    """按首次出现顺序返回分组及其阶段。"""
    ordered: list[tuple[str, list[Stage]]] = []
    for stage in _STAGES:
        for name, stages in ordered:
            if name == stage.group:
                stages.append(stage)
                break
        else:
            ordered.append((stage.group, [stage]))
    return ordered


def _print_groups(groups: list[tuple[str, list[Stage]]]) -> None:
    """打印一级菜单：分组编号、阶段数量和内容提示。"""
    print("\n请选择分组：")
    for index, (name, stages) in enumerate(groups, start=1):
        print(f"  {index}. {name}（{len(stages)} 个阶段）")
        hint = _GROUP_HINTS.get(name)
        if hint:
            print(f"     {hint}")
    print(f"  a. 显示全部阶段（{len(_STAGES)} 个）")
    aliases = "  ".join(f"{alias}={key}" for alias, key in _ALIASES.items())
    print(f"\n常用别名：{aliases}")
    print("也可以直接输入阶段名或别名。")


def _print_group_stages(name: str, stages: list[Stage]) -> None:
    """打印二级菜单：指定分组内的阶段。"""
    width = max(len(stage.key) for stage in stages)
    print(f"\n【{name}】")
    for index, stage in enumerate(stages, start=1):
        print(f"  {index}. {stage.key:<{width}} {stage.title}")
    print("  0. 返回分组")


def _ask(prompt: str) -> str | None:
    """读取一行输入；遇到 EOF 时打印提示并返回 None。"""
    try:
        return input(prompt).strip()
    except EOFError:
        print("当前运行环境不支持交互输入；请用 --list 查看分组，"
              "再用 `python tools/run_pipeline.py <阶段名>` 运行。")
        return None


def _interactive(extra: list[str], dry_run: bool) -> int:
    """两级菜单：先选分组，再选组内阶段。"""
    groups = _groups()
    while True:
        _print_groups(groups)
        token = _ask("\n输入分组编号、阶段名或 a（回车退出）：")
        if token is None:
            return 2
        if not token:
            return 0
        if token.lower() in ("help", "?", "h"):
            continue
        if token.lower() in ("a", "all"):
            _print_stages()
            choice = _ask("\n输入阶段编号或名称（回车返回）：")
            if choice is None:
                return 2
            if not choice:
                continue
            stage = _resolve(choice)
            if stage is None:
                print(f"没有阶段 {choice!r}")
                continue
            return _run(stage, extra, dry_run)
        if token.isdigit():
            index = int(token) - 1
            if not 0 <= index < len(groups):
                print(f"没有分组 {token}")
                continue
            name, stages = groups[index]
            _print_group_stages(name, stages)
            choice = _ask("\n输入编号或阶段名（0 返回）：")
            if choice is None:
                return 2
            if choice in ("", "0"):
                continue
            if choice.isdigit():
                local = int(choice) - 1
                if not 0 <= local < len(stages):
                    print(f"没有阶段 {choice}")
                    continue
                return _run(stages[local], extra, dry_run)
            stage = _resolve(choice)
            if stage is None:
                print(f"没有阶段 {choice!r}")
                continue
            return _run(stage, extra, dry_run)
        stage = _resolve(token)
        if stage is None:
            print(f"没有分组或阶段 {token!r}")
            continue
        return _run(stage, extra, dry_run)


def _build_command(
    stage: Stage, extra: list[str]
) -> tuple[list[str] | None, str]:
    """拼出实际命令；自动补 --run 失败时返回错误说明。"""
    arguments = list(stage.arguments)
    if stage.use_latest_run and "--run" not in extra:
        latest = _latest(_MATCH_ROOT)
        if latest is None:
            return None, f"{_MATCH_ROOT} 下没有可用的匹配评测运行，请先跑 match-eval"
        arguments.extend(["--run", str(latest)])
    command = [sys.executable, "-m", stage.module, *arguments, *extra]
    return command, ""


def _run(stage: Stage, extra: list[str], dry_run: bool) -> int:
    """执行阶段并打印产物目录。"""
    command, error = _build_command(stage, extra)
    if command is None:
        print(_paint(f"无法运行 {stage.key}：{error}", _COLOR_ERROR))
        return 2
    print(_paint("命令：" + " ".join(shlex.quote(part) for part in command),
                 _COLOR_COMMAND))
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=_PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        print(_paint(
            f"阶段 {stage.key} 失败，退出码 {completed.returncode}", _COLOR_ERROR
        ))
        return completed.returncode
    newest = _resolve_artifact(stage, extra)
    if newest is not None:
        print(_paint(f"产物目录：{newest}", _COLOR_ARTIFACT))
    return 0


_STAGE_PARAMS: dict[str, str] = {
    "yolo-train": "--config <训练配置>",
    "yolo-batch": "--manifest <清单>；--keep-going（单个失败继续）",
    "p2-init": "--source <权重>、--model <模型配置>、--output <权重>、--report <路径前缀>",
    "yolo-valid": "--run <实验目录>；--config <预测配置>；--overlays",
    "yolo-test": "--run <实验目录>；--config <预测配置>；--overlays",
    "yolo-train-predict": "--run <实验目录>",
    "seg-report": "无（按实验清单运行）",
    "yolo-watch": "--run <实验目录>；--interval <秒>；--text；--stop-when-done",
    "seg-compare": "无（读取已有 valid 指标）",
    "yolo-board": "无（固定端口 6006）",
    "sam3-valid": "--overlays",
    "sam3-test": "--overlays",
    "sam3-train-predict": "--overlays",
    "match-eval": (
        "--algorithms <算法名...>；--max-pairs <N>；--review-limit <N>；"
        "--pair-manifest <csv>；--gmsa-config <yaml>"
    ),
    "gt-eval-latest": "--run <匹配评测目录>；--output <目录>",
    "gt-eval-algo": "--algorithms <算法名...>；--run <匹配评测目录>",
    "match-demo": "无（模块内固定流程）",
    "rail-match-demo": (
        "--limit <数量>、--weights <pt>、--predict-config <yaml>、"
        "--postprocess-config <yaml>、--roi-mode <raw|predicted>、"
        "--gt-split <valid|test>、--save-masks、--no-previews"
    ),
    "mask-analyze": "--run <实验目录或名称>；--show",
    "mask-bundle": (
        "--run <实验...>（可多个）；--splits <划分...>；--output <目录>；--no-zip"
    ),
    "gas-preview": "--count <N>；--split <划分>；--seed <N>；--output <目录>",
    "postprocess-ablation": (
        "--valid-pred-dir <目录>；--test-pred-dir <目录>；--limit <N>；--no-figures"
    ),
    "postprocess-matching": (
        "--valid-pred-dir <目录>；--test-pred-dir <目录>；--splits <划分...>；--limit <N>"
    ),
    "view-data": "--image <图片名>",
    "pair-export": "无（按数据配置运行）",
    "bbox-export": "--config <训练配置>、--refresh（重建已有导出目录）",
    "geometry": "无（按数据配置运行）",
    "gt-manifest": "无",
    "gt-export": "无",
    "gt-annotate": "--split <划分>；--pair-id <P0001>；--annotator <名字>",
}


_GROUP_HINTS: dict[str, str] = {
    "训练": "YOLO 训练、按清单批量训练、P2 权重初始化、GAS-Rail 目标预览",
    "评测与监控": (
        "独立验证/测试、训练集预测、汇总报告、跨模型对比、SAM3、训练曲线、TensorBoard"
    ),
    "匹配与真值": "匹配评测、真值评测、MSLC 双视角预览、模型→去重→匹配 Demo",
    "掩码与预览": "掩码几何对比、掩码包导出",
    "后处理": "Mask NMS 消融、关系级匹配评测",
    "数据": "标注检查、YOLO 数据集导出、配对预览、内容裁剪和几何分析",
    "真值标注": "刷新真值清单、导出真值 CSV、打开标注工具",
}


def _show_stage_help(stage: Stage) -> None:
    """打印阶段的静态说明，不执行阶段脚本。"""
    print(f"\n阶段：{stage.key}")
    print(f"说明：{stage.title}")
    print(f"模块：{stage.module}")
    fixed = " ".join(stage.arguments) if stage.arguments else "无"
    print(f"固定参数：{fixed}")
    print(f"可追加参数：{_STAGE_PARAMS.get(stage.key, '无（按配置运行）')}")
    print(f"用法：python tools/run_pipeline.py {stage.key} [参数]")


def main() -> None:
    """命令行入口：无参数时进入交互菜单。"""
    parser = argparse.ArgumentParser(
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "stage", nargs="?", help="阶段名、短别名或菜单编号；省略时进入两级菜单"
    )
    parser.add_argument(
        "-l", "--list", action="store_true",
        help=f"只列出 {len(_groups())} 个分组",
    )
    parser.add_argument(
        "-a", "--list-all", "--all", dest="list_all", action="store_true",
        help=f"列出全部 {len(_STAGES)} 个阶段",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    parser.add_argument(
        "-h", "--help", action="store_true", dest="show_help",
        help="显示本帮助；带阶段名时显示该阶段脚本的参数",
    )
    # 未识别的参数原样转发给阶段脚本，例如 --run out/evaluate_rail_match/<时间戳>。
    args, forwarded = parser.parse_known_args()
    if args.show_help:
        if args.stage:
            stage = _resolve(args.stage)
            if stage is None:
                parser.error(f"未知阶段 {args.stage!r}")
            _show_stage_help(stage)
            return
        parser.print_help()
        return
    if args.list:
        _print_groups(_groups())
        return
    if args.list_all:
        _print_stages()
        return
    if args.stage:
        if args.stage.strip().lower() in ("help", "?", "h"):
            _print_groups(_groups())
            return
        stage = _resolve(args.stage)
        if stage is None:
            parser.error(f"未知阶段 {args.stage!r}，可用 --list 或 --list-all 查看")
        raise SystemExit(_run(stage, forwarded, args.dry_run))

    raise SystemExit(_interactive(forwarded, args.dry_run))


if __name__ == "__main__":
    main()
