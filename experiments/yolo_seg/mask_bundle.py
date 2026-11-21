"""把一次或多次实验的 GT 与预测掩码导出为紧凑 LabelMe 包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preprocess.common.config import PROJECT_ROOT
from experiments.yolo_seg.valid_rail_seg import latest_experiment
from utils.mask_bundle import build_mask_bundle, build_mask_bundle_collection

_SPLIT_CHOICES = ("train", "valid", "test")


def _resolve_run_dirs(values: list[Path] | None) -> list[Path]:
    """把参数换算成实验目录；省略时取 out/yolo_seg 下最新一次。"""
    if not values:
        return [latest_experiment()]
    return [Path(PROJECT_ROOT, value) for value in values]


def main() -> None:
    """命令行入口：默认对最新一次实验生成单实验包。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, nargs="+",
        help="实验目录（相对项目根目录），可给多个；省略时取最新一次",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("out/mask_bundle"),
        help="产物根目录，相对项目根目录",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=_SPLIT_CHOICES,
        default=list(_SPLIT_CHOICES), help="要导出的划分，默认全部",
    )
    parser.add_argument(
        "--polygon-epsilon", type=float, default=1.0,
        help="预测掩码转多边形的简化容差，单位像素",
    )
    parser.add_argument("--no-zip", action="store_true", help="不生成 ZIP")
    args = parser.parse_args()

    run_dirs = _resolve_run_dirs(args.run)
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise SystemExit(f"实验目录不存在：{run_dir}")
    print("实验目录：" + "、".join(path.name for path in run_dirs))
    if len(run_dirs) == 1:
        result = build_mask_bundle(
            run_dirs[0], args.output, tuple(args.splits),
            polygon_epsilon_px=args.polygon_epsilon,
            make_zip=not args.no_zip,
        )
    else:
        result = build_mask_bundle_collection(
            run_dirs, args.output, tuple(args.splits),
            polygon_epsilon_px=args.polygon_epsilon,
            make_zip=not args.no_zip,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
