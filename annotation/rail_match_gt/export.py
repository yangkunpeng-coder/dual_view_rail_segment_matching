"""把人工匹配真值导出为评测可读的扁平表。

产物写在 `<root>/export/` 下：`manual_match_ground_truth.csv` 一行一条关系，
`gt_progress.csv` 一行一对图，`gt_summary.md` 汇总进度与异常。

`skipped` 的图对整对排除；`in_progress`、缺失或校验失败的图对也会被排除，
并列入汇总的异常清单，不会静默计入指标。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from annotation.rail_match_gt.prelabel import load_settings
from data_preprocess.common.config import PROJECT_ROOT
from utils.match_gt import (
    GT_COLUMNS,
    PROGRESS_COLUMNS,
    SCHEMA_VERSION,
    build_gt_rows,
    read_pair_annotation,
    validate_pair_annotation,
)

_DEFAULT_CONFIG = "config/eval/rail_match_gt.yaml"
_FINISHED = ("done", "skipped")


def export_ground_truth(
    config_path: str | Path = _DEFAULT_CONFIG,
) -> Path:
    """读取全部逐对 JSON，写出真值 CSV、进度表与汇总。

    Args:
        config_path: 真值配置路径。

    Returns:
        导出目录的绝对路径。

    Raises:
        OSError: 清单或真值文件不可读。
        ValueError: 清单缺失或没有任何可用图对。
    """
    settings = load_settings(config_path)
    root = PROJECT_ROOT / str(settings["root"])
    rows = _read_manifest(root / "manifest.csv")
    gt_version = str(settings.get("gt_version", "v1"))
    export_dir = root / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    relation_rows: list[dict[str, object]] = []
    progress_rows: list[dict[str, object]] = []
    problems: list[str] = []
    skipped: list[str] = []
    for row in rows:
        pair_id = str(row["pair_id"])
        split = str(row["split"])
        path = root / split / f"{pair_id}_association.json"
        if not path.is_file():
            problems.append(f"{pair_id}（{split}）缺少真值文件，按未完成排除")
            progress_rows.append(_progress_row(row, "未开始"))
            continue
        payload = read_pair_annotation(path)
        status = str((payload.get("review") or {}).get("status", ""))
        if status == "skipped":
            skipped.append(f"{pair_id}（{split}）")
            progress_rows.append(_progress_row(row, "skipped", payload))
            continue
        if status != "done":
            problems.append(f"{pair_id}（{split}）状态为 {status}，按未完成排除")
            progress_rows.append(_progress_row(row, status or "未知", payload))
            continue
        if payload.get("schema_version") != SCHEMA_VERSION:
            problems.append(
                f"{pair_id}（{split}）schema 为 {payload.get('schema_version')}，"
                "与当前版本不一致"
            )
            progress_rows.append(_progress_row(row, "旧版本", payload))
            continue
        errors = _validate(payload)
        if errors:
            problems.extend(f"{pair_id}（{split}）{text}" for text in errors[:3])
            progress_rows.append(_progress_row(row, "校验失败", payload))
            continue
        crossing = _crossing_pairs(payload)
        if crossing:
            problems.append(
                f"{pair_id}（{split}）左右顺序交叉：{crossing}（建议复核）"
            )
        relation_rows.extend(build_gt_rows(payload))
        progress_rows.append(_progress_row(row, "done", payload))

    if not relation_rows:
        raise ValueError("没有任何可用的已完成图对，无法导出")
    _write_csv(
        export_dir / "manual_match_ground_truth.csv", GT_COLUMNS, relation_rows
    )
    _write_csv(export_dir / "gt_progress.csv", PROGRESS_COLUMNS, progress_rows)
    summary_path = export_dir / "gt_summary.md"
    summary_path.write_text(
        _summary_text(rows, relation_rows, progress_rows, skipped, problems,
                      gt_version),
        encoding="utf-8", newline="\n",
    )
    print(
        f"已导出 {len(relation_rows)} 条真值关系到 {export_dir}；"
        f"可用图对 {sum(1 for r in progress_rows if r['status'] == 'done')} 对，"
        f"跳过 {len(skipped)} 对，异常 {len(problems)} 条",
        flush=True,
    )
    return export_dir


def _read_manifest(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"缺少清单 {path}，请先运行清单脚本")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _validate(payload: dict[str, object]) -> list[str]:
    expected = {
        side: [
            str(record.get("instance_id"))
            for record in (payload.get("instances") or {}).get(side, [])
        ]
        for side in ("south", "north")
    }
    return [
        error
        for error in validate_pair_annotation(payload, expected)
        if not error.endswith("没有状态")
    ]


def _crossing_pairs(payload: dict[str, object]) -> str:
    """返回左右顺序交叉的实例对描述，空串表示单调。"""
    order = {
        side: {
            str(record.get("instance_id")): index
            for index, record in enumerate(
                (payload.get("instances") or {}).get(side, [])
            )
        }
        for side in ("south", "north")
    }
    pairs = sorted(
        (order["south"][str(match["south_instance_id"])],
         order["north"][str(match["north_instance_id"])],
         str(match["south_instance_id"]), str(match["north_instance_id"]))
        for match in payload.get("matches", [])
    )
    problems = [
        f"{first[2]} 与 {second[2]}"
        for first in pairs for second in pairs
        if first[0] < second[0] and first[1] > second[1]
    ]
    return "；".join(problems[:3])


def _progress_row(
    row: dict[str, object], status: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """组装进度表的一行；没有 payload 时只记状态。"""
    south_count = len(((payload or {}).get("instances") or {}).get("south", []))
    north_count = len(((payload or {}).get("instances") or {}).get("north", []))
    matches = list((payload or {}).get("matches", []))
    unmatched = list((payload or {}).get("unmatched", []))
    ambiguous = list((payload or {}).get("ambiguous", []))
    south_only = sum(1 for item in unmatched if item.get("camera") == "south")
    north_only = len(unmatched) - south_only
    review = (payload or {}).get("review") or {}
    return {
        "pair_id": str(row["pair_id"]),
        "split": str(row["split"]),
        "status": status,
        "south_instances": south_count,
        "north_instances": north_count,
        "paired_count": len(matches) if payload else "",
        "south_only_count": south_only if payload else "",
        "north_only_count": north_only if payload else "",
        "uncertain_count": len(ambiguous) if payload else "",
        "pair_shift": (payload or {}).get("pair_shift", ""),
        "updated_at": review.get("reviewed_at", ""),
        "annotator": review.get("annotator", ""),
    }


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(columns), lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _summary_text(
    manifest: list[dict[str, object]],
    relations: list[dict[str, object]],
    progress: list[dict[str, object]],
    skipped: list[str],
    problems: list[str],
    gt_version: str,
) -> str:
    totals: dict[str, int] = {}
    for item in relations:
        status = str(item["gt_status"])
        totals[status] = totals.get(status, 0) + 1
    done = [item for item in progress if item["status"] == "done"]
    lines = [
        "# 钢轨匹配真值汇总",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"真值版本：{gt_version}",
        "",
        "## 图对",
        "",
        f"- 清单图对：{len(manifest)}",
        f"- 已确认：{len(done)}",
        f"- 无法标注（整对排除）：{len(skipped)}",
        f"- 未完成或异常（排除）：{len(progress) - len(done) - len(skipped)}",
        "",
        "## 关系",
        "",
    ]
    for status in ("paired", "south_only", "north_only", "uncertain"):
        lines.append(f"- {status}：{totals.get(status, 0)}")
    lines.append(f"- 合计：{len(relations)}")
    if skipped:
        lines += ["", "## 无法标注的图对", ""]
        lines += [f"- {item}" for item in skipped]
    if problems:
        lines += ["", "## 异常与提示", ""]
        lines += [f"- {item}" for item in problems]
    lines += [
        "",
        "## 评测口径",
        "",
        "- 只有 `status = done` 的图对参与指标；",
        "- `skipped` 与未完成图对整对排除，只在进度表里保留记录；",
        "- `uncertain` 的关系不计入指标的分子分母，单独统计数量。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """命令行入口：导出真值。"""
    export_ground_truth()


if __name__ == "__main__":
    main()
