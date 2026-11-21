"""南北钢轨匹配真值的字段定义、读写与校验。

真值的唯一来源是逐对 association JSON。本模块不依赖具体算法，也不做绘图，
供标注工具、导出脚本与评测脚本共用。
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import TypedDict

SCHEMA_VERSION = "1.1"

PAIRED = "paired"
SOUTH_ONLY = "south_only"
NORTH_ONLY = "north_only"
UNCERTAIN = "uncertain"
GT_STATUSES = (PAIRED, SOUTH_ONLY, NORTH_ONLY, UNCERTAIN)

UNMATCHED_REASONS = ("out_of_view", "occluded", "missing_label", "unknown")
AMBIGUOUS_REASONS = (
    "insufficient_geometry", "similar_appearance", "occlusion",
    "order_conflict", "unknown",
)

MANIFEST_COLUMNS = (
    "pair_id", "split", "south_image_path", "north_image_path",
    "south_label_sha256", "north_label_sha256",
    "south_image_sha256", "north_image_sha256",
    "south_instance_count", "north_instance_count",
)

GT_COLUMNS = (
    "pair_id", "split", "gt_version", "gt_rail_id",
    "south_group_id", "south_shape_index", "north_group_id", "north_shape_index",
    "gt_status", "pair_shift", "review_status", "reviewer",
    "source_algorithm", "source_shared_id", "notes",
)

PROGRESS_COLUMNS = (
    "pair_id", "split", "status", "south_instances", "north_instances",
    "paired_count", "south_only_count", "north_only_count", "uncertain_count",
    "pair_shift", "updated_at", "annotator",
)

# 双侧置信水平到标准正态分位点的映射，只保留常用档位。
_Z_BY_LEVEL = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}

Instances = dict[str, list[dict[str, object]]]
Payload = dict[str, object]


class GtRow(TypedDict):
    """汇总 CSV 的一行：一条关系、一个单侧实例或一个不确定实例。"""

    pair_id: str
    split: str
    gt_version: str
    gt_rail_id: str
    south_group_id: object
    south_shape_index: object
    north_group_id: object
    north_shape_index: object
    gt_status: str
    pair_shift: object
    review_status: str
    reviewer: str
    source_algorithm: str
    source_shared_id: str
    notes: str


def file_sha256(path: Path) -> str:
    """返回文件的 SHA-256 十六进制摘要。

    Args:
        path: 待计算的文件路径。

    Returns:
        小写十六进制摘要字符串。
    """
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def instance_id(side: str, group_id: int) -> str:
    """把侧别与 group_id 合成界面和真值共用的实例编号，例如 S#3。"""
    return f"{side[0].upper()}#{group_id}"


def read_pair_annotation(path: Path) -> Payload:
    """读取逐对 association JSON。

    Args:
        path: JSON 路径。

    Returns:
        解析后的字典。

    Raises:
        OSError: 文件无法读取。
        ValueError: 内容不是 JSON 对象。
    """
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 的顶层必须是 JSON 对象")
    return payload


def write_pair_annotation(path: Path, payload: Payload) -> None:
    """以固定缩进和 LF 换行写入逐对 JSON，先写临时文件再替换。

    Args:
        path: 目标 JSON 路径。
        payload: 待写入的字典。

    Raises:
        OSError: 目录或文件不可写。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def validate_pair_annotation(
    payload: Payload, expected_ids: dict[str, list[str]]
) -> list[str]:
    """校验一对一、完整性、字段合法性与实例覆盖。

    Args:
        payload: 逐对 JSON 内容。
        expected_ids: 两侧应出现的实例编号，例如 {"south": ["S#1", ...]}。

    Returns:
        错误信息列表；为空表示通过。
    """
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version 应为 {SCHEMA_VERSION}")
    instances = payload.get("instances")
    if not isinstance(instances, dict):
        return ["缺少 instances"]
    for side in ("south", "north"):
        records = instances.get(side)
        if not isinstance(records, list):
            errors.append(f"instances.{side} 必须是列表")
            continue
        seen = [str(record.get("instance_id")) for record in records]
        if len(seen) != len(set(seen)):
            errors.append(f"{side} 实例编号重复")
        expected = set(expected_ids.get(side, []))
        if set(seen) != expected:
            missing = sorted(expected - set(seen))
            extra = sorted(set(seen) - expected)
            errors.append(f"{side} 实例集合不一致，缺失 {missing}，多出 {extra}")

    known = {
        side: {str(record.get("instance_id")) for record in instances.get(side, [])}
        for side in ("south", "north")
    }
    used: dict[str, str] = {}
    matches = payload.get("matches")
    if not isinstance(matches, list):
        return errors + ["缺少 matches"]
    rail_ids: set[str] = set()
    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            errors.append(f"matches[{index}] 必须是对象")
            continue
        south = str(match.get("south_instance_id"))
        north = str(match.get("north_instance_id"))
        rail_id = str(match.get("gt_rail_id"))
        if south not in known["south"]:
            errors.append(f"matches[{index}] 南侧实例 {south} 不存在")
        if north not in known["north"]:
            errors.append(f"matches[{index}] 北侧实例 {north} 不存在")
        for key, value in ((south, "南"), (north, "北")):
            if key in used:
                errors.append(f"matches[{index}] 的{value}侧实例 {key} 重复配对")
            used[key] = rail_id
        if rail_id in rail_ids:
            errors.append(f"gt_rail_id {rail_id} 重复")
        rail_ids.add(rail_id)

    for kind, camera_field in (("unmatched", True), ("ambiguous", False)):
        entries = payload.get(kind, [])
        if not isinstance(entries, list):
            errors.append(f"{kind} 必须是列表")
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"{kind}[{index}] 必须是对象")
                continue
            side = str(entry.get("camera"))
            if side not in ("south", "north"):
                errors.append(f"{kind}[{index}] 的 camera 无效")
                continue
            item = str(entry.get("instance_id"))
            if item not in known[side]:
                errors.append(f"{kind}[{index}] 的实例 {item} 不存在")
            if item in used:
                errors.append(f"{kind}[{index}] 的实例 {item} 同时出现在配对中")
            used[item] = kind
            if camera_field and str(entry.get("reason")) not in UNMATCHED_REASONS:
                errors.append(f"{kind}[{index}] 的 reason 不在约定枚举内")
            if not camera_field:
                reason = str(entry.get("reason"))
                if reason not in AMBIGUOUS_REASONS:
                    errors.append(f"{kind}[{index}] 的 reason 不在约定枚举内")
                candidates = entry.get("candidate_instance_ids", [])
                if not isinstance(candidates, list):
                    errors.append(f"{kind}[{index}] 的候选实例必须是列表")
                else:
                    other = "north" if side == "south" else "south"
                    for candidate in candidates:
                        if str(candidate) not in known[other]:
                            errors.append(
                                f"{kind}[{index}] 的候选 {candidate} 不存在"
                            )

    for side in ("south", "north"):
        for item in sorted(known[side] - set(used)):
            errors.append(f"{side} 实例 {item} 没有状态")
    if payload.get("pair_shift") is not None and not isinstance(
        payload.get("pair_shift"), int
    ):
        errors.append("pair_shift 必须是整数或 null")
    return errors


def build_gt_rows(payload: Payload) -> list[GtRow]:
    """把逐对 JSON 摊平成汇总 CSV 行，主键使用 group_id。

    Args:
        payload: 通过校验的逐对 JSON 内容。

    Returns:
        关系、单侧与不确定实例各一行。
    """
    instances = payload["instances"]
    index: dict[str, dict[str, dict[str, object]]] = {
        side: {
            str(record["instance_id"]): record
            for record in instances.get(side, [])
        }
        for side in ("south", "north")
    }
    review = payload.get("review") or {}
    base = {
        "pair_id": str(payload.get("pair_id", "")),
        "split": str(payload.get("split", "")),
        "gt_version": str(payload.get("gt_version", "")),
        "pair_shift": payload.get("pair_shift"),
        "review_status": str(review.get("status", "")),
        "reviewer": str(review.get("reviewer", "")),
    }
    rows: list[GtRow] = []
    for match in payload.get("matches", []):
        south = index["south"][str(match["south_instance_id"])]
        north = index["north"][str(match["north_instance_id"])]
        rows.append(GtRow(
            **base,
            gt_rail_id=str(match.get("gt_rail_id", "")),
            south_group_id=south.get("group_id"),
            south_shape_index=south.get("shape_index"),
            north_group_id=north.get("group_id"),
            north_shape_index=north.get("shape_index"),
            gt_status=PAIRED,
            source_algorithm=str(match.get("source_algorithm", "")),
            source_shared_id=str(match.get("source_shared_id", "")),
            notes=str(match.get("note", "")),
        ))
    for index, entry in enumerate(payload.get("unmatched", []), start=1):
        side = str(entry["camera"])
        other = "north" if side == "south" else "south"
        record = index_by_id(payload, side, str(entry["instance_id"]))
        rail_id = str(entry.get("gt_rail_id") or f"{base['pair_id']}_S{index:02d}")
        values: GtRow = GtRow(
            **base,
            gt_rail_id=rail_id,
            south_group_id=record.get("group_id") if side == "south" else None,
            south_shape_index=record.get("shape_index") if side == "south" else None,
            north_group_id=record.get("group_id") if other == "north" else None,
            north_shape_index=record.get("shape_index") if other == "north" else None,
            gt_status=SOUTH_ONLY if side == "south" else NORTH_ONLY,
            source_algorithm="",
            source_shared_id="",
            notes=f"{entry.get('reason', '')} {entry.get('note', '')}".strip(),
        )
        rows.append(values)
    for index, entry in enumerate(payload.get("ambiguous", []), start=1):
        side = str(entry["camera"])
        record = index_by_id(payload, side, str(entry["instance_id"]))
        candidates = entry.get("candidate_instance_ids", [])
        note = str(entry.get("note", ""))
        rail_id = str(entry.get("gt_rail_id") or f"{base['pair_id']}_U{index:02d}")
        rows.append(GtRow(
            **base,
            gt_rail_id=rail_id,
            south_group_id=record.get("group_id") if side == "south" else None,
            south_shape_index=record.get("shape_index") if side == "south" else None,
            north_group_id=record.get("group_id") if side == "north" else None,
            north_shape_index=record.get("shape_index") if side == "north" else None,
            gt_status=UNCERTAIN,
            source_algorithm="",
            source_shared_id="",
            notes=f"candidates={','.join(map(str, candidates))} {note}".strip(),
        ))
    return rows


def index_by_id(
    payload: Payload, side: str, identifier: str
) -> dict[str, object]:
    """按实例编号取回该实例的映射记录。

    Args:
        payload: 逐对 JSON 内容。
        side: `south` 或 `north`。
        identifier: 实例编号，例如 `S#3`。

    Returns:
        `instances` 中对应的记录。

    Raises:
        KeyError: 该侧不存在此实例编号。
    """
    for record in payload["instances"].get(side, []):
        if str(record.get("instance_id")) == identifier:
            return record
    raise KeyError(f"{side} 缺少实例 {identifier}")


def wilson_interval(
    successes: int, total: int, level: float = 0.95
) -> tuple[float, float]:
    """按 Wilson 方法给出比例的双侧置信区间。

    Args:
        successes: 成功计数。
        total: 总计数。
        level: 置信水平，取值在 0 与 1 之间。

    Returns:
        (下限, 上限)；total 为 0 时返回 (nan, nan)。
    """
    if total <= 0:
        return math.nan, math.nan
    z = _normal_quantile(0.5 + level / 2)
    ratio = successes / total
    denominator = 1 + z * z / total
    center = (ratio + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(ratio * (1 - ratio) / total + z * z / (4 * total ** 2))
    margin /= denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _normal_quantile(probability: float) -> float:
    """取双侧置信水平对应的标准正态分位点。

    Args:
        probability: 单侧累计概率，例如 0.975 对应双侧 95%。

    Returns:
        分位点数值。

    Raises:
        ValueError: 该置信水平不在预设档位内。
    """
    level = round(2 * probability - 1, 2)
    try:
        return _Z_BY_LEVEL[level]
    except KeyError as error:
        raise ValueError(f"不支持的置信水平 {level:.2f}") from error
