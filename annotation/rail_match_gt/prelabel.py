"""用三套匹配算法生成标注骨架。

骨架必须完整且单调：先按多数票为每根南侧实例选伙伴，再用最长递增子序列投影成
单调子集，最后按整数错移补齐剩余实例。证据等级记录支持该关系的算法个数，
竞争候选记录其他算法提出的不同伙伴。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from core_algorithm.common.preview import (
    LabelMeInstanceRef,
    load_labelme_view_details,
    measure_section,
    section_rows,
)
from core_algorithm.matching import (
    rail_matching,
    rail_matching_gmsa,
    rail_matching_mslc,
)
from data_preprocess.common.config import (
    PROJECT_ROOT,
    config_path as resolve_config,
    load_yaml_mapping,
)
from utils.match_gt import instance_id

_DEFAULT_CONFIG = "config/eval/rail_match_gt.yaml"
_OUTPUT_ROOT = PROJECT_ROOT / "out" / "rail_match_label" / "prelabel"
_MAIN_SECTION = 1
_DEFAULT_ALGORITHMS = ("rail_matching", "rail_matching_mslc", "rail_matching_gmsa")
_DEFAULT_TIE_BREAK = "rail_matching_mslc"


@dataclass(frozen=True)
class SideView:
    """一侧原图、逐实例 Mask 与按横向位置排好的实例顺序。"""

    key: str
    image_path: str
    image: NDArray[np.uint8]
    masks: NDArray[np.bool_]
    references: tuple[LabelMeInstanceRef, ...]
    label_sha256: str
    order: tuple[int, ...]
    centers: tuple[tuple[float, float] | None, ...]

    @property
    def identifiers(self) -> tuple[str, ...]:
        """按显示顺序返回实例编号。"""
        labels: list[str] = []
        for mask_index in self.order:
            reference = self.references[mask_index]
            if reference.group_id is None:
                labels.append(f"{self.key[0].upper()}G{reference.shape_index}")
            else:
                labels.append(instance_id(self.key, int(reference.group_id)))
        return tuple(labels)

    @property
    def mask_of_order(self) -> tuple[int, ...]:
        """按显示顺序返回输入 Mask 索引。"""
        return self.order


@dataclass(frozen=True)
class PairView:
    """一对图的两侧视图与来源信息。"""

    row: dict[str, object]
    south: SideView
    north: SideView

    @property
    def pair_id(self) -> str:
        """采集事件编号。"""
        return str(self.row["pair_id"])

    @property
    def split(self) -> str:
        """所属划分。"""
        return str(self.row["split"])


def load_pair(row: dict[str, object]) -> PairView:
    """按清单行读取两侧原图与逐实例 Mask，并排好显示顺序。

    Args:
        row: 清单中的一行，含两侧项目相对路径。

    Returns:
        两侧视图。

    Raises:
        OSError: 原图或标注不可读。
        ValueError: 标注与图像尺寸不符。
    """
    south = _build_side("south", str(row["south_image_path"]))
    north = _build_side("north", str(row["north_image_path"]))
    return PairView(row, south, north)


def _build_side(key: str, image_path: str) -> SideView:
    """读取单侧视图，并按 50% 截面中心排序。"""
    view = load_labelme_view_details(PROJECT_ROOT / image_path)
    height, width = view.image.shape[:2]
    rows = section_rows(height)
    main_row = rows[_MAIN_SECTION]
    centers: list[tuple[float, float] | None] = []
    positions: list[float] = []
    for mask in view.masks:
        section = measure_section(mask[main_row])
        if section.reason is not None:
            centers.append(None)
            positions.append(float("inf"))
            continue
        coordinate = section.center_x / (width - 1)
        position = 1 - coordinate if key == "south" else coordinate
        centers.append((section.center_x, float(main_row)))
        positions.append(position)
    order = tuple(sorted(range(len(view.masks)), key=lambda index: positions[index]))
    return SideView(
        key=key,
        image_path=image_path,
        image=view.image,
        masks=view.masks,
        references=view.references,
        label_sha256=view.label_sha256,
        order=order,
        centers=tuple(centers[index] for index in order),
    )


def collect_votes(
    pair: PairView, algorithms: tuple[str, ...] = _DEFAULT_ALGORITHMS
) -> dict[tuple[int, int], set[str]]:
    """运行指定算法，把配对结果换算成显示顺序上的投票。

    Args:
        pair: 一对图的两侧视图。
        algorithms: 参与投票的算法名。

    Returns:
        键为 (南侧顺序, 北侧顺序)，值为支持该关系的算法名集合。
    """
    south_position = {mask: order for order, mask in enumerate(pair.south.order)}
    north_position = {mask: order for order, mask in enumerate(pair.north.order)}
    votes: dict[tuple[int, int], set[str]] = {}
    for name in algorithms:
        result = _ALGORITHM_RUNNERS[name](pair)
        for match in result["matches"]:
            south_mask = int(match["south"]["mask_index"])
            north_mask = int(match["north"]["mask_index"])
            key = (south_position[south_mask], north_position[north_mask])
            votes.setdefault(key, set()).add(name)
    return votes


def _run_rail_matching(pair: PairView) -> dict[str, object]:
    return rail_matching.match_south_north_rail(
        pair.south.image_path, pair.north.image_path,
        pair.south.masks, pair.north.masks,
    )


def _run_mslc(pair: PairView) -> dict[str, object]:
    return rail_matching_mslc.match_south_north_rail_mslc(
        pair.south.image_path, pair.north.image_path,
        pair.south.masks, pair.north.masks,
    )


def _run_gmsa(pair: PairView) -> dict[str, object]:
    return rail_matching_gmsa.match_south_north_rail_gmsa_masks(
        pair.south.image_path, pair.north.image_path,
        pair.south.masks, pair.north.masks, return_diagnostics=False,
    )


_ALGORITHM_RUNNERS: dict[str, Callable[[PairView], dict[str, object]]] = {
    "rail_matching": _run_rail_matching,
    "rail_matching_mslc": _run_mslc,
    "rail_matching_gmsa": _run_gmsa,
}


def merge_alignments(
    south_count: int, north_count: int,
    votes: dict[tuple[int, int], set[str]], tie_break: str = _DEFAULT_TIE_BREAK,
) -> dict[int, int]:
    """把三套算法的部分配对合并成完整且单调的骨架。

    步骤：多数票选伙伴 → 最长递增子序列投影成单调子集 → 按中位错移补齐。

    Args:
        south_count: 南侧实例数。
        north_count: 北侧实例数。
        votes: 关系投票。
        tie_break: 平票时优先采用的算法。

    Returns:
        键为南侧顺序、值为北侧顺序的单调配对。
    """
    chosen: dict[int, int] = {}
    for south in range(south_count):
        candidates = [
            (north, names) for (side, north), names in votes.items() if side == south
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                -len(item[1]), 0 if tie_break in item[1] else 1, item[0]
            )
        )
        chosen[south] = candidates[0][0]
    anchors = _monotone_subset(chosen)
    pairs = dict(anchors)
    used_north = set(pairs.values())
    offset = (
        round(median([north - south for south, north in anchors.items()]))
        if anchors else 0
    )
    for south in range(south_count):
        if south in pairs:
            continue
        north = south + offset
        if not 0 <= north < north_count or north in used_north:
            continue
        if any(
            (other_south < south and other_north > north)
            or (other_south > south and other_north < north)
            for other_south, other_north in pairs.items()
        ):
            continue
        pairs[south] = north
        used_north.add(north)
    return dict(sorted(pairs.items()))


def _monotone_subset(chosen: dict[int, int]) -> dict[int, int]:
    """用最长递增子序列保留单调关系。"""
    items = sorted(chosen.items())
    if not items:
        return {}
    lengths = [1] * len(items)
    previous = [-1] * len(items)
    for index in range(len(items)):
        for earlier in range(index):
            north_before = items[earlier][1]
            if north_before < items[index][1] and lengths[earlier] + 1 > lengths[index]:
                lengths[index] = lengths[earlier] + 1
                previous[index] = earlier
    end = max(range(len(items)), key=lambda index: lengths[index])
    chain: list[int] = []
    while end >= 0:
        chain.append(end)
        end = previous[end]
    return {items[index][0]: items[index][1] for index in reversed(chain)}


def build_prelabel(pair: PairView, settings: dict[str, object]) -> dict[str, object]:
    """生成一对图的预标注骨架与证据记录。

    Args:
        pair: 一对图的两侧视图。
        settings: 真值配置，读取算法名与平票优先项。

    Returns:
        可直接写入 JSON 的预标注字典。
    """
    algorithms = tuple(settings.get("prelabel_algorithms", _DEFAULT_ALGORITHMS))
    tie_break = str(settings.get("prelabel_tie_break", _DEFAULT_TIE_BREAK))
    unknown = [name for name in algorithms if name not in _ALGORITHM_RUNNERS]
    if unknown:
        raise ValueError(f"未实现的预标注算法 {', '.join(unknown)}")
    votes = collect_votes(pair, algorithms)
    pairs = merge_alignments(
        len(pair.south.order), len(pair.north.order), votes, tie_break
    )
    competing: dict[str, list[dict[str, object]]] = {}
    for (south, north), names in votes.items():
        if pairs.get(south) == north:
            continue
        competing.setdefault(str(south), []).append({
            "order": north,
            "support": sorted(names),
        })
    return {
        "pair_id": pair.pair_id,
        "split": pair.split,
        "algorithms": list(algorithms),
        "tie_break": tie_break,
        "south": {
            "instance_ids": list(pair.south.identifiers),
            "mask_index": list(pair.south.order),
        },
        "north": {
            "instance_ids": list(pair.north.identifiers),
            "mask_index": list(pair.north.order),
        },
        "pairs": [
            {
                "south": south,
                "north": north,
                "support": sorted(votes.get((south, north), set())),
                "level": len(votes.get((south, north), set())),
            }
            for south, north in pairs.items()
        ],
        "competing": competing,
    }


def load_settings(config_path: str | Path = _DEFAULT_CONFIG) -> dict[str, object]:
    """读取真值配置。"""
    return load_yaml_mapping(resolve_config(config_path))


def _main() -> None:
    """命令行入口：为清单内全部图对生成预标注。"""
    settings = load_settings()
    manifest = PROJECT_ROOT / str(settings["root"]) / "manifest.csv"
    rows = _read_manifest(manifest)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = _OUTPUT_ROOT / timestamp
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        pair = load_pair(row)
        payload = build_prelabel(pair, settings)
        destination = output / f"{pair.pair_id}_prelabel.json"
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        levels = [int(entry["level"]) for entry in payload["pairs"]]
        print(
            f"{pair.pair_id} {pair.split} 骨架 {len(levels)} 条"
            f"（3/3 {levels.count(3)}，2/3 {levels.count(2)}，1/3 {levels.count(1)}）",
            flush=True,
        )
    print(f"预标注已写入 {output}", flush=True)


def _read_manifest(path: Path) -> list[dict[str, object]]:
    """读取清单 CSV。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


if __name__ == "__main__":
    _main()
