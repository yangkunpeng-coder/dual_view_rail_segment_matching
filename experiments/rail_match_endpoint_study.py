"""端部证据可分性实验：端部位置/可见长度能否选出真值整列错移。

结论（2026-11-12，33 对真值）：**不可分，端部证据在本数据上为零信息**。

- 钢轨掩码几乎都是满高：南侧 top 恒 0.00、bottom 恒 1.00、长度恒 1.00；
  北侧只是相机取景的固定偏置（top≈0.10、长度≈0.89），不是逐轨变化；
- 端部相关性在 31/33 图对上无定义（端部方差为 0）；绝对/居中评分命中率 0.152，
  远低于"恒选 k=0"的 0.939 基线，且 26–27 个图对会把正确的 0 错选成非 0；
- 5 个难点图对（P0097/P0091/P0099/P0104/P0113）三种评分全部选错；
- 真值关系本身也不是恒定错移：每对"北侧 rank − 南侧 rank"有 8–25 种不同取值，
  是随轨进出占用带而变化的单调映射，不是整列平移。

因此"整列错移 k 裁决"对剩余误差不成立；V5（图匹配）的端部证据方向到此关闭。
"""

from __future__ import annotations

import statistics

import numpy as np

from core_algorithm.matching import rail_matching_mslc as mslc
from experiments import rail_match_eval as ev
from experiments.rail_match_gt_eval import (
    _GT_CONFIG, _GT_MANIFEST, _apply_exclusions, _excluded_pairs, _load_truth,
)

SHIFT_RANGE = range(-3, 4)
SCORE_NAMES = ("absolute", "centered", "correlation")
HARD_PAIRS = ("P0097", "P0091", "P0099", "P0104", "P0113")


def _rank_by_x(masks: np.ndarray) -> dict[int, int]:
    """按掩码像素质心横坐标从左到右给实例编号。"""
    centers = []
    for index, mask in enumerate(masks):
        columns = np.flatnonzero(mask.any(axis=0))
        center = float(np.median(columns)) if len(columns) else float("inf")
        centers.append((center, index))
    centers.sort()
    return {index: rank for rank, (_, index) in enumerate(centers)}


def _endpoints(masks: np.ndarray) -> dict[int, tuple[float, float, float]]:
    """返回每个实例的相对上端、下端与可见长度（按图高归一）。"""
    height = masks.shape[1]
    spans = {}
    for index, mask in enumerate(masks):
        rows = np.flatnonzero(mask.any(axis=1))
        if not len(rows):
            continue
        top = rows[0] / height
        bottom = rows[-1] / height
        spans[index] = (top, bottom, bottom - top)
    return spans


def _deltas(
    truth, south_rank: dict[int, int], north_rank: dict[int, int]
) -> list[int]:
    """真值关系在两侧顺序号上的差值。"""
    return [
        north_rank[north] - south_rank[south]
        for south, north in truth.paired
        if south in south_rank and north in north_rank
    ]


def _scores(
    south_span: dict[int, tuple[float, float, float]],
    north_span: dict[int, tuple[float, float, float]],
    south_rank: dict[int, int],
    north_rank: dict[int, int],
    shift: int,
) -> dict[str, float]:
    """给定错移，计算三种端部一致性评分（越小越好；无效项不返回）。"""
    north_by_rank = {rank: index for index, rank in north_rank.items()}
    south_by_rank = {rank: index for index, rank in south_rank.items()}
    pairs = []
    for rank, index in south_by_rank.items():
        partner = north_by_rank.get(rank + shift)
        if partner is None:
            continue
        if index not in south_span or partner not in north_span:
            continue
        pairs.append((south_span[index], north_span[partner]))
    if len(pairs) < 4:
        return {}
    south_tops = np.array([span[0] for span, _ in pairs])
    north_tops = np.array([span[0] for _, span in pairs])
    south_bottoms = np.array([span[1] for span, _ in pairs])
    north_bottoms = np.array([span[1] for _, span in pairs])
    south_lengths = np.array([span[2] for span, _ in pairs])
    north_lengths = np.array([span[2] for _, span in pairs])

    result = {
        "absolute": float(
            np.mean(np.abs(south_tops - north_tops))
            + np.mean(np.abs(south_bottoms - north_bottoms))
            + np.mean(np.abs(south_lengths - north_lengths))
        ) / 3,
        "centered": float(
            np.mean(np.abs(
                (south_tops - np.median(south_tops))
                - (north_tops - np.median(north_tops))
            ))
            + np.mean(np.abs(
                (south_bottoms - np.median(south_bottoms))
                - (north_bottoms - np.median(north_bottoms))
            ))
            + np.mean(np.abs(
                (south_lengths - np.median(south_lengths))
                - (north_lengths - np.median(north_lengths))
            ))
        ) / 3,
    }
    # 端部全部相同（都是满高）时相关系数无定义，不能当成"支持 k=0"。
    if np.std(south_tops) > 1e-9 and np.std(north_tops) > 1e-9 \
            and np.std(south_bottoms) > 1e-9 and np.std(north_bottoms) > 1e-9:
        result["correlation"] = 1.0 - float(
            np.corrcoef(south_tops, north_tops)[0, 1]
            + np.corrcoef(south_bottoms, north_bottoms)[0, 1]
        ) / 2
    return result


def _pick(score_by_shift: dict[int, dict[str, float]], name: str) -> int | None:
    """取该评分最小的错移；并列取 |k| 更小者。"""
    values = [
        (scores[name], abs(shift), shift)
        for shift, scores in score_by_shift.items() if name in scores
    ]
    if not values:
        return None
    return min(values)[2]


def main() -> None:
    """输出三档评分的 k 选择结果、常数基线，以及局部漂移对照。"""
    truths = _load_truth()
    truths, _, _ = _apply_exclusions(truths, _excluded_pairs(_GT_CONFIG))
    pairs = {pair.pair_id: pair for pair in ev._read_pairs(_GT_MANIFEST)}

    rows = []
    for pair_id, truth in sorted(truths.items()):
        data = ev._load_pair(pairs[pair_id])
        south_rank = _rank_by_x(data.south_masks)
        north_rank = _rank_by_x(data.north_masks)
        deltas = _deltas(truth, south_rank, north_rank)
        if not deltas:
            continue
        true_shift = int(statistics.median(deltas))
        score_by_shift = {
            shift: _scores(
                _endpoints(data.south_masks), _endpoints(data.north_masks),
                south_rank, north_rank, shift,
            )
            for shift in SHIFT_RANGE
        }
        result = mslc.match_south_north_rail_mslc(
            data.south_path, data.north_path,
            data.south_masks, data.north_masks,
        )
        predicted = {
            (m["south"]["mask_index"], m["north"]["mask_index"])
            for m in result["matches"]
        }
        swapped = sum(
            1 for key in predicted - truth.paired
            if key[0] in {s for s, _ in truth.paired}
        )
        rows.append({
            "pair": pair_id, "split": truth.split, "gold": len(truth.paired),
            "true_k": true_shift, "deltas": deltas,
            "drift": len(set(deltas)), "swapped": swapped,
            "picks": {name: _pick(score_by_shift, name) for name in SCORE_NAMES},
        })

    print("图对  划分  真值k 漂移 swap  绝对  居中  相关  gold")
    for row in rows:
        def mark(name: str) -> str:
            value = row["picks"][name]
            if value is None:
                return "  - "
            flag = "✔" if value == row["true_k"] else " "
            return f"{flag}{value:+d}"

        print("%s %-5s %+3d %3d %4d  %5s %5s %5s  %d" % (
            row["pair"], row["split"], row["true_k"], row["drift"],
            row["swapped"], mark("absolute"), mark("centered"),
            mark("correlation"), row["gold"]))

    total = len(rows)
    constant_zero = sum(1 for row in rows if row["true_k"] == 0) / total
    print("\n常数基线（恒选 k=0）: %.3f (%d/%d 真值 k=0)" % (
        constant_zero, sum(1 for row in rows if row["true_k"] == 0), total))
    for name in SCORE_NAMES:
        usable = [row for row in rows if row["picks"][name] is not None]
        hits = sum(1 for row in usable if row["picks"][name] == row["true_k"])
        false_shift = sum(
            1 for row in usable
            if row["true_k"] == 0 and row["picks"][name] not in (0, None)
        )
        print("%-12s 可用 %2d 对，命中 %2d = %.3f，真值0却选非0 %2d" % (
            name, len(usable), hits, hits / max(1, len(usable)), false_shift))

    print("\n难点图对（真值 k 与三种评分）：")
    for row in rows:
        if row["pair"] in HARD_PAIRS:
            print("  %s 真值 k=%+d 漂移=%d MSLC swap=%d 绝对=%s 居中=%s 相关=%s" % (
                row["pair"], row["true_k"], row["drift"], row["swapped"],
                row["picks"]["absolute"], row["picks"]["centered"],
                row["picks"]["correlation"]))
    drift_distribution = {k: sum(1 for row in rows if row["drift"] == k)
                          for k in sorted({row["drift"] for row in rows})}
    print("\n真值差值集合大小分布:", drift_distribution)


if __name__ == "__main__":
    main()
