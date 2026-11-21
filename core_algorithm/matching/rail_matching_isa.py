"""ISA-Rail：用整数错移假设与精确裕度建立南北钢轨的一对一关联。

ISA-Rail（Integer-Shift Alignment，整数错移对齐）把"整列错配"看成唯一的
离散自由度：在仿射映射 ``u_N ≈ a·u_S + b`` 下，单调对齐只能被确定到整数个
轨位的平移，因此不需要在连续空间里搜索映射，而是显式枚举错移 ``k``。

算法分为四步：

1. 测量：在 10%-95% 高度取 18 个采样行，逐实例追踪主交段，得到归一化横向
   位置、宽度、可见区间、排序质量与相邻间距指纹（``rail_measurement``）。
2. 假设：种子来自整数错移 ``k`` 与两侧稳定大间隙；每个种子先用相邻配对斜率
   的中位数给出 ``(a, b)``（个别错配只影响两侧斜率），再迭代两轮"单调动态
   规划 → 加权重估"，候选映射按 ``(a, b)`` 分桶去重。
3. 求解：在固定映射下做 ``O(mn)`` 的单调动态规划，允许跳过任意实例但不允许
   交叉；跳过代价按"映射后是否落在对侧占用带内"区分真漏检与视野截断。
4. 判定：竞争者裕度要求"换一个伙伴"的完整路径代价明显更差；假设裕度要求
   "换一个错移"的完整路径代价明显更差，两者都不满足时输出歧义而不是硬判。

与同级的 V1、MSLC、GMSA 相比，ISA-Rail 不做跨度曲线的 DTW，也不依赖映射
投票：代价是 O(K × mn)，判定由两个可解释的裕度给出。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite, sqrt
from pathlib import Path
import sys
from time import perf_counter
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import Image

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core_algorithm.common.match_result import (
    RailMatch,
    RailMatchingResult,
    UnmatchedRail,
    make_unmatched_rail,
)
from core_algorithm.common.rail_measurement import (
    FloatArray,
    MeasuredRail,
    RailMeasurementConfig,
    SideMeasurement,
    measure_side,
)
from data_preprocess.common.config import PROJECT_ROOT
from utils.instance_masks import as_mask_stack

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.engine.results import Masks

_EPSILON = 1e-9
_VERSION = "ISA-Rail V1"
_CANDIDATE_LIMIT = 300


class ISAMatchingResult(RailMatchingResult):
    """在项目结果协议上增加待确认、歧义关系与诊断。"""

    tentative_matches: list[dict]
    ambiguous_candidates: list[dict]
    diagnostics: dict | None


@dataclass(frozen=True)
class ISARailConfig:
    """假设枚举、代价权重与判定门限。位置与代价均已归一化。"""

    measurement: RailMeasurementConfig = field(default_factory=RailMeasurementConfig)
    shift_limit: int = 4
    max_hypotheses: int = 12
    refine_iterations: int = 2
    position_weight: float = 0.45
    spacing_weight: float = 0.30
    width_weight: float = 0.10
    quality_weight: float = 0.15
    max_position_tolerance: float = 0.025
    gap_tolerance_fraction: float = 0.48
    spacing_log_cap: float = 1.5
    width_log_cap: float = 1.0
    skip_base: float = 0.30
    skip_inside_band: float = 0.80
    min_pair_margin: float = 0.03
    min_relative_margin: float = 0.25
    min_hypothesis_margin: float = 0.10
    min_confidence_pairs: int = 3

    def __post_init__(self) -> None:
        integers = (
            (self.shift_limit, 1, 12),
            (self.max_hypotheses, 1, 64),
            (self.refine_iterations, 0, 8),
            (self.min_confidence_pairs, 1, 4096),
        )
        if any(type(value) is not int or not low <= value <= high
               for value, low, high in integers):
            raise ValueError("ISA-Rail 的整数配置或范围无效")
        weights = (
            self.position_weight, self.spacing_weight, self.width_weight,
            self.quality_weight, self.max_position_tolerance,
            self.gap_tolerance_fraction, self.spacing_log_cap, self.width_log_cap,
            self.skip_base, self.skip_inside_band, self.min_pair_margin,
            self.min_relative_margin, self.min_hypothesis_margin,
        )
        if any(not isfinite(value) or value < 0 for value in weights):
            raise ValueError("ISA-Rail 的代价、门限须为有限非负数")
        if not 0 < self.gap_tolerance_fraction <= 1:
            raise ValueError("gap_tolerance_fraction 须在 (0, 1] 内")


@dataclass(frozen=True)
class _Hypothesis:
    """一套横向映射及其下的最优单调路径。"""

    shift_k: int | None
    scale: float
    offset: float
    cost: float
    pairs: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class _Solve:
    """一次单调动态规划的最优值表与前后缀代价。"""

    prefix: FloatArray
    suffix: FloatArray
    cost: float
    pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Costs:
    """节点代价矩阵与两侧跳过代价。"""

    matrix: FloatArray
    skip_south: FloatArray
    skip_north: FloatArray


def _median_u(rail: MeasuredRail) -> float:
    """实例的横向位置取有效采样行的中位数。"""
    valid = rail.u[np.isfinite(rail.u)]
    return float(np.median(valid)) if len(valid) else float("nan")


def _neighbor_gaps(side: SideMeasurement) -> FloatArray:
    """每个实例到左右邻轨的最小归一化轨距，用于位置门限与间距项。"""
    positions = np.asarray([_median_u(rail) for rail in side.rails], dtype=np.float64)
    if len(positions) < 2:
        return np.full(len(positions), np.inf)
    gaps = np.abs(np.diff(positions))
    return np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf])


def _rail_gap_logs(side: SideMeasurement, index: int) -> tuple[FloatArray, FloatArray]:
    """实例左右相邻间距的对数指纹，缺失侧返回全 NaN。"""
    rows = side.log_gaps.shape[1]
    left = np.full(rows, np.nan)
    right = np.full(rows, np.nan)
    if index > 0:
        left = side.log_gaps[index - 1]
    if index < side.count - 1:
        right = side.log_gaps[index]
    return left, right


def _spacing_cost(
    south: SideMeasurement, north: SideMeasurement,
    south_index: int, north_index: int, config: ISARailConfig,
) -> float | None:
    """比较两侧左右相邻间距的对数指纹，取可用项平均。"""
    s_left, s_right = _rail_gap_logs(south, south_index)
    n_left, n_right = _rail_gap_logs(north, north_index)
    terms: list[float] = []
    for first, second in ((s_left, n_left), (s_right, n_right)):
        valid = np.isfinite(first) & np.isfinite(second)
        if np.any(valid):
            difference = np.abs(first[valid] - second[valid])
            terms.append(float(np.mean(np.minimum(difference, config.spacing_log_cap))))
    if not terms:
        return None
    return float(np.mean(terms))


def _width_cost(
    south_rail: MeasuredRail, north_rail: MeasuredRail, config: ISARailConfig
) -> float | None:
    """比较宽度相对图内中位的偏差，作为弱深度证据。"""
    south_width = np.median(south_rail.widths[np.isfinite(south_rail.widths)])
    north_width = np.median(north_rail.widths[np.isfinite(north_rail.widths)])
    if not (np.isfinite(south_width) and np.isfinite(north_width)):
        return None
    if south_width <= 0 or north_width <= 0:
        return None
    ratio = np.log(south_width / north_width)
    return float(min(abs(ratio), config.width_log_cap))


def _skip_costs(
    side: SideMeasurement, other: SideMeasurement, scale: float, offset: float,
    config: ISARailConfig,
) -> FloatArray:
    """按"映射后是否落在对侧占用带内"区分边界截断与真漏检。"""
    costs = np.zeros(side.count, dtype=np.float64)
    low, high = other.band
    for index, rail in enumerate(side.rails):
        position = _median_u(rail)
        mapped = scale * position + offset
        inside = bool(np.isfinite(low) and low <= mapped <= high)
        penalty = config.skip_base + (config.skip_inside_band if inside else 0.0)
        costs[index] = penalty * (1 - 0.5 * rail.quality)
    return costs


def _cost_matrix(
    south: SideMeasurement, north: SideMeasurement,
    scale: float, offset: float, config: ISARailConfig,
) -> _Costs:
    """按给定映射计算节点代价与两侧跳过代价。"""
    matrix = np.full((south.count, north.count), np.inf, dtype=np.float64)
    if not south.count or not north.count:
        return _Costs(matrix, np.zeros(south.count), np.zeros(north.count))
    if not isfinite(scale) or scale <= _EPSILON:
        return _Costs(
            matrix,
            _skip_costs(south, north, 1.0, 0.0, config),
            _skip_costs(north, south, 1.0, 0.0, config),
        )
    south_gaps = _neighbor_gaps(south)
    north_gaps = _neighbor_gaps(north)
    rail_south = [_median_u(rail) for rail in south.rails]
    rail_north = [_median_u(rail) for rail in north.rails]
    for i, first in enumerate(south.rails):
        if not first.geometry_valid or not isfinite(rail_south[i]):
            continue
        for j, second in enumerate(north.rails):
            if not second.geometry_valid or not isfinite(rail_north[j]):
                continue
            residual = abs(rail_north[j] - (scale * rail_south[i] + offset))
            local_gap = min(north_gaps[j], scale * south_gaps[i])
            tolerance = min(config.max_position_tolerance,
                            config.gap_tolerance_fraction * local_gap)
            if not isfinite(tolerance) or tolerance <= _EPSILON or residual > tolerance:
                continue
            terms = [(config.position_weight, residual / tolerance)]
            spacing = _spacing_cost(south, north, i, j, config)
            if spacing is not None:
                terms.append((config.spacing_weight, spacing))
            width = _width_cost(first, second, config)
            if width is not None:
                terms.append((config.width_weight, width))
            terms.append((config.quality_weight,
                          1 - sqrt(max(first.quality, 0.0) * max(second.quality, 0.0))))
            total_weight = sum(weight for weight, _ in terms)
            matrix[i, j] = sum(weight * value for weight, value in terms) / total_weight
    return _Costs(
        matrix,
        _skip_costs(south, north, scale, offset, config),
        _skip_costs(north, south, 1 / scale, -offset / scale, config),
    )


def _monotone_dp(costs: _Costs) -> _Solve:
    """在配对、跳南、跳北之间求最小代价的单调路径与前后缀代价表。"""
    matrix = costs.matrix
    m, n = matrix.shape
    prefix = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    moves = np.zeros((m + 1, n + 1), dtype=np.uint8)
    prefix[0, 0] = 0.0
    if m:
        prefix[1:, 0] = np.cumsum(costs.skip_south)
    if n:
        prefix[0, 1:] = np.cumsum(costs.skip_north)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            pair = prefix[i - 1, j - 1] + matrix[i - 1, j - 1]
            south_skip = prefix[i - 1, j] + costs.skip_south[i - 1]
            north_skip = prefix[i, j - 1] + costs.skip_north[j - 1]
            if pair <= min(south_skip, north_skip):
                prefix[i, j] = pair
            elif south_skip <= north_skip:
                prefix[i, j] = south_skip
                moves[i, j] = 1
            else:
                prefix[i, j] = north_skip
                moves[i, j] = 2

    pairs: list[tuple[int, int]] = []
    i, j = m, n
    while i and j:
        move = moves[i, j]
        if move == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    suffix = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    suffix[m, n] = 0.0
    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            if i == m and j == n:
                continue
            value = np.inf
            if i < m:
                value = min(value, costs.skip_south[i] + suffix[i + 1, j])
            if j < n:
                value = min(value, costs.skip_north[j] + suffix[i, j + 1])
            if i < m and j < n and isfinite(matrix[i, j]):
                value = min(value, matrix[i, j] + suffix[i + 1, j + 1])
            suffix[i, j] = value
    return _Solve(prefix, suffix, float(prefix[m, n]), tuple(pairs))


def _theil_sen(
    pairs: tuple[tuple[int, int], ...], south: SideMeasurement, north: SideMeasurement
) -> tuple[float, float]:
    """给出稳健的 (a, b) 初值。

    种子配对里可能夹带个别错配（例如缺轨处）。用全部两两斜率时，一对错配会
    污染大量斜率并推翻中位数，因此这里优先取**相邻种子配对**的斜率中位数：
    个别错配只影响它两侧的两个斜率。随后按残差修剪并重估一次。
    """
    if len(pairs) < 2:
        return 1.0, 0.0
    positions_s = np.asarray([_median_u(south.rails[i]) for i, _ in pairs])
    positions_n = np.asarray([_median_u(north.rails[j]) for _, j in pairs])
    deltas_s = np.diff(positions_s)
    deltas_n = np.diff(positions_n)
    slopes = [
        float(delta_n / delta_s)
        for delta_s, delta_n in zip(deltas_s, deltas_n, strict=True)
        if abs(delta_s) > _EPSILON and isfinite(delta_n)
    ]
    if not slopes:
        slopes = [
            float((positions_n[second] - positions_n[first])
                  / (positions_s[second] - positions_s[first]))
            for first in range(len(pairs))
            for second in range(first + 1, len(pairs))
            if abs(positions_s[second] - positions_s[first]) > _EPSILON
        ]
    if not slopes:
        return 1.0, float(np.median(positions_n - positions_s))
    scale = float(np.median(slopes))
    offset = float(np.median(positions_n - scale * positions_s))
    residuals = np.abs(positions_n - (scale * positions_s + offset))
    threshold = max(3.0 * float(np.median(residuals)), _EPSILON)
    keep = residuals <= threshold
    if np.count_nonzero(keep) >= 2:
        kept_s, kept_n = positions_s[keep], positions_n[keep]
        variance = float(np.var(kept_s))
        if variance > _EPSILON:
            fitted = float(np.cov(kept_s, kept_n, bias=True)[0, 1]) / variance
            if isfinite(fitted) and fitted > 0:
                return fitted, float(np.mean(kept_n) - fitted * np.mean(kept_s))
    return scale, offset


def _weighted_refit(
    pairs: tuple[tuple[int, int], ...], south: SideMeasurement, north: SideMeasurement,
    scale: float, offset: float,
) -> tuple[float, float]:
    """按残差的稳健尺度加权重估 (a, b)，抑制个别错配的杠杆作用。"""
    if len(pairs) < 2:
        return scale, offset
    x = np.asarray([_median_u(south.rails[i]) for i, _ in pairs])
    y = np.asarray([_median_u(north.rails[j]) for _, j in pairs])
    residuals = np.abs(y - (scale * x + offset))
    sigma = float(np.median(residuals))
    weights = 1.0 / (1.0 + residuals / max(sigma, 1e-6))
    total = float(weights.sum())
    if total <= _EPSILON:
        return scale, offset
    mean_x = float(np.average(x, weights=weights))
    mean_y = float(np.average(y, weights=weights))
    variance = float(np.average((x - mean_x) ** 2, weights=weights))
    if variance <= _EPSILON:
        return scale, offset
    fitted_scale = float(np.average((x - mean_x) * (y - mean_y), weights=weights))
    fitted_scale /= variance
    if not isfinite(fitted_scale) or fitted_scale <= 0:
        return scale, offset
    return fitted_scale, mean_y - fitted_scale * mean_x


def _shift_seeds(
    south: SideMeasurement, north: SideMeasurement, config: ISARailConfig
) -> list[tuple[int | None, tuple[tuple[int, int], ...]]]:
    """整数错移种子：把两侧有序序列整体相对平移 k 个位置。"""
    m, n = south.count, north.count
    bound = min(config.shift_limit, max(2, abs(m - n) + 1))
    seeds: list[tuple[int | None, tuple[tuple[int, int], ...]]] = []
    for shift in range(-bound, bound + 1):
        pairs = tuple(
            (i, i + shift) for i in range(m)
            if 0 <= i + shift < n
            and south.rails[i].geometry_valid and north.rails[i + shift].geometry_valid
        )
        if pairs:
            seeds.append((shift, pairs))
    return seeds


def _boundary_positions(side: SideMeasurement) -> list[float]:
    """稳定大间隙所在位置的中点，作为结构锚点。"""
    positions: list[float] = []
    for index, is_large in enumerate(side.stable_large_gaps):
        if not is_large:
            continue
        left = _median_u(side.rails[index])
        right = _median_u(side.rails[index + 1])
        if np.isfinite(left) and np.isfinite(right):
            positions.append((left + right) / 2)
    return positions


def _anchor_seeds(
    south: SideMeasurement, north: SideMeasurement
) -> list[tuple[int | None, tuple[tuple[int, int], ...]]]:
    """结构种子：按顺序配对两侧稳定大间隙，用其位置差给出 (a, b) 初值。"""
    south_anchors = _boundary_positions(south)
    north_anchors = _boundary_positions(north)
    if len(south_anchors) < 2 or len(north_anchors) < 2:
        return []
    count = min(len(south_anchors), len(north_anchors))
    seeds: list[tuple[int | None, tuple[tuple[int, int], ...]]] = []
    for offset_s in sorted({0, len(south_anchors) - count}):
        for offset_n in sorted({0, len(north_anchors) - count}):
            xs = np.asarray(south_anchors[offset_s:offset_s + count])
            ys = np.asarray(north_anchors[offset_n:offset_n + count])
            if np.ptp(xs) <= _EPSILON:
                continue
            scale = float(np.polyfit(xs, ys, 1)[0])
            if not isfinite(scale) or scale <= 0:
                continue
            offset = float(np.median(ys - scale * xs))
            seeds.append((None, tuple(_nearest_pairs(south, north, scale, offset))))
    return seeds


def _nearest_pairs(
    south: SideMeasurement, north: SideMeasurement, scale: float, offset: float
) -> list[tuple[int, int]]:
    """在给定映射下按单调顺序做一次最近邻配对，用于生成种子配对集合。"""
    pairs: list[tuple[int, int]] = []
    cursor = 0
    for i, rail in enumerate(south.rails):
        position = _median_u(rail)
        if not isfinite(position):
            continue
        mapped = scale * position + offset
        best: tuple[float, int] | None = None
        for j in range(cursor, north.count):
            target = _median_u(north.rails[j])
            if not isfinite(target):
                continue
            distance = abs(target - mapped)
            if best is None or distance < best[0]:
                best = (distance, j)
        if best is not None:
            pairs.append((i, best[1]))
            cursor = best[1] + 1
    return pairs


def _hypotheses(
    south: SideMeasurement, north: SideMeasurement, config: ISARailConfig
) -> list[_Hypothesis]:
    """枚举并重估候选映射，按 (a, b) 去重后返回。"""
    seeds = _shift_seeds(south, north, config) + _anchor_seeds(south, north)
    north_positions = np.asarray(
        [_median_u(rail) for rail in north.rails], dtype=np.float64
    )
    north_positions = north_positions[np.isfinite(north_positions)]
    bin_width = (
        0.25 * float(np.median(np.diff(north_positions)))
        if len(north_positions) > 1 else 0.01
    )
    bin_width = max(bin_width, 1e-4)
    seen: dict[tuple[int, int], _Hypothesis] = {}
    for shift, seed_pairs in seeds:
        scale, offset = _theil_sen(seed_pairs, south, north)
        if not isfinite(scale) or scale <= 0:
            continue
        costs = _cost_matrix(south, north, scale, offset, config)
        solve = _monotone_dp(costs)
        for _ in range(config.refine_iterations):
            if len(solve.pairs) < 2:
                break
            scale, offset = _weighted_refit(solve.pairs, south, north, scale, offset)
            costs = _cost_matrix(south, north, scale, offset, config)
            solve = _monotone_dp(costs)
        if not solve.pairs or not isfinite(solve.cost):
            continue
        key = (round(scale / bin_width), round(offset / bin_width))
        candidate = _Hypothesis(
            shift, scale, offset, solve.cost, solve.pairs,
            tuple(costs.matrix[i, j] for i, j in solve.pairs),
        )
        if key not in seen or candidate.cost < seen[key].cost:
            seen[key] = candidate
    ranked = sorted(seen.values(), key=lambda item: item.cost)
    return ranked[:config.max_hypotheses]


def _finite_or_none(value: float) -> float | None:
    """诊断表里用空值表示"没有竞争者"等无界量。"""
    return float(value) if isfinite(value) else None


def _through_table(costs: _Costs, solve: _Solve) -> FloatArray:
    """每个配对"经过它"的最优完整路径代价，形状 (南, 北)。"""
    m, n = costs.matrix.shape
    return solve.prefix[:m, :n] + costs.matrix + solve.suffix[1:, 1:]


def _competitor_margin(
    south_index: int, north_index: int, through: FloatArray, best_cost: float
) -> tuple[float, int]:
    """换一个伙伴的完整路径代价增量，以及最小的替代伙伴编号。

    禁用代价增量（跳过一个实例）由跳过代价主导，恒在 0.5–1.1 之间，不能反映
    身份是否可分辨。身份的真正竞争者是同一实例的其它可行配对：这里取经过
    `(i, j')` 与 `(i', j)` 的最优完整路径代价，减去当前路径代价，取最小者。
    """
    best = np.inf
    partner = -1
    row = through[south_index]
    for index in range(through.shape[1]):
        if index == north_index or not isfinite(row[index]):
            continue
        value = float(row[index] - best_cost)
        if value < best:
            best, partner = value, index
    column = through[:, north_index]
    for index in range(through.shape[0]):
        if index == south_index or not isfinite(column[index]):
            continue
        value = float(column[index] - best_cost)
        if value < best:
            best, partner = value, index
    if partner < 0:
        return float("inf"), -1
    return max(best, 0.0), partner


def _hypothesis_margin(
    south_index: int, north_index: int, best: _Hypothesis,
    hypotheses: list[_Hypothesis],
) -> float | None:
    """换用其它错移假设时该配对的相对代价差；没有竞争假设时为 None。"""
    margin: float | None = None
    for other in hypotheses:
        if other is best:
            continue
        partner_south = dict(other.pairs).get(south_index)
        partner_north = {j: i for i, j in other.pairs}.get(north_index)
        if partner_south == north_index and partner_north == south_index:
            continue
        relative = (other.cost - best.cost) / max(abs(best.cost), _EPSILON)
        if margin is None or relative < margin:
            margin = relative
    return margin


def _decide(
    south: SideMeasurement, north: SideMeasurement, costs: _Costs, solve: _Solve,
    best: _Hypothesis, hypotheses: list[_Hypothesis], config: ISARailConfig,
) -> tuple[list[RailMatch], list[dict], list[dict], dict[int, dict]]:
    """按配对裕度与假设裕度给出接受、待确认和歧义关系。"""
    matches: list[RailMatch] = []
    tentative: list[dict] = []
    ambiguous: list[dict] = []
    decided: dict[int, dict] = {}
    through = _through_table(costs, solve)
    for south_index, north_index in solve.pairs:
        first, second = south.rails[south_index], north.rails[north_index]
        key = south_index * north.count + north_index
        node_cost = float(costs.matrix[south_index, north_index])
        margin, partner = _competitor_margin(
            south_index, north_index, through, solve.cost
        )
        relative = margin / max(node_cost, _EPSILON)
        hypothesis_margin = _hypothesis_margin(
            south_index, north_index, best, hypotheses
        )
        row = {
            "south_mask_index": first.record["mask_index"],
            "north_mask_index": second.record["mask_index"],
            "south_order_index": first.rank,
            "north_order_index": second.rank,
            "position_residual_norm": abs(
                _median_u(second) - (best.scale * _median_u(first) + best.offset)
            ),
            "node_cost": node_cost,
            "pair_margin": _finite_or_none(margin),
            "pair_margin_relative": _finite_or_none(relative),
            "hypothesis_margin": hypothesis_margin,
            "alternative_partner_index": None if partner < 0 else partner,
            "decision": "",
            "shared_id": None,
        }
        if margin < config.min_pair_margin or relative < config.min_relative_margin:
            row["decision"] = "ambiguous/near_optimal_alternative"
            ambiguous.append({**row, "status": "ambiguous",
                              "reason": "near_optimal_alternative"})
        elif (hypothesis_margin is not None
              and hypothesis_margin < config.min_hypothesis_margin):
            row["decision"] = "ambiguous/shift_ambiguous"
            ambiguous.append(
                {**row, "status": "ambiguous", "reason": "shift_ambiguous"}
            )
        elif (first.order_uncertain or second.order_uncertain
              or not first.geometry_valid or not second.geometry_valid):
            row["decision"] = "tentative/weak_geometry"
            tentative.append({**row, "status": "tentative", "reason": "weak_geometry"})
        elif len(solve.pairs) < config.min_confidence_pairs:
            row["decision"] = "tentative/short_path"
            tentative.append({**row, "status": "tentative", "reason": "short_path"})
        else:
            shared_id = f"R{len(matches) + 1:03d}"
            row["decision"] = "accepted"
            row["shared_id"] = shared_id
            matches.append({
                "shared_id": shared_id,
                "south": first.record,
                "north": second.record,
                "position_error": row["position_residual_norm"],
            })
        decided[key] = row
    return matches, tentative, ambiguous, decided


def _unmatched(
    side: SideMeasurement, other: SideMeasurement, decided: dict[int, dict],
    solve: _Solve | None, costs: _Costs | None, config: ISARailConfig,
) -> tuple[list[UnmatchedRail], list[dict]]:
    """保留未被接受实例的原因，并标出仍有近优候选的实例。"""
    decided_south = {
        row["south_mask_index"] for row in decided.values()
        if row["decision"] == "accepted"
    }
    decided_north = {
        row["north_mask_index"] for row in decided.values()
        if row["decision"] == "accepted"
    }
    accepted = decided_south if side.side == "south" else decided_north
    unmatched: list[UnmatchedRail] = []
    rows: list[dict] = []
    near_limit = None
    if solve is not None:
        near_limit = solve.cost + config.min_pair_margin
    for rail in side.rails:
        if rail.record["mask_index"] in accepted:
            continue
        reason = "no_reliable_partner"
        status = "unmatched"
        if not other.rails:
            reason = "other_side_empty"
        elif not rail.geometry_valid:
            status, reason = "invalid_geometry", "invalid_geometry"
        elif solve is None or costs is None:
            reason = "insufficient_alignment_evidence"
        elif near_limit is not None and _has_near_candidate(
            side, rail, costs, solve, near_limit
        ):
            status, reason = "ambiguous", "near_optimal_alternative"
        unmatched.append(make_unmatched_rail(rail.record, reason))
        rows.append({
            "side": side.side,
            "instance_id": rail.record["instance_id"],
            "mask_index": rail.record["mask_index"],
            "order_index": rail.rank,
            "valid_section_count": rail.valid_count,
            "geometry_quality": rail.quality,
            "status": status,
            "reason": reason,
        })
    return unmatched, rows


def _has_near_candidate(
    side: SideMeasurement, rail: MeasuredRail, costs: _Costs, solve: _Solve,
    near_limit: float,
) -> bool:
    """按完整经过代价判断该实例是否仍有近优候选。"""
    index = rail.rank - 1
    matrix = costs.matrix
    if side.side == "south":
        through = (
            solve.prefix[index, :matrix.shape[1]]
            + matrix[index]
            + solve.suffix[index + 1, 1:]
        )
    else:
        through = (
            solve.prefix[:matrix.shape[0], index]
            + matrix[:, index]
            + solve.suffix[1:, index + 1]
        )
    values = through[np.isfinite(through)]
    return bool(len(values) and float(values.min()) <= near_limit)


def _image_shape(image_path: str | Path) -> tuple[int, int]:
    """读取项目内原图的高度和宽度，供掩码坐标校验。"""
    relative_path = Path(image_path)
    absolute_path = (PROJECT_ROOT / relative_path).resolve()
    if relative_path.anchor or not absolute_path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"请提供相对于项目根目录的图片路径 {image_path}")
    with Image.open(absolute_path) as image:
        width, height = image.size
    if width < 2:
        raise ValueError(f"原图宽度至少为两个像素 {image_path}")
    return height, width


def _instance_rows(side: SideMeasurement) -> list[dict]:
    """单侧实例的几何质量记录。"""
    rows: list[dict] = []
    for rail in side.rails:
        rows.append({
            "side": side.side,
            "instance_id": rail.record["instance_id"],
            "mask_index": rail.record["mask_index"],
            "order_index": rail.rank,
            "valid_section_count": rail.valid_count,
            "geometry_quality": rail.quality,
            "geometry_valid": rail.geometry_valid,
            "order_uncertain": rail.order_uncertain,
            "order_stability": rail.order_stability,
            "overlap_section_count": int(np.count_nonzero(rail.overlap)),
            "order_conflict_section_count": int(np.count_nonzero(rail.conflict)),
        })
    return rows


def match_south_north_rail_isa(
    south_image_path: str | Path,
    north_image_path: str | Path,
    south_masks: np.ndarray | Tensor | Masks | None,
    north_masks: np.ndarray | Tensor | Masks | None,
    *,
    config: ISARailConfig | None = None,
    return_diagnostics: bool = True,
) -> ISAMatchingResult:
    """用整数错移假设与精确配对裕度建立当前图对的一对一对应。

    Args:
        south_image_path: 南侧原图的项目相对路径。
        north_image_path: 北侧原图的项目相对路径。
        south_masks: 南侧 YOLO Results.masks、原图尺寸 ``(N, H, W)`` 数组或
            Tensor；二维数组表示单实例，``None`` 表示该侧为空。
        north_masks: 北侧掩码，约定同 south_masks。
        config: 假设数、代价权重与判定门限，省略时使用默认值。
        return_diagnostics: 是否保留假设、逐对代价与几何诊断。

    Returns:
        与项目结果协议兼容的配对、未匹配实例、待确认与歧义关系。
        ``position_error`` 是该配对在北图宽度归一化后的横向残差。

    Raises:
        OSError: 无法读取原图。
        ValueError: 路径、图像尺寸或掩码尺寸无效。
    """
    settings = config if config is not None else ISARailConfig()
    start = perf_counter()
    south_shape = _image_shape(south_image_path)
    north_shape = _image_shape(north_image_path)
    south = measure_side(as_mask_stack(south_masks, south_shape), "south",
                         settings.measurement)
    north = measure_side(as_mask_stack(north_masks, north_shape), "north",
                         settings.measurement)
    feature_ms = 1000 * (perf_counter() - start)
    return _assemble(
        south, north, Path(south_image_path).as_posix(),
        Path(north_image_path).as_posix(), settings, feature_ms, return_diagnostics,
    )


def _assemble(
    south: SideMeasurement, north: SideMeasurement,
    south_image_path: str, north_image_path: str,
    settings: ISARailConfig, feature_ms: float, return_diagnostics: bool,
) -> ISAMatchingResult:
    """在已完成测量的两侧上做假设、求解、判定并组装结果。"""
    start = perf_counter()
    hypotheses = _hypotheses(south, north, settings)
    best = hypotheses[0] if hypotheses else None
    costs = solve = None
    matches: list[RailMatch] = []
    tentative: list[dict] = []
    ambiguous: list[dict] = []
    decided: dict[int, dict] = {}
    if best is not None:
        costs = _cost_matrix(south, north, best.scale, best.offset, settings)
        solve = _monotone_dp(costs)
        matches, tentative, ambiguous, decided = _decide(
            south, north, costs, solve, best, hypotheses, settings
        )
    hypothesis_ms = 1000 * (perf_counter() - start)

    start = perf_counter()
    south_unmatched, south_rows = _unmatched(
        south, north, decided, solve, costs, settings
    )
    north_unmatched, north_rows = _unmatched(
        north, south, decided, solve, costs, settings
    )
    decision_ms = 1000 * (perf_counter() - start)

    diagnostics = None
    if return_diagnostics:
        costs_rows = _pair_rows(south, north, costs, solve, best, hypotheses, decided)
        summary = {
            "algorithm_version": _VERSION,
            "instance_count_south": south.count,
            "instance_count_north": north.count,
            "hypothesis_count": len(hypotheses),
            "selected_shift_k": None if best is None else best.shift_k,
            "selected_scale": None if best is None else best.scale,
            "selected_offset": None if best is None else best.offset,
            "path_cost": None if best is None else best.cost,
            "path_match_count": 0 if best is None else len(best.pairs),
            "accepted_count": len(matches),
            "tentative_count": len(tentative),
            "ambiguous_count": len(ambiguous),
            "unmatched_count": len(south_rows) + len(north_rows),
            "median_pair_margin": _median([
                row["pair_margin"] for row in decided.values()
            ]),
            "min_hypothesis_margin": _min([
                row["hypothesis_margin"] for row in decided.values()
                if row["hypothesis_margin"] is not None
            ]),
        }
        summary.update({
            f"config_{name}": value
            for name, value in asdict(settings).items() if name != "measurement"
        })
        summary.update({
            "feature_time_ms": feature_ms,
            "hypothesis_time_ms": hypothesis_ms,
            "decision_time_ms": decision_ms,
        })
        diagnostics = {
            "summary": summary,
            "hypotheses": _hypothesis_rows(best, hypotheses),
            "pair_costs": costs_rows,
            "unmatched": south_rows + north_rows,
            "instances": _instance_rows(south) + _instance_rows(north),
        }
    result: ISAMatchingResult = {
        "south_image_path": south_image_path,
        "north_image_path": north_image_path,
        "matches": matches,
        "unmatched_south": south_unmatched,
        "unmatched_north": north_unmatched,
        "tentative_matches": tentative,
        "ambiguous_candidates": ambiguous,
        "diagnostics": diagnostics,
    }
    return result


def _median(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and isfinite(value)]
    return float(np.median(finite)) if finite else None


def _min(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and isfinite(value)]
    return float(np.min(finite)) if finite else None


def _hypothesis_rows(
    best: _Hypothesis | None, hypotheses: list[_Hypothesis]
) -> list[dict]:
    """候选映射的诊断表。"""
    rows: list[dict] = []
    for rank, hypothesis in enumerate(hypotheses, start=1):
        rows.append({
            "rank": rank,
            "shift_k": hypothesis.shift_k,
            "scale": hypothesis.scale,
            "offset": hypothesis.offset,
            "path_cost": hypothesis.cost,
            "match_count": len(hypothesis.pairs),
            "is_selected": hypothesis is best,
            "mask_index_pairs": "|".join(
                f"{i}:{j}" for i, j in hypothesis.pairs
            ),
        })
    return rows


def _pair_rows(
    south: SideMeasurement, north: SideMeasurement, costs: _Costs | None,
    solve: _Solve | None, best: _Hypothesis | None, hypotheses: list[_Hypothesis],
    decided: dict[int, dict],
) -> list[dict]:
    """逐候选配对的代价分项、两个裕度与最终判定。"""
    if costs is None or solve is None:
        return []
    through = _through_table(costs, solve)
    candidates = [
        (i, j) for i in range(south.count) for j in range(north.count)
        if isfinite(costs.matrix[i, j])
    ][:_CANDIDATE_LIMIT]
    rows: list[dict] = []
    for i, j in candidates:
        node_cost = float(costs.matrix[i, j])
        margin, _ = _competitor_margin(i, j, through, solve.cost)
        relative = margin / max(node_cost, _EPSILON)
        hypothesis_margin = _hypothesis_margin(i, j, best, hypotheses) \
            if best is not None else None
        decision = decided.get(i * north.count + j)
        rows.append({
            "south_mask_index": south.rails[i].record["mask_index"],
            "north_mask_index": north.rails[j].record["mask_index"],
            "south_order_index": south.rails[i].rank,
            "north_order_index": north.rails[j].rank,
            "node_cost": node_cost,
            "pair_margin": _finite_or_none(margin),
            "pair_margin_relative": _finite_or_none(relative),
            "hypothesis_margin": hypothesis_margin,
            "decision": "not_selected" if decision is None else decision["decision"],
            "shared_id": None if decision is None else decision.get("shared_id"),
        })
    return rows


def _run_demo(show_missed_matches_only: bool = False) -> None:
    """预览 ISA-Rail 的自动接受关联。"""
    from core_algorithm.common.preview import preview_matching_results

    preview_matching_results(
        match_south_north_rail_isa, show_missed_matches_only=show_missed_matches_only
    )


if __name__ == "__main__":
    _run_demo(show_missed_matches_only=False)
