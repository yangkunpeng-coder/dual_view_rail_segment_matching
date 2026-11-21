"""CGM-Rail：链式图约束下的整列错移裁决。

CGM-Rail（Chain-Graph Matching）把南北匹配看成一张**链式图**：节点是钢轨实例，
边是相邻间距关系，约束是顺序单调与一对一。图的自由指标只有一个离散量——整列
错移 ``k``（南侧第 i 根对应北侧第 i+k 根）。算法显式枚举 ``k``，在每个 ``k`` 下
拟合横向映射 ``u_N ≈ a·u_S + b``，用链式动态规划求最优对齐，再用**整束大间隙
一致性**裁决 ``k``：

1. 节点代价：位置残差（以局部轨距为门限）、宽度比、几何质量；
2. 边代价：左右相邻间距比的对数指纹（1–2 个邻域）；
3. 整束代价：沿对齐路径比较两侧"稳定大间隙"的出现数量与位置；
4. 置信：配对裕度（换伙伴的代价差）与错移裕度（换 k 的相对代价差）。

与 V1/MSLC/GMSA/ISA 的关系：求解器仍是 ``O(mn)`` 的单调 DP（真值实验证明在这批
数据上无约束指派与 DP 结果相同），新增的是"整列错移作为显式变量"与"整束证据"。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil, isfinite, sqrt
from pathlib import Path
import sys
from time import perf_counter
from typing import TYPE_CHECKING, TypedDict

import numpy as np
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
_VERSION = "CGM-Rail V1.2"
_CANDIDATE_LIMIT = 300


class CGMMatchingResult(RailMatchingResult):
    """在项目结果协议上增加待确认、歧义关系与诊断。"""

    tentative_matches: list[dict]
    ambiguous_candidates: list[dict]
    diagnostics: dict | None


@dataclass(frozen=True)
class CGMRailConfig:
    """错移枚举范围、代价权重与判定门限。位置与代价均已归一化。"""

    measurement: RailMeasurementConfig = field(default_factory=RailMeasurementConfig)
    shift_limit: int = 2
    max_hypotheses: int = 12
    refine_iterations: int = 1
    seed_expand_coverage: float = 0.90
    position_weight: float = 0.45
    spacing_weight: float = 0.30
    width_weight: float = 0.10
    quality_weight: float = 0.15
    max_position_tolerance: float = 0.025
    gap_tolerance_fraction: float = 0.55
    spacing_log_cap: float = 1.5
    width_log_cap: float = 1.0
    skip_base: float = 0.30
    skip_inside_band: float = 0.80
    bundle_weight: float = 0.35
    min_pair_margin: float = 0.03
    min_relative_margin: float = 0.25
    min_shift_margin: float = 0.05
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
            raise ValueError("CGM 的整数配置或范围无效")
        values = (
            self.position_weight, self.spacing_weight, self.width_weight,
            self.quality_weight, self.max_position_tolerance,
            self.gap_tolerance_fraction, self.spacing_log_cap, self.width_log_cap,
            self.skip_base, self.skip_inside_band, self.bundle_weight,
            self.min_pair_margin, self.min_relative_margin, self.min_shift_margin,
            self.seed_expand_coverage,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("CGM 的代价、门限须为有限非负数")
        if not 0 < self.gap_tolerance_fraction <= 1:
            raise ValueError("gap_tolerance_fraction 须在 (0, 1] 内")
        if not 0 <= self.seed_expand_coverage <= 1:
            raise ValueError("seed_expand_coverage 须在 [0, 1] 内")


@dataclass(frozen=True)
class _Costs:
    """节点+边代价矩阵与两侧跳过代价。"""

    matrix: FloatArray
    skip_south: FloatArray
    skip_north: FloatArray


@dataclass(frozen=True)
class _Solve:
    """一次链式 DP 的最优值表、路径与前后缀代价。"""

    prefix: FloatArray
    suffix: FloatArray
    cost: float
    pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Hypothesis:
    """一个整列错移假设及其对齐结果。"""

    shift: int
    scale: float
    offset: float
    dp_cost: float
    bundle_cost: float
    total_cost: float
    pairs: tuple[tuple[int, int], ...]


def _median_u(rail: MeasuredRail) -> float:
    """实例的横向位置取有效采样行的中位数。"""
    valid = rail.u[np.isfinite(rail.u)]
    return float(np.median(valid)) if len(valid) else float("nan")


def _neighbor_gaps(side: SideMeasurement) -> FloatArray:
    """每个实例到左右邻轨的最小归一化轨距。"""
    positions = np.asarray([_median_u(rail) for rail in side.rails], dtype=np.float64)
    if len(positions) < 2:
        return np.full(len(positions), np.inf)
    gaps = np.abs(np.diff(positions))
    return np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf])


def _rail_gap_logs(side: SideMeasurement, index: int) -> list[FloatArray]:
    """实例左右两侧间距的对数指纹（各两列，缺失为全 NaN）。"""
    rows = side.log_gaps.shape[1]
    missing = np.full(rows, np.nan)
    outputs: list[FloatArray] = []
    for offset in (1, 2):
        row = index - offset
        outputs.append(side.log_gaps[row] if row >= 0 else missing)
    for offset in (0, 1):
        row = index + offset
        outputs.append(
            side.log_gaps[row] if row < side.count - 1 else missing
        )
    return outputs


def _spacing_cost(
    south: SideMeasurement, north: SideMeasurement,
    south_index: int, north_index: int, config: CGMRailConfig,
) -> float | None:
    """比较两侧邻域间距的对数指纹，按可用项平均。"""
    south_logs = _rail_gap_logs(south, south_index)
    north_logs = _rail_gap_logs(north, north_index)
    terms: list[float] = []
    for first, second in zip(south_logs, north_logs, strict=True):
        valid = np.isfinite(first) & np.isfinite(second)
        if np.any(valid):
            difference = np.abs(first[valid] - second[valid])
            terms.append(float(np.mean(np.minimum(difference, config.spacing_log_cap))))
    if not terms:
        return None
    return float(np.mean(terms))


def _width_cost(
    south_rail: MeasuredRail, north_rail: MeasuredRail, config: CGMRailConfig
) -> float | None:
    """比较宽度中位数的相对偏差，作为弱深度证据。"""
    south_width = np.median(south_rail.widths[np.isfinite(south_rail.widths)])
    north_width = np.median(north_rail.widths[np.isfinite(north_rail.widths)])
    if not np.isfinite(south_width) or not np.isfinite(north_width):
        return None
    if south_width <= 0 or north_width <= 0:
        return None
    return float(min(abs(np.log(south_width / north_width)), config.width_log_cap))


def _skip_costs(
    side: SideMeasurement, other: SideMeasurement, scale: float, offset: float,
    config: CGMRailConfig,
) -> FloatArray:
    """按"映射后是否落在对侧占用带内"区分边界截断与真漏检。"""
    costs = np.zeros(side.count, dtype=np.float64)
    low, high = other.band
    for index, rail in enumerate(side.rails):
        mapped = scale * _median_u(rail) + offset
        inside = bool(np.isfinite(low) and low <= mapped <= high)
        penalty = config.skip_base + (config.skip_inside_band if inside else 0.0)
        costs[index] = penalty * (1 - 0.5 * rail.quality)
    return costs


def _cost_matrix(
    south: SideMeasurement, north: SideMeasurement,
    scale: float, offset: float, config: CGMRailConfig,
) -> _Costs:
    """按给定映射计算节点+边代价与两侧跳过代价。"""
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
    south_positions = [_median_u(rail) for rail in south.rails]
    north_positions = [_median_u(rail) for rail in north.rails]
    for i, first in enumerate(south.rails):
        if not first.geometry_valid or not isfinite(south_positions[i]):
            continue
        for j, second in enumerate(north.rails):
            if not second.geometry_valid or not isfinite(north_positions[j]):
                continue
            residual = abs(north_positions[j] - (scale * south_positions[i] + offset))
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
    """链式动态规划：配对 / 跳南 / 跳北，回溯得单调路径与前后缀代价。"""
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


def _through_table(costs: _Costs, solve: _Solve) -> FloatArray:
    """每个配对"经过它"的最优完整路径代价。"""
    m, n = costs.matrix.shape
    return solve.prefix[:m, :n] + costs.matrix + solve.suffix[1:, 1:]


def _competitor_margin(
    south_index: int, north_index: int, through: FloatArray, best_cost: float
) -> float:
    """换一个伙伴的完整路径代价增量（无竞争者时为无穷）。"""
    best = np.inf
    row = through[south_index]
    for index in range(through.shape[1]):
        if index != north_index and isfinite(row[index]):
            best = min(best, float(row[index] - best_cost))
    column = through[:, north_index]
    for index in range(through.shape[0]):
        if index != south_index and isfinite(column[index]):
            best = min(best, float(column[index] - best_cost))
    return max(best, 0.0)


def _fit_line(
    x: FloatArray, y: FloatArray, *, adjacent: bool
) -> tuple[float, float] | None:
    """用斜率中位数拟合 a、b。相邻斜率对个别错配不敏感，全斜率更稳。"""
    slopes: list[float] = []
    if adjacent:
        for index in range(len(x) - 1):
            if abs(x[index + 1] - x[index]) > _EPSILON:
                slopes.append(float((y[index + 1] - y[index])
                                    / (x[index + 1] - x[index])))
    else:
        for first in range(len(x)):
            for second in range(first + 1, len(x)):
                if abs(x[second] - x[first]) > _EPSILON:
                    slopes.append(float((y[second] - y[first])
                                        / (x[second] - x[first])))
    if not slopes:
        return None
    scale = float(np.median(slopes))
    if not isfinite(scale) or scale <= 0:
        return None
    return scale, float(np.median(y - scale * x))


def _mapping_seeds(
    south: SideMeasurement, north: SideMeasurement, shift: int
) -> list[tuple[float, float]]:
    """一个整列错移下的多套映射种子。

    单一全局仿射在密集或存在透视的轨束上无法把每根轨都放进位置门限，因此为
    同一错移提供多个候选映射：相邻斜率全局、全斜率全局、两个半区、以及按残差
    修剪后的重估。每个种子各自跑一次链式 DP，取代价最低者作为该错移的解。
    """
    indices = [
        (index, index + shift) for index in range(south.count)
        if 0 <= index + shift < north.count
        and south.rails[index].geometry_valid
        and north.rails[index + shift].geometry_valid
    ]
    if len(indices) < 2:
        return []
    x = np.asarray([_median_u(south.rails[i]) for i, _ in indices], dtype=np.float64)
    y = np.asarray([_median_u(north.rails[j]) for _, j in indices], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return []
    seeds: list[tuple[float, float]] = []
    for adjacent in (True, False):
        fitted = _fit_line(x, y, adjacent=adjacent)
        if fitted is not None:
            seeds.append(fitted)
    half = max(2, len(x) // 2)
    for start, stop in ((0, half), (len(x) - half, len(x))):
        fitted = _fit_line(x[start:stop], y[start:stop], adjacent=True)
        if fitted is not None:
            seeds.append(fitted)
    if len(x) >= 4:
        base = _fit_line(x, y, adjacent=True)
        if base is not None:
            scale, offset = base
            residual = np.abs(y - (scale * x + offset))
            keep = residual <= max(2.0 * float(np.median(residual)), _EPSILON)
            if np.count_nonzero(keep) >= 2:
                trimmed = _fit_line(x[keep], y[keep], adjacent=False)
                if trimmed is not None:
                    seeds.append(trimmed)
    unique: dict[tuple[float, float], tuple[float, float]] = {}
    for scale, offset in seeds:
        unique[(round(scale, 3), round(offset, 4))] = (scale, offset)
    return list(unique.values())


def _shift_fit(
    south: SideMeasurement, north: SideMeasurement, shift: int
) -> tuple[float, float] | None:
    """按"南侧第 i 根对应北侧第 i+shift 根"拟合 (a, b) 的单一全局解。"""
    seeds = _mapping_seeds(south, north, shift)
    return seeds[0] if seeds else None


def _weighted_refit(
    pairs: tuple[tuple[int, int], ...], south: SideMeasurement,
    north: SideMeasurement, scale: float, offset: float,
) -> tuple[float, float]:
    """按残差稳健尺度加权重估 (a, b)。"""
    if len(pairs) < 2:
        return scale, offset
    x = np.asarray([_median_u(south.rails[i]) for i, _ in pairs])
    y = np.asarray([_median_u(north.rails[j]) for _, j in pairs])
    residuals = np.abs(y - (scale * x + offset))
    sigma = max(float(np.median(residuals)), 1e-6)
    weights = 1.0 / (1.0 + residuals / sigma)
    total = float(weights.sum())
    if total <= _EPSILON:
        return scale, offset
    mean_x = float(np.average(x, weights=weights))
    mean_y = float(np.average(y, weights=weights))
    variance = float(np.average((x - mean_x) ** 2, weights=weights))
    if variance <= _EPSILON:
        return scale, offset
    fitted = float(np.average((x - mean_x) * (y - mean_y), weights=weights)) / variance
    if not isfinite(fitted) or fitted <= 0:
        return scale, offset
    return fitted, mean_y - fitted * mean_x


def _bundle_cost(
    south: SideMeasurement, north: SideMeasurement,
    pairs: tuple[tuple[int, int], ...],
) -> float:
    """沿对齐路径比较"稳定大间隙"的数量，作为整束证据。

    正确错移下，两侧的大间隙应落在相同的对齐区间内；错移一层时，大间隙会错位，
    这个计数差变大。它只看大间隙的分布，不重复计算局部间距代价。
    """
    if len(pairs) < 2:
        return 0.0
    total = 0.0
    for (first_s, first_n), (next_s, next_n) in zip(pairs, pairs[1:]):
        south_count = int(np.count_nonzero(south.stable_large_gaps[first_s:next_s]))
        north_count = int(np.count_nonzero(north.stable_large_gaps[first_n:next_n]))
        total += abs(south_count - north_count)
    head_s, head_n = pairs[0]
    tail_s, tail_n = pairs[-1]
    total += abs(
        int(np.count_nonzero(south.stable_large_gaps[:head_s]))
        - int(np.count_nonzero(north.stable_large_gaps[:head_n]))
    )
    total += abs(
        int(np.count_nonzero(south.stable_large_gaps[tail_s:]))
        - int(np.count_nonzero(north.stable_large_gaps[tail_n:]))
    )
    return total / (len(pairs) + 1)


def _hypotheses(
    south: SideMeasurement, north: SideMeasurement, config: CGMRailConfig
) -> list[_Hypothesis]:
    """枚举整列错移与多套映射，取每个错移的最优对齐后按总代价排序。

    种子剪枝：先跑首套映射，若它已经覆盖到 ``seed_expand_coverage`` 比例的实例，
    就跳过该错移下其余映射。多套映射只用于补"首套映射覆盖不足"的情形，达标时
    继续展开只是重复 DP。
    """
    hypotheses: list[_Hypothesis] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    target = min(south.count, north.count)
    threshold = max(1, ceil(target * config.seed_expand_coverage))
    for shift in range(-config.shift_limit, config.shift_limit + 1):
        best: _Hypothesis | None = None
        for index, (scale, offset) in enumerate(_mapping_seeds(south, north, shift)):
            solve = _monotone_dp(_cost_matrix(south, north, scale, offset, config))
            for _ in range(config.refine_iterations):
                if len(solve.pairs) < 2:
                    break
                scale, offset = _weighted_refit(
                    solve.pairs, south, north, scale, offset
                )
                solve = _monotone_dp(_cost_matrix(south, north, scale, offset, config))
            if not solve.pairs:
                continue
            bundle = _bundle_cost(south, north, solve.pairs)
            candidate = _Hypothesis(
                shift, scale, offset, solve.cost, bundle,
                solve.cost + config.bundle_weight * bundle, solve.pairs,
            )
            if best is None or candidate.total_cost < best.total_cost:
                best = candidate
            if index == 0 and len(solve.pairs) >= threshold:
                break
        if best is None:
            continue
        key = frozenset(best.pairs)
        if key in seen:
            continue
        seen.add(key)
        hypotheses.append(best)
    ranked = sorted(hypotheses, key=lambda item: item.total_cost)
    return ranked[:config.max_hypotheses]


def _shift_margin(hypotheses: list[_Hypothesis], best: _Hypothesis) -> float | None:
    """次优错移与该错移的相对代价差；另一种关系集合相同则视为无竞争。"""
    competitors = [
        item.total_cost for item in hypotheses
        if item is not best and set(item.pairs) != set(best.pairs)
    ]
    if not competitors:
        return None
    second = min(competitors)
    return (second - best.total_cost) / max(abs(best.total_cost), _EPSILON)


def _hypothesis_margin(
    south_index: int, north_index: int, best: _Hypothesis,
    hypotheses: list[_Hypothesis],
) -> float | None:
    """换一个错移假设后该配对的相对代价差。"""
    margin: float | None = None
    for other in hypotheses:
        if other is best:
            continue
        partner_south = dict(other.pairs).get(south_index)
        partner_north = {j: i for i, j in other.pairs}.get(north_index)
        if partner_south == north_index and partner_north == south_index:
            continue
        relative = (other.total_cost - best.total_cost) / max(
            abs(best.total_cost), _EPSILON
        )
        if margin is None or relative < margin:
            margin = relative
    return margin


def _decide(
    south: SideMeasurement, north: SideMeasurement, costs: _Costs, solve: _Solve,
    best: _Hypothesis, hypotheses: list[_Hypothesis], config: CGMRailConfig,
) -> tuple[list[RailMatch], list[dict], list[dict], dict[int, dict]]:
    """按配对裕度、错移裕度、几何质量与路径长度给出接受/待确认/歧义。"""
    matches: list[RailMatch] = []
    tentative: list[dict] = []
    ambiguous: list[dict] = []
    decided: dict[int, dict] = {}
    through = _through_table(costs, solve)
    shift_margin = _shift_margin(hypotheses, best)
    for south_index, north_index in solve.pairs:
        first, second = south.rails[south_index], north.rails[north_index]
        node_cost = float(costs.matrix[south_index, north_index])
        margin = _competitor_margin(south_index, north_index, through, solve.cost)
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
            "pair_margin": None if not isfinite(margin) else margin,
            "pair_margin_relative": None if not isfinite(relative) else relative,
            "hypothesis_margin": hypothesis_margin,
            "shift_margin": shift_margin,
            "decision": "",
            "shared_id": None,
        }
        if margin < config.min_pair_margin or relative < config.min_relative_margin:
            row["decision"] = "ambiguous/near_optimal_alternative"
            ambiguous.append({**row, "status": "ambiguous",
                              "reason": "near_optimal_alternative"})
        elif (shift_margin is not None and shift_margin < config.min_shift_margin):
            row["decision"] = "ambiguous/shift_ambiguous"
            ambiguous.append(
                {**row, "status": "ambiguous", "reason": "shift_ambiguous"}
            )
        elif (hypothesis_margin is not None
              and hypothesis_margin < config.min_shift_margin):
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
        decided[south_index * north.count + north_index] = row
    return matches, tentative, ambiguous, decided


def _unmatched(
    side: SideMeasurement, other: SideMeasurement, decided: dict[int, dict],
    solve: _Solve | None, costs: _Costs | None, config: CGMRailConfig,
) -> tuple[list[UnmatchedRail], list[dict]]:
    """保留未接受实例的原因，并标出仍有近优候选的实例。"""
    accepted = {
        row["south_mask_index"] if side.side == "south" else row["north_mask_index"]
        for row in decided.values() if row["decision"] == "accepted"
    }
    unmatched: list[UnmatchedRail] = []
    rows: list[dict] = []
    near_limit = None
    if solve is not None:
        near_limit = solve.cost + config.min_pair_margin
    for rail in side.rails:
        if rail.record["mask_index"] in accepted:
            continue
        status, reason = "unmatched", "no_reliable_partner"
        if not other.rails:
            reason = "other_side_empty"
        elif not rail.geometry_valid:
            status, reason = "invalid_geometry", "invalid_geometry"
        elif solve is None or costs is None:
            reason = "insufficient_alignment_evidence"
        elif near_limit is not None and _has_near_candidate(side, rail, costs, solve,
                                                            near_limit):
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
        through = (solve.prefix[index, :matrix.shape[1]] + matrix[index]
                   + solve.suffix[index + 1, 1:])
    else:
        through = (solve.prefix[:matrix.shape[0], index] + matrix[:, index]
                   + solve.suffix[1:, index + 1])
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
    return [
        {
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
        }
        for rail in side.rails
    ]


def match_south_north_rail_cgm(
    south_image_path: str | Path,
    north_image_path: str | Path,
    south_masks: np.ndarray | Tensor | Masks | None,
    north_masks: np.ndarray | Tensor | Masks | None,
    *,
    config: CGMRailConfig | None = None,
    return_diagnostics: bool = True,
) -> CGMMatchingResult:
    """用链式图约束与整列错移裁决建立当前图对的一对一对应。

    Args:
        south_image_path: 南侧原图的项目相对路径。
        north_image_path: 北侧原图的项目相对路径。
        south_masks: 南侧 YOLO Results.masks、原图尺寸 ``(N, H, W)`` 数组或
            Tensor；二维数组表示单实例，``None`` 表示该侧为空。
        north_masks: 北侧掩码，约定同 south_masks。
        config: 错移范围、代价权重与判定门限，省略时使用默认值。
        return_diagnostics: 是否保留假设、逐对代价与几何诊断。

    Returns:
        与项目结果协议兼容的配对、未匹配实例、待确认与歧义关系。
        ``position_error`` 是该配对在北图宽度归一化后的横向残差。

    Raises:
        OSError: 无法读取原图。
        ValueError: 路径、图像尺寸或掩码尺寸无效。
    """
    settings = config if config is not None else CGMRailConfig()
    start = perf_counter()
    south_shape = _image_shape(south_image_path)
    north_shape = _image_shape(north_image_path)
    south = measure_side(as_mask_stack(south_masks, south_shape), "south",
                         settings.measurement)
    north = measure_side(as_mask_stack(north_masks, north_shape), "north",
                         settings.measurement)
    feature_ms = 1000 * (perf_counter() - start)

    start = perf_counter()
    hypotheses = _hypotheses(south, north, settings)
    best = hypotheses[0] if hypotheses else None
    costs = solve = None
    matches: list[RailMatch] = []
    tentative: list[dict] = []
    ambiguous: list[dict] = []
    decided: dict[int, dict] = {}
    shift_margin = None
    if best is not None:
        costs = _cost_matrix(south, north, best.scale, best.offset, settings)
        solve = _monotone_dp(costs)
        shift_margin = _shift_margin(hypotheses, best)
        matches, tentative, ambiguous, decided = _decide(
            south, north, costs, solve, best, hypotheses, settings
        )
    hypothesis_ms = 1000 * (perf_counter() - start)

    start = perf_counter()
    south_unmatched, south_rows = _unmatched(south, north, decided, solve, costs,
                                             settings)
    north_unmatched, north_rows = _unmatched(north, south, decided, solve, costs,
                                             settings)
    decision_ms = 1000 * (perf_counter() - start)

    diagnostics = None
    if return_diagnostics:
        summary = {
            "algorithm_version": _VERSION,
            "instance_count_south": south.count,
            "instance_count_north": north.count,
            "hypothesis_count": len(hypotheses),
            "selected_shift_k": None if best is None else best.shift,
            "selected_scale": None if best is None else best.scale,
            "selected_offset": None if best is None else best.offset,
            "dp_cost": None if best is None else best.dp_cost,
            "bundle_cost": None if best is None else best.bundle_cost,
            "total_cost": None if best is None else best.total_cost,
            "shift_margin": shift_margin,
            "path_match_count": 0 if best is None else len(best.pairs),
            "accepted_count": len(matches),
            "tentative_count": len(tentative),
            "ambiguous_count": len(ambiguous),
            "unmatched_count": len(south_rows) + len(north_rows),
            "median_pair_margin": _median(
                [row["pair_margin"] for row in decided.values()]
            ),
            "feature_time_ms": feature_ms,
            "hypothesis_time_ms": hypothesis_ms,
            "decision_time_ms": decision_ms,
        }
        summary.update({
            f"config_{name}": value
            for name, value in asdict(settings).items() if name != "measurement"
        })
        diagnostics = {
            "summary": summary,
            "hypotheses": [
                {
                    "rank": rank,
                    "shift_k": item.shift,
                    "scale": item.scale,
                    "offset": item.offset,
                    "dp_cost": item.dp_cost,
                    "bundle_cost": item.bundle_cost,
                    "total_cost": item.total_cost,
                    "match_count": len(item.pairs),
                    "is_selected": item is best,
                    "mask_index_pairs": "|".join(f"{i}:{j}" for i, j in item.pairs),
                }
                for rank, item in enumerate(hypotheses, start=1)
            ],
            "pair_costs": _pair_rows(south, north, costs, solve, best, hypotheses,
                                     decided),
            "unmatched": south_rows + north_rows,
            "instances": _instance_rows(south) + _instance_rows(north),
        }
    return {
        "south_image_path": Path(south_image_path).as_posix(),
        "north_image_path": Path(north_image_path).as_posix(),
        "matches": matches,
        "unmatched_south": south_unmatched,
        "unmatched_north": north_unmatched,
        "tentative_matches": tentative,
        "ambiguous_candidates": ambiguous,
        "diagnostics": diagnostics,
    }


def _median(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and isfinite(value)]
    return float(np.median(finite)) if finite else None


def _pair_rows(
    south: SideMeasurement, north: SideMeasurement, costs: _Costs | None,
    solve: _Solve | None, best: _Hypothesis | None, hypotheses: list[_Hypothesis],
    decided: dict[int, dict],
) -> list[dict]:
    """逐候选配对的代价、两个裕度与最终判定。"""
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
        margin = _competitor_margin(i, j, through, solve.cost)
        relative = margin / max(node_cost, _EPSILON)
        decision = decided.get(i * north.count + j)
        rows.append({
            "south_mask_index": south.rails[i].record["mask_index"],
            "north_mask_index": north.rails[j].record["mask_index"],
            "south_order_index": south.rails[i].rank,
            "north_order_index": north.rails[j].rank,
            "node_cost": node_cost,
            "pair_margin": None if not isfinite(margin) else margin,
            "pair_margin_relative": None if not isfinite(relative) else relative,
            "hypothesis_margin": _hypothesis_margin(i, j, best, hypotheses)
            if best is not None else None,
            "decision": "not_selected" if decision is None else decision["decision"],
            "shared_id": None if decision is None else decision.get("shared_id"),
        })
    return rows


def _run_demo(show_missed_matches_only: bool = False) -> None:
    """预览 CGM-Rail 的自动接受关联。"""
    from core_algorithm.common.preview import preview_matching_results

    preview_matching_results(
        match_south_north_rail_cgm, show_missed_matches_only=show_missed_matches_only
    )


if __name__ == "__main__":
    _run_demo(show_missed_matches_only=False)
