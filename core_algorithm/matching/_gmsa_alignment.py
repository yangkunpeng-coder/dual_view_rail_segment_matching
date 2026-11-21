"""GMSA 的纵向 DTW、锚点代价缓存和有序路径求解。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core_algorithm.matching._gmsa_geometry import (
    Direction,
    FloatArray,
    GMSARailConfig,
    IndexArray,
    ViewFeatures,
)

_EPSILON = 1e-9
_ROBUST_ERROR_CAP = 2.0
_LOW_QUALITY_COST = 0.40


@dataclass(frozen=True)
class DtwPoint:
    """DTW 中的有效采样点，索引均对应原始图高方向。"""

    south_index: int
    north_index: int
    local_cost: float


@dataclass(frozen=True)
class DtwTrace:
    """采用的纵向路径及各归一化代价分项。"""

    points: tuple[DtwPoint, ...]
    distance_cost: float
    warp_cost: float
    coverage_cost: float
    coverage: float
    warp_fraction: float


@dataclass(frozen=True)
class DtwResult:
    """批量 DTW 的代价、覆盖率及按需回溯的有效路径。"""

    costs: FloatArray
    coverage: FloatArray
    traces: dict[int, DtwTrace]


def _walk_dtw(
    dp: FloatArray, end_s: int, end_n: int, batch_index: int, repeats: int
) -> list[tuple[int, int]]:
    """回溯已保存的重复步状态，按固定优先级处理并列代价。"""
    i, j = end_s, end_n
    state = int(np.argmin(dp[i, j, :, batch_index]))
    path = [(i, j)]
    while i or j:
        if state == 0:
            i, j = i - 1, j - 1
            state = int(np.argmin(dp[i, j, :, batch_index]))
        elif state <= repeats:
            j -= 1
            if state == 1:
                states = (0, *range(repeats + 1, 2 * repeats + 1))
                state = min(states, key=lambda value: dp[i, j, value, batch_index])
            else:
                state -= 1
        else:
            i -= 1
            if state == repeats + 1:
                state = int(np.argmin(dp[i, j, :repeats + 1, batch_index]))
            else:
                state -= 1
        path.append((i, j))
    path.reverse()
    return path


def constrained_dtw(
    south: FloatArray,
    north: FloatArray,
    south_weights: FloatArray,
    north_weights: FloatArray,
    config: GMSARailConfig,
    direction: Direction,
    error_scale: float,
    *,
    trace: bool = False,
) -> DtwResult:
    """在原始采样索引窗口内批量对齐对数曲线，跳过无效样本。

    输入均为 (批量, 截面数)。移除双方在窗口内没有可比对象的采样，
    再覆盖剩余的两个有效序列。状态 0 为对角，后两组分别记录同一南
    或北采样点的连续重复步数。找不到满足条件的路径时返回无穷代价。
    """
    batch_size, sample_count = south.shape
    costs = np.full(batch_size, np.inf, dtype=np.float64)
    coverage = np.full(batch_size, 0.0, dtype=np.float64)
    traces: dict[int, DtwTrace] = {}
    if not batch_size:
        return DtwResult(costs, coverage, traces)
    if direction == "reverse":
        north, north_weights = north[:, ::-1], north_weights[:, ::-1]
    indices = np.arange(sample_count)
    band = np.abs(indices[:, None] - indices) <= config.dtw_window
    south_valid = np.isfinite(south) & (south_weights > 0)
    north_valid = np.isfinite(north) & (north_weights > 0)
    comparable = south_valid[:, :, None] & north_valid[:, None, :] & band
    south_keep, north_keep = comparable.any(axis=2), comparable.any(axis=1)
    south_counts = np.count_nonzero(south_keep, axis=1)
    north_counts = np.count_nonzero(north_keep, axis=1)
    common_counts = np.minimum(south_counts, north_counts)
    coverage = common_counts / sample_count
    usable = np.flatnonzero(
        (common_counts >= config.min_valid_samples)
        & (coverage >= config.min_dtw_coverage)
    )
    if not len(usable):
        return DtwResult(costs, coverage, traces)

    south_count, north_count = south_counts[usable], north_counts[usable]
    south_length, north_length = int(south_count.max()), int(north_count.max())
    si = np.argsort(~south_keep[usable], axis=1, kind="stable")[:, :south_length]
    ni = np.argsort(~north_keep[usable], axis=1, kind="stable")[:, :north_length]
    distance = np.abs(si[:, :, None] - ni[:, None, :])
    allowed = (
        (distance <= config.dtw_window)
        & (np.arange(south_length)[None, :, None] < south_count[:, None, None])
        & (np.arange(north_length)[None, None, :] < north_count[:, None, None])
    )
    south_values, north_values = south[usable[:, None], si], north[usable[:, None], ni]
    reliability = np.sqrt(
        south_weights[usable[:, None], si][:, :, None]
        * north_weights[usable[:, None], ni][:, None, :]
    )
    residual = np.minimum(
        np.abs(south_values[:, :, None] - north_values[:, None, :]) / error_scale,
        _ROBUST_ERROR_CAP,
    )
    local_cost = residual * reliability + _LOW_QUALITY_COST * (1 - reliability)
    lag_cost = config.dtw_warp_weight * distance / max(1, config.dtw_window)
    local = np.where(allowed, local_cost + lag_cost, np.inf).transpose(1, 2, 0)
    repeats = config.dtw_max_repeat
    dp = np.full(
        (south_length, north_length, 1 + 2 * repeats, len(usable)),
        np.inf, dtype=np.float64,
    )
    occupied = allowed.any(axis=0)
    for i, j in np.argwhere(occupied):
        current = dp[i, j]
        point = local[i, j]
        if i == j == 0:
            current[0] = point
        elif i and j:
            current[0] = dp[i - 1, j - 1].min(axis=0) + point
        if repeats:
            step = point + config.dtw_warp_weight
            if j:
                previous = dp[i, j - 1]
                current[1] = np.minimum(previous[0], previous[repeats + 1:].min(axis=0)) + step
                current[2:repeats + 1] = previous[1:repeats] + step
            if i:
                previous = dp[i - 1, j]
                current[repeats + 1] = previous[:repeats + 1].min(axis=0) + step
                current[repeats + 2:] = previous[repeats + 1:-1] + step

    batch = np.arange(len(usable))
    end_s, end_n = south_count - 1, north_count - 1
    denominator = np.maximum(south_count, north_count)
    cover_penalty = config.dtw_coverage_weight * (1 - coverage[usable])
    costs[usable] = dp[end_s, end_n, :, batch].min(axis=1) / denominator + cover_penalty
    if trace:
        for index, original in enumerate(usable):
            if not np.isfinite(costs[original]):
                continue
            ranks = _walk_dtw(dp, int(end_s[index]), int(end_n[index]), index, repeats)
            repeated_steps = sum(
                a == c or b == d for (a, b), (c, d) in zip(ranks, ranks[1:])
            )
            points: list[DtwPoint] = []
            lag = 0.0
            for i, j in ranks:
                north_index = int(ni[index, j])
                if direction == "reverse":
                    north_index = sample_count - 1 - north_index
                points.append(DtwPoint(
                    int(si[index, i]), north_index, float(local_cost[index, i, j]),
                ))
                lag += float(lag_cost[index, i, j])
            traces[int(original)] = DtwTrace(
                tuple(points),
                sum(point.local_cost for point in points) / int(denominator[index]),
                (lag + config.dtw_warp_weight * repeated_steps) / int(denominator[index]),
                float(cover_penalty[index]), float(coverage[original]),
                repeated_steps / max(1, len(points) - 1),
            )
    return DtwResult(costs, coverage, traces)


@dataclass(frozen=True)
class SpanCurves:
    """一个视角所有几何有效锚点区间的跨度曲线。"""

    start: IndexArray
    end: IndexArray
    ratios: FloatArray
    log_values: FloatArray
    weights: FloatArray
    anchor_counts: IndexArray
    anchor_positions: FloatArray


def span_curves(view: ViewFeatures) -> SpanCurves:
    """生成任意有序锚点间的整体跨度，与中间实例数量解耦。"""
    eligible = np.flatnonzero([instance.geometry_valid for instance in view.instances])
    p, i = np.triu_indices(len(eligible), k=1)
    start, end = eligible[p], eligible[i]
    delta = view.centers[end] - view.centers[start]
    ratios = np.where(delta > 0, delta / (view.scales + _EPSILON), np.nan)
    weights = np.where(
        np.isfinite(ratios), np.minimum(view.weights[start], view.weights[end]), 0.0
    )
    gap_index = np.arange(len(view.stable_large_gaps))
    anchors = (
        (gap_index[None] >= start[:, None])
        & (gap_index[None] < end[:, None])
        & view.stable_large_gaps
    )
    counts = np.count_nonzero(anchors, axis=1)
    relative_position = (
        (gap_index[None] + 0.5 - start[:, None]) / (end - start)[:, None]
    )
    positions = np.full(len(start), np.nan, dtype=np.float64)
    np.divide(np.sum(relative_position * anchors, axis=1), counts,
              out=positions, where=counts > 0)
    return SpanCurves(start, end, ratios, np.log(ratios + _EPSILON), weights,
                      counts, positions)


def _span_pair_costs(
    south: SpanCurves,
    north: SpanCurves,
    config: GMSARailConfig,
    direction: Direction,
) -> FloatArray:
    """只计算跳过数相容的跨度对，其余位置保持无穷代价。

    过渡 (p,q)->(i,j) 隐含南侧跳过 i-p-1 根、北侧跳过 j-q-1 根。两端跨度在
    "本行中位间距"单位下的长度相差约为 ``log(1 + 差值 / 较短跨度)``，差值越大
    代价越高；超过 ``max_span_skip_difference`` 的配对通常已被 ``max_span_cost``
    拒绝，因此不必进入批量对齐。

    这是按跳过数差的近似剪枝，不是可证明的等价：局部间距差异很大时，长跳过边
    仍可能得到可用代价。默认值在项目 123 对数据上与关闭剪枝的结果逐位一致，
    可用 ``max_span_skip_difference=4096`` 关闭。
    """
    south_steps = south.end - south.start
    north_steps = north.end - north.start
    compatible = (
        np.abs(south_steps[:, None] - north_steps[None, :])
        <= config.max_span_skip_difference
    )
    costs = np.full(compatible.shape, np.inf, dtype=np.float64)
    rows, columns = np.nonzero(compatible)
    if not len(rows):
        return costs
    # 逐对比较：这里只需要 (p,i) 与 (q,j) 的组合，不是两张表的笛卡尔积。
    for start in range(0, len(rows), config.dtw_batch_size):
        stop = start + config.dtw_batch_size
        pair_rows, pair_columns = rows[start:stop], columns[start:stop]
        result = constrained_dtw(
            south.log_values[pair_rows], north.log_values[pair_columns],
            south.weights[pair_rows], north.weights[pair_columns],
            config, direction, config.span_error_scale,
        )
        costs[pair_rows, pair_columns] = result.costs
    return costs


def _curve_pair_costs(
    south: FloatArray,
    north: FloatArray,
    south_weights: FloatArray,
    north_weights: FloatArray,
    config: GMSARailConfig,
    direction: Direction,
    error_scale: float,
) -> FloatArray:
    """按资源配置批量比较两组曲线的笛卡尔积。"""
    pair_count = len(south) * len(north)
    costs = np.full(pair_count, np.inf, dtype=np.float64)
    for start in range(0, pair_count, config.dtw_batch_size):
        stop = min(start + config.dtw_batch_size, pair_count)
        s, n = np.divmod(np.arange(start, stop), len(north))
        result = constrained_dtw(
            south[s], north[n], south_weights[s], north_weights[n],
            config, direction, error_scale,
        )
        costs[start:stop] = result.costs
    return costs.reshape(len(south), len(north))


def _gap_group(view: ViewFeatures, size: int) -> tuple[FloatArray, FloatArray]:
    if size == 1:
        return view.log_gaps, view.gap_weights
    # 合并的是线性归一化中心距，不能相加其对数。
    return (
        np.log(view.ratios[:-1] + view.ratios[1:] + _EPSILON),
        np.minimum(view.gap_weights[:-1], view.gap_weights[1:]),
    )


def gap_subsequence_costs(
    south: ViewFeatures,
    north: ViewFeatures,
    s_spans: SpanCurves,
    n_spans: SpanCurves,
    config: GMSARailConfig,
    direction: Direction,
) -> FloatArray:
    """按 1:1、1:2、2:1 对齐各锚点之间完整的相邻间距子序列。"""
    m, n = len(south.instances), len(north.instances)
    allowed = ((1, 1), (1, 2), (2, 1)) if config.gap_merge_limit == 2 else ((1, 1),)
    steps: dict[tuple[int, int], FloatArray] = {}
    for a, b in allowed:
        s, sw = _gap_group(south, a)
        t, tw = _gap_group(north, b)
        distances = _curve_pair_costs(s, t, sw, tw, config, direction, config.gap_error_scale)
        steps[a, b] = max(a, b) * distances + config.gap_merge_penalty * (a + b - 2)
    # 外层按终点递增，最后两维同时保存各锚点起点的最小代价。
    dp = np.full((m, n, m, n), np.inf, dtype=np.float64)
    for i in range(m):
        for j in range(n):
            current = dp[i, j]
            current[i, j] = 0.0
            for (a, b), costs in steps.items():
                if i >= a and j >= b and np.isfinite(costs[i - a, j - b]):
                    np.minimum(
                        current, dp[i - a, j - b] + costs[i - a, j - b], out=current
                    )
    length = np.maximum(
        (s_spans.end - s_spans.start)[:, None],
        (n_spans.end - n_spans.start)[None, :],
    )
    result = dp[
        s_spans.end[:, None], n_spans.end[None, :],
        s_spans.start[:, None], n_spans.start[None, :],
    ]
    return result / length


@dataclass(frozen=True)
class Graph:
    """固定纵向方向的节点和锚点转移代价缓存。"""

    direction: Direction
    north_count: int
    node_cost: FloatArray
    leading_cost: FloatArray
    trailing_cost: FloatArray
    empty_cost: float
    s_spans: SpanCurves
    n_spans: SpanCurves
    span_costs: FloatArray
    gap_costs: FloatArray
    anchor_costs: FloatArray
    transition_costs: FloatArray
    predecessors: IndexArray
    edge_costs: FloatArray
    offsets: IndexArray


def build_graph(
    south: ViewFeatures,
    north: ViewFeatures,
    s_spans: SpanCurves,
    n_spans: SpanCurves,
    node_costs: FloatArray,
    config: GMSARailConfig,
    direction: Direction,
) -> Graph:
    """融合可靠跨度、间距与双侧稳定大间隙，并缓存合法单调边。"""
    m, n = node_costs.shape
    skip_s, skip_n = (
        np.r_[0.0, np.cumsum([
            config.skip_cost_min + instance.quality * (config.skip_cost_max - config.skip_cost_min)
            for instance in view.instances
        ])]
        for view in (south, north)
    )
    leading = (skip_s[:-1, None] + skip_n[None, :-1]).ravel()
    trailing = (skip_s[-1] - skip_s[1:, None] + skip_n[-1] - skip_n[None, 1:]).ravel()
    spans = _span_pair_costs(s_spans, n_spans, config, direction)
    if spans.size and config.gap_weight:
        gaps = gap_subsequence_costs(south, north, s_spans, n_spans, config, direction)
    else:
        gaps = np.full(spans.shape, np.inf, dtype=np.float64)
    anchor_count_delta = np.abs(s_spans.anchor_counts[:, None] - n_spans.anchor_counts)
    anchor_count_scale = np.maximum(s_spans.anchor_counts[:, None], n_spans.anchor_counts)
    count_error = anchor_count_delta / np.maximum(anchor_count_scale, 1)
    position_error = np.abs(s_spans.anchor_positions[:, None] - n_spans.anchor_positions)
    anchor_supported = (
        (s_spans.anchor_counts[:, None] > 0) & (n_spans.anchor_counts > 0)
        & (spans <= config.anchor_max_span_cost) & (gaps <= config.anchor_max_span_cost)
    )
    anchors = np.where(anchor_supported, (count_error + position_error) / 2, np.inf)

    numerator = np.full(spans.shape, 0.0, dtype=np.float64)
    denominator = np.full(spans.shape, 0.0, dtype=np.float64)
    for values, weight in (
        (spans, config.span_weight), (gaps, config.gap_weight),
        (anchors, config.anchor_weight),
    ):
        present = np.isfinite(values)
        numerator += np.where(present, values, 0.0) * weight
        denominator += present * weight
    transitions = np.full(spans.shape, np.inf, dtype=np.float64)
    # 有效跨度明确冲突时拒绝该边。缺失项只从可用权重中移除。
    usable = (denominator > 0) & (~np.isfinite(spans) | (spans <= config.max_span_cost))
    np.divide(numerator, denominator, out=transitions, where=usable)
    s, t = np.nonzero(usable)
    p, q = s_spans.start[s], n_spans.start[t]
    i, j = s_spans.end[s], n_spans.end[t]
    destination = i * n + j
    order = np.argsort(destination, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(destination, minlength=m * n))]
    skip_middle = skip_s[i] - skip_s[p + 1] + skip_n[j] - skip_n[q + 1]
    return Graph(
        direction, n, node_costs.ravel(), leading, trailing, float(skip_s[-1] + skip_n[-1]),
        s_spans, n_spans, spans, gaps, anchors, transitions,
        (p * n + q)[order], (transitions[s, t] + skip_middle)[order], offsets,
    )


@dataclass(frozen=True)
class PathOption:
    """一条非空完整路径，节点编码为 south_index * north_count + north_index。"""

    cost: float
    nodes: tuple[int, ...]
    direction: Direction


def solve_paths(
    graph: Graph,
    k_best: int,
    *,
    forbidden: int | None = None,
    measure_through: bool = False,
) -> tuple[list[PathOption], FloatArray | None]:
    """求 DAG 的精确 K 条不同路径，可禁止一个配对节点后复算。"""
    count = len(graph.node_cost)
    costs = np.full((count, k_best), np.inf, dtype=np.float64)
    back = np.full((count, k_best), -1, dtype=np.intp)
    active = np.flatnonzero(np.isfinite(graph.node_cost))
    for node in active:
        if node == forbidden:
            continue
        start, stop = graph.offsets[node:node + 2]
        previous = graph.predecessors[start:stop]
        prefix = costs[previous] + graph.edge_costs[start:stop, None]
        candidates = np.r_[graph.leading_cost[node], prefix.ravel()]
        if k_best == 1:
            selected = np.asarray([int(np.argmin(candidates))])
        else:
            selected = np.argsort(candidates, kind="stable")[:k_best]
        valid = selected[np.isfinite(candidates[selected])]
        costs[node, :len(valid)] = candidates[valid] + graph.node_cost[node]
        ranks = np.flatnonzero(valid > 0)
        links = valid[ranks] - 1
        back[node, ranks] = previous[links // k_best] * k_best + links % k_best

    complete = (costs + graph.trailing_cost[:, None]).ravel()
    if k_best == 1 and len(complete):
        ending = np.asarray([int(np.argmin(complete))])
    else:
        ending = np.argsort(complete, kind="stable")[:k_best]
    paths: list[PathOption] = []
    for index in ending:
        if not np.isfinite(complete[index]):
            continue
        trace: list[int] = []
        link = int(index)
        while link >= 0:
            node, rank = divmod(link, k_best)
            trace.append(node)
            link = int(back[node, rank])
        paths.append(PathOption(float(complete[index]), tuple(reversed(trace)), graph.direction))

    if not measure_through:
        return paths, None
    # 逆向最短后缀与最优前缀拼接，给每个候选得到完整路径代价。
    tail = graph.trailing_cost.copy()
    for node in active[::-1]:
        if node == forbidden:
            continue
        start, stop = graph.offsets[node:node + 2]
        previous = graph.predecessors[start:stop]
        alternatives = graph.edge_costs[start:stop] + graph.node_cost[node] + tail[node]
        tail[previous] = np.minimum(tail[previous], alternatives)
    return paths, costs[:, 0] + tail


def solve_paths_avoiding(
    graph: Graph, forbidden_nodes: Sequence[int]
) -> list[PathOption | None]:
    """批量求"禁止某个配对节点"后的最优完整路径。

    逐节点调用 :func:`solve_paths` 时，每个节点都要重跑一遍全图动态规划；
    这里把被禁止的节点放在批量维上，一次遍历即可得到所有结果。递推、来源
    选择和并列代价的优先级与 ``solve_paths(k_best=1, forbidden=node)`` 完全
    相同，因此代价与路径逐位一致。

    Args:
        graph: 固定纵向方向的有向无环图。
        forbidden_nodes: 依次被禁止的配对节点。

    Returns:
        与 ``forbidden_nodes`` 等长的路径列表；无法避开或没有可行路径时为 None。
    """
    if not forbidden_nodes:
        return []
    count = len(graph.node_cost)
    batch = len(forbidden_nodes)
    costs = np.full((count, batch), np.inf, dtype=np.float64)
    back = np.full((count, batch), -1, dtype=np.intp)
    blocked_at = {int(node): index for index, node in enumerate(forbidden_nodes)}
    for node in np.flatnonzero(np.isfinite(graph.node_cost)):
        start, stop = graph.offsets[node:node + 2]
        if stop > start:
            previous = graph.predecessors[start:stop]
            candidates = costs[previous] + graph.edge_costs[start:stop, None]
            best_edge = candidates.min(axis=0)
            best_source = previous[candidates.argmin(axis=0)]
        else:
            best_edge = np.full(batch, np.inf, dtype=np.float64)
            best_source = np.full(batch, -1, dtype=np.intp)
        # 并列时优先使用起始代价，与 solve_paths 的候选顺序一致。
        use_edge = best_edge < graph.leading_cost[node]
        value = np.where(use_edge, best_edge, graph.leading_cost[node])
        value = value + graph.node_cost[node]
        source = np.where(use_edge, best_source, -1)
        blocked = blocked_at.get(int(node))
        if blocked is not None:
            value[blocked] = np.inf
            source[blocked] = -1
        costs[node] = value
        back[node] = source

    complete = costs + graph.trailing_cost[:, None]
    endings = complete.argmin(axis=0)
    paths: list[PathOption | None] = []
    for index in range(batch):
        ending = int(endings[index])
        total = float(complete[ending, index])
        if not np.isfinite(total):
            paths.append(None)
            continue
        trace: list[int] = []
        node = ending
        while node >= 0:
            trace.append(node)
            node = int(back[node, index])
        paths.append(PathOption(total, tuple(reversed(trace)), graph.direction))
    return paths


def path_transitions(
    graph: Graph, path: PathOption
) -> list[tuple[int, int, int, int, int, int]]:
    """定位该路径实际采用的锚点转移及其跨度曲线索引。"""
    by_south = {
        (int(a), int(b)): index
        for index, (a, b) in enumerate(zip(graph.s_spans.start, graph.s_spans.end))
    }
    by_north = {
        (int(a), int(b)): index
        for index, (a, b) in enumerate(zip(graph.n_spans.start, graph.n_spans.end))
    }
    edges = []
    for before, after in zip(path.nodes, path.nodes[1:]):
        p, q = divmod(before, graph.north_count)
        i, j = divmod(after, graph.north_count)
        edges.append((p, q, i, j, by_south[p, i], by_north[q, j]))
    return edges
