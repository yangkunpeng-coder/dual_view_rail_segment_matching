"""GMSA-Rail V3 全局多截面结构对齐。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import comb, isfinite
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
from core_algorithm.matching._gmsa_alignment import (
    DtwTrace,
    Graph,
    PathOption,
    build_graph,
    constrained_dtw,
    path_transitions,
    solve_paths,
    solve_paths_avoiding,
    span_curves,
)
from core_algorithm.matching._gmsa_geometry import (
    Direction,
    FloatArray,
    GMSARailConfig,
    Instance,
    RailView,
    Row,
    ViewFeatures,
    geometry_rows,
    instance_row,
    measure_view,
)
from data_preprocess.common.config import PROJECT_ROOT
from utils.instance_masks import as_mask_stack

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.engine.results import Masks

_EPSILON = 1e-9
_NODE_COST_FLOOR = 0.15
_WIDTH_VARIATION_WEIGHT = 0.10
_CANDIDATE_LIMIT = 3
_VERSION = "GMSA-Rail V3.1"


class GMSAMatch(RailMatch):
    """自动接受的匹配和可追溯的结构证据。"""

    evidence: Row


class GMSADiagnostics(TypedDict):
    """可序列化的逐图对、路径、几何和耗时诊断。"""

    config: dict[str, object]
    summary: Row
    timings_ms: dict[str, float]
    accepted: list[Row]
    unmatched: list[Row]
    paths: list[Row]
    transitions: list[Row]
    dtw_alignment: list[Row]
    instances: list[Row]
    sections: list[Row]
    gap_fingerprints: list[Row]


class GMSAMatchingResult(RailMatchingResult):
    """兼容项目结果协议，按状态保留完整的配对证据。"""

    tentative_matches: list[Row]
    ambiguous_candidates: list[Row]
    diagnostics: GMSADiagnostics | None


@dataclass(frozen=True)
class _Search:
    """两个方向的候选路径、逐边排除和最优经过代价。"""

    best: PathOption | None
    best_cost: float
    second_cost: float | None
    global_margin: float | None
    competition_count: int
    support_status: str
    direction_best_paths: dict[Direction, PathOption | None]
    through: FloatArray
    exclusions: dict[int, float]
    near_paths: tuple[PathOption, ...]
    near_margin: float
    direction_ambiguous: bool
    direction_margin: float | None
    direction_costs: dict[Direction, float]


@dataclass(frozen=True)
class _Transition:
    """路径上实际使用的边、代价分项及回溯后的纵向对齐。"""

    direction: Direction
    p: int
    q: int
    i: int
    j: int
    cost: float
    span_cost: float | None
    gap_cost: float | None
    anchor_cost: float | None
    trace: DtwTrace | None
    position_error: float | None


def _options(
    config: GMSARailConfig | Mapping[str, object] | None,
) -> GMSARailConfig:
    if config is None:
        return GMSARailConfig()
    if isinstance(config, GMSARailConfig):
        return config
    return GMSARailConfig(**config)


def _node_costs(
    south: ViewFeatures, north: ViewFeatures, config: GMSARailConfig
) -> FloatArray:
    """只用可用性和低权重宽度形态评分，身份由转移结构决定。"""
    costs = np.full((len(south.instances), len(north.instances)), np.inf, dtype=np.float64)
    for i, first in enumerate(south.instances):
        if not first.geometry_valid:
            continue
        for j, second in enumerate(north.instances):
            if not second.geometry_valid:
                continue
            quality = (first.quality * second.quality) ** 0.5
            width_error = abs(first.track.width_variation - second.track.width_variation)
            costs[i, j] = config.node_weight * (
                _NODE_COST_FLOOR + 1 - quality + _WIDTH_VARIATION_WEIGHT * min(1.0, width_error)
            )
    return costs


def _finite(value: float) -> float | None:
    return float(value) if isfinite(value) else None


def _path_key(path: PathOption) -> tuple[float, tuple[int, ...], bool]:
    return path.cost, path.nodes, path.direction == "reverse"


def _relative_margin(best: float, alternative: float) -> float:
    return max(0.0, alternative - best) / max(abs(best), _EPSILON)


def _search_paths(
    graphs: Sequence[Graph], node_costs: FloatArray, empty_cost: float,
    config: GMSARailConfig, resource_limited: bool,
) -> _Search:
    """允许全跳过，并用相同缓存复算排除每个最优配对后的最优值。"""
    through = np.full(node_costs.size, np.inf, dtype=np.float64)
    found: list[PathOption] = []
    direction_costs: dict[Direction, float] = {}
    direction_best_paths: dict[Direction, PathOption | None] = {}
    for graph in graphs:
        paths, through_direction = solve_paths(graph, config.k_best, measure_through=True)
        if through_direction is not None:
            np.minimum(through, through_direction, out=through)
        direction_best_paths[graph.direction] = paths[0] if paths else None
        if paths:
            direction_costs[graph.direction] = paths[0].cost
            found.extend(paths)
    found.sort(key=_path_key)
    best = found[0] if found and found[0].cost < empty_cost else None
    best_cost = empty_cost if best is None else best.cost
    near_margin = max(config.pair_exclusion_margin, config.path_relative_margin * best_cost)
    exclusions: dict[int, float] = {}
    if best is not None:
        nodes = tuple(int(node) for node in best.nodes)
        avoided = {
            graph.direction: solve_paths_avoiding(graph, nodes) for graph in graphs
        }
        for index, node in enumerate(nodes):
            alternatives = [
                paths[index] for paths in avoided.values() if paths[index] is not None
            ]
            alternative = min(alternatives, key=_path_key, default=None)
            cost = min(empty_cost, alternative.cost) if alternative is not None else empty_cost
            exclusions[node] = max(0.0, cost - best_cost)
            if alternative is not None:
                found.append(alternative)

    by_relation: dict[tuple[int, ...], PathOption] = {}
    by_direction: dict[tuple[Direction, tuple[int, ...]], PathOption] = {}
    for path in sorted(found, key=_path_key):
        by_relation.setdefault(path.nodes, path)
        by_direction.setdefault((path.direction, path.nodes), path)
    competition_count = len(by_relation)
    second_cost = None
    if resource_limited:
        support = "resource_limit"
    elif not competition_count:
        support = "no_feasible_path"
    elif competition_count == 1:
        support = "single_path_no_competition"
    elif best is None:
        support = "empty_path_preferred"
        second_cost = min(path.cost for path in by_relation.values())
    else:
        support = "competing_paths"
        second_cost = min(
            empty_cost, min(
                path.cost for path in by_relation.values() if path.nodes != best.nodes
            ),
        )
    direction_margin = None
    if best is not None and len(best.nodes) >= 2 and len(direction_costs) == 2:
        direction_margin = _relative_margin(
            min(direction_costs.values()), max(direction_costs.values())
        )
    direction_ambiguous = (
        direction_margin is not None and direction_margin <= config.direction_relative_margin
    )
    near_paths = tuple(
        path for path in by_direction.values()
        if best is not None and path.cost - best_cost <= near_margin + _EPSILON
    )
    return _Search(
        best, best_cost, second_cost,
        None if second_cost is None else _relative_margin(best_cost, second_cost),
        competition_count, support, direction_best_paths, through, exclusions,
        near_paths, near_margin, direction_ambiguous, direction_margin, direction_costs,
    )


def _transition_evidence(
    paths: Sequence[PathOption],
    graphs: Mapping[Direction, Graph],
    north: ViewFeatures,
    config: GMSARailConfig,
) -> tuple[dict[tuple[Direction, int, int], _Transition], list[list[_Transition]]]:
    """只对保留路径实际经过的边回溯 DTW，复用重复边。"""
    edge_lists = [path_transitions(graphs[path.direction], path) for path in paths]
    cache: dict[tuple[Direction, int, int], _Transition] = {}
    for direction, graph in graphs.items():
        edges = list(dict.fromkeys(
            edge for path, path_edges in zip(paths, edge_lists, strict=True)
            if path.direction == direction for edge in path_edges
        ))
        for start in range(0, len(edges), config.dtw_batch_size):
            batch = edges[start:start + config.dtw_batch_size]
            south_indices = [edge[4] for edge in batch]
            north_indices = [edge[5] for edge in batch]
            spans_s, spans_n = graph.s_spans, graph.n_spans
            alignment = constrained_dtw(
                spans_s.log_values[south_indices], spans_n.log_values[north_indices],
                spans_s.weights[south_indices], spans_n.weights[north_indices],
                config, direction, config.span_error_scale, trace=True,
            )
            for index, (p, q, i, j, s, n) in enumerate(batch):
                trace = alignment.traces.get(index)
                error = None
                if trace is not None:
                    # 将南侧无量纲跨度还原到北侧局部尺度再计算宽度归一化残差。
                    errors = [
                        abs(spans_s.ratios[s, point.south_index]
                            - spans_n.ratios[n, point.north_index])
                        * north.scales[point.north_index] / max(1, north.width_px - 1)
                        for point in trace.points
                    ]
                    error = float(np.median(errors))
                cache[direction, s, n] = _Transition(
                    direction, p, q, i, j, float(graph.transition_costs[s, n]),
                    _finite(graph.span_costs[s, n]), _finite(graph.gap_costs[s, n]),
                    _finite(graph.anchor_costs[s, n]), trace, error,
                )
    return cache, [
        [cache[path.direction, s, n] for _, _, _, _, s, n in edges]
        for path, edges in zip(paths, edge_lists, strict=True)
    ]


def _instance_keys(south: Instance, north: Instance) -> Row:
    return {
        f"{side}_{name}": value
        for side, instance in (("south", south), ("north", north))
        for name, value in (
            ("mask_index", instance.record["mask_index"]),
            ("instance_id", instance.record["instance_id"]),
            ("shape_index", instance.shape_index), ("group_id", instance.group_id),
            ("order_index", instance.rank),
        )
    }


def _median(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(np.median(finite)) if finite else None


def _candidate_nodes(through: FloatArray, node: int, north_count: int) -> list[int]:
    """给出同南侧或同北侧的可行替代配对。"""
    i, j = divmod(node, north_count)
    candidates = set(range(i * north_count, (i + 1) * north_count))
    candidates.update(range(j, len(through), north_count))
    candidates.discard(node)
    return sorted(
        (index for index in candidates if np.isfinite(through[index])),
        key=lambda index: (through[index], index),
    )


def _direction_nodes(
    search: _Search,
) -> tuple[set[int], set[int], set[int]]:
    """返回两个方向最优路径的节点集合及其交集。

    纵向方向只表示南北采样方向，不代表实例身份。两条最优路径共有的
    关系对方向选择不敏感，可作为方向共识继续执行配对级检查。
    """
    same_path = search.direction_best_paths.get("same")
    reverse_path = search.direction_best_paths.get("reverse")
    same = set(same_path.nodes) if same_path is not None else set()
    reverse = set(reverse_path.nodes) if reverse_path is not None else set()
    return same, reverse, same & reverse


def _direction_support(node: int, same: set[int], reverse: set[int]) -> str:
    """标注单条关系的方向支持来源。"""
    if node in same and node in reverse:
        return "both_directions"
    if node in same:
        return "same_only"
    if node in reverse:
        return "reverse_only"
    return "not_applicable"


def _direction_summary(
    search: _Search, evidence_count: int, minimum_transitions: int
) -> Row:
    """生成 Pair 级方向诊断，供复核方向共识与冲突规模。"""
    same, reverse, consensus = _direction_nodes(search)
    union = same | reverse
    has_both = bool(search.direction_best_paths.get("same")) and bool(
        search.direction_best_paths.get("reverse")
    )
    if not has_both:
        decision = "single_direction"
    elif evidence_count < minimum_transitions:
        decision = "insufficient_evidence"
    elif consensus == union:
        decision = "direction_consensus"
    else:
        decision = "direction_conflict"
    return {
        "same_best_path_match_count": len(same),
        "reverse_best_path_match_count": len(reverse),
        "direction_consensus_pair_count": len(consensus),
        "direction_conflict_pair_count": len(union - consensus),
        # 只有一个方向时不存在跨方向比较，比率留空而不报 0。
        "direction_consensus_ratio": (
            len(consensus) / len(union) if has_both and union else None
        ),
        "direction_decision": decision,
    }


def _near_optimal_partner(costs: FloatArray, near_limit: float) -> bool:
    """判断该实例是否存在落在近优门限内的完整路径候选。"""
    if not isfinite(near_limit):
        return False
    return bool(np.any(costs <= near_limit + _EPSILON))


def _decide(
    south: ViewFeatures, north: ViewFeatures, node_costs: FloatArray,
    search: _Search, edges: Sequence[_Transition], config: GMSARailConfig,
) -> tuple[list[RailMatch], list[Row], list[Row], list[Row], dict[int, Row]]:
    """按方向共识、有效跨度、排除代价和排序质量决定接受状态。"""
    matches: list[RailMatch] = []
    accepted: list[Row] = []
    tentative: list[Row] = []
    ambiguous: list[Row] = []
    decisions: dict[int, Row] = {}
    best = search.best
    if best is None:
        return matches, accepted, tentative, ambiguous, decisions
    n = len(north.instances)
    evidence_count = sum(edge.trace is not None for edge in edges)
    near_limit = best.cost + search.near_margin + _EPSILON
    same_nodes, reverse_nodes, consensus_nodes = _direction_nodes(search)
    # 方向共识只在证据量达标时生效：方向接近不是身份正确的证据。
    is_direction_gated = (
        search.direction_ambiguous
        and evidence_count >= config.min_span_transitions
        and bool(same_nodes)
        and bool(reverse_nodes)
    )
    for index, node in enumerate(best.nodes):
        i, j = divmod(node, n)
        first, second = south.instances[i], north.instances[j]
        adjacent = edges[max(0, index - 1):index + 1]
        span = _median([edge.span_cost for edge in adjacent])
        gap = _median([edge.gap_cost for edge in adjacent])
        anchor = _median([edge.anchor_cost for edge in adjacent])
        quality = float(node_costs[i, j])
        margin = search.exclusions[node]
        alternatives = [
            other for other in _candidate_nodes(search.through, node, n)
            if search.through[other] <= near_limit
        ]
        direction_support = _direction_support(node, same_nodes, reverse_nodes)
        status, reason = "accepted", ""
        if search.competition_count == 1:
            status, reason = "tentative", "single_path_no_competition"
        elif is_direction_gated and node not in consensus_nodes:
            # 只有一种方向支持该关系，保留歧义；两种方向共有的关系继续检查。
            status, reason = "ambiguous", "direction_relation_conflict"
        elif alternatives or margin <= search.near_margin + _EPSILON:
            status, reason = "ambiguous", "near_optimal_alternative"
        elif evidence_count < config.min_span_transitions or span is None:
            status, reason = "tentative", "insufficient_span_transitions"
        elif first.order_uncertain or second.order_uncertain:
            status, reason = "tentative", "order_uncertain"
        row: Row = {
            **_instance_keys(first, second), "status": status, "reason": reason,
            "total_cost": quality + (_median([edge.cost for edge in adjacent]) or 0.0),
            "quality_cost": quality, "span_cost": span, "gap_cost": gap,
            "anchor_cost": anchor, "path_cost": best.cost, "cost_gap": 0.0,
            "valid_dtw_points": sum(
                len(edge.trace.points) for edge in adjacent if edge.trace is not None
            ),
            "evidence_transition_count": evidence_count,
            "best_path_cost": best.cost, "second_path_cost": search.second_cost,
            "global_path_margin": search.global_margin, "pair_exclusion_margin": margin,
            "alternative_partner_count": len(alternatives),
            "longitudinal_direction": best.direction,
            "direction_support": direction_support,
            "direction_consensus": node in consensus_nodes,
            "evidence_status": "structural_competition" if status == "accepted" else reason,
            "candidate_rank": 1,
        }
        decisions[node] = row
        if status == "accepted":
            position_error = _median([edge.position_error for edge in adjacent])
            if position_error is None:
                raise RuntimeError("已接受配对缺少有效跨度残差")
            shared_id = f"R{len(matches) + 1:03d}"
            row["shared_id"] = shared_id
            match: GMSAMatch = {
                "shared_id": shared_id, "south": first.record, "north": second.record,
                "position_error": position_error, "evidence": row,
            }
            matches.append(match)
            accepted.append(row)
        elif status == "tentative":
            tentative.append(row)
        else:
            ambiguous.append(row)

    # 经过代价是完整路径代价，可发现 K-best 清单以外的近似替代对象。
    ordered = sorted(
        (int(node) for node in np.flatnonzero(search.through <= near_limit)
         if node not in decisions),
        key=lambda node: (search.through[node], node),
    )
    for node in ordered:
        i, j = divmod(node, n)
        cost = float(search.through[node])
        row_costs = search.through[i * n:(i + 1) * n]
        rank = 1 + int(np.count_nonzero(row_costs < cost - _EPSILON))
        ambiguous.append({
            **_instance_keys(south.instances[i], north.instances[j]),
            "status": "ambiguous", "reason": "near_optimal_alternative",
            "candidate_rank": rank, "path_cost": cost, "cost_gap": max(0.0, cost - best.cost),
            "best_path_cost": best.cost, "second_path_cost": search.second_cost,
            "global_path_margin": search.global_margin, "evidence_status": "alternative_path",
            "direction_support": "not_applicable", "direction_consensus": False,
        })
    return matches, accepted, tentative, ambiguous, decisions


def _unmatched(
    view: ViewFeatures, other: ViewFeatures, decisions: Mapping[int, Row],
    search: _Search, north_count: int,
) -> tuple[list[UnmatchedRail], list[Row]]:
    """保留所有未自动接受的输入实例及最多三个候选。"""
    best_by_index = {
        (node // north_count if view.side == "south" else node % north_count): row
        for node, row in decisions.items()
    }
    unmatched: list[UnmatchedRail] = []
    rows: list[Row] = []
    costs = search.through.reshape(
        (len(view.instances), len(other.instances)) if view.side == "south"
        else (len(other.instances), len(view.instances))
    )
    if view.side == "north":
        costs = costs.T
    # 近优门限与 _decide() 一致，保证近优关系不会同时出现在两张表里。
    near_limit = search.best_cost + search.near_margin + _EPSILON
    for index, instance in enumerate(view.instances):
        decision = best_by_index.get(index, {})
        if decision.get("status") == "accepted":
            continue
        status, reason = "unmatched", "no_reliable_partner"
        if not instance.geometry_valid:
            status, reason = "invalid_geometry", "invalid_geometry"
        elif not other.instances:
            reason = "other_side_empty"
        elif search.support_status == "resource_limit":
            reason = "resource_limit"
        elif decision.get("status") in ("tentative", "ambiguous"):
            status = str(decision["status"])
            reason = str(decision["reason"])
        elif search.best is None:
            reason = "insufficient_alignment_evidence"
        elif _near_optimal_partner(costs[index], near_limit):
            # 最优路径没有该实例，但仍存在近优完整路径经过它：应记为歧义候选。
            status, reason = "ambiguous", "near_optimal_alternative"
        common_reason = "ambiguous_alignment" if status == "ambiguous" else reason
        unmatched.append(make_unmatched_rail(instance.record, common_reason))
        row: Row = {
            **instance_row(instance, view.side, len(view.rows)),
            "status": status, "reason": reason,
        }
        candidates = np.argsort(costs[index], kind="stable")[:_CANDIDATE_LIMIT]
        for rank, partner_index in enumerate(candidates, start=1):
            cost = float(costs[index, partner_index])
            if not isfinite(cost):
                continue
            partner = other.instances[partner_index]
            row[f"candidate_{rank}_mask_index"] = partner.record["mask_index"]
            row[f"candidate_{rank}_group_id"] = partner.group_id
            row[f"candidate_{rank}_order_index"] = partner.rank
            row[f"candidate_{rank}_path_cost"] = cost
        rows.append(row)
    return unmatched, rows


def _trace_rows(
    paths: Sequence[PathOption], path_edges: Sequence[Sequence[_Transition]],
    south: ViewFeatures, north: ViewFeatures, config: GMSARailConfig,
) -> tuple[list[Row], list[Row], list[Row]]:
    """将近优路径使用的转移和有效采样写成可联查的行。"""
    ranks: dict[_Transition, list[int]] = {}
    path_rows: list[Row] = []
    for rank, (path, edges) in enumerate(zip(paths, path_edges, strict=True), start=1):
        path_rows.append({
            "path_rank": rank, "path_cost": path.cost, "direction": path.direction,
            "match_count": len(path.nodes),
            "evidence_transition_count": sum(edge.trace is not None for edge in edges),
            "mask_index_pairs": "|".join(
                f"{south.instances[i].record['mask_index']}:"
                f"{north.instances[j].record['mask_index']}"
                for i, j in (divmod(node, len(north.instances)) for node in path.nodes)
            ),
        })
        for edge in edges:
            ranks.setdefault(edge, []).append(rank)
    transitions: list[Row] = []
    dtw_rows: list[Row] = []
    for index, (edge, used_by) in enumerate(ranks.items(), start=1):
        transition_id = f"T{index:04d}"
        trace = edge.trace
        transitions.append({
            "transition_id": transition_id, "path_ranks": "|".join(map(str, used_by)),
            "is_best_path": 1 in used_by, "direction": edge.direction,
            "south_start_mask_index": south.instances[edge.p].record["mask_index"],
            "south_end_mask_index": south.instances[edge.i].record["mask_index"],
            "north_start_mask_index": north.instances[edge.q].record["mask_index"],
            "north_end_mask_index": north.instances[edge.j].record["mask_index"],
            "south_start_order_index": edge.p + 1, "south_end_order_index": edge.i + 1,
            "north_start_order_index": edge.q + 1, "north_end_order_index": edge.j + 1,
            "south_skipped_count": edge.i - edge.p - 1,
            "north_skipped_count": edge.j - edge.q - 1,
            "transition_cost": edge.cost, "span_cost": edge.span_cost,
            "gap_cost": edge.gap_cost, "anchor_cost": edge.anchor_cost,
            "position_error_norm": edge.position_error,
            "valid_dtw_points": 0 if trace is None else len(trace.points),
            "dtw_coverage": None if trace is None else trace.coverage,
            "dtw_warp_fraction": None if trace is None else trace.warp_fraction,
            "dtw_distance_cost": None if trace is None else trace.distance_cost,
            "dtw_warp_cost": None if trace is None else trace.warp_cost,
            "dtw_coverage_cost": None if trace is None else trace.coverage_cost,
        })
        if trace is not None:
            for step, point in enumerate(trace.points):
                s, n = point.south_index, point.north_index
                dtw_rows.append({
                    "transition_id": transition_id, "step": step,
                    "direction": edge.direction, "south_sample_index": s,
                    "north_sample_index": n, "south_height": config.sample_positions[s],
                    "north_height": config.sample_positions[n],
                    "south_y_px": south.rows[s], "north_y_px": north.rows[n],
                    "valid": True, "local_cost": point.local_cost,
                })
    return path_rows, transitions, dtw_rows


def match_south_north_rail_gmsa(
    south_instances: RailView,
    north_instances: RailView,
    config: GMSARailConfig | Mapping[str, object] | None = None,
    return_diagnostics: bool = True,
) -> GMSAMatchingResult:
    """在同一采集图对中以多截面跨度和单调路径关联钢轨。

    Args:
        south_instances: 南侧现有解析器生成的逐 Polygon 布尔 Mask 及来源编号。
        north_instances: 北侧实例，约定同 south_instances。
        config: 工程阈值及资源上限，接受配置对象或同名字段字典。
        return_diagnostics: 为 True 时保存采样、路径和耗时诊断。

    Returns:
        自动接受、待确认和歧义关系及未自动匹配的实例。自动接受的
        position_error 是局部跨度在北图宽度归一化后的残差。

    Raises:
        ValueError: GMSA 配置无效。
    """
    settings = _options(config)
    start = perf_counter()
    south = measure_view(south_instances, "south", settings)
    north = measure_view(north_instances, "north", settings)
    node_costs = _node_costs(south, north, settings)
    feature_ms = 1000 * (perf_counter() - start)
    valid_south = sum(instance.geometry_valid for instance in south.instances)
    valid_north = sum(instance.geometry_valid for instance in north.instances)
    transition_count = comb(valid_south, 2) * comb(valid_north, 2)
    resource_limited = (
        node_costs.size > settings.max_candidate_nodes
        or transition_count > settings.max_transitions
    )
    start = perf_counter()
    graphs: dict[Direction, Graph] = {}
    if valid_south and valid_north and not resource_limited:
        s_spans, n_spans = span_curves(south), span_curves(north)
        directions: tuple[Direction, ...] = (
            ("same", "reverse") if settings.longitudinal_direction == "auto"
            else (settings.longitudinal_direction,)
        )
        graphs = {
            direction: build_graph(
                south, north, s_spans, n_spans, node_costs, settings, direction
            )
            for direction in directions
        }
    transition_ms = 1000 * (perf_counter() - start)
    empty_cost = sum(
        settings.skip_cost_min + instance.quality * (
            settings.skip_cost_max - settings.skip_cost_min
        )
        for view in (south, north) for instance in view.instances
    )
    start = perf_counter()
    search = _search_paths(tuple(graphs.values()), node_costs, empty_cost, settings,
                           resource_limited)
    dp_ms = 1000 * (perf_counter() - start)
    start = perf_counter()
    best = search.best
    paths = search.near_paths if return_diagnostics else (() if best is None else (best,))
    _, path_edges = _transition_evidence(paths, graphs, north, settings)
    best_edges = path_edges[0] if path_edges else []
    matches, accepted, tentative, ambiguous, decisions = _decide(
        south, north, node_costs, search, best_edges, settings
    )
    unmatched_south, south_rows = _unmatched(
        south, north, decisions, search, len(north.instances)
    )
    unmatched_north, north_rows = _unmatched(
        north, south, decisions, search, len(north.instances)
    )
    diagnostics = None
    if return_diagnostics:
        s_instances, s_sections, s_gaps = geometry_rows(south, settings)
        n_instances, n_sections, n_gaps = geometry_rows(north, settings)
        path_rows, transition_rows, dtw_rows = _trace_rows(
            paths, path_edges, south, north, settings
        )
        span_count = sum(edge.trace is not None for edge in best_edges)
        summary: Row = {
            "algorithm_version": _VERSION, "support_status": search.support_status,
            "resource_limited": resource_limited,
            "candidate_pair_count": valid_south * valid_north,
            "theoretical_transition_count": transition_count,
            "best_path_cost": search.best_cost, "second_path_cost": search.second_cost,
            "empty_path_cost": empty_cost, "global_path_margin": search.global_margin,
            "path_competition_count": search.competition_count,
            "evidence_transition_count": span_count,
            "best_path_transition_count": len(best_edges),
            "effective_span_transition_fraction": (
                span_count / len(best_edges) if best_edges else None
            ),
            "longitudinal_direction": best.direction if best is not None else None,
            "direction_ambiguous": search.direction_ambiguous,
            "direction_relative_margin": search.direction_margin,
            "same_best_path_cost": search.direction_costs.get("same"),
            "reverse_best_path_cost": search.direction_costs.get("reverse"),
            "accepted_count": len(matches), "tentative_count": len(tentative),
            "ambiguous_candidate_count": len(ambiguous),
            "ambiguous_best_count": sum(row["status"] == "ambiguous"
                                        for row in decisions.values()),
            "unmatched_instance_count": len(south_rows) + len(north_rows),
            "invalid_geometry_count": len(south.instances) + len(north.instances)
            - valid_south - valid_north,
            "order_uncertain_count": sum(
                instance.order_uncertain
                for view in (south, north) for instance in view.instances
            ),
            "median_dtw_coverage": _median([
                edge.trace.coverage for edge in best_edges if edge.trace is not None
            ]),
            "median_dtw_warp_fraction": _median([
                edge.trace.warp_fraction for edge in best_edges if edge.trace is not None
            ]),
            "median_span_cost": _median([edge.span_cost for edge in best_edges]),
            "median_gap_cost": _median([edge.gap_cost for edge in best_edges]),
        }
        summary.update(
            _direction_summary(search, span_count, settings.min_span_transitions)
        )
        diagnostics = GMSADiagnostics(
            config=asdict(settings), summary=summary, accepted=accepted,
            unmatched=south_rows + north_rows, paths=path_rows, transitions=transition_rows,
            dtw_alignment=dtw_rows, instances=s_instances + n_instances,
            sections=s_sections + n_sections, gap_fingerprints=s_gaps + n_gaps,
            timings_ms={
                "feature_time_ms": feature_ms, "transition_time_ms": transition_ms,
                "dp_time_ms": dp_ms, "diagnostics_time_ms": 1000 * (perf_counter() - start),
            },
        )
    return {
        "south_image_path": Path(south_instances.image_path).as_posix(),
        "north_image_path": Path(north_instances.image_path).as_posix(),
        "matches": matches, "unmatched_south": unmatched_south,
        "unmatched_north": unmatched_north, "tentative_matches": tentative,
        "ambiguous_candidates": ambiguous, "diagnostics": diagnostics,
    }


def _view_from_masks(
    image_path: str | Path, masks: np.ndarray | Tensor | Masks | None,
) -> RailView:
    """按项目已有预测 Mask 转换规则构造输入视图。"""
    relative = Path(image_path)
    absolute = (PROJECT_ROOT / relative).resolve()
    if relative.anchor or not absolute.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"请提供相对于项目根目录的图片路径 {image_path}")
    with Image.open(absolute) as image:
        width, height = image.size
    return RailView(relative, as_mask_stack(masks, (height, width)))


def match_south_north_rail_gmsa_masks(
    south_image_path: str | Path,
    north_image_path: str | Path,
    south_masks: np.ndarray | Tensor | Masks | None,
    north_masks: np.ndarray | Tensor | Masks | None,
    *,
    config: GMSARailConfig | Mapping[str, object] | None = None,
    return_diagnostics: bool = True,
) -> GMSAMatchingResult:
    """用旧项目的路径和 Mask 调用协议运行 GMSA。

    Args:
        south_image_path: 南侧原图的项目相对路径。
        north_image_path: 北侧原图的项目相对路径。
        south_masks: 南侧 YOLO Mask、NumPy 数组、Tensor 或 None。
        north_masks: 北侧掩码，约定同 south_masks。
        config: GMSA 配置对象或字段映射。
        return_diagnostics: 是否保留详细诊断。

    Returns:
        与视图接口相同的匹配结果。预测 Mask 仅用输入索引回溯。
    """
    return match_south_north_rail_gmsa(
        _view_from_masks(south_image_path, south_masks),
        _view_from_masks(north_image_path, north_masks),
        config, return_diagnostics,
    )


def _run_demo(show_missed_matches_only: bool = False) -> None:
    """按项目统一预览协议显示 GMSA 自动接受的关联。"""
    from core_algorithm.common.preview import preview_matching_results

    preview_matching_results(
        match_south_north_rail_gmsa_masks,
        show_missed_matches_only=show_missed_matches_only,
    )


if __name__ == "__main__":
    _run_demo(show_missed_matches_only=False)
