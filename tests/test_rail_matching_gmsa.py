"""覆盖 GMSA 的合成几何、DTW 和全局路径约束。"""

from dataclasses import replace
from itertools import combinations
import json
import unittest

import numpy as np

from core_algorithm.common.preview import LabelMeInstanceRef
from core_algorithm.matching._gmsa_alignment import (
    PathOption,
    build_graph,
    constrained_dtw,
    gap_subsequence_costs,
    solve_paths,
    solve_paths_avoiding,
    span_curves,
)
from core_algorithm.matching._gmsa_geometry import (
    FloatArray,
    GMSARailConfig,
    RailView,
    measure_view,
)
from core_algorithm.matching.rail_matching_gmsa import (
    GMSADiagnostics,
    GMSAMatchingResult,
    _Search,
    _direction_nodes,
    _direction_summary,
    _direction_support,
    _node_costs,
    match_south_north_rail_gmsa,
)

_HEIGHT = 201
_WIDTH = 1600
_GAPS = (71, 137, 54, 208, 82, 126, 60, 184)
_SAME = GMSARailConfig(longitudinal_direction="same")


def _centers(gaps: tuple[int, ...] = _GAPS, *, dynamic: bool = False) -> FloatArray:
    """生成间距非均匀的横向中心序列。"""
    y = np.linspace(0, 1, _HEIGHT)
    distances = np.repeat(np.asarray(gaps, dtype=np.float64)[:, None], _HEIGHT, axis=1)
    if dynamic:
        amplitudes = np.sin(np.arange(len(gaps)) * 2.4 + 1.0)[:, None]
        distances *= 1 + amplitudes * (1.1 * (y - 0.5) + 0.35 * y ** 2)
    return 100 + np.vstack((np.zeros(_HEIGHT), np.cumsum(distances, axis=0)))


def _view(
    centers: FloatArray,
    side: str,
    *,
    keep: tuple[int, ...] | None = None,
    reverse: bool = False,
    radius: int = 5,
    visible: tuple[int, int] = (0, _HEIGHT),
) -> RailView:
    """将已知中心和可见范围转换为原项目的逐实例 Mask。"""
    indices = tuple(range(len(centers))) if keep is None else keep
    masks = np.zeros((len(indices), _HEIGHT, _WIDTH), dtype=np.bool_)
    for mask, index in zip(masks, indices, strict=True):
        line = centers[index]
        if reverse:
            line = line[np.clip(210 - np.arange(_HEIGHT), 0, _HEIGHT - 1)]
        if side == "south":
            line = _WIDTH - 1 - line
        for row in range(*visible):
            x = round(float(line[row]))
            mask[row, max(0, x - radius):min(_WIDTH, x + radius + 1)] = True
    refs = tuple(LabelMeInstanceRef(index, index + 1000) for index in indices)
    return RailView(f"{side}.png", masks, refs)


def _diagnostics(result: GMSAMatchingResult) -> GMSADiagnostics:
    details = result["diagnostics"]
    if details is None:
        raise AssertionError("本次测试需要诊断")
    return details


def _search(
    same: tuple[int, ...], reverse: tuple[int, ...], *, node_count: int = 9
) -> _Search:
    """构造只用于方向决策核对的搜索结果。"""
    return _Search(
        best=PathOption(1.0, same, "same"),
        best_cost=1.0,
        second_cost=1.4,
        global_margin=0.4,
        competition_count=2,
        support_status="competing_paths",
        direction_best_paths={
            "same": PathOption(1.0, same, "same"),
            "reverse": PathOption(1.1, reverse, "reverse"),
        },
        through=np.full(node_count, np.inf),
        exclusions={},
        near_paths=(),
        near_margin=0.08,
        direction_ambiguous=True,
        direction_margin=0.01,
        direction_costs={"same": 1.0, "reverse": 1.1},
    )


class GMSAMatchingTests(unittest.TestCase):
    """以受控 Mask 校验身份索引、证据不足和歧义。"""

    def test_empty_and_one_sided_inputs(self) -> None:
        empty = RailView("empty.png", None)
        result = match_south_north_rail_gmsa(empty, empty)
        self.assertEqual(result["matches"], [])
        self.assertEqual(_diagnostics(result)["summary"]["path_competition_count"], 0)
        south = _view(_centers(), "south")
        for first, second, side in ((south, empty, "south"), (empty, south, "north")):
            with self.subTest(side=side):
                result = match_south_north_rail_gmsa(first, second, _SAME)
                self.assertFalse(result["matches"])
                key = "unmatched_south" if side == "south" else "unmatched_north"
                self.assertEqual(len(result[key]), len(_GAPS) + 1)
                self.assertEqual({item["reason"] for item in result[key]}, {"other_side_empty"})

    def test_single_path_does_not_imply_support(self) -> None:
        centers = _centers(())
        result = match_south_north_rail_gmsa(_view(centers, "south"), _view(centers, "north"))
        summary = _diagnostics(result)["summary"]
        self.assertEqual(summary["path_competition_count"], 1)
        self.assertEqual(summary["support_status"], "single_path_no_competition")
        self.assertIsNone(summary["second_path_cost"])
        self.assertIsNone(summary["global_path_margin"])
        self.assertEqual(len(result["tentative_matches"]), 1)
        self.assertFalse(result["matches"])

    def test_two_instances_have_only_one_span(self) -> None:
        centers = _centers((71,))
        result = match_south_north_rail_gmsa(
            _view(centers, "south"), _view(centers, "north"), _SAME,
        )
        self.assertEqual(len(result["tentative_matches"]), 2)
        self.assertFalse(result["matches"])
        self.assertEqual(_diagnostics(result)["summary"]["evidence_transition_count"], 1)

    def test_exact_order_with_permuted_input(self) -> None:
        centers = _centers()
        south = _view(centers, "south", keep=(3, 6, 0, 2, 8, 4, 5, 7, 1))
        north = _view(centers, "north", keep=(8, 1, 7, 0, 5, 4, 3, 2, 6))
        north = replace(north, references=tuple(
            LabelMeInstanceRef(ref.shape_index, ref.group_id + 1000)
            for ref in north.references if ref.group_id is not None
        ))
        result = match_south_north_rail_gmsa(south, north, _SAME)
        rows = _diagnostics(result)["accepted"]
        self.assertEqual(len(rows), len(centers))
        self.assertEqual([row["south_order_index"] for row in rows], list(range(1, 10)))
        self.assertEqual([row["north_order_index"] for row in rows], list(range(1, 10)))
        self.assertTrue(all(row["south_shape_index"] == row["north_shape_index"] for row in rows))
        self.assertTrue(all(row["south_group_id"] != row["north_group_id"] for row in rows))
        self.assertEqual(len({row["south_mask_index"] for row in rows}), len(rows))
        self.assertEqual(len({row["north_mask_index"] for row in rows}), len(rows))

    def test_missing_one_two_and_both_ends(self) -> None:
        centers = _centers()
        for missing in ((2,), (2, 3), (0, 8)):
            with self.subTest(missing=missing):
                keep = tuple(index for index in range(len(centers)) if index not in missing)
                result = match_south_north_rail_gmsa(
                    _view(centers, "south"), _view(centers, "north", keep=keep), _SAME,
                )
                rows = _diagnostics(result)["accepted"]
                self.assertTrue(rows)
                self.assertTrue(all(row["south_shape_index"] == row["north_shape_index"] for row in rows))
                self.assertTrue(all(row["south_shape_index"] not in missing for row in rows))
                self.assertEqual(len(rows) + len(result["unmatched_south"]), len(centers))
                self.assertEqual(len(rows) + len(result["unmatched_north"]), len(keep))

    def test_even_spacing_has_near_optimal_alternatives(self) -> None:
        centers = _centers((70,) * 6)
        result = match_south_north_rail_gmsa(
            _view(centers, "south"),
            _view(centers, "north", keep=tuple(range(1, 7))),
            _SAME,
        )
        summary = _diagnostics(result)["summary"]
        self.assertAlmostEqual(summary["global_path_margin"], 0, places=8)
        self.assertTrue(result["ambiguous_candidates"])
        self.assertFalse(result["matches"])

    def test_two_directions_and_auto(self) -> None:
        centers = _centers(dynamic=True)
        south = _view(centers, "south")
        for reverse, expected in ((False, "same"), (True, "reverse")):
            with self.subTest(direction=expected):
                north = _view(centers, "north", reverse=reverse)
                result = match_south_north_rail_gmsa(south, north)
                summary = _diagnostics(result)["summary"]
                self.assertEqual(summary["longitudinal_direction"], expected)
                self.assertFalse(summary["direction_ambiguous"])
                self.assertTrue(result["matches"])
        flat = _centers()
        ambiguous = match_south_north_rail_gmsa(_view(flat, "south"), _view(flat, "north"))
        detail = _diagnostics(ambiguous)
        summary = detail["summary"]
        self.assertTrue(summary["direction_ambiguous"])
        # 两方向最优路径相同：方向不确定不等于身份不确定，共识关系应继续配对级检查。
        self.assertEqual(summary["direction_decision"], "direction_consensus")
        self.assertEqual(summary["direction_conflict_pair_count"], 0)
        self.assertTrue(ambiguous["matches"])
        self.assertTrue(all(
            match["evidence"]["direction_support"] == "both_directions"
            for match in ambiguous["matches"]
        ))

    def test_different_visible_ranges(self) -> None:
        centers = _centers(dynamic=True)
        result = match_south_north_rail_gmsa(
            _view(centers, "south", visible=(16, 169)),
            _view(centers, "north", visible=(35, 193)),
            _SAME,
        )
        detail = _diagnostics(result)
        self.assertGreaterEqual(detail["summary"]["evidence_transition_count"], 2)
        self.assertLess(detail["summary"]["median_dtw_coverage"], 1)
        self.assertTrue(all(row["valid"] for row in detail["dtw_alignment"]))

    def test_multi_interval_break_and_overlap(self) -> None:
        centers = _centers((14, 86, 52))
        north = _view(centers, "north", radius=8)
        masks = north.masks
        if masks is None:
            self.fail("测试夹具缺少 Mask")
        masks[0, 90, 200:211] = True
        masks[0, 100] = False
        result = match_south_north_rail_gmsa(_view(centers, "south"), north, _SAME)
        details = _diagnostics(result)
        rail = next(row for row in details["instances"]
                    if row["side"] == "north" and row["mask_index"] == 0)
        self.assertEqual(rail["multi_interval_section_count"], 1)
        self.assertEqual(rail["broken_section_count"], 1)
        self.assertGreater(rail["overlap_section_count"], 0)
        selected = next(row for row in details["sections"]
                        if row["side"] == "north" and row["mask_index"] == 0 and row["y_px"] == 90)
        self.assertEqual(selected["center_x_px"], 100)
        self.assertLess(selected["weight"], 1)
        self.assertTrue(any(row["boundary_gap_px"] is not None and row["boundary_gap_px"] < 0
                            for row in details["gap_fingerprints"]))
        self.assertEqual(sum(row["side"] == "north" for row in details["instances"]), 4)

    def test_order_conflict_remains_visible(self) -> None:
        centers = _centers((71, 137, 54, 208))
        crossing = centers.copy()
        crossing[0, 140:] += 95
        result = match_south_north_rail_gmsa(
            _view(centers, "south"), _view(crossing, "north"), _SAME,
        )
        rows = _diagnostics(result)["instances"]
        bad = {row["mask_index"] for row in rows if row["side"] == "north" and row["order_uncertain"]}
        self.assertTrue({0, 1} <= bad)
        self.assertTrue(all(match["north"]["mask_index"] not in bad for match in result["matches"]))

    def test_stable_large_gap_is_soft_anchor(self) -> None:
        centers = _centers((70, 72, 231, 65, 73, 66))
        result = match_south_north_rail_gmsa(
            _view(centers, "south"), _view(centers, "north"), _SAME,
        )
        detail = _diagnostics(result)
        self.assertTrue(any(row["stable_large_gap"] for row in detail["gap_fingerprints"]))
        self.assertTrue(any(row["anchor_cost"] == 0 for row in detail["transitions"]))

    def test_one_sided_pseudo_gap_cannot_be_soft_anchor(self) -> None:
        centers = _centers((70,) * 6)
        result = match_south_north_rail_gmsa(
            _view(centers, "south"),
            _view(centers, "north", keep=(0, 1, 3, 4, 5, 6)), _SAME,
        )
        detail = _diagnostics(result)
        sides = {row["side"] for row in detail["gap_fingerprints"] if row["stable_large_gap"]}
        self.assertEqual(sides, {"north"})
        self.assertTrue(all(row["anchor_cost"] is None for row in detail["transitions"]))

    def test_merge_groups_do_not_create_instance_pairs(self) -> None:
        centers = _centers()
        south = measure_view(_view(centers, "south"), "south", _SAME)
        north = measure_view(
            _view(centers, "north", keep=(0, 1, 3, 4, 5, 6, 7, 8)), "north", _SAME,
        )
        s, n = span_curves(south), span_curves(north)
        s_edge = int(np.flatnonzero((s.start == 1) & (s.end == 3))[0])
        n_edge = int(np.flatnonzero((n.start == 1) & (n.end == 2))[0])
        merged = gap_subsequence_costs(south, north, s, n, _SAME, "same")
        unmerged = gap_subsequence_costs(
            south, north, s, n, replace(_SAME, gap_merge_limit=1), "same",
        )
        self.assertTrue(np.isfinite(merged[s_edge, n_edge]))
        self.assertTrue(np.isinf(unmerged[s_edge, n_edge]))

    def test_reproducible_and_json_safe(self) -> None:
        centers = _centers(dynamic=True)
        south, north = _view(centers, "south"), _view(centers, "north")
        first = match_south_north_rail_gmsa(south, north, {"longitudinal_direction": "same"})
        second = match_south_north_rail_gmsa(south, north, _SAME)
        for result in (first, second):
            json.dumps(result, allow_nan=False)
            _diagnostics(result)["timings_ms"].clear()
        self.assertEqual(first, second)
        minimal = match_south_north_rail_gmsa(south, north, _SAME, False)
        self.assertIsNone(minimal["diagnostics"])
        self.assertEqual(minimal["matches"], first["matches"])

    def test_invalid_geometry_and_resources(self) -> None:
        with self.assertRaises(ValueError):
            replace(_SAME, span_weight=float("nan"))
        with self.assertRaises(ValueError):
            RailView("invalid.png", np.full((1, 5, 5), np.nan))
        centers = _centers()
        limited = match_south_north_rail_gmsa(
            _view(centers, "south"), _view(centers, "north"),
            replace(_SAME, max_transitions=0),
        )
        self.assertTrue(_diagnostics(limited)["summary"]["resource_limited"])
        self.assertFalse(limited["matches"])
        invalid = match_south_north_rail_gmsa(
            _view(centers, "south", visible=(0, 30)), _view(centers, "north"), _SAME,
        )
        self.assertEqual({row["reason"] for row in invalid["unmatched_south"]}, {"invalid_geometry"})
        self.assertFalse(invalid["matches"])

    def test_skip_every_instance_is_a_feasible_solution(self) -> None:
        centers = _centers((70, 120, 80))
        result = match_south_north_rail_gmsa(
            _view(centers, "south"), _view(centers, "north"), replace(_SAME, node_weight=20),
        )
        summary = _diagnostics(result)["summary"]
        self.assertEqual(summary["best_path_cost"], summary["empty_path_cost"])
        self.assertEqual(summary["support_status"], "empty_path_preferred")
        self.assertFalse(result["matches"] or result["tentative_matches"])

    def test_unmatched_and_ambiguous_states_stay_in_sync(self) -> None:
        even = (70,) * 6
        fixtures = (
            ("flat", _view(_centers(), "south"), _view(_centers(), "north")),
            ("missing_end", _view(_centers(), "south"),
             _view(_centers(), "north", keep=(0, 1, 3, 4, 5, 6, 7, 8))),
            ("even_spacing", _view(_centers(even), "south"),
             _view(_centers(even), "north", keep=tuple(range(1, 7)))),
        )
        for name, south, north in fixtures:
            with self.subTest(fixture=name):
                detail = _diagnostics(match_south_north_rail_gmsa(south, north, _SAME))
                rows = detail["unmatched"]
                self.assertEqual(
                    detail["summary"]["unmatched_instance_count"], len(rows)
                )
                best_cost = detail["summary"]["best_path_cost"]
                near_limit = best_cost + max(
                    _SAME.pair_exclusion_margin, _SAME.path_relative_margin * best_cost
                )
                for row in rows:
                    self.assertIn(
                        row["status"], ("unmatched", "tentative", "ambiguous")
                    )
                    if row["status"] == "unmatched":
                        self.assertEqual(row["reason"], "no_reliable_partner")
                        nearest = row.get("candidate_1_path_cost")
                        if nearest is not None:
                            # 没有近优替代才允许标记为无可靠伙伴。
                            self.assertGreater(nearest, near_limit)


class DirectionDecisionTests(unittest.TestCase):
    """核对方向共识划分与 Pair 级方向诊断。"""

    def test_consensus_support_and_conflict_partition(self) -> None:
        search = _search((0, 4), (4, 8))
        same, reverse, consensus = _direction_nodes(search)
        self.assertEqual(same, {0, 4})
        self.assertEqual(reverse, {4, 8})
        self.assertEqual(consensus, {4})
        self.assertEqual(_direction_support(4, same, reverse), "both_directions")
        self.assertEqual(_direction_support(0, same, reverse), "same_only")
        self.assertEqual(_direction_support(8, same, reverse), "reverse_only")
        self.assertEqual(_direction_support(2, same, reverse), "not_applicable")
        summary = _direction_summary(search, 3, 2)
        self.assertEqual(summary["same_best_path_match_count"], 2)
        self.assertEqual(summary["reverse_best_path_match_count"], 2)
        self.assertEqual(summary["direction_consensus_pair_count"], 1)
        self.assertEqual(summary["direction_conflict_pair_count"], 2)
        self.assertAlmostEqual(float(summary["direction_consensus_ratio"]), 1 / 3)
        self.assertEqual(summary["direction_decision"], "direction_conflict")

    def test_single_direction_and_insufficient_evidence(self) -> None:
        single = _search((0, 1), (0, 1))
        single.direction_best_paths["reverse"] = None
        summary = _direction_summary(single, 2, 2)
        self.assertEqual(summary["direction_decision"], "single_direction")
        self.assertIsNone(summary["direction_consensus_ratio"])
        weak = _direction_summary(_search((0, 4), (0, 4)), 1, 2)
        self.assertEqual(weak["direction_decision"], "insufficient_evidence")


class AlignmentTests(unittest.TestCase):
    """独立核对带窗口 DTW 和 K-best 的边界条件。"""

    def test_dtw_reverse_window_repeats_and_invalid_samples(self) -> None:
        sample = np.arange(18, dtype=np.float64)[None] ** 1.1
        shifted = np.roll(sample, 1, axis=1)
        sample[0, 3] = np.nan
        shifted[0, 5] = np.inf
        weights = np.ones_like(sample)
        config = replace(_SAME, dtw_window=2, dtw_max_repeat=1)
        result = constrained_dtw(sample, shifted[:, ::-1], weights, weights,
                                 config, "reverse", 0.7, trace=True)
        points = result.traces[0].points
        for point in points:
            self.assertNotEqual(point.south_index, 3)
            self.assertNotEqual(point.north_index, 12)
            self.assertLessEqual(abs(point.south_index - (17 - point.north_index)), 2)
        for side in ("south_index", "north_index"):
            runs = np.diff([getattr(point, side) for point in points]) == 0
            self.assertFalse(np.any(runs[:-1] & runs[1:]))

    def test_empty_overlap_and_extreme_scales(self) -> None:
        a = np.full((1, 18), np.nan)
        b = a.copy()
        a[0, :5], b[0, 13:] = 1, 1
        weights = np.ones_like(a)
        empty = constrained_dtw(a, b, weights, weights, _SAME, "same", 0.7, trace=True)
        self.assertTrue(np.isinf(empty.costs[0]))
        self.assertEqual(empty.traces, {})
        a.fill(np.log(1e-300))
        b.fill(np.log(1e300))
        extreme = constrained_dtw(a, b, weights, weights, _SAME, "same", 0.7)
        self.assertTrue(np.isfinite(extreme.costs[0]))

    def test_batched_exclusion_matches_single_forbidden_solves(self) -> None:
        centers = _centers()
        south = measure_view(_view(centers, "south"), "south", _SAME)
        north = measure_view(_view(centers, "north"), "north", _SAME)
        node_costs = _node_costs(south, north, _SAME)
        s_spans, n_spans = span_curves(south), span_curves(north)
        for direction in ("same", "reverse"):
            graph = build_graph(
                south, north, s_spans, n_spans, node_costs, _SAME, direction
            )
            best, _ = solve_paths(graph, _SAME.k_best, measure_through=True)
            self.assertTrue(best)
            nodes = tuple(int(node) for node in best[0].nodes)
            batched = solve_paths_avoiding(graph, nodes)
            self.assertEqual(len(batched), len(nodes))
            for node, path in zip(nodes, batched, strict=True):
                single, _ = solve_paths(graph, 1, forbidden=node)
                self.assertTrue(single)
                self.assertIsNotNone(path)
                self.assertAlmostEqual(single[0].cost, path.cost, places=9)
                self.assertEqual(single[0].nodes, path.nodes)

    def test_span_pruning_only_drops_long_skip_edges(self) -> None:
        centers = _centers(dynamic=True)
        south = measure_view(_view(centers, "south"), "south", _SAME)
        north = measure_view(
            _view(centers, "north", keep=(0, 1, 3, 4, 5, 6, 7, 8)), "north", _SAME
        )
        node_costs = _node_costs(south, north, _SAME)
        s_spans, n_spans = span_curves(south), span_curves(north)
        for direction in ("same", "reverse"):
            pruned = build_graph(
                south, north, s_spans, n_spans, node_costs, _SAME, direction
            )
            full = build_graph(
                south, north, s_spans, n_spans, node_costs,
                replace(_SAME, max_span_skip_difference=4096), direction,
            )
            kept = np.isfinite(pruned.transition_costs)
            # 剪掉的一定是长跳过边：保留边是完整可用边的子集。
            self.assertFalse(np.any(kept & ~np.isfinite(full.transition_costs)))
            np.testing.assert_allclose(
                pruned.transition_costs[kept], full.transition_costs[kept],
                rtol=0, atol=1e-12,
            )
            dropped = np.isfinite(full.transition_costs) & ~kept
            if dropped.any():
                south_steps = (s_spans.end - s_spans.start)
                north_steps = (n_spans.end - n_spans.start)
                steps = np.abs(south_steps[:, None] - north_steps[None, :])
                self.assertTrue(np.all(steps[dropped] > _SAME.max_span_skip_difference))

    def test_k_best_exclusion_and_through_costs_match_exhaustive_search(self) -> None:
        centers = _centers((70, 125, 87))
        south = measure_view(_view(centers, "south"), "south", _SAME)
        north = measure_view(_view(centers, "north"), "north", _SAME)
        count = len(centers)
        nodes = np.arange(count * count, dtype=np.float64).reshape(count, count) / 100
        graph = build_graph(south, north, span_curves(south), span_curves(north),
                            nodes, _SAME, "same")
        edges = {
            (int(graph.predecessors[edge]), node): float(graph.edge_costs[edge])
            for node in range(count * count)
            for edge in range(graph.offsets[node], graph.offsets[node + 1])
        }
        paths: list[tuple[float, tuple[int, ...]]] = []
        for length in range(1, count + 1):
            for s in combinations(range(count), length):
                for n in combinations(range(count), length):
                    path = tuple(i * count + j for i, j in zip(s, n, strict=True))
                    transitions = [edges.get(edge, np.inf) for edge in zip(path, path[1:])]
                    cost = (graph.leading_cost[path[0]] + graph.trailing_cost[path[-1]]
                            + sum(graph.node_cost[node] for node in path) + sum(transitions))
                    if np.isfinite(cost):
                        paths.append((float(cost), path))
        paths.sort()
        result, through = solve_paths(graph, 5, measure_through=True)
        self.assertEqual(len({path.nodes for path in result}), 5)
        np.testing.assert_allclose([path.cost for path in result], [cost for cost, _ in paths[:5]])
        for node in range(count * count):
            banned, _ = solve_paths(graph, 1, forbidden=node)
            self.assertAlmostEqual(banned[0].cost, min(cost for cost, path in paths if node not in path))
            self.assertAlmostEqual(through[node], min(cost for cost, path in paths if node in path))


if __name__ == "__main__":
    unittest.main()
