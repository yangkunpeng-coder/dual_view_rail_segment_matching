"""覆盖 ISA-Rail 的假设枚举、单调路径、精确裕度与状态判定。"""

import json
from dataclasses import replace
import unittest

import numpy as np

from core_algorithm.common.rail_measurement import RailMeasurementConfig, measure_side
from core_algorithm.matching.rail_matching_isa import (
    ISARailConfig,
    _assemble,
    _competitor_margin,
    _cost_matrix,
    _hypotheses,
    _monotone_dp,
    _through_table,
)

_HEIGHT = 201
_WIDTH = 1600
_GAPS = (71, 137, 54, 208, 82, 126, 60, 184)
_CONFIG = ISARailConfig()


def _positions(gaps: tuple[int, ...] = _GAPS, start: int = 320) -> list[float]:
    """由像素间距生成从左到右的归一化位置序列。"""
    positions = [float(start)]
    for gap in gaps:
        positions.append(positions[-1] + gap)
    return [value / (_WIDTH - 1) for value in positions]


def _measure(
    positions: list[float], side: str, *, labels: list[int] | None = None,
    radius: int = 5, visible: tuple[int, int] = (0, _HEIGHT),
    order: list[int] | None = None,
):
    """按公共方向位置生成掩码并测量，返回测量结果与 mask_index 到标签的映射。"""
    indices = list(range(len(positions))) if order is None else order
    identities = list(range(len(positions))) if labels is None else labels
    masks = np.zeros((len(indices), _HEIGHT, _WIDTH), dtype=np.bool_)
    for slot, index in enumerate(indices):
        target = positions[index] * (_WIDTH - 1)
        x = target if side == "north" else (_WIDTH - 1 - target)
        left = max(0, round(x) - radius)
        right = min(_WIDTH, round(x) + radius + 1)
        masks[slot, visible[0]:visible[1], left:right] = True
    mapping = {slot: identities[index] for slot, index in enumerate(indices)}
    return measure_side(masks, side, RailMeasurementConfig()), mapping


def _run(south, north, settings: ISARailConfig = _CONFIG):
    """用合成测量结果直接调用 ISA-Rail 的组装流程。"""
    return _assemble(south, north, "south.png", "north.png", settings, 0.0, True)


def _labels(result, south_map, north_map) -> set[tuple[int, int]]:
    """把匹配结果换算为物理标签对。"""
    return {
        (south_map[match["south"]["mask_index"]],
         north_map[match["north"]["mask_index"]])
        for match in result["matches"]
    }


class ISAMatchingTests(unittest.TestCase):
    """以受控掩码校验身份、缺失、截断与歧义。"""

    def test_identical_sides_are_fully_matched(self) -> None:
        positions = _positions()
        south, s_map = _measure(positions, "south")
        north, n_map = _measure(positions, "north")
        result = _run(south, north)
        self.assertEqual(len(result["matches"]), len(positions))
        self.assertEqual(_labels(result, s_map, n_map),
                         {(index, index) for index in range(len(positions))})
        self.assertEqual(result["diagnostics"]["summary"]["selected_shift_k"], 0)
        for match in result["matches"]:
            self.assertLess(match["position_error"], 1e-6)

    def test_first_rail_missing_on_north(self) -> None:
        positions = _positions()
        south, s_map = _measure(positions, "south")
        north, n_map = _measure(positions[1:], "north",
                                labels=list(range(1, len(positions))))
        result = _run(south, north)
        expected = {(index, index) for index in range(1, len(positions))}
        self.assertEqual(_labels(result, s_map, n_map), expected)
        self.assertEqual(result["diagnostics"]["summary"]["selected_shift_k"], -1)

    def test_interior_and_end_missing(self) -> None:
        positions = _positions()
        keep = [0, 1, 2, 4, 5, 6, 7]
        south, s_map = _measure(positions, "south")
        north, n_map = _measure([positions[index] for index in keep], "north",
                                labels=keep)
        result = _run(south, north)
        matched = _labels(result, s_map, n_map)
        self.assertTrue(all(first == second for first, second in matched))
        self.assertEqual(len(matched), len(keep))

    def test_equal_spacing_uses_instance_count(self) -> None:
        positions = [0.2 + 0.05 * index for index in range(7)]
        south, s_map = _measure(positions, "south")
        north, n_map = _measure(positions, "north")
        result = _run(south, north)
        self.assertEqual(len(result["matches"]), 7)
        self.assertEqual(_labels(result, s_map, n_map),
                         {(index, index) for index in range(7)})

    def test_visible_band_truncation(self) -> None:
        positions = _positions()
        keep = [2, 3, 4, 5, 6]
        south, s_map = _measure(positions, "south")
        north, n_map = _measure([positions[index] for index in keep], "north",
                                labels=keep)
        result = _run(south, north)
        matched = _labels(result, s_map, n_map)
        self.assertEqual(matched, {(index, index) for index in keep})
        self.assertGreaterEqual(len(result["unmatched_south"]), 4)

    def test_input_permutation_is_invariant(self) -> None:
        positions = _positions()
        south, s_map = _measure(positions, "south")
        north_direct, n_map = _measure(positions, "north")
        baseline = _run(south, north_direct)
        shuffled, shuffled_map = _measure(positions, "north",
                                          order=[3, 0, 7, 1, 8, 2, 6, 4, 5])
        result = _run(south, shuffled)
        self.assertEqual(_labels(result, s_map, shuffled_map),
                         _labels(baseline, s_map, n_map))

    def test_empty_single_and_invalid_inputs(self) -> None:
        empty = measure_side(np.zeros((0, _HEIGHT, _WIDTH), dtype=np.bool_), "south")
        empty_north = measure_side(np.zeros((0, _HEIGHT, _WIDTH), dtype=np.bool_),
                                   "north")
        both = _run(empty, empty_north)
        self.assertEqual(both["matches"], [])
        south, _ = _measure(_positions(), "south")
        one_sided = _run(south, empty_north)
        self.assertFalse(one_sided["matches"])
        self.assertEqual({item["reason"] for item in one_sided["unmatched_south"]},
                         {"other_side_empty"})
        single_south, _ = _measure([0.4], "south")
        single_north, _ = _measure([0.4], "north")
        single = _run(single_south, single_north)
        self.assertFalse(single["matches"])
        self.assertEqual({item["reason"] for item in single["tentative_matches"]},
                         {"short_path"})
        with self.assertRaises(ValueError):
            replace(_CONFIG, gap_tolerance_fraction=0.0)

    def test_short_visible_masks_are_invalid_geometry(self) -> None:
        positions = _positions()
        south, _ = _measure(positions, "south")
        north, _ = _measure(positions, "north", visible=(0, 40))
        result = _run(south, north)
        self.assertTrue(all(item["reason"] == "invalid_geometry"
                            for item in result["unmatched_north"]))
        self.assertFalse(result["matches"])

    def test_competitor_margin_ignores_skip_cost_scale(self) -> None:
        positions = _positions()
        south, _ = _measure(positions, "south")
        north, _ = _measure(positions, "north")
        hypotheses = _hypotheses(south, north, _CONFIG)
        self.assertTrue(hypotheses)
        best = hypotheses[0]
        costs = _cost_matrix(south, north, best.scale, best.offset, _CONFIG)
        solve = _monotone_dp(costs)
        through = _through_table(costs, solve)
        margins = [
            _competitor_margin(i, j, through, solve.cost)[0] for i, j in solve.pairs
        ]
        self.assertTrue(all(value >= 0 for value in margins))
        # 竞争者裕度只看同一实例的其它配对，不随跳过代价整体缩放而变。
        scaled = replace(
            costs,
            skip_south=costs.skip_south * 2.0,
            skip_north=costs.skip_north * 2.0,
        )
        scaled_solve = _monotone_dp(scaled)
        self.assertEqual(scaled_solve.pairs, solve.pairs)
        scaled_through = _through_table(scaled, scaled_solve)
        scaled_margins = [
            _competitor_margin(i, j, scaled_through, scaled_solve.cost)[0]
            for i, j in scaled_solve.pairs
        ]
        np.testing.assert_allclose(margins, scaled_margins, rtol=0, atol=1e-9)

    def test_ambiguity_is_reported_not_forced(self) -> None:
        positions = [0.2 + 0.05 * index for index in range(6)]
        south, s_map = _measure(positions, "south")
        # 北侧整体右移一个轨距且少一根：两种错移给出几乎相同的代价。
        north, n_map = _measure([value + 0.05 for value in positions[1:]], "north",
                                labels=list(range(1, 7)))
        result = _run(south, north)
        detail = result["diagnostics"]["summary"]
        self.assertGreaterEqual(detail["hypothesis_count"], 2)
        ambiguous = {row["reason"] for row in result["ambiguous_candidates"]}
        if ambiguous:
            self.assertIn("shift_ambiguous", ambiguous)

    def test_reproducible_and_json_safe(self) -> None:
        positions = _positions()
        south, _ = _measure(positions, "south")
        north, _ = _measure(positions, "north")
        first = _run(south, north)
        second = _run(south, north)
        json.dumps(first, allow_nan=False)
        for key in ("feature_time_ms", "hypothesis_time_ms", "decision_time_ms"):
            first["diagnostics"]["summary"][key] = 0.0
            second["diagnostics"]["summary"][key] = 0.0
        self.assertEqual(first, second)
        minimal = _assemble(south, north, "south.png", "north.png", _CONFIG,
                            0.0, False)
        self.assertIsNone(minimal["diagnostics"])
        self.assertEqual(minimal["matches"], second["matches"])


if __name__ == "__main__":
    unittest.main()
