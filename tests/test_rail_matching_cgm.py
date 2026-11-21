"""覆盖 CGM-Rail 的错移枚举、整束代价与判定约束。"""

import json
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from core_algorithm.common.rail_measurement import RailMeasurementConfig, measure_side
from core_algorithm.matching.rail_matching_cgm import (
    CGMRailConfig,
    _bundle_cost,
    _cost_matrix,
    _hypotheses,
    _monotone_dp,
    match_south_north_rail_cgm,
)
from data_preprocess.common.config import PROJECT_ROOT

_HEIGHT = 200
_WIDTH = 600
_GAPS = (60, 110, 50, 170, 70)
_CONFIG = CGMRailConfig()


def _positions(gaps: tuple[int, ...] = _GAPS, start: int = 80) -> list[float]:
    """由像素间距生成从左到右的归一化位置。"""
    values = [float(start)]
    for gap in gaps:
        values.append(values[-1] + gap)
    return [value / (_WIDTH - 1) for value in values]


def _measure(positions: list[float], side: str) -> object:
    """按公共方向位置生成竖条掩码并测量。"""
    masks = np.zeros((len(positions), _HEIGHT, _WIDTH), dtype=np.bool_)
    for index, position in enumerate(positions):
        target = position * (_WIDTH - 1)
        x = target if side == "north" else (_WIDTH - 1 - target)
        left = max(0, round(x) - 6)
        masks[index, :, left:left + 13] = True
    return measure_side(masks, side, RailMeasurementConfig())


class CGMTests(unittest.TestCase):
    """用合成测量结果校验错移选择与约束。"""

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(_CONFIG, shift_limit=0)
        with self.assertRaises(ValueError):
            replace(_CONFIG, bundle_weight=-1.0)
        with self.assertRaises(ValueError):
            replace(_CONFIG, gap_tolerance_fraction=0.0)

    def test_shift_is_selected_on_translated_fixture(self) -> None:
        positions = _positions()
        south = _measure(positions, "south")
        # 北侧少最左一根：真值是南侧第 i 根对应北侧第 i-1 根，即 k=-1。
        north = _measure(positions[1:], "north")
        hypotheses = _hypotheses(south, north, _CONFIG)
        self.assertTrue(hypotheses)
        best = hypotheses[0]
        self.assertEqual(best.shift, -1)
        self.assertEqual(len(best.pairs), len(positions) - 1)
        # 选中的错移下，配对顺序与掩码顺序一致。
        self.assertEqual(
            [first for first, _ in best.pairs], list(range(1, len(positions)))
        )

    def test_identity_shift_on_identical_sides(self) -> None:
        positions = _positions()
        south = _measure(positions, "south")
        north = _measure(positions, "north")
        best = _hypotheses(south, north, _CONFIG)[0]
        self.assertEqual(best.shift, 0)
        self.assertEqual(len(best.pairs), len(positions))

    def test_bundle_cost_reacts_to_large_gap_position(self) -> None:
        positions = _positions()
        south = _measure(positions, "south")
        north = _measure(positions, "north")
        bundle = _bundle_cost(
            south, north, tuple((i, i) for i in range(len(positions)))
        )
        self.assertGreaterEqual(bundle, 0.0)
        # 错移一层的对齐应得到不小于正确对齐的整束代价。
        shifted = tuple((i, i + 1) for i in range(len(positions) - 1))
        self.assertGreaterEqual(_bundle_cost(south, north, shifted), bundle)

    def test_path_is_monotone_and_one_to_one(self) -> None:
        positions = _positions()
        south = _measure(positions, "south")
        north = _measure(positions[1:], "north")
        best = _hypotheses(south, north, _CONFIG)[0]
        south_indices = [first for first, _ in best.pairs]
        north_indices = [second for _, second in best.pairs]
        self.assertEqual(south_indices, sorted(south_indices))
        self.assertEqual(north_indices, sorted(north_indices))
        self.assertEqual(len(set(south_indices)), len(south_indices))
        self.assertEqual(len(set(north_indices)), len(north_indices))

    def test_empty_and_single_sides(self) -> None:
        empty = measure_side(np.zeros((0, _HEIGHT, _WIDTH), dtype=np.bool_), "south")
        empty_north = measure_side(
            np.zeros((0, _HEIGHT, _WIDTH), dtype=np.bool_), "north"
        )
        self.assertEqual(_hypotheses(empty, empty_north, _CONFIG), [])
        single = _measure([0.5], "south")
        single_north = _measure([0.5], "north")
        self.assertEqual(_hypotheses(single, single_north, _CONFIG), [])

    def test_cost_matrix_prefers_correct_shift(self) -> None:
        positions = _positions()
        south = _measure(positions, "south")
        north = _measure(positions[1:], "north")
        correct = _cost_matrix(south, north, 1.0, 0.0, _CONFIG)
        wrong = _cost_matrix(south, north, 1.0, 0.10, _CONFIG)
        self.assertGreater(
            np.isfinite(correct.matrix).sum(), np.isfinite(wrong.matrix).sum()
        )
        self.assertLess(
            _monotone_dp(correct).cost, _monotone_dp(wrong).cost
        )


class CGMEntryTests(unittest.TestCase):
    """校验公开入口的返回协议与可复现性。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = {}
        for side in ("south", "north"):
            relative = f"temp/cgm_test_{side}.png"
            absolute = PROJECT_ROOT / relative
            absolute.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (_WIDTH, _HEIGHT), (32, 32, 32)).save(absolute)
            cls.paths[side] = relative
        cls.masks = {}
        for side in ("south", "north"):
            masks = np.zeros((len(_GAPS) + 1, _HEIGHT, _WIDTH), dtype=np.bool_)
            for index, position in enumerate(_positions()):
                target = position * (_WIDTH - 1)
                x = target if side == "north" else (_WIDTH - 1 - target)
                left = max(0, round(x) - 6)
                masks[index, :, left:left + 13] = True
            cls.masks[side] = masks

    @classmethod
    def tearDownClass(cls) -> None:
        for relative in cls.paths.values():
            (PROJECT_ROOT / relative).unlink(missing_ok=True)

    def _run(self, **overrides):
        return match_south_north_rail_cgm(
            self.paths["south"], self.paths["north"],
            self.masks["south"], self.masks["north"],
            config=replace(_CONFIG, **overrides),
        )

    def test_result_protocol_and_reproducibility(self) -> None:
        first = self._run()
        second = self._run()
        json.dumps(first, allow_nan=False)
        for key in ("feature_time_ms", "hypothesis_time_ms", "decision_time_ms"):
            first["diagnostics"]["summary"][key] = 0.0
            second["diagnostics"]["summary"][key] = 0.0
        self.assertEqual(first, second)
        self.assertIn("matches", first)
        self.assertIn("shift_margin", first["diagnostics"]["summary"])
        matches = first["matches"]
        south = [match["south"]["mask_index"] for match in matches]
        north = [match["north"]["mask_index"] for match in matches]
        self.assertEqual(south, sorted(south))
        self.assertEqual(north, sorted(north))
        self.assertEqual(len(set(south)), len(south))
        self.assertEqual(len(set(north)), len(north))


if __name__ == "__main__":
    unittest.main()
