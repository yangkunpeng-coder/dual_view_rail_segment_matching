"""覆盖 MSLC 的 25% 截面通道（S1）与配置校验。"""

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from core_algorithm.matching.rail_matching_mslc import (
    MSLCRailConfig,
    match_south_north_rail_mslc,
)
from data_preprocess.common.config import PROJECT_ROOT

_HEIGHT = 200
_WIDTH = 400
_POSITIONS = (60, 160, 260)


def _write_fixture(name: str) -> str:
    """在 temp 目录生成一张纯色原图，便于按项目相对路径读取尺寸。"""
    relative = f"temp/mslc_test_{name}.png"
    absolute = PROJECT_ROOT / relative
    absolute.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (_WIDTH, _HEIGHT), (32, 32, 32)).save(absolute)
    return relative


def _masks() -> np.ndarray:
    """三根等宽竖条，作为两侧实例。"""
    masks = np.zeros((len(_POSITIONS), _HEIGHT, _WIDTH), dtype=np.bool_)
    for index, position in enumerate(_POSITIONS):
        masks[index, :, position:position + 12] = True
    return masks


class MSLCTopSectionTests(unittest.TestCase):
    """校验新增通道不破坏一对一约束，且可以关闭。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.south_path = _write_fixture("south")
        cls.north_path = _write_fixture("north")
        cls.masks = _masks()

    @classmethod
    def tearDownClass(cls) -> None:
        for name in (cls.south_path, cls.north_path):
            (PROJECT_ROOT / name).unlink(missing_ok=True)

    def _run(self, config: MSLCRailConfig):
        return match_south_north_rail_mslc(
            self.south_path, self.north_path, self.masks, self.masks, config=config
        )

    def test_invalid_threshold_is_rejected(self) -> None:
        for value in (-0.1, 1.5, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MSLCRailConfig(min_top_section_support=value)

    def test_channel_is_reported_and_can_be_disabled(self) -> None:
        enabled = self._run(MSLCRailConfig(min_top_section_support=0.40))
        rows = enabled["diagnostics"]["candidate_pair_support"]
        self.assertTrue(all(
            isinstance(row["top_section_eligible"], bool) for row in rows
        ))
        disabled = self._run(MSLCRailConfig(min_top_section_support=0.0))
        self.assertTrue(all(
            row["top_section_eligible"] is False
            for row in disabled["diagnostics"]["candidate_pair_support"]
        ))

    def test_one_to_one_and_monotonic(self) -> None:
        for threshold in (0.0, 0.40):
            with self.subTest(threshold=threshold):
                result = self._run(MSLCRailConfig(min_top_section_support=threshold))
                matches = result["matches"]
                south = [match["south"]["mask_index"] for match in matches]
                north = [match["north"]["mask_index"] for match in matches]
                self.assertEqual(len(south), len(set(south)))
                self.assertEqual(len(north), len(set(north)))
                self.assertEqual(south, sorted(south))
                self.assertEqual(north, sorted(north))
                self.assertLessEqual(len(matches), len(_POSITIONS))

    def test_enabling_channel_never_loses_matches(self) -> None:
        disabled = self._run(MSLCRailConfig(min_top_section_support=0.0))
        enabled = self._run(MSLCRailConfig(min_top_section_support=0.40))
        self.assertGreaterEqual(len(enabled["matches"]), len(disabled["matches"]))

    def test_empty_side_reports_visible_instances_as_unmatched(self) -> None:
        """端到端预测只有一侧有实例时，保留该侧的未匹配结果。"""
        result = match_south_north_rail_mslc(
            self.south_path, self.north_path, self.masks, self.masks[:0]
        )
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["unmatched_north"], [])
        self.assertEqual(
            [instance["reason"] for instance in result["unmatched_south"]],
            ["other_side_empty"] * len(self.masks),
        )


if __name__ == "__main__":
    unittest.main()
