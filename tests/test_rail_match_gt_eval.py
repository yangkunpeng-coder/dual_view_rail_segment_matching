"""验证真值评测的图对剔除与实例计数。"""

from pathlib import Path
import unittest

from experiments.rail_match_gt_eval import (
    _GT_CONFIG,
    _Truth,
    _apply_exclusions,
    _excluded_pairs,
    _score_pair,
)


class ExclusionTests(unittest.TestCase):
    """剔除只作用于指标，配置与记录必须可追溯。"""

    def test_config_lists_three_problem_pairs(self) -> None:
        excluded = _excluded_pairs(_GT_CONFIG)
        self.assertEqual(sorted(excluded), ["P0089", "P0092", "P0094"])
        self.assertTrue(all(reason for reason in excluded.values()))

    def test_missing_config_means_no_exclusions(self) -> None:
        self.assertEqual(_excluded_pairs(Path("config/absent.yaml")), {})

    def test_apply_exclusions_keeps_other_pairs_and_reports_unknown(self) -> None:
        truths = {"P0001": _Truth("valid"), "P0002": _Truth("test")}
        kept, dropped, unknown = _apply_exclusions(
            truths, {"P0002": "ambiguous_shift_annotation", "P9999": "typo"}
        )
        self.assertEqual(sorted(kept), ["P0001"])
        self.assertEqual(dropped, {"P0002": "ambiguous_shift_annotation"})
        self.assertEqual(unknown, ["P9999"])


class ScoringTests(unittest.TestCase):
    """验证关系输出覆盖的实例数。"""

    def test_same_mask_index_counts_on_both_sides(self) -> None:
        """相同下标在南北两侧各代表一个实例。"""
        truth = _Truth(
            split="valid",
            paired={(0, 0)},
            south_instances={0},
            north_instances={0},
        )

        score, _, _ = _score_pair(truth, {(0, 0)})

        self.assertEqual(score.instances, 2)
        self.assertEqual(score.covered_instances, 2)


if __name__ == "__main__":
    unittest.main()
