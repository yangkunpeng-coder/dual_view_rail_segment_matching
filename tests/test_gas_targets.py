"""辅助监督目标生成的单元测试。

只构造合成多边形，不加载数据集与模型，可在 CPU 上直接运行。
"""

import unittest

import numpy as np
import torch

from core_algorithm.gas_rail.losses import align_to_target
from data_preprocess.gas_targets import (
    GAS_TARGET_KEYS,
    GasTargetSpec,
    GasFormat,
    build_gas_targets,
)

_CANVAS = 128
_RATIO = 4
_GRID = _CANVAS // _RATIO
_SPEC = GasTargetSpec(mask_ratio=_RATIO)
_VERTICAL = [(60, 10), (68, 10), (68, 118), (60, 118)]
_HORIZONTAL = [(10, 60), (118, 60), (118, 68), (10, 68)]


def _polygon(points: list[tuple[float, float]]) -> np.ndarray:
    return np.array(points, dtype=np.float32)


def _build(segments: list[np.ndarray]) -> dict[str, np.ndarray]:
    targets = build_gas_targets(segments, _CANVAS, _CANVAS, _SPEC)
    return {key: value.numpy() for key, value in targets.items()}


class GasTargetTests(unittest.TestCase):
    """覆盖空输入、两种走向、宽度、遮挡断段与格式器校验。"""

    def test_empty_input_returns_zero_targets(self) -> None:
        """没有实例时返回形状正确、全零的目标。"""
        targets = _build([])
        self.assertEqual(set(targets), set(GAS_TARGET_KEYS))
        self.assertEqual(targets["gas_center"].shape, (1, _GRID, _GRID))
        self.assertEqual(targets["gas_direction"].shape, (2, _GRID, _GRID))
        for key in GAS_TARGET_KEYS:
            self.assertEqual(float(np.abs(targets[key]).sum()), 0.0)

    def test_vertical_rail_peak_and_direction(self) -> None:
        """竖直轨的峰值落在轨身中心列，双角方向为 (cos π, sin π)。"""
        targets = _build([_polygon(_VERTICAL)])
        center = targets["gas_center"][0]
        row, column = np.unravel_index(int(center.argmax()), center.shape)
        self.assertAlmostEqual(float(center[row, column]), 1.0, places=5)
        self.assertAlmostEqual(column, 63.5 / _RATIO - 0.5, delta=1.0)
        direction = targets["gas_direction"][:, row, column]
        self.assertAlmostEqual(float(direction[0]), -1.0, delta=0.1)
        self.assertAlmostEqual(float(direction[1]), 0.0, delta=0.1)

    def test_horizontal_rail_direction(self) -> None:
        """水平轨的双角方向为 (1, 0)，与竖直轨相差 90°。"""
        targets = _build([_polygon(_HORIZONTAL)])
        center = targets["gas_center"][0]
        row, column = np.unravel_index(int(center.argmax()), center.shape)
        direction = targets["gas_direction"][:, row, column]
        self.assertAlmostEqual(float(direction[0]), 1.0, delta=0.1)
        self.assertAlmostEqual(float(direction[1]), 0.0, delta=0.1)

    def test_direction_is_unit_norm_where_valid(self) -> None:
        """有效域内方向向量必须归一化；无效域必须为 0。"""
        targets = _build([_polygon(_VERTICAL)])
        norm = np.hypot(
            targets["gas_direction"][0], targets["gas_direction"][1]
        )
        valid = targets["gas_direction_valid"][0] > 0
        self.assertGreater(int(valid.sum()), 0)
        np.testing.assert_allclose(norm[valid], 1.0, atol=1e-5)
        np.testing.assert_allclose(norm[~valid], 0.0, atol=1e-6)

    def test_width_target_is_log_of_grid_width(self) -> None:
        """8 像素宽、栅格比 4 → 2 个网格单元，目标为 log(2)。"""
        targets = _build([_polygon(_VERTICAL)])
        width = targets["gas_width"][0]
        valid = targets["gas_center_valid"][0] > 0
        self.assertGreater(int(valid.sum()), 0)
        self.assertAlmostEqual(
            float(np.median(width[valid])), float(np.log(2.0)), delta=0.15
        )

    def test_occlusion_gap_leaves_invalid_cells(self) -> None:
        """同一根轨被遮断成两段时，遮挡区不产生监督。"""
        first = _polygon([(60, 10), (68, 10), (68, 55), (60, 55)])
        second = _polygon([(60, 80), (68, 80), (68, 118), (60, 118)])
        targets = _build([first, second])
        center_valid = targets["gas_center_valid"][0]
        self.assertEqual(float(center_valid[15:18, 14:17].sum()), 0.0)
        self.assertGreater(float(center_valid[4:12, 14:16].sum()), 0.0)
        self.assertGreater(float(center_valid[21:29, 14:16].sum()), 0.0)

    def test_format_decouples_target_ratio(self) -> None:
        """目标栅格与数据管线解耦：管线 /2 时目标仍按 /4 生成。"""
        formatter = GasFormat(
            bbox_format="xywh",
            normalize=True,
            return_mask=True,
            mask_ratio=2,
            target_spec=GasTargetSpec(mask_ratio=4),
        )
        self.assertEqual(formatter.mask_ratio, 2)
        self.assertEqual(formatter.target_spec.mask_ratio, 4)

    def test_align_to_target_pools_by_area(self) -> None:
        """辅助输出按面积平均对齐到目标栅格。"""
        logits = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
        pooled = align_to_target(logits, (1, 1))
        self.assertEqual(tuple(pooled.shape), (1, 1, 1, 1))
        self.assertAlmostEqual(float(pooled[0, 0, 0, 0]), 4.0, places=5)
        unchanged = align_to_target(logits, (2, 2))
        self.assertTrue(torch.equal(unchanged, logits))


if __name__ == "__main__":
    unittest.main()
