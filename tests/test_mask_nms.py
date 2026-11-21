"""Mask 级去重的单元测试：使用合成掩码，不依赖原图、标签或 GPU。"""

import unittest

import numpy as np

from core_algorithm.postprocess.mask_nms import suppress_duplicate_masks


def _bar_mask(height: int, width: int, start: int, end: int) -> np.ndarray:
    """生成指定列的布尔横条。"""
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[:, start:end] = True
    return mask


class MaskNmsTests(unittest.TestCase):
    """覆盖去重的保留、抑制、预筛和输入校验。"""

    def test_empty_input_returns_empty(self) -> None:
        result = suppress_duplicate_masks(
            np.zeros((0, 8, 8), dtype=np.bool_), np.empty(0, dtype=np.float32)
        )
        self.assertEqual(len(result.keep_indices), 0)
        self.assertEqual(len(result.suppressed_indices), 0)
        self.assertEqual(result.duplicate_of, {})

    def test_single_mask_is_kept(self) -> None:
        mask = _bar_mask(4, 10, 2, 6)[None]
        result = suppress_duplicate_masks(mask, np.array([0.8]))
        np.testing.assert_array_equal(result.keep_indices, [0])
        self.assertEqual(len(result.suppressed_indices), 0)

    def test_identical_masks_keep_highest_score(self) -> None:
        mask = _bar_mask(4, 10, 2, 8)
        masks = np.stack([mask, mask])
        result = suppress_duplicate_masks(masks, np.array([0.9, 0.8]))
        np.testing.assert_array_equal(result.keep_indices, [0])
        np.testing.assert_array_equal(result.suppressed_indices, [1])
        self.assertEqual(result.duplicate_of, {1: 0})

    def test_iou_below_threshold_keeps_both(self) -> None:
        masks = np.stack([
            _bar_mask(4, 12, 0, 6),
            _bar_mask(4, 12, 5, 11),
        ])
        result = suppress_duplicate_masks(
            masks, np.array([0.9, 0.8]), iou_threshold=0.6
        )
        np.testing.assert_array_equal(result.keep_indices, [0, 1])

    def test_threshold_one_disables_suppression(self) -> None:
        mask = _bar_mask(4, 8, 1, 7)
        masks = np.stack([mask, mask])
        result = suppress_duplicate_masks(
            masks, np.array([0.9, 0.8]), iou_threshold=1.0
        )
        np.testing.assert_array_equal(result.keep_indices, [0, 1])

    def test_chain_duplicates_only_compare_against_kept(self) -> None:
        masks = np.stack([
            _bar_mask(2, 30, 0, 10),
            _bar_mask(2, 30, 2, 12),
            _bar_mask(2, 30, 4, 14),
        ])
        result = suppress_duplicate_masks(
            masks, np.array([0.9, 0.8, 0.7]), iou_threshold=0.6
        )
        np.testing.assert_array_equal(result.keep_indices, [0, 2])
        np.testing.assert_array_equal(result.suppressed_indices, [1])

    def test_adjacent_rails_with_overlapping_boxes_are_kept(self) -> None:
        masks = np.stack([
            _bar_mask(10, 20, 0, 2),
            _bar_mask(10, 20, 12, 14),
        ])
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [4.0, 0.0, 14.0, 10.0]])
        result = suppress_duplicate_masks(
            masks, np.array([0.9, 0.85]), iou_threshold=0.3, boxes=boxes
        )
        np.testing.assert_array_equal(result.keep_indices, [0, 1])

    def test_provided_boxes_match_computed_boxes(self) -> None:
        mask = _bar_mask(6, 12, 3, 9)
        masks = np.stack([mask, mask])
        boxes = np.array([[3.0, 0.0, 9.0, 6.0], [3.0, 0.0, 9.0, 6.0]])
        with_boxes = suppress_duplicate_masks(
            masks, np.array([0.9, 0.8]), boxes=boxes
        )
        without_boxes = suppress_duplicate_masks(masks, np.array([0.9, 0.8]))
        np.testing.assert_array_equal(
            with_boxes.keep_indices, without_boxes.keep_indices
        )

    def test_tiny_mask_does_not_participate(self) -> None:
        tiny = np.zeros((6, 12), dtype=np.bool_)
        tiny[0, 0] = True
        big = _bar_mask(6, 12, 0, 8)
        masks = np.stack([tiny, big])
        result = suppress_duplicate_masks(
            masks, np.array([0.9, 0.8]), min_area_px=8
        )
        np.testing.assert_array_equal(result.keep_indices, [0, 1])

    def test_equal_scores_keep_stable_order(self) -> None:
        mask = _bar_mask(4, 8, 2, 6)
        masks = np.stack([mask, mask])
        result = suppress_duplicate_masks(masks, np.array([0.9, 0.9]))
        np.testing.assert_array_equal(result.keep_indices, [0])
        np.testing.assert_array_equal(result.suppressed_indices, [1])

    def test_inputs_are_not_mutated(self) -> None:
        masks = np.stack([
            _bar_mask(4, 10, 2, 8),
            _bar_mask(4, 10, 2, 8),
        ])
        scores = np.array([0.9, 0.8])
        masks_before = masks.copy()
        scores_before = scores.copy()
        suppress_duplicate_masks(masks, scores)
        np.testing.assert_array_equal(masks, masks_before)
        np.testing.assert_array_equal(scores, scores_before)

    def test_invalid_masks_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            suppress_duplicate_masks(
                np.zeros((4, 5), dtype=np.bool_), np.array([0.9])
            )

    def test_non_boolean_masks_raise(self) -> None:
        with self.assertRaises(ValueError):
            suppress_duplicate_masks(
                np.zeros((1, 4, 5), dtype=np.uint8), np.array([0.9])
            )

    def test_invalid_scores_raise(self) -> None:
        masks = np.zeros((2, 4, 5), dtype=np.bool_)
        with self.assertRaises(ValueError):
            suppress_duplicate_masks(masks, np.array([0.9]))
        with self.assertRaises(ValueError):
            suppress_duplicate_masks(masks, np.array([0.9, 1.2]))
        with self.assertRaises(ValueError):
            suppress_duplicate_masks(masks, np.array([0.9, np.nan]))

    def test_invalid_threshold_raises(self) -> None:
        masks = np.zeros((1, 4, 5), dtype=np.bool_)
        with self.assertRaises(ValueError):
            suppress_duplicate_masks(masks, np.array([0.9]), iou_threshold=1.5)


if __name__ == "__main__":
    unittest.main()
