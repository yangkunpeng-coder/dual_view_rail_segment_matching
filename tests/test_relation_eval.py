"""关系级匹配评测的单元测试，不依赖原图、标注或 GPU。"""

import unittest

import numpy as np

from utils.relation_eval import mask_iou_matrix, score_relations


class RelationScoreTests(unittest.TestCase):
    """覆盖关系匹配、单侧误配和覆盖计数。"""

    def test_perfect_relations_are_true_positive(self) -> None:
        score = score_relations(
            {(0, 0), (1, 1)}, {(0, 0), (1, 1)},
            np.eye(2), np.eye(2), set(), set(), 0.5,
        )
        self.assertEqual(score.true_positive, 2)
        self.assertEqual(score.false_positive, 0)
        self.assertEqual(score.false_negative, 0)
        self.assertTrue(score.pair_exact)
        self.assertAlmostEqual(score.f1, 1.0)
        self.assertEqual(score.south_gt_covered, 2)
        self.assertEqual(score.south_pred_covered, 2)

    def test_swapped_relation_is_not_matched(self) -> None:
        score = score_relations(
            {(0, 0), (1, 1)}, {(0, 1), (1, 0)},
            np.eye(2), np.eye(2), set(), set(), 0.5,
        )
        self.assertEqual(score.true_positive, 0)
        self.assertEqual(score.false_negative, 2)
        self.assertEqual(score.false_positive, 2)
        self.assertFalse(score.pair_exact)

    def test_one_prediction_cannot_cover_two_relations(self) -> None:
        south = np.array([[1.0, 0.9]])
        north = np.array([[1.0, 0.9]])
        score = score_relations(
            {(0, 0), (1, 1)}, {(0, 0)}, south, north, set(), set(), 0.5,
        )
        self.assertEqual(score.true_positive, 1)
        self.assertEqual(score.false_negative, 1)
        self.assertEqual(score.false_positive, 0)
        self.assertFalse(score.pair_exact)

    def test_extra_relation_on_single_side_is_counted(self) -> None:
        south = np.zeros((2, 3))
        north = np.zeros((2, 3))
        south[1, 2] = 1.0
        north[1, 2] = 1.0
        score = score_relations(
            set(), {(1, 1)}, south, north, {2}, {2}, 0.5,
        )
        self.assertEqual(score.false_positive, 1)
        self.assertEqual(score.extra_on_single, 1)
        self.assertEqual(score.unmapped_extra, 0)

    def test_unmapped_prediction_is_counted(self) -> None:
        score = score_relations(
            set(), {(0, 0)}, np.zeros((1, 1)), np.zeros((1, 1)),
            set(), set(), 0.5,
        )
        self.assertEqual(score.false_positive, 1)
        self.assertEqual(score.extra_on_single, 0)
        self.assertEqual(score.unmapped_extra, 1)

    def test_threshold_boundary_is_inclusive(self) -> None:
        score = score_relations(
            {(0, 0)}, {(0, 0)},
            np.array([[0.5]]), np.array([[0.5]]), set(), set(), 0.5,
        )
        self.assertEqual(score.true_positive, 1)
        below = score_relations(
            {(0, 0)}, {(0, 0)},
            np.array([[0.49]]), np.array([[0.49]]), set(), set(), 0.5,
        )
        self.assertEqual(below.true_positive, 0)

    def test_invalid_relation_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            score_relations(
                {(2, 0)}, {(0, 0)},
                np.eye(2), np.eye(2), set(), set(), 0.5,
            )


class MaskIouMatrixTests(unittest.TestCase):
    """覆盖 Mask IoU 矩阵的尺寸与基本数值。"""

    def test_identical_and_disjoint_masks(self) -> None:
        gt = np.zeros((1, 10, 10), dtype=np.bool_)
        gt[0, 2:6, 2:6] = True
        pred = np.zeros((2, 10, 10), dtype=np.bool_)
        pred[0, 2:6, 2:6] = True
        pred[1, 7:9, 7:9] = True
        matrix = mask_iou_matrix(pred, gt)
        self.assertEqual(matrix.shape, (2, 1))
        self.assertAlmostEqual(matrix[0, 0], 1.0)
        self.assertEqual(matrix[1, 0], 0.0)

    def test_mask_size_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            mask_iou_matrix(
                np.zeros((1, 10, 10), dtype=np.bool_),
                np.zeros((1, 8, 8), dtype=np.bool_),
            )


if __name__ == "__main__":
    unittest.main()
