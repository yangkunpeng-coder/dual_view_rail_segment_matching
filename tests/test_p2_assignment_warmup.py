"""用 CPU 合成样本验证 P2 热身分配与旧配置的兼容性。"""

from pathlib import Path
import unittest

import torch
from torch import Tensor
from ultralytics.utils.tal import SegTaskAlignedAssigner, make_anchors

from core_algorithm.p2_assignment_warmup import (
    P2AssignmentWarmupSettings,
    _P2WarmupAssigner,
)
from experiments.yolo_seg.train_rail_seg import _pop_p2

_STRIDES = [4, 8, 16, 32]
_P2_COUNT = 16 * 16
_SETTINGS = P2AssignmentWarmupSettings(
    epochs=3, max_positives_per_gt=3, target_score=0.5
)


def _assignment_inputs() -> dict[str, Tensor]:
    """构造框内有足够粗尺度匹配、P2 预测框偏小的单实例图。"""
    feats = [torch.empty((1, 1, 64 // stride, 64 // stride)) for stride in _STRIDES]
    centers, strides = make_anchors(feats, _STRIDES)
    centers = centers * strides
    gt_box = torch.tensor([12.0, 12.0, 52.0, 52.0])
    predictions = gt_box.expand(1, len(centers), 4).clone()
    p2 = centers[:_P2_COUNT]
    predictions[0, :_P2_COUNT] = torch.cat((p2 - 2, p2 + 2), dim=-1)
    masks = torch.zeros((1, 1, 32, 32), dtype=torch.bool)
    masks[0, 0, 9:24, 17:21] = True
    return {
        "pd_scores": torch.full((1, len(centers), 2), 0.7),
        "pd_bboxes": predictions,
        "anc_points": centers,
        "gt_labels": torch.ones((1, 1, 1)),
        "gt_bboxes": gt_box.view(1, 1, 4),
        "mask_gt": torch.ones((1, 1, 1), dtype=torch.bool),
        "pd_coef": torch.ones((1, len(centers), 1)),
        "mask_scale": torch.tensor([[0.5]]),
        "proto": torch.ones((1, 1, 32, 32)),
        "gt_masks": masks,
    }


def _assigners(
    topk: int, topk2: int
) -> tuple[SegTaskAlignedAssigner, _P2WarmupAssigner]:
    """用和两个检测头一致的 top-k 参数构造原生和热身分配器。"""
    native = SegTaskAlignedAssigner(
        topk=topk, topk2=topk2, num_classes=2, alpha=0.5, stride=_STRIDES
    )
    warmup = _P2WarmupAssigner(native, _SETTINGS)
    warmup.anchor_count = _P2_COUNT
    return native, warmup


class P2WarmupAssignerTests(unittest.TestCase):
    """覆盖补位、数量约束和多个 GT 的冲突处理。"""

    def test_adds_in_mask_positives_within_each_head_budget(self) -> None:
        inputs = _assignment_inputs()
        original_predictions = inputs["pd_bboxes"].clone()
        for topk, cap, p2_count in ((10, 10, 3), (7, 1, 1)):
            with self.subTest(cap=cap):
                native, warmup = _assigners(topk, cap)
                _, _, _, native_fg, _ = native(**inputs)
                labels, boxes, scores, foreground, gt_indices = warmup(**inputs)
                p2 = foreground[0, :_P2_COUNT].nonzero().squeeze(1)
                self.assertEqual(int(native_fg[:, :_P2_COUNT].sum()), 0)
                self.assertEqual(int(native_fg.sum()), cap)
                self.assertEqual(int(foreground.sum()), cap)
                self.assertEqual(len(p2), p2_count)
                self.assertTrue((labels[0, p2] == 1).all())
                self.assertTrue((gt_indices[0, p2] == 0).all())
                self.assertTrue((boxes[0, p2] == inputs["gt_bboxes"][0, 0]).all())
                self.assertTrue((scores[0, p2, 1] == _SETTINGS.target_score).all())
                self.assertTrue((scores[0, p2, 0] == 0).all())
                self.assertTrue((scores[~foreground] == 0).all())
                pixels = (inputs["anc_points"][p2] * 0.5).long()
                is_on_mask = inputs["gt_masks"][0, 0, pixels[:, 1], pixels[:, 0]]
                self.assertTrue(is_on_mask.all())
        torch.testing.assert_close(inputs["pd_bboxes"], original_predictions)

    def test_keeps_native_assignment_with_sufficient_p2_positives(self) -> None:
        inputs = _assignment_inputs()
        inputs["pd_bboxes"][0, :_P2_COUNT] = inputs["gt_bboxes"][0, 0]
        inputs["pd_scores"][0, _P2_COUNT:] = 0.01
        native, warmup = _assigners(10, 10)
        expected = native(**inputs)
        self.assertGreaterEqual(int(expected[3][:, :_P2_COUNT].sum()), 3)
        torch.testing.assert_close(warmup(**inputs), expected)

    def test_keeps_native_assignment_without_mask_candidates(self) -> None:
        inputs = _assignment_inputs()
        inputs["gt_masks"].zero_()
        native, warmup = _assigners(7, 1)
        torch.testing.assert_close(warmup(**inputs), native(**inputs))

    def test_shared_mask_point_goes_to_nearer_gt_only(self) -> None:
        inputs = _assignment_inputs()
        inputs["gt_masks"] = torch.zeros((1, 2, 32, 32), dtype=torch.bool)
        inputs["gt_masks"][0, :, 15, 17] = True
        inputs["gt_labels"] = torch.tensor([[[0.0], [1.0]]])
        inputs["gt_bboxes"] = torch.tensor(
            [[[12.0, 12.0, 52.0, 52.0], [14.0, 10.0, 54.0, 50.0]]]
        )
        inputs["mask_gt"] = torch.ones((1, 2, 1), dtype=torch.bool)
        _, warmup = _assigners(7, 1)
        _, _, _, foreground, gt_indices = warmup(**inputs)
        p2 = foreground[0, :_P2_COUNT].nonzero().squeeze(1)
        self.assertEqual(len(p2), 1)
        self.assertEqual(gt_indices[0, p2].item(), 1)

    def test_empty_batch_keeps_native_assignment(self) -> None:
        inputs = _assignment_inputs()
        for key in ("gt_labels", "gt_bboxes", "mask_gt", "gt_masks"):
            inputs[key] = inputs[key][:, :0]
        native, warmup = _assigners(7, 1)
        torch.testing.assert_close(warmup(**inputs), native(**inputs))


class P2WarmupConfigTests(unittest.TestCase):
    """验证新设置必须显式开启，旧配置仍解析成原生四尺度。"""

    def test_omission_and_explicit_disable_are_equivalent(self) -> None:
        block = {
            "enabled": True,
            "model": "p2.yaml",
            "pretrained": "init.pt",
            "mask_ratio": 2,
        }
        source = Path("train.yaml")
        native = _pop_p2({"p2": block}, source)
        disabled = _pop_p2(
            {"p2": {**block, "assignment_warmup": {"enabled": False}}}, source
        )
        self.assertIsNotNone(native)
        self.assertIsNone(native.assignment_warmup)
        self.assertEqual(native, disabled)

    def test_rejects_enabled_warmup_without_p2(self) -> None:
        options = {
            "p2": {
                "enabled": False,
                "assignment_warmup": {
                    "enabled": True, "epochs": 3, "max_positives_per_gt": 1,
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "p2.enabled"):
            _pop_p2(options, Path("train.yaml"))


if __name__ == "__main__":
    unittest.main()
