"""用临时原图、NPZ 和合成预测验证实例适配。"""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from core_algorithm.postprocess import PostprocessConfig
from data_preprocess.common.config import PROJECT_ROOT
from data_preprocess.common.roi_config import load_roi_settings
from utils.segmentation_instances import PredictedViewLoader, load_archived_view


class ArchivedViewTests(unittest.TestCase):
    """覆盖归档 NPZ 读取、去重和尺寸校验。"""

    def setUp(self) -> None:
        self.directory = Path(tempfile.gettempdir())
        self.image_path = self.directory / "pp_instance_test.png"
        self.npz_path = self.directory / "pp_instance_test.npz"
        self.addCleanup(self.image_path.unlink, missing_ok=True)
        self.addCleanup(self.npz_path.unlink, missing_ok=True)

    def _write_fixture(self, masks: np.ndarray) -> None:
        image = np.full((20, 30, 3), 127, dtype=np.uint8)
        if not cv2.imwrite(str(self.image_path), image):
            raise OSError("无法写入测试图片")
        np.savez_compressed(
            self.npz_path,
            masks=masks,
            scores=np.array([0.9, 0.8], dtype=np.float32)[:len(masks)],
        )

    def test_deduplicates_archived_masks(self) -> None:
        mask = np.zeros((20, 30), dtype=np.bool_)
        mask[5:15, 5:15] = True
        self._write_fixture(np.stack([mask, mask]))
        view = load_archived_view(
            self.npz_path, self.image_path,
            PostprocessConfig(enabled=True, mask_nms_iou=0.6),
        )
        self.assertEqual(view.raw_count, 2)
        self.assertEqual(len(view.masks), 1)
        self.assertEqual(view.suppressed_count, 1)
        self.assertEqual(view.image_rgb.shape, (20, 30, 3))
        self.assertEqual(view.masks.dtype, np.bool_)

    def test_rejects_mask_size_mismatch(self) -> None:
        mask = np.zeros((10, 10), dtype=np.bool_)
        mask[2:5, 2:5] = True
        self._write_fixture(mask[None])
        with self.assertRaises(ValueError):
            load_archived_view(self.npz_path, self.image_path)


class PredictedCorridorTests(unittest.TestCase):
    """覆盖二次分割的原图坐标和首遍结果回退。"""

    def setUp(self) -> None:
        scratch = PROJECT_ROOT / "temp"
        if scratch.resolve() != scratch:
            raise ValueError(f"临时目录指向意外位置 {scratch}")
        scratch.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(
            dir=scratch, prefix="rail_corridor_test_"
        )
        self.addCleanup(directory.cleanup)
        self.image_path = Path(directory.name) / "original.png"
        weights = Path(directory.name) / "test.pt"
        weights.touch()
        corridor = replace(
            load_roi_settings().corridor,
            margins=(1, 1),
            smoothing_kernel=1,
            outside_fill_rgb=(10, 20, 30),
        )
        with patch("ultralytics.YOLO", return_value=Mock(task="segment")):
            self.loader = PredictedViewLoader(
                weights,
                postprocess_config=PostprocessConfig(
                    enabled=True, quality_enabled=True,
                ),
                corridor=corridor,
            )
        self.image_bgr = np.full((20, 30, 3), (17, 29, 61), dtype=np.uint8)
        self.scout = np.zeros((1, 20, 30), dtype=np.bool_)
        self.scout[0, 3:8, 5:9] = True
        self.scout[0, 8:15, 9:13] = True
        self.scout_result = (
            self.image_bgr, self.scout, np.array([0.7], dtype=np.float32), None,
        )

    def test_maps_masks_and_boxes_before_postprocessing(self) -> None:
        """回映后去重，过滤空实例并保持置信度、质量与 Mask 对齐。"""
        masks = np.zeros((3, 12, 10), dtype=np.bool_)
        masks[:2, 2:9, 2:5] = True
        scores = np.array([0.8, 0.9, 0.95], dtype=np.float32)
        boxes = np.array([[2, 2, 5, 9]] * 3, dtype=np.float32)
        with patch.object(
            self.loader, "_predict",
            side_effect=[self.scout_result, (None, masks, scores, boxes)],
        ) as predict:
            view = self.loader.load(self.image_path)
        crop = predict.call_args.args[0]
        self.assertEqual(crop.shape, (12, 10, 3))
        self.assertEqual(tuple(crop[0, -1]), (30, 20, 10))
        self.assertEqual(tuple(crop[0, 1]), (17, 29, 61))
        self.assertEqual(tuple(view.image_rgb[0, 0]), (61, 29, 17))
        self.assertEqual(view.roi_bbox, (4, 3, 14, 15))
        self.assertEqual(view.roi_status, "cropped")
        expected = np.zeros_like(self.scout)
        expected[0, 5:12, 6:9] = True
        np.testing.assert_array_equal(view.masks, expected)
        np.testing.assert_array_equal(view.scores, scores[1:2])
        self.assertEqual(view.suppressed_count, 1)
        self.assertEqual(view.qualities[0].bbox, (6, 5, 9, 12))

    def test_falls_back_when_second_pass_is_empty(self) -> None:
        """空二次结果保留首遍 Mask 和候选裁剪框。"""
        empty = (
            None, np.zeros((0, 12, 10), dtype=np.bool_),
            np.empty(0, dtype=np.float32), None,
        )
        with patch.object(
            self.loader, "_predict", side_effect=[self.scout_result, empty]
        ):
            view = self.loader.load(self.image_path)
        self.assertEqual(view.roi_bbox, (4, 3, 14, 15))
        self.assertEqual(view.roi_status, "empty_crop")
        np.testing.assert_array_equal(view.masks, self.scout)

    def test_skips_second_pass_when_no_corridor(self) -> None:
        """无首遍实例时保留原图尺寸的空 Mask 栈。"""
        empty = (
            self.image_bgr, self.scout[:0], np.empty(0, dtype=np.float32), None,
        )
        with patch.object(self.loader, "_predict", return_value=empty) as predict:
            view = self.loader.load(self.image_path)
        predict.assert_called_once()
        self.assertIsNone(view.roi_bbox)
        self.assertEqual(view.roi_status, "no_corridor")
        self.assertEqual(view.masks.shape, (0, 20, 30))


if __name__ == "__main__":
    unittest.main()
