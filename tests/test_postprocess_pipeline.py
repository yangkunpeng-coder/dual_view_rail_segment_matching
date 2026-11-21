"""分割后处理管线与配置的单元测试。"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core_algorithm.postprocess import (
    PostprocessConfig,
    measure_mask_quality,
    run_postprocess,
)
from utils.segmentation_postprocess import (
    POSTPROCESS_CONFIG,
    PostprocessImageStat,
    load_postprocess_config,
    parse_postprocess_config,
    save_instance_counts,
    save_postprocess_report,
)


def _square_mask(size: int, top: int, left: int) -> np.ndarray:
    """在给定位置生成 size×size 的方形 Mask。"""
    mask = np.zeros((10, 10), dtype=np.bool_)
    mask[top:top + size, left:left + size] = True
    return mask


class PostprocessPipelineTests(unittest.TestCase):
    """覆盖开关、去重结果对齐和质量元数据。"""

    def test_disabled_keeps_all_instances(self) -> None:
        masks = np.stack([_square_mask(4, 0, 0), _square_mask(4, 0, 0)])
        result = run_postprocess(
            masks,
            np.array([0.9, 0.8], dtype=np.float32),
            PostprocessConfig(enabled=False),
        )
        self.assertEqual(len(result.masks), 2)
        self.assertEqual(result.suppressed_count, 0)
        np.testing.assert_array_equal(result.keep_indices, [0, 1])

    def test_enabled_suppresses_duplicate(self) -> None:
        masks = np.stack([_square_mask(4, 0, 0), _square_mask(4, 0, 0)])
        result = run_postprocess(
            masks,
            np.array([0.9, 0.8], dtype=np.float32),
            PostprocessConfig(enabled=True, mask_nms_iou=0.6),
        )
        self.assertEqual(len(result.masks), 1)
        self.assertEqual(result.suppressed_count, 1)
        np.testing.assert_array_equal(result.keep_indices, [0])

    def test_result_masks_and_scores_follow_keep_indices(self) -> None:
        masks = np.stack([
            _square_mask(4, 0, 0),
            _square_mask(4, 0, 0),
            _square_mask(2, 6, 6),
        ])
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        result = run_postprocess(
            masks, scores, PostprocessConfig(mask_nms_iou=0.6)
        )
        np.testing.assert_array_equal(result.masks, masks[result.keep_indices])
        np.testing.assert_array_equal(
            result.scores, scores[result.keep_indices]
        )
        self.assertEqual(result.masks.dtype, np.bool_)
        self.assertEqual(result.scores.dtype, np.float32)
        self.assertGreaterEqual(result.latency_ms, 0.0)

    def test_quality_metadata_reports_components(self) -> None:
        mask = np.zeros((10, 10), dtype=np.bool_)
        mask[1:3, 1:3] = True
        mask[5:7, 5:7] = True
        quality = measure_mask_quality(mask)
        self.assertEqual(quality.area_px, 8)
        self.assertEqual(quality.component_count, 2)
        self.assertAlmostEqual(quality.main_component_ratio, 0.5)
        self.assertEqual(quality.bbox, (1, 1, 7, 7))
        self.assertFalse(quality.touches_border)

    def test_quality_empty_mask(self) -> None:
        quality = measure_mask_quality(np.zeros((5, 5), dtype=np.bool_))
        self.assertEqual(quality.area_px, 0)
        self.assertEqual(quality.component_count, 0)
        self.assertEqual(quality.bbox, (0, 0, 0, 0))

    def test_invalid_arrays_raise(self) -> None:
        with self.assertRaises(ValueError):
            run_postprocess(
                np.zeros((4, 5), dtype=np.bool_), np.array([0.9])
            )
        with self.assertRaises(ValueError):
            run_postprocess(
                np.zeros((2, 4, 5), dtype=np.bool_), np.array([0.9])
            )
        with self.assertRaises(ValueError):
            run_postprocess(
                np.zeros((2, 4, 5), dtype=np.bool_),
                np.array([0.9, 1.5]),
            )


class PostprocessConfigTests(unittest.TestCase):
    """覆盖 YAML 配置读取与错误提示。"""

    def test_loads_project_config(self) -> None:
        config = load_postprocess_config(POSTPROCESS_CONFIG)
        self.assertTrue(config.enabled)
        self.assertAlmostEqual(config.mask_nms_iou, 0.65)
        self.assertEqual(config.min_area_px, 8)
        self.assertFalse(config.quality_enabled)

    def test_parses_values(self) -> None:
        config = parse_postprocess_config({
            "enabled": True,
            "mask_nms": {"iou": 0.55, "min_area_px": 12},
            "quality": {"enabled": False},
        })
        self.assertTrue(config.enabled)
        self.assertAlmostEqual(config.mask_nms_iou, 0.55)
        self.assertEqual(config.min_area_px, 12)
        self.assertFalse(config.quality_enabled)

    def test_rejects_unknown_key(self) -> None:
        with self.assertRaises(ValueError):
            parse_postprocess_config({"enabled": True, "unknown": 1})

    def test_rejects_out_of_range_iou(self) -> None:
        with self.assertRaises(ValueError):
            parse_postprocess_config({"mask_nms": {"iou": 1.5}})

    def test_rejects_non_boolean_switch(self) -> None:
        with self.assertRaises(ValueError):
            parse_postprocess_config({"enabled": 1})


class PostprocessReportTests(unittest.TestCase):
    """覆盖统计文件和计数文件的字段与编码。"""

    def test_saves_report_and_counts(self) -> None:
        metrics_dir = Path(tempfile.gettempdir())
        report_path = metrics_dir / "postprocess.json"
        counts_path = metrics_dir / "instance_counts.csv"
        for path in (report_path, counts_path):
            path.unlink(missing_ok=True)
            self.addCleanup(path.unlink, missing_ok=True)
        stats = [
            PostprocessImageStat("north_1.png", "north", 20, 18, 2, 12.5),
            PostprocessImageStat("south_1.png", "south", 15, 15, 0, 8.0),
        ]
        saved_report = save_postprocess_report(
            metrics_dir, PostprocessConfig(), stats
        )
        report = json.loads(saved_report.read_text(encoding="utf-8"))
        self.assertEqual(report["raw_instances"], 35)
        self.assertEqual(report["kept_instances"], 33)
        self.assertEqual(report["suppressed_instances"], 2)
        self.assertEqual(report["image_count"], 2)

        rows = [{
            "side": "north", "image": "north_1.png", "gt": 18,
            "pred": 18, "tp": 17, "fp": 1, "missed": 1,
        }]
        saved_counts = save_instance_counts(metrics_dir, "val", rows)
        with saved_counts.open(encoding="utf-8-sig", newline="") as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["split"], "val")
        self.assertEqual(saved[0]["missed"], "1")


if __name__ == "__main__":
    unittest.main()
