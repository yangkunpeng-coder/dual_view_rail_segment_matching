"""验证钢轨有效区域算法和 YOLO 数据集导出。"""

import csv
from dataclasses import replace
import json
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from data_preprocess import create_YOLO_dataset as exporter
from data_preprocess.common.config import PROJECT_ROOT, config_path
from data_preprocess.common.roi_config import load_roi_settings
from data_preprocess.crop_image import crop_valid_image, load_crop_sample

_WIDTH = 80
_HEIGHT = 60
_SOURCE_COLOR = (35, 70, 150)
_STRAIGHT = [[10, 4], [14, 4], [30, 43], [26, 43]]
_CURVED = [
    [35, 4], [40, 4], [47, 14], [49, 26], [44, 43],
    [39, 43], [44, 26], [42, 14],
]
_THIRD = [[55, 4], [60, 4], [58, 43], [53, 43]]


class CropImageTests(unittest.TestCase):
    """用小尺寸图片覆盖栅格化、导出和异常样本。"""

    def setUp(self) -> None:
        scratch = PROJECT_ROOT / "temp"
        if scratch.resolve() != scratch:
            raise ValueError(f"临时目录指向意外位置 {scratch}")
        scratch.mkdir(exist_ok=True)
        directory = TemporaryDirectory(dir=scratch, prefix="roi_test_")
        self.root = Path(directory.name).resolve()
        self.assertEqual(self.root.parent, scratch)
        self.addCleanup(directory.cleanup)
        self.settings = load_roi_settings()
        self.corridor = replace(
            self.settings.corridor,
            margins=(2, 3),
            outside_fill_rgb=(80, 90, 100),
            max_interp_gap=3,
            smoothing_kernel=9,
        )

    def test_mask_coverage_and_dynamic_edges(self) -> None:
        """覆盖直线、弯曲、不同实例数、平滑和中性区域填充。"""
        for instance_count in (1, 2, 3):
            with self.subTest(instance_count=instance_count):
                polygons = [_STRAIGHT, _CURVED, _THIRD][:instance_count]
                image, annotation = _make_sample(self.root / "image.png", polygons)
                sample = load_crop_sample(image, annotation, self.corridor)
                self.assertEqual(len(sample.polygons), instance_count)
                self.assertEqual(sample.processed_image.shape, (_HEIGHT, _WIDTH, 3))

                for points in polygons:
                    instance = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
                    cv2.fillPoly(instance, [np.asarray(points, dtype=np.int32)], 1)
                    self.assertTrue(np.all(sample.valid_mask[instance != 0] == 1))
                inside = sample.valid_mask == 1
                self.assertTrue(np.all(sample.processed_image[inside] == _SOURCE_COLOR))
                outside_pixels = sample.processed_image[~inside]
                self.assertTrue(
                    np.all(outside_pixels == self.corridor.outside_fill_rgb)
                )
                self.assertNotEqual(
                    np.flatnonzero(inside[4])[0], np.flatnonzero(inside[43])[0]
                )
                self.assertFalse(inside[0].any())
                self.assertFalse(inside[-1].any())

    def test_order_group_id_and_repeatability(self) -> None:
        """更改实例顺序和分组编号时，逐行有效区一致。"""
        image, annotation = _make_sample(
            self.root / "sample.png", [_STRAIGHT, _CURVED, _THIRD]
        )
        image_bytes = image.read_bytes()
        annotation_bytes = annotation.read_bytes()
        processed = crop_valid_image(str(image), annotation)
        self.assertEqual(processed.dtype, np.uint8)
        self.assertTrue(np.array_equal(processed, crop_valid_image(image, annotation)))
        self.assertEqual(image.read_bytes(), image_bytes)
        self.assertEqual(annotation.read_bytes(), annotation_bytes)

        content = json.loads(annotation_bytes)
        content["shapes"].reverse()
        for shape in content["shapes"]:
            shape["group_id"] = 42
        _write_json(annotation, content)
        self.assertTrue(np.array_equal(processed, crop_valid_image(image, annotation)))
        sample = load_crop_sample(image, annotation, self.corridor)
        self.assertEqual(len(sample.polygons), 3)

    def test_short_and_long_empty_rows(self) -> None:
        """插值三个空行，并保留更长的无效间隔。"""
        polygons = [
            [[8, 2], [11, 2], [11, 5], [8, 5]],
            [[16, 9], [20, 9], [20, 12], [16, 12]],
            [[23, 22], [27, 22], [27, 26], [23, 26]],
        ]
        image, annotation = _make_sample(self.root / "gaps.png", polygons)
        settings = replace(self.corridor, margins=(0, 0), smoothing_kernel=1)
        sample = load_crop_sample(image, annotation, settings)
        self.assertEqual(np.flatnonzero(sample.valid_mask[7])[0], 12)
        self.assertTrue(np.all(sample.valid_mask[6:9].any(axis=1)))
        self.assertFalse(sample.valid_mask[13:22].any())

    def test_per_image_margins(self) -> None:
        """应用单图左右边距覆盖。"""
        polygon = [[15, 4], [19, 4], [19, 14], [15, 14]]
        image, annotation = _make_sample(self.root / "north" / "image.png", [polygon])
        settings = replace(
            self.corridor,
            margin_overrides={"north/image.png": (4, 0)},
        )
        sample = load_crop_sample(image, annotation, settings)
        self.assertEqual(sample.margins, (4, 0))
        actual_columns = np.flatnonzero(sample.valid_mask[8]).tolist()
        self.assertEqual(actual_columns, list(range(11, 20)))

    def test_invalid_sample(self) -> None:
        """拒绝空标注、损坏 JSON、尺寸不匹配和无效 Polygon。"""
        image, annotation = _make_sample(self.root / "image.png", [_STRAIGHT])
        baseline = json.loads(annotation.read_bytes())
        shape = baseline["shapes"][0]
        invalid = (
            {**baseline, "shapes": []},
            {**baseline, "imageWidth": _WIDTH + 1},
            {**baseline, "shapes": [{**shape, "shape_type": "rectangle"}]},
            {**baseline, "shapes": [{**shape, "points": [[0, 0], [0, 1]]}]},
            {**baseline, "shapes": [{**shape, "points": [[0, 0], [1, 1], [2, 2]]}]},
            {**baseline, "shapes": [{**shape, "points": [[-1, 0], [1, 0], [0, 1]]}]},
            {
                **baseline,
                "shapes": [{**shape, "points": [[79.6, 1], [80, 1], [80, 5]]}],
            },
        )
        for content in invalid:
            with self.subTest(content=content):
                _write_json(annotation, content)
                with self.assertRaises(ValueError):
                    load_crop_sample(image, annotation, self.corridor)
        annotation.write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_crop_sample(image, annotation, self.corridor)

    def test_export_in_one_step(self) -> None:
        """按已有划分直接导出有效样本，并记录拒绝原因。"""
        config_dir = self.root / "config" / "data"
        config_dir.mkdir(parents=True)
        for name in ("rail_seg.yaml", "clop_seg.yaml"):
            copy2(config_path(name), config_dir / name)
        relative_path = Path("data/rail_data")
        original_root = self.root / relative_path
        reference_root = self.root / "data" / "rail_seg" / "images"
        for split in ("train", "val", "test"):
            (reference_root / split).mkdir(parents=True)
        samples = (
            ("train", "north", "20260101030000.png", [_STRAIGHT, _CURVED]),
            ("train", "south", "20260101030000.jpg", []),
            ("train", "south", "20260101030002.png", [[[-3, 0], [1, 0], [0, 1]]]),
            ("val", "north", "20260101010000.png", [_STRAIGHT]),
            ("val", "south", "20260101010000.png", [_STRAIGHT]),
            ("test", "north", "20260101020000.png", [_STRAIGHT]),
            ("test", "south", "20260101020000.png", [_STRAIGHT]),
        )
        for split, side, name, polygons in samples:
            image, annotation = _make_sample(original_root / side / name, polygons)
            (reference_root / split / f"{side}_{image.stem}.png").touch()
            if split == "val" and side == "south":
                annotation.write_text("{", encoding="utf-8")
            if split == "test" and side == "south":
                annotation.unlink()

        artifacts = self.root / "data" / "clop_seg_artifacts"
        output = self.root / "data" / "clop_seg"
        legacy_review = (
            self.root / "data" / "clop_seg_candidates" / "audit" / "review.csv"
        )
        legacy_review.parent.mkdir(parents=True)
        legacy_review.write_bytes(b"review_status\npending\n")
        with patch.object(exporter, "PROJECT_ROOT", self.root):
            artifacts.mkdir()
            with self.assertRaises(FileExistsError):
                exporter.create_YOLOseg_dataset(relative_path)
            artifacts.rmdir()
            exporter.create_YOLOseg_dataset(relative_path)

            rejected = _read_csv(artifacts / "audit" / "rejected_samples.csv")
            rejected_splits = {row["split"] for row in rejected}
            self.assertEqual(rejected_splits, {"train", "val", "test"})
            self.assertEqual(len(rejected), 4)
            manifest = _read_csv(artifacts / "audit" / "manifest.csv")
            self.assertEqual(len(manifest), 3)
            self.assertFalse((artifacts / "audit" / "review.csv").exists())
            output_dirs = {path.name for path in output.iterdir()}
            self.assertEqual(output_dirs, {"images", "labels"})
            expected = {
                ("train", "north_20260101030000.png"),
                ("val", "north_20260101010000.png"),
                ("test", "north_20260101020000.png"),
            }
            self.assertEqual(
                {(path.parent.name, path.name) for path in output.rglob("*.png")},
                expected,
            )
            self.assertEqual(
                {(path.parent.name, path.stem) for path in output.rglob("*.txt")},
                {(split, Path(name).stem) for split, name in expected},
            )

            filename = "north_20260101030000.png"
            exported_annotation = artifacts / "annotations" / "train" / (
                Path(filename).with_suffix(".json")
            )
            new = json.loads(exported_annotation.read_bytes())
            source = original_root / "north" / "20260101030000.json"
            original = json.loads(source.read_bytes())
            self.assertEqual(new["shapes"], original["shapes"])
            self.assertIsNone(new["imageData"])
            self.assertEqual(
                (exported_annotation.parent / new["imagePath"]).resolve(),
                output / "images" / "train" / filename,
            )
            self.assertTrue((artifacts / "valid_masks" / "train" / filename).is_file())
            preview_name = Path(filename).with_suffix(".jpg")
            preview = artifacts / "previews" / "train" / preview_name
            self.assertTrue(preview.is_file())
            labels = output / "labels" / "train" / Path(filename).with_suffix(".txt")
            self.assertEqual(len(labels.read_text(encoding="utf-8").splitlines()), 2)

            stale_image = output / "images" / "train" / "stale.png"
            stale_preview = artifacts / "previews" / "train" / "stale.jpg"
            stale_image.touch()
            stale_preview.touch()
            exporter.create_YOLOseg_dataset(relative_path)
            self.assertFalse(stale_image.exists())
            self.assertFalse(stale_preview.exists())
        self.assertEqual(legacy_review.read_bytes(), b"review_status\npending\n")


def _make_sample(
    image_path: Path, polygons: list[list[list[int]]]
) -> tuple[Path, Path]:
    """在临时目录写出彩色原图和对应的 LabelMe 测试标注。"""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (_WIDTH, _HEIGHT), _SOURCE_COLOR).save(image_path)
    annotation_path = image_path.with_suffix(".json")
    _write_json(
        annotation_path,
        {
            "imageWidth": _WIDTH,
            "imageHeight": _HEIGHT,
            "imagePath": image_path.name,
            "imageData": "placeholder",
            "shapes": [
                {
                    "label": "rail",
                    "shape_type": "polygon",
                    "points": points,
                    "group_id": index,
                }
                for index, points in enumerate(polygons, start=1)
            ],
        },
    )
    return image_path, annotation_path


def _write_json(path: Path, annotation: dict[str, object]) -> None:
    """保存测试用标注。"""
    path.write_text(json.dumps(annotation), encoding="utf-8", newline="\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取测试用导出或拒绝清单。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))
