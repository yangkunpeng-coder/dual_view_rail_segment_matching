"""裁剪并一次性导出 YOLO 钢轨实例分割数据集。"""

from collections import Counter
import csv
import json
from os.path import relpath
from pathlib import Path
from shutil import rmtree
import sys

import numpy as np
from PIL import Image

# 支持从 CMD 或 IDE 直接运行文件时使用项目内的绝对导入。
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preprocess.common.config import (
    PROJECT_ROOT,
    load_dataset_relative_path,
    load_yaml_mapping,
)
from data_preprocess.common.image import review_rgb_image
from data_preprocess.common.roi_config import RoiSettings, load_roi_settings
from data_preprocess.common.roi_preview import render_crop_preview
from data_preprocess.crop_image import CropSample, load_crop_sample
from data_preprocess.match_pair_image import match_pair_image

_SEGMENTATION_CONFIG = Path("config/data/clop_seg.yaml")
_EXPORT_RELATIVE_PATH = Path("data/clop_seg")
_EMPTY_LABEL_RELATIVE_PATH = Path("data/empty_label")
_SPLITS = ("train", "val", "test")
_CAMERA_SIDES = ("south", "north")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_TRAIN_RATIO = 0.7
_TRAIN_VAL_RATIO = 0.9
_RAIL_CLASS_ID = 0
_SAMPLE_COLUMNS = ("split", "image_name", "source_image", "source_json")
_MANIFEST_COLUMNS = (
    *_SAMPLE_COLUMNS,
    "yolo_image",
    "yolo_label",
    "annotation",
    "valid_mask",
    "preview",
    "width",
    "height",
    "instance_count",
    "margin_left",
    "margin_right",
    "mask_coverage",
)
_REJECTED_COLUMNS = (*_SAMPLE_COLUMNS, "reason")


def create_YOLOseg_dataset(dataset_relative_path: str | Path) -> None:
    """排除空标签清单中的图片后，直接重建 data/clop_seg。

    Args:
        dataset_relative_path: 原始数据集相对于项目根目录的路径。

    Raises:
        OSError: 无法读取输入目录或重建输出目录。
        ValueError: 配置或数据集划分无效。单张输入无效时记入拒绝表。
    """
    settings = load_roi_settings()
    relative_path = _validate_dataset_path(dataset_relative_path)
    output_root, artifacts_root = _output_roots(relative_path, settings)
    _check_artifacts_directory(artifacts_root)

    splits, rejected = _plan_export(relative_path, settings)
    json_root = PROJECT_ROOT / relative_path / settings.json_dir
    manifest: list[dict[str, object]] = []

    _recreate_directory(artifacts_root)
    audit = artifacts_root / "audit"
    audit.mkdir()
    manifest_path = audit / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_MANIFEST_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        # 即使处理途中被终止，下次也可用表头识别本脚本生成的目录。
        stream.flush()
        _recreate_directory(output_root)
        for split, images in splits.items():
            _create_split_dirs(output_root, artifacts_root, split, settings)
            print(f"处理 {split} 的 {len(images)} 张图片", flush=True)
            for image_path in images:
                json_path = (
                    json_root / image_path.parent.name / f"{image_path.stem}.json"
                )
                source = _source_record(split, image_path, json_path)
                try:
                    sample = load_crop_sample(image_path, json_path, settings.corridor)
                except (OSError, ValueError) as error:
                    rejected.append({**source, "reason": str(error)})
                    continue

                row = _save_sample(
                    output_root, artifacts_root, source, sample, settings
                )
                writer.writerow(row)
                manifest.append(row)
                del sample

    _write_csv(audit / "rejected_samples.csv", _REJECTED_COLUMNS, rejected)
    _write_statistics(audit, relative_path, settings, manifest, rejected)

    print(f"已导出 {len(manifest)} 张，拒绝 {len(rejected)} 张。")
    print(f"YOLO 数据集位于 {output_root}")
    print(f"导出统计和拒绝清单位于 {artifacts_root / 'audit'}")


def _validate_dataset_path(dataset_relative_path: str | Path) -> Path:
    """检查原始数据根目录位于项目内。"""
    relative_path = Path(dataset_relative_path)
    if relative_path.anchor or ".." in relative_path.parts:
        raise ValueError("原始数据集路径必须相对于项目根目录")
    dataset_root = (PROJECT_ROOT / relative_path).resolve()
    if not dataset_root.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"原始数据集目录超出项目 {dataset_root}")
    return relative_path


def _output_roots(relative_path: Path, settings: RoiSettings) -> tuple[Path, Path]:
    """核对 YOLO 配置及输出路径，避免影响原始数据或旧数据集。"""
    config_path = PROJECT_ROOT / _SEGMENTATION_CONFIG
    config = load_yaml_mapping(config_path)
    path = config.get("path")
    if not isinstance(path, str) or Path(path) != _EXPORT_RELATIVE_PATH:
        raise ValueError(f"{config_path} 的 path 必须指向 data/clop_seg")
    if config.get("names") != {_RAIL_CLASS_ID: settings.corridor.target_label}:
        raise ValueError(f"{config_path} 的类别须与 annotation.target_label 一致")
    for split in _SPLITS:
        if config.get(split) != f"images/{split}":
            raise ValueError(f"{config_path} 的 {split} 必须指向 images/{split}")

    output_root = PROJECT_ROOT / _EXPORT_RELATIVE_PATH
    artifacts_root = PROJECT_ROOT / settings.output_dir
    data_root = PROJECT_ROOT / "data"
    if artifacts_root == data_root or not artifacts_root.is_relative_to(data_root):
        raise ValueError(f"诊断目录必须位于项目的 data 目录内 {artifacts_root}")
    dataset_root = PROJECT_ROOT / relative_path
    protected = (
        dataset_root.resolve(),
        (dataset_root / settings.image_dir).resolve(),
        (dataset_root / settings.json_dir).resolve(),
        (data_root / "rail_seg").resolve(),
        (PROJECT_ROOT / _EMPTY_LABEL_RELATIVE_PATH).resolve(),
    )
    for root in (output_root, artifacts_root):
        if root.resolve() != root or any(
            _paths_overlap(root, path) for path in protected
        ):
            raise ValueError(f"输出目录与输入、旧数据集重叠或指向意外位置 {root}")
        if root.exists() and not root.is_dir():
            raise NotADirectoryError(f"输出路径已被文件占用 {root}")
    if _paths_overlap(output_root, artifacts_root):
        raise ValueError(f"诊断目录与数据集导出目录重叠 {artifacts_root}")
    return output_root, artifacts_root


def _paths_overlap(first: Path, second: Path) -> bool:
    """判断两条路径是否相同或存在包含关系。"""
    return first.is_relative_to(second) or second.is_relative_to(first)


def _check_artifacts_directory(artifacts_root: Path) -> None:
    """仅允许重建由本脚本生成的诊断目录。"""
    if not artifacts_root.exists():
        return
    manifest_path = artifacts_root / "audit" / "manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            if next(csv.reader(stream), []) == list(_MANIFEST_COLUMNS):
                return
    statistics_path = artifacts_root / "audit" / "statistics.json"
    if statistics_path.is_file():
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        if isinstance(statistics, dict) and statistics.get("yolo_dataset") == (
            _EXPORT_RELATIVE_PATH.as_posix()
        ):
            return
    raise FileExistsError(f"诊断路径被其他目录占用 {artifacts_root}")


def _recreate_directory(output_root: Path) -> None:
    """清空已校验的输出目录并直接在原路径重建。"""
    data_root = PROJECT_ROOT / "data"
    if (
        output_root.resolve() != output_root
        or output_root == data_root
        or not output_root.is_relative_to(data_root)
    ):
        raise ValueError(f"导出路径指向意外位置 {output_root}")
    if output_root.exists():
        if not output_root.is_dir():
            raise NotADirectoryError(f"输出路径已被文件占用 {output_root}")
        rmtree(output_root)
    output_root.mkdir(parents=True)


def _create_split_dirs(
    output_root: Path, artifacts_root: Path, split: str, settings: RoiSettings
) -> None:
    """创建 YOLO 目录和启用的诊断附件目录。"""
    for directory in ("images", "labels"):
        (output_root / directory / split).mkdir(parents=True)
    artifact_options = (
        (settings.preserve_original_json, "annotations"),
        (settings.save_valid_mask, "valid_masks"),
        (settings.save_preview, "previews"),
    )
    for should_save, directory in artifact_options:
        if should_save:
            (artifacts_root / directory / split).mkdir(parents=True)


def _load_empty_label_images() -> set[Path]:
    """读取南北空标签清单，返回需要排除的原图绝对路径。"""
    excluded_images: set[Path] = set()
    for side in _CAMERA_SIDES:
        list_path = PROJECT_ROOT / _EMPTY_LABEL_RELATIVE_PATH / f"{side}.txt"
        try:
            lines = list_path.read_text(encoding="utf-8-sig").splitlines()
        except FileNotFoundError:
            continue
        for line_number, line in enumerate(lines, start=1):
            entry = line.strip()
            if not entry:
                continue
            image = Path(entry)
            if image.anchor or ".." in image.parts:
                raise ValueError(f"{list_path} 第 {line_number} 行须为项目内相对路径")
            excluded_images.add(PROJECT_ROOT / image)
    return excluded_images


def _plan_export(
    relative_path: Path, settings: RoiSettings
) -> tuple[dict[str, list[Path]], list[dict[str, object]]]:
    """排除清单内的原图，优先沿用其余图片的已有划分。"""
    images_relative_path = relative_path / settings.image_dir
    images_root = PROJECT_ROOT / images_relative_path
    json_root = PROJECT_ROOT / relative_path / settings.json_dir
    excluded_images = _load_empty_label_images()
    previous = _load_reference_splits(settings.split_config)
    rejected: list[dict[str, object]] = []
    images_by_side: dict[str, dict[str, Path]] = {}
    for side in _CAMERA_SIDES:
        images_by_side[side] = {}
        image_paths = list((images_root / side).iterdir())
        image_paths.sort()
        for image in image_paths:
            if (
                image.suffix.lower() not in settings.image_extensions
                or not image.is_file()
            ):
                continue
            if image in excluded_images:
                split = (
                    previous.pop(_export_name(image), "") if previous is not None
                    else ""
                )
                json_path = json_root / side / f"{image.stem}.json"
                record = _source_record(split, image, json_path)
                rejected.append({**record, "reason": "列入空标签清单"})
                continue
            if image.stem in images_by_side[side]:
                raise ValueError(f"相机目录中存在同名图片，导出文件将冲突 {image}")
            images_by_side[side][image.stem] = image

    if previous is None:
        return _split_images(images_relative_path, images_by_side, settings), rejected

    splits: dict[str, list[Path]] = {split: [] for split in _SPLITS}
    for images in images_by_side.values():
        for image in images.values():
            name = _export_name(image)
            split = previous.pop(name, None)
            if split is None:
                json_path = json_root / image.parent.name / f"{image.stem}.json"
                record = _source_record("", image, json_path)
                rejected.append({**record, "reason": "样本在已有划分中没有归属"})
                continue
            splits[split].append(image)

    for name, split in previous.items():
        side, _, stem = Path(name).stem.partition("_")
        image = images_root / side / f"{stem}.png"
        json_path = json_root / side / f"{stem}.json"
        record = _source_record(split, image, json_path)
        rejected.append({**record, "reason": "已有划分中的样本缺少原图"})
    return splits, rejected


def _load_reference_splits(config_path: Path | None) -> dict[str, str] | None:
    """读取已有 YOLO 图像的划分归属，保持原采集组边界。"""
    if config_path is None:
        return None
    config = load_yaml_mapping(PROJECT_ROOT / config_path)
    path = config.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{config_path} 的 path 必须是非空字符串")
    reference_root = PROJECT_ROOT / path
    if not reference_root.exists():
        return None

    splits: dict[str, str] = {}
    for split in _SPLITS:
        directory = config.get(split)
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError(f"{config_path} 的 {split} 必须是目录路径")
        for image in (reference_root / directory).iterdir():
            if image.suffix.lower() not in _IMAGE_SUFFIXES or not image.is_file():
                continue
            name = f"{image.stem}.png"
            if name in splits:
                raise ValueError(f"已有划分中发现重复图像名称 {image}")
            splits[name] = split
    return splits or None


def _split_images(
    images_relative_path: Path,
    images_by_side: dict[str, dict[str, Path]],
    settings: RoiSettings,
) -> dict[str, list[Path]]:
    """没有已有划分时，按采集组边界划分约 7:2:1 的三组数据。"""
    paired_names = match_pair_image(
        images_relative_path, image_extensions=settings.image_extensions
    )
    paired_stems = zip(paired_names["south"], paired_names["north"], strict=True)
    groups: list[tuple[Path, ...]] = [
        (images_by_side["south"].pop(south), images_by_side["north"].pop(north))
        for south, north in paired_stems
        if south in images_by_side["south"] and north in images_by_side["north"]
    ]
    # 同一采集组只保留一侧时，该图片作为单图组参与划分。
    for images in images_by_side.values():
        groups.extend((image,) for image in images.values())
    groups.sort(key=lambda group: min(path.stem for path in group))
    group_count = len(groups)
    if group_count < len(_SPLITS):
        raise ValueError("至少需要三个采集组才能划分训练、验证和测试集")

    image_counts = [0]
    for group in groups:
        image_counts.append(image_counts[-1] + len(group))
    train_target = image_counts[-1] * _TRAIN_RATIO
    train_val_target = image_counts[-1] * _TRAIN_VAL_RATIO
    train_end = min(
        range(1, group_count - 1),
        key=lambda index: abs(train_target - image_counts[int(index)]),
    )
    val_end = min(
        range(train_end + 1, group_count),
        key=lambda index: abs(train_val_target - image_counts[int(index)]),
    )
    split_groups = (groups[:train_end], groups[train_end:val_end], groups[val_end:])
    return {
        split: [image for group in partition for image in group]
        for split, partition in zip(_SPLITS, split_groups, strict=True)
    }


def _export_name(image_path: Path) -> str:
    """加相机前缀，统一无损 PNG 导出文件名。"""
    return f"{image_path.parent.name}_{image_path.stem}.png"


def _source_record(split: str, image_path: Path, json_path: Path) -> dict[str, str]:
    """建立可追溯至原图和原始标注的记录。"""
    return {
        "split": split,
        "image_name": _export_name(image_path),
        "source_image": image_path.relative_to(PROJECT_ROOT).as_posix(),
        "source_json": json_path.relative_to(PROJECT_ROOT).as_posix(),
    }


def _save_sample(
    output_root: Path,
    artifacts_root: Path,
    source: dict[str, str],
    sample: CropSample,
    settings: RoiSettings,
) -> dict[str, object]:
    """保存裁剪后的 PNG、YOLO 标签和已启用的诊断附件。"""
    name = Path(source["image_name"])
    split = source["split"]
    image_path = Path("images", split) / name
    label_path = Path("labels", split) / name.with_suffix(".txt")
    Image.fromarray(sample.processed_image).save(
        output_root / image_path, compress_level=1
    )

    height, width = sample.valid_mask.shape
    bounds = np.asarray((width, height), dtype=np.float64)
    lines = []
    for polygon in sample.polygons:
        coordinates = " ".join(f"{value:.8f}" for value in (polygon / bounds).ravel())
        lines.append(f"{_RAIL_CLASS_ID} {coordinates}")
    label_text = "\n".join(lines) + "\n"
    (output_root / label_path).write_text(label_text, encoding="utf-8", newline="\n")

    row: dict[str, object] = {
        **source,
        "yolo_image": (_EXPORT_RELATIVE_PATH / image_path).as_posix(),
        "yolo_label": (_EXPORT_RELATIVE_PATH / label_path).as_posix(),
        "width": width,
        "height": height,
        "instance_count": len(lines),
        "margin_left": sample.margins[0],
        "margin_right": sample.margins[1],
        "mask_coverage": 1.0,
    }
    if settings.preserve_original_json:
        annotation_path = Path("annotations", split) / name.with_suffix(".json")
        annotation_dir = PROJECT_ROOT / settings.output_dir / annotation_path.parent
        exported_image = PROJECT_ROOT / _EXPORT_RELATIVE_PATH / image_path
        updated_annotation = {
            **sample.annotation,
            "imagePath": Path(relpath(exported_image, annotation_dir)).as_posix(),
            "imageData": None,
        }
        _write_json(artifacts_root / annotation_path, updated_annotation)
        row["annotation"] = (settings.output_dir / annotation_path).as_posix()
    if settings.save_valid_mask:
        mask_path = Path("valid_masks", split) / name
        Image.fromarray(sample.valid_mask * 255).save(artifacts_root / mask_path)
        row["valid_mask"] = (settings.output_dir / mask_path).as_posix()
    if settings.save_preview:
        preview_path = Path("previews", split) / name.with_suffix(".jpg")
        preview = render_crop_preview(sample, name.name)
        preview.save(artifacts_root / preview_path, quality=88)
        row["preview"] = (settings.output_dir / preview_path).as_posix()
    return row


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    """写入固定表头的 UTF-8 导出或拒绝清单。"""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_statistics(
    audit: Path,
    relative_path: Path,
    settings: RoiSettings,
    manifest: list[dict[str, object]],
    rejected: list[dict[str, object]],
) -> None:
    """记录原始数据来源、目标路径和各集合导出数量。"""
    accepted_counts = Counter(row["split"] for row in manifest)
    rejected_counts = Counter(row["split"] for row in rejected)
    _write_json(
        audit / "statistics.json",
        {
            "dataset_relative_path": relative_path.as_posix(),
            "yolo_dataset": _EXPORT_RELATIVE_PATH.as_posix(),
            "target_label": settings.corridor.target_label,
            "input_count": len(manifest) + len(rejected),
            "exported_count": len(manifest),
            "rejected_count": len(rejected),
            "unassigned_count": rejected_counts[""],
            "splits": {
                split: {
                    "exported_count": accepted_counts[split],
                    "rejected_count": rejected_counts[split],
                }
                for split in _SPLITS
            },
        },
    )


def _write_json(path: Path, data: dict[str, object]) -> None:
    """保存 UTF-8 标注或统计文件。"""
    content = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(content + "\n", encoding="utf-8", newline="\n")


def _show_previews(dataset_relative_path: Path) -> None:
    """逐张显示已保存的裁剪对照图。"""
    settings = load_roi_settings()
    _, artifacts_root = _output_roots(
        _validate_dataset_path(dataset_relative_path), settings
    )
    if not settings.save_preview:
        raise ValueError("当前配置关闭了裁剪预览图")
    preview_root = artifacts_root / "previews"
    if not preview_root.is_dir():
        raise FileNotFoundError(f"请先运行导出脚本生成预览图 {preview_root}")
    previews = [
        path
        for split in _SPLITS
        for path in sorted((preview_root / split).glob("*.jpg"))
    ]
    print(f"共 {len(previews)} 张预览。关闭窗口查看下一张，按 Esc 退出。")
    for index, path in enumerate(previews, 1):
        with Image.open(path) as image:
            preview = np.asarray(image.convert("RGB")).astype(np.uint8, copy=False)
        title = f"{index}/{len(previews)} | {path.parent.name} | {path.stem}"
        if not review_rgb_image(preview, title):
            break


if __name__ == "__main__":
    dataset_path = load_dataset_relative_path()
    # 浏览已有对照图时，将下方调用改为 _show_previews(dataset_path)。
    create_YOLOseg_dataset(dataset_path)
