"""把一次或多次实验的 GT 与预测掩码导出为紧凑 LabelMe 包。"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
from numpy.typing import NDArray

from data_preprocess.common.config import PROJECT_ROOT
from utils.instance_masks import load_yolo_polygons

BUNDLE_VERSION = "mask_bundle_v2"
_LABELME_VERSION = "5.0.1"
_DEFAULT_EPSILON_PX = 1.0
_SPLIT_NAMES = ("train", "valid", "test")
_SIDE_NAMES = ("north", "south")


@dataclass(frozen=True)
class _SplitSource:
    """一个划分的掩码目录和图片清单。"""

    split: str
    source_dir: Path
    images: list[dict[str, object]]


def build_mask_bundle(
    run_dir: str | Path,
    output_root: str | Path = "out/mask_bundle",
    splits: Sequence[str] = _SPLIT_NAMES,
    *,
    polygon_epsilon_px: float = _DEFAULT_EPSILON_PX,
    make_zip: bool = True,
) -> dict[str, object]:
    """把一次实验的 GT 与预测掩码导出为 LabelMe JSON 包。

    只导出 ``gt``、``pred`` 和根目录 ``README.md``，不生成 CSV、指标、
    manifest、checksums 或训练上下文。缺失的划分打印提示后跳过。

    Args:
        run_dir: 实验目录，例如 ``out/yolo_seg/20261111_152443``。
        output_root: 汇总包根目录，相对路径以项目根目录为基准。
        splits: 需要尝试的划分，默认 train、valid、test。
        polygon_epsilon_px: 预测掩码转多边形的简化容差，单位像素。
        make_zip: 为 True 时生成 ZIP。

    Returns:
        汇总包目录、ZIP、实际划分、跳过划分和文件数。没有可用划分时，
        目录和 ZIP 为 None，不会创建空包。

    Raises:
        FileNotFoundError: 实验目录不存在。
        ValueError: 划分名称或掩码文件无效。
        OSError: 无法创建目录或写入文件。
    """
    experiment_dir = _resolve_under_project(run_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"实验目录不存在 {experiment_dir}")
    sources, skipped = _collect_splits(experiment_dir, splits)
    if not sources:
        print("没有可汇总的划分，未生成汇总包")
        return {
            "bundle_dir": None,
            "archive_path": None,
            "splits": [],
            "skipped": skipped,
            "file_count": 0,
        }
    if skipped:
        print(f"已跳过没有结果的划分：{', '.join(skipped)}")
    output_dir = _resolve_under_project(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_dir / (
        f"mask_{experiment_dir.name}_{_clock(include_date=False)}"
    )
    bundle_dir.mkdir(exist_ok=False)
    try:
        _export_splits(sources, bundle_dir, polygon_epsilon_px)
        _write_single_readme(
            bundle_dir, experiment_dir, sources, polygon_epsilon_px
        )
    except Exception:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise
    archive_path = _zip_bundle(bundle_dir) if make_zip else None
    return {
        "bundle_dir": _project_path(bundle_dir),
        "archive_path": (
            _project_path(archive_path) if archive_path is not None else None
        ),
        "splits": [source.split for source in sources],
        "skipped": skipped,
        "file_count": sum(
            1 for path in bundle_dir.rglob("*") if path.is_file()
        ),
    }


def build_mask_bundle_collection(
    run_dirs: Sequence[str | Path],
    output_root: str | Path = "out/mask_bundle",
    splits: Sequence[str] = _SPLIT_NAMES,
    *,
    polygon_epsilon_px: float = _DEFAULT_EPSILON_PX,
    make_zip: bool = True,
) -> dict[str, object]:
    """把多个实验导出到同一父目录，并写一个跨实验 README。

    每个实验一个子目录，子目录内只有 ``gt``、``pred``；父目录只有
    ``README.md`` 和实验子目录。缺失划分或没有结果的实验会跳过。

    Args:
        run_dirs: 实验目录列表，相对路径以项目根目录为基准。
        output_root: 集合包根目录，相对路径以项目根目录为基准。
        splits: 每个实验需要尝试的划分。
        polygon_epsilon_px: 预测掩码转多边形的简化容差，单位像素。
        make_zip: 为 True 时生成集合级 ZIP。

    Returns:
        集合目录、ZIP、实验数、各实验信息和跳过项。没有可用实验时，
        目录和 ZIP 为 None。

    Raises:
        ValueError: 实验列表为空。
        OSError: 无法创建目录或写入文件。
    """
    values = list(run_dirs)
    if not values:
        raise ValueError("至少需要一个实验目录")
    output_dir = _resolve_under_project(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_dir = output_dir / (
        f"mask_col_{_clock(include_date=True)}"
    )
    collection_dir.mkdir(exist_ok=False)
    experiments: list[dict[str, object]] = []
    skipped_experiments: list[dict[str, str]] = []
    try:
        for value in values:
            experiment_dir = _resolve_under_project(value)
            if not experiment_dir.is_dir():
                print(f"跳过实验：目录不存在 {experiment_dir}")
                skipped_experiments.append({
                    "experiment_id": experiment_dir.name,
                    "run_dir": _project_path(experiment_dir),
                    "reason": "目录不存在",
                })
                continue
            sources, skipped = _collect_splits(experiment_dir, splits)
            if not sources:
                print(f"跳过实验：{experiment_dir.name} 没有可汇总的划分")
                skipped_experiments.append({
                    "experiment_id": experiment_dir.name,
                    "run_dir": _project_path(experiment_dir),
                    "reason": "没有可汇总的划分",
                })
                continue
            target_dir = collection_dir / experiment_dir.name
            _export_splits(sources, target_dir, polygon_epsilon_px)
            experiments.append({
                "experiment_id": experiment_dir.name,
                "run_dir": _project_path(experiment_dir),
                "splits": [source.split for source in sources],
                "skipped_splits": skipped,
            })
        if not experiments:
            print("没有可汇总的实验，未生成集合包")
            shutil.rmtree(collection_dir, ignore_errors=True)
            return {
                "bundle_dir": None,
                "archive_path": None,
                "experiment_count": 0,
                "experiments": [],
                "skipped_experiments": skipped_experiments,
                "file_count": 0,
            }
        _write_collection_readme(
            collection_dir, experiments, skipped_experiments,
            polygon_epsilon_px,
        )
    except Exception:
        shutil.rmtree(collection_dir, ignore_errors=True)
        raise
    archive_path = _zip_bundle(collection_dir) if make_zip else None
    return {
        "bundle_dir": _project_path(collection_dir),
        "archive_path": (
            _project_path(archive_path) if archive_path is not None else None
        ),
        "experiment_count": len(experiments),
        "experiments": experiments,
        "skipped_experiments": skipped_experiments,
        "file_count": sum(
            1 for path in collection_dir.rglob("*") if path.is_file()
        ),
    }


def _collect_splits(
    experiment_dir: Path, splits: Sequence[str]
) -> tuple[list[_SplitSource], list[str]]:
    """读取可用的划分，并记录跳过的划分。"""
    sources: list[_SplitSource] = []
    skipped: list[str] = []
    for split in _normalize_splits(splits):
        source = _load_split_source(experiment_dir, split)
        if source is None:
            skipped.append(split)
        else:
            sources.append(source)
    return sources, skipped


def _load_split_source(
    experiment_dir: Path, split: str
) -> _SplitSource | None:
    """读取划分目录和输入清单；缺失时打印并返回 None。"""
    source_dir = _find_split_dir(experiment_dir, split)
    if source_dir is None:
        return None
    manifest_path = source_dir / "metrics" / "evaluation_manifest.json"
    if not manifest_path.is_file():
        print(f"跳过 {split}：缺少 {manifest_path}")
        return None
    manifest = _read_json(manifest_path)
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        print(f"跳过 {split}：{manifest_path} 的 images 无效")
        return None
    return _SplitSource(split, source_dir, images)


def _find_split_dir(experiment_dir: Path, split: str) -> Path | None:
    """定位划分目录；没有结果时打印提示并返回 None。"""
    base = experiment_dir / split
    if not base.is_dir():
        detail = (
            "先运行 experiments/yolo_seg/predict_train_rail_seg.py"
            if split == "train" else "先运行对应的预测入口"
        )
        print(f"跳过 {split}：未发现 {base}，{detail}")
        return None
    candidates = sorted(
        path for path in base.iterdir()
        if path.is_dir()
        and (path / "metrics" / "evaluation_manifest.json").is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates and (base / "masks").is_dir():
        return base
    if not candidates:
        print(f"跳过 {split}：{base} 下没有可用的掩码")
        return None
    latest = candidates[-1]
    print(f"{split} 有多个批次，使用最新 {latest.name}")
    return latest


def _export_splits(
    sources: Sequence[_SplitSource],
    target_dir: Path,
    epsilon: float,
) -> None:
    """把若干划分写到目标目录。"""
    if not 0 < epsilon <= 20:
        raise ValueError("polygon_epsilon_px 必须在 (0, 20] 内")
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        _export_split(source, target_dir / source.split, epsilon)


def _export_split(
    source: _SplitSource, split_dir: Path, epsilon: float
) -> None:
    """逐图导出 GT 和预测的 LabelMe JSON。"""
    split_dir.mkdir()
    for entry in source.images:
        if not isinstance(entry, dict):
            raise ValueError(f"{source.split} 的图片清单项必须为字典")
        image_value = entry.get("image")
        label_value = entry.get("label")
        if not isinstance(image_value, str) or not isinstance(label_value, str):
            raise ValueError(f"{source.split} 的图片清单缺少 image 或 label")
        image_path = _resolve_under_project(image_value)
        label_path = _resolve_under_project(label_value)
        image_name = image_path.name
        stem = image_path.stem
        side = _side_name(image_name)
        masks, scores = _load_masks(
            _mask_path(source.source_dir, side, _mask_stem(image_name))
        )
        height, width = masks.shape[1:]
        polygons = load_yolo_polygons(label_path, (width, height))
        _write_labelme(
            split_dir / "gt" / side / f"{stem}.json",
            image_name, width, height, "gt",
            _gt_shapes(polygons),
        )
        _write_labelme(
            split_dir / "pred" / side / f"{stem}.json",
            image_name, width, height, "pred",
            _prediction_shapes(masks, scores, epsilon),
        )


def _write_single_readme(
    bundle_dir: Path,
    experiment_dir: Path,
    sources: Sequence[_SplitSource],
    epsilon: float,
) -> None:
    """写单实验包的目录、格式和读取说明。"""
    split_text = "、".join(source.split for source in sources)
    lines = [
        "# 实验掩码包",
        "",
        f"- 实验：`{experiment_dir.name}`",
        f"- 来源：`{_project_path(experiment_dir)}`",
        f"- 生成时间：{_now_text()}",
        f"- 实际包含：{split_text}",
        f"- 预测多边形简化容差：{epsilon} 像素",
        "",
        "本包只包含标注掩码（gt）和模型预测掩码（pred），格式为 LabelMe",
        "JSON。不含原图、CSV、指标、manifest 或训练记录。",
        "",
        "## 目录",
        "",
        "```text",
        f"{bundle_dir.name}/",
        "├── README.md",
        "├── <split>/                实际存在的划分",
        "│   ├── gt/north/<图片>.json",
        "│   ├── gt/south/<图片>.json",
        "│   ├── pred/north/<图片>.json",
        "│   └── pred/south/<图片>.json",
        "```",
        "",
        "`gt` 是标注，`pred` 是模型预测。`north`、`south` 是相机侧别，",
        "取自原图文件名前缀。文件名保留原图名，只把后缀改为 `.json`。",
        "",
        "## LabelMe 字段",
        "",
        "| 字段 | 含义 |",
        "| --- | --- |",
        "| `version` | `5.0.1` |",
        "| `imagePath` | 原图文件名，不含图片数据 |",
        "| `imageWidth`、`imageHeight` | 原图像素尺寸 |",
        "| `source` | `gt` 或 `pred` |",
        "| `shapes[].label` | 固定为 `rail` |",
        "| `shapes[].points` | 多边形顶点，原图像素坐标 `[x, y]` |",
        "| `shapes[].group_id` | 实例编号；预测实例的多个连通段共享同一值 |",
        "| `shapes[].score` | 仅预测有，模型置信度 |",
        "| `shapes[].description` | `instance_id`、来源和分数 |",
        "| `shapes[].shape_type` | 固定为 `polygon` |",
        "| `imageData` | 固定为 `null` |",
        "",
        "GT 顶点直接来自 YOLO 标签，不做简化。预测顶点由掩码外轮廓经",
        "`approxPolyDP` 简化，容差见导出参数；同一预测实例的多个独立",
        "连通段会写成多个形状并共享 `group_id`。外轮廓方式不保留孔洞。",
        "",
        "## 如何还原掩码",
        "",
        "```python",
        "import json",
        "import cv2",
        "import numpy as np",
        "",
        "path = 'valid/gt/north/north_20260911224626.json'",
        "with open(path, encoding='utf-8') as stream:",
        "    data = json.load(stream)",
        "mask = np.zeros((data['imageHeight'], data['imageWidth']), np.uint8)",
        "for shape in data['shapes']:",
        "    if shape['group_id'] != 1:",
        "        continue",
        "    points = np.rint(shape['points']).astype(np.int32)",
        "    cv2.fillPoly(mask, [points], 1)",
        "```",
        "",
        "## 注意",
        "",
        "- 包内没有原图和 imageData，`imagePath` 只用于对应原始图片。",
        "- 预测保留模型输出中的全部实例，`score` 可用于自行筛选。",
        "- 预测多边形是近似表示，极细结构和孔洞可能丢失；需要精确",
        "  像素统计时请回到原实验的 npz 或指标文件。",
        "- 没有结果的划分不会出现在包内，也不会单独建空目录。",
    ]
    (bundle_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _write_collection_readme(
    collection_dir: Path,
    experiments: Sequence[dict[str, object]],
    skipped_experiments: Sequence[dict[str, str]],
    epsilon: float,
) -> None:
    """写多实验集合包的实验清单和目录说明。"""
    lines = [
        "# 多实验掩码包",
        "",
        f"- 实验数：{len(experiments)}",
        f"- 生成时间：{_now_text()}",
        f"- 预测多边形简化容差：{epsilon} 像素",
        "",
        "每个实验一个子目录，子目录内只有 `gt`、`pred` 和该实验实际",
        "存在的划分；父目录只有本 README 和各实验子目录。",
        "",
        "## 实验",
        "",
        "| 实验 | 来源 | 包含划分 | 跳过划分 |",
        "| --- | --- | --- | --- |",
    ]
    for item in experiments:
        lines.append(
            f"| {item['experiment_id']} | `{item['run_dir']}` | "
            f"{_split_text(item.get('splits'))} | "
            f"{_split_text(item.get('skipped_splits'))} |"
        )
    if skipped_experiments:
        lines.extend(("", "## 跳过的实验", ""))
        for item in skipped_experiments:
            lines.append(
                f"- `{item['experiment_id']}`：{item['reason']}，"
                f"{item['run_dir']}"
            )
    lines.extend((
        "",
        "## 目录",
        "",
        "```text",
        f"{collection_dir.name}/",
        "├── README.md",
        "└── <实验ID>/",
        "    ├── <split>/gt/north|south/<图片>.json",
        "    └── <split>/pred/north|south/<图片>.json",
        "```",
        "",
        "各实验子目录没有单独的 README，字段和还原方式与单实验包相同：",
        "`gt` 是标注，`pred` 是预测，`source` 标明来源，`group_id` 标识",
        "实例，`score` 仅预测有。预测多边形由掩码外轮廓简化得到，不保留",
        "孔洞；需要精确像素统计时回到原实验的 npz 或指标文件。",
    ))
    (collection_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _write_labelme(
    path: Path,
    image_name: str,
    width: int,
    height: int,
    source: str,
    shapes: list[dict[str, object]],
) -> None:
    """写出不含 imageData 的 LabelMe JSON。"""
    payload = {
        "version": _LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "bundleVersion": BUNDLE_VERSION,
        "source": source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _gt_shapes(
    polygons: Sequence[NDArray[np.float64]]
) -> list[dict[str, object]]:
    """把 YOLO 多边形转成 LabelMe 形状，保留原始顶点。"""
    shapes: list[dict[str, object]] = []
    for instance_id, polygon in enumerate(polygons, start=1):
        points = [
            [round(float(x), 2), round(float(y), 2)] for x, y in polygon
        ]
        shapes.append(_labelme_shape(
            "rail", points, instance_id,
            f"instance_id={instance_id};source=gt",
        ))
    return shapes


def _prediction_shapes(
    masks: NDArray[np.bool_],
    scores: NDArray[np.float32],
    epsilon: float,
) -> list[dict[str, object]]:
    """把预测掩码转成 LabelMe 多边形，同一实例的多段共享 group_id。"""
    shapes: list[dict[str, object]] = []
    for instance_id in range(len(masks)):
        contours, _ = cv2.findContours(
            masks[instance_id].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            if len(contour) < 3:
                continue
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue
            points = [
                [int(x), int(y)] for x, y in approx.reshape(-1, 2)
            ]
            score = float(scores[instance_id])
            shapes.append(_labelme_shape(
                "rail", points, instance_id + 1,
                f"instance_id={instance_id + 1};source=pred;"
                f"score={score:.6f}",
                score,
            ))
    return shapes


def _labelme_shape(
    label: str,
    points: list[list[float]],
    group_id: int,
    description: str,
    score: float | None = None,
) -> dict[str, object]:
    """构造单个 LabelMe 形状。"""
    shape: dict[str, object] = {
        "label": label,
        "points": points,
        "group_id": group_id,
        "description": description,
        "shape_type": "polygon",
        "flags": {},
    }
    if score is not None:
        shape["score"] = round(score, 6)
    return shape


def _load_masks(path: Path) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
    """读取 npz 中的预测掩码和分数，保留全部实例。"""
    with np.load(path) as data:
        masks = data["masks"]
        scores = data["scores"]
    if masks.ndim != 3 or masks.dtype != np.bool_:
        raise ValueError(f"{path} 的 masks 必须为 (N,H,W) 布尔数组")
    if scores.shape != (len(masks),) or not np.isfinite(scores).all():
        raise ValueError(f"{path} 的 scores 必须为有限一维数组")
    return masks, scores.astype(np.float32)


def _mask_path(source_dir: Path, side: str, stem: str) -> Path:
    """兼容按南北分组和直接放在 masks 根目录的 npz。"""
    candidates = (
        source_dir / "masks" / side / f"{stem}.npz",
        source_dir / "masks" / f"{stem}.npz",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"缺少预测掩码 {candidates[0]}")


def _zip_bundle(bundle_dir: Path) -> Path:
    """在汇总包旁创建 ZIP，保留顶层目录名。"""
    archive_path = bundle_dir.parent / f"{bundle_dir.name}.zip"
    if archive_path.exists():
        raise FileExistsError(f"汇总压缩包已存在 {archive_path}")
    pending = archive_path.with_suffix(".zip.part")
    try:
        with ZipFile(pending, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(bundle_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(bundle_dir).as_posix()
                archive.write(path, f"{bundle_dir.name}/{relative}")
        with ZipFile(pending) as archive:
            bad_name = archive.testzip()
            if bad_name is not None:
                raise OSError(f"汇总压缩包校验失败 {bad_name}")
        pending.rename(archive_path)
    finally:
        pending.unlink(missing_ok=True)
    return archive_path


def _read_json(path: Path) -> dict[str, object]:
    """读取 JSON 对象。"""
    loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON 顶层必须为对象 {path}")
    return loaded


def _normalize_splits(splits: Sequence[str]) -> tuple[str, ...]:
    """检查划分名称并保持顺序。"""
    normalized = tuple(dict.fromkeys(splits))
    unknown = [split for split in normalized if split not in _SPLIT_NAMES]
    if unknown:
        raise ValueError(f"不支持的划分 {unknown}，可选 {_SPLIT_NAMES}")
    if not normalized:
        raise ValueError("至少需要一个划分")
    return normalized


def _side_name(image_name: str) -> str:
    """从文件名前缀识别南北相机。"""
    side, _, rest = Path(image_name).stem.partition("_")
    return side if side in _SIDE_NAMES and rest else "unknown"


def _mask_stem(image_name: str) -> str:
    """返回掩码归档使用的文件名，去掉 north_ 或 south_ 前缀。"""
    side, _, rest = Path(image_name).stem.partition("_")
    return rest if side in _SIDE_NAMES and rest else Path(image_name).stem


def _split_text(value: object) -> str:
    """把划分列表写成可读文本。"""
    if isinstance(value, list):
        return "、".join(str(item) for item in value) or "无"
    return "无"


def _resolve_under_project(value: str | Path) -> Path:
    """把相对路径解析到项目根目录，拒绝项目外绝对路径。"""
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (
        PROJECT_ROOT / path
    ).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"路径必须位于项目内 {value}")
    return resolved


def _project_path(path: Path) -> str:
    """优先记录项目相对路径。"""
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(resolved)


def _clock(include_date: bool) -> str:
    """生成短时间戳；单实验用时分秒，集合包带日期。"""
    pattern = "%Y%m%d_%H%M%S" if include_date else "%H%M%S"
    return datetime.now().strftime(pattern)


def _now_text() -> str:
    """返回带时区的可读时间。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")
