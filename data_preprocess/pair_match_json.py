"""导出南北成对且钢轨 Mask 非空的 LabelMe 标注。"""

import json
from os.path import relpath
from pathlib import Path
from shutil import make_archive, rmtree
import sys

import cv2
import numpy as np
from PIL import Image

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preprocess.common.config import PROJECT_ROOT, load_dataset_relative_path
from data_preprocess.common.labelme import (
    extract_rail_polygons,
    load_labelme_annotation,
)
from data_preprocess.match_pair_image import match_pair_image

_EXPORT_NAME = "pair_match_json"
_README_CONTENT = """# 南北成对钢轨 LabelMe 标注

图对沿用项目的时间配对规则。两侧图片同秒或相差一秒，且双方互为唯一候选。
每对图的南北侧都须有 LabelMe 标注，并且至少有一个钢轨 Polygon 在原图内
形成前景像素。任一侧缺少 JSON 或 Mask 为空时，整对跳过。

`Pxxxx` 按筛选前的图对顺序从 `P0001` 编号，与评测工具默认配对的编号对应。
筛选后编号可能跳号。各目录中的南北 JSON 以原文件名加视角后缀区分。

```text
pair_match_json/
├── README.md
└── P0004/
    ├── <南侧原文件名>_south.json
    └── <北侧原文件名>_north.json
```

JSON 保留原始形状、标注尺寸和其他元信息。`imageData` 为 `null`，
`imagePath` 从该 JSON 所在目录相对指向原图。将 ZIP 解压到项目的
`data/` 下可保留原图引用。

每次导出都会重建此目录，并重新生成 `data/pair_match_json.zip`。
"""


def _reset_output(output_dir: Path, archive_path: Path) -> None:
    """核对固定输出位置后清空旧产物。"""
    if (
        output_dir != PROJECT_ROOT / "data" / _EXPORT_NAME
        or archive_path != output_dir.with_suffix(".zip")
        or output_dir.resolve() != output_dir
        or archive_path.resolve() != archive_path
    ):
        raise ValueError(f"标注导出路径指向意外位置 {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"标注导出目录被文件占用 {output_dir}")
    if archive_path.exists() and not archive_path.is_file():
        raise ValueError(f"ZIP 路径被目录或其他对象占用 {archive_path}")

    if output_dir.exists():
        rmtree(output_dir)
    archive_path.unlink(missing_ok=True)
    output_dir.mkdir(parents=True)


def _load_nonempty_annotation(image_path: Path) -> dict[str, object] | None:
    """仅加载有钢轨前景像素的标注，原图只读取尺寸。"""
    annotation_path = image_path.with_suffix(".json")
    if not annotation_path.is_file():
        return None
    with Image.open(image_path) as image:
        width, height = image.size

    annotation = load_labelme_annotation(annotation_path, (width, height))
    polygons = extract_rail_polygons(annotation, annotation_path)
    if not polygons:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        vertices = np.rint(polygon["points"]).astype(np.int32)
        cv2.fillPoly(mask, [vertices], 1)
    return annotation if mask.any() else None


def _write_annotation(
    image_path: Path, annotation: dict[str, object], target: Path
) -> None:
    """写入仅通过路径引用原图的 LabelMe JSON。"""
    annotation["imageData"] = None
    annotation["imagePath"] = Path(relpath(image_path, start=target.parent)).as_posix()
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(annotation, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def match_pair_json(dataset_relative_path: str | Path | None = None) -> Path:
    """重建非空南北图对的标注目录和 ZIP。

    Args:
        dataset_relative_path: 原始数据集相对于项目根目录的路径。
            省略时读取 config/data/rail_data.yaml。

    Returns:
        data/pair_match_json.zip 的绝对路径。
    Raises:
        OSError: 无法读取输入或重建导出文件。
        ValueError: 数据集路径、输出位置或 LabelMe 标注无效。
    """
    relative_path = (
        load_dataset_relative_path()
        if dataset_relative_path is None
        else Path(dataset_relative_path)
    )
    dataset_root = (PROJECT_ROOT / relative_path).resolve()
    if relative_path.anchor or not dataset_root.is_relative_to(PROJECT_ROOT):
        raise ValueError("原始数据集必须使用项目内的相对路径")

    output_dir = PROJECT_ROOT / "data" / _EXPORT_NAME
    archive_path = output_dir.with_suffix(".zip")
    if any(
        (dataset_root / side).resolve().is_relative_to(output_dir)
        for side in ("south", "north")
    ):
        raise ValueError(f"原始数据不能位于标注导出目录中 {output_dir}")

    pairs = match_pair_image(dataset_root.relative_to(PROJECT_ROOT))
    _reset_output(output_dir, archive_path)
    (output_dir / "README.md").write_text(
        _README_CONTENT, encoding="utf-8", newline="\n"
    )

    names = zip(pairs["south"], pairs["north"], strict=True)
    for pair_index, (south_name, north_name) in enumerate(names, start=1):
        south_image = dataset_root / "south" / f"{south_name}.png"
        south_annotation = _load_nonempty_annotation(south_image)
        if south_annotation is None:
            continue
        north_image = dataset_root / "north" / f"{north_name}.png"
        north_annotation = _load_nonempty_annotation(north_image)
        if north_annotation is None:
            continue

        pair_dir = output_dir / f"P{pair_index:04d}"
        pair_dir.mkdir()
        _write_annotation(
            south_image, south_annotation, pair_dir / f"{south_name}_south.json"
        )
        _write_annotation(
            north_image, north_annotation, pair_dir / f"{north_name}_north.json"
        )

    return Path(
        make_archive(
            str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
        )
    )


if __name__ == "__main__":
    print(f"已生成 {match_pair_json()}")
