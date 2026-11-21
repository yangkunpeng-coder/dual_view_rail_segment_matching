"""记录原始数据中空标注的南北图片。"""

import json
from pathlib import Path

from data_preprocess.common.config import PROJECT_ROOT, load_dataset_relative_path

_CAMERA_SIDES = ("south", "north")
_EMPTY_LABEL_DIR = PROJECT_ROOT / "data" / "empty_label"


def check_empty_labels(dataset_relative_path: str | Path) -> None:
    """将空标注图片的项目相对路径分别写入南北清单。

    Args:
        dataset_relative_path: 相对于项目根目录的原始数据集路径。

    Raises:
        FileNotFoundError: 缺少相机目录或图片对应的 JSON。
        ValueError: 数据集路径不是相对路径，或标注 JSON 格式无效。
    """
    relative_path = Path(dataset_relative_path)
    if relative_path.anchor:
        raise ValueError("数据集路径必须相对于项目根目录")

    empty_images = {
        side: _find_empty_images(relative_path / side) for side in _CAMERA_SIDES
    }
    _EMPTY_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    for side, image_paths in empty_images.items():
        destination = _EMPTY_LABEL_DIR / f"{side}.txt"
        destination.write_text(
            "".join(f"{path}\n" for path in image_paths),
            encoding="utf-8",
            newline="\n",
        )


def _find_empty_images(camera_relative_path: Path) -> list[str]:
    """筛选原始 LabelMe shapes 为空列表的 PNG。"""
    empty_paths: list[str] = []
    image_paths = list((PROJECT_ROOT / camera_relative_path).iterdir())
    image_paths.sort()
    for image_path in image_paths:
        if image_path.suffix.lower() != ".png" or not image_path.is_file():
            continue

        label_path = image_path.with_suffix(".json")
        with label_path.open(encoding="utf-8-sig") as label_file:
            try:
                annotation = json.load(label_file)
            except json.JSONDecodeError as error:
                raise ValueError(f"无法解析标注文件 {label_path}") from error

        shapes = annotation.get("shapes") if isinstance(annotation, dict) else None
        if not isinstance(shapes, list):
            raise ValueError(f"{label_path} 的 shapes 必须为列表")
        if not shapes:
            empty_paths.append((camera_relative_path / image_path.name).as_posix())

    return empty_paths


if __name__ == "__main__":
    check_empty_labels(load_dataset_relative_path())
