"""按采集时间配对钢轨数据集的南北图片。"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

from data_preprocess.common.config import PROJECT_ROOT, load_dataset_relative_path

_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
_TIMESTAMP_NAME_LENGTH = 14
_TIMESTAMP_EPOCH = datetime(1970, 1, 1)
_ONE_SECOND = timedelta(seconds=1)
_PAIR_OFFSETS_SECONDS = (-1, 0, 1)
_DEFAULT_IMAGE_EXTENSIONS = (".png",)


class PairedImageNames(TypedDict):
    """按索引一一对应的南北图片文件名。"""

    south: list[str]
    north: list[str]


def match_pair_image(
    dataset_relative_path: str | Path,
    *,
    image_extensions: tuple[str, ...] = _DEFAULT_IMAGE_EXTENSIONS,
) -> PairedImageNames:
    """按文件名时间配对南北图片，保留双方互为唯一候选的图片。

    Args:
        dataset_relative_path: 相对于项目根目录的数据集路径，包含 south
            和 north 目录。图片文件名为 YYYYMMDDHHMMSS 格式。
        image_extensions: 扫描的图片扩展名，默认只读取 PNG。

    Returns:
        以 south 和 north 为键的字典，两个列表按采集时间排列，
        相同索引处是一对图片。文件名不包含 PNG 后缀。

    Raises:
        FileNotFoundError: south 或 north 目录不存在。
        ValueError: 数据集路径不是相对路径，或图片文件名时间无效。
    """
    relative_path = Path(dataset_relative_path)
    if relative_path.anchor:
        raise ValueError("数据集路径必须相对于项目根目录")

    dataset_root = PROJECT_ROOT / relative_path
    south_images = _read_capture_times(dataset_root / "south", image_extensions)
    north_images = _read_capture_times(dataset_root / "north", image_extensions)
    pairs: PairedImageNames = {"south": [], "north": []}

    for south_second, south_name in south_images.items():
        north_candidates = _find_nearby_seconds(south_second, north_images)
        if len(north_candidates) != 1:
            continue

        north_second = north_candidates[0]
        if len(_find_nearby_seconds(north_second, south_images)) == 1:
            pairs["south"].append(south_name)
            pairs["north"].append(north_images[north_second])

    return pairs


def _read_capture_times(
    camera_dir: Path, image_extensions: tuple[str, ...]
) -> dict[int, str]:
    """按时间顺序读取图片文件名，并用整数秒索引。"""
    images: dict[int, str] = {}
    for image_path in sorted(camera_dir.iterdir()):
        if (
            image_path.suffix.lower() not in image_extensions
            or not image_path.is_file()
        ):
            continue

        name = image_path.stem
        if (
            len(name) != _TIMESTAMP_NAME_LENGTH
            or not name.isascii()
            or not name.isdigit()
        ):
            raise ValueError(f"图片文件名须使用 YYYYMMDDHHMMSS 格式 {image_path}")
        try:
            capture_time = datetime.strptime(name, _TIMESTAMP_FORMAT)
        except ValueError as error:
            raise ValueError(f"图片文件名包含无效的采集时间 {image_path}") from error

        capture_second = (capture_time - _TIMESTAMP_EPOCH) // _ONE_SECOND
        if capture_second in images:
            raise ValueError(f"{camera_dir} 的采集时间 {name} 对应多张图片")
        images[capture_second] = name

    return images


def _find_nearby_seconds(capture_second: int, images: dict[int, str]) -> list[int]:
    """查询同秒或相差一秒的图片。"""
    nearby_seconds = (capture_second + offset for offset in _PAIR_OFFSETS_SECONDS)
    return [second for second in nearby_seconds if second in images]


if __name__ == "__main__":
    dataset_relative_path = load_dataset_relative_path()
    pair = match_pair_image(dataset_relative_path)
    for i in range(len(pair['south'])):
        print('{}, {}'.format(pair['south'][i], pair['north'][i]))
