"""在同一画布预览南北两侧的钢轨实例 Mask 叠加图。"""

import argparse
from colorsys import hsv_to_rgb
from pathlib import Path
import sys
from typing import TypedDict

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from data_preprocess.common.config import PROJECT_ROOT, load_dataset_relative_path
from data_preprocess.common.image import pad_to_size, review_rgb_image
from data_preprocess.common.labelme import load_rail_image_and_mask
from data_preprocess.match_pair_image import match_pair_image
from utils.image_layout import compose_image_pair

_MASK_HUE_STEP = 0.61803398875
_MASK_SATURATION = 0.8
_MASK_BRIGHTNESS = 0.85
_MASK_OPACITY = 0.45
_PREVIEW_WINDOW_TITLE = "Rail Masks (South | North)"


class RailPairData(TypedDict):
    """南北 RGB 图像和对应的单通道实例 Mask。"""

    south_image: NDArray[np.uint8]
    south_mask: NDArray[np.int32]
    north_image: NDArray[np.uint8]
    north_mask: NDArray[np.int32]


def view_image_mask(
    south_image_name: str, north_image_name: str, should_show: bool = False
) -> RailPairData:
    """读取并对齐南北图片与 Mask，按需显示叠加预览。

    Args:
        south_image_name: south 目录中的图片名，可省略 .png 后缀。
        north_image_name: north 目录中的图片名，可省略 .png 后缀。
        should_show: 为 True 时在 OpenCV 最大化预览左南右北的 Mask 叠加图，
            两图上下居中。按空格、Esc 或关闭窗口后返回。

    Returns:
        包含 south_image、south_mask、north_image、north_mask 的字典。
        图片为 RGB uint8 数组 (H, W, 3)，Mask 为 int32 数组 (H, W)，
        正整数表示实例 ID，0 表示背景。H 和 W 分别取南北最大高和宽。
        返回的图像和 Mask 在右侧及底部补零，保留原图坐标。

    Raises:
        FileNotFoundError: 缺少配置、所选图片或其同名标注。
        ValueError: 数据集配置无效、图片名不合法或标注数据无效。
    """
    pair = _load_pair(south_image_name, north_image_name)
    if should_show:
        _show_pair(pair)

    south_height, south_width = map(int, pair["south_image"].shape[:2])
    north_height, north_width = map(int, pair["north_image"].shape[:2])
    height = max(south_height, north_height)
    width = max(south_width, north_width)
    return {
        "south_image": pad_to_size(pair["south_image"], height, width),
        "south_mask": pad_to_size(pair["south_mask"], height, width),
        "north_image": pad_to_size(pair["north_image"], height, width),
        "north_mask": pad_to_size(pair["north_mask"], height, width),
    }


def _load_pair(south_image_name: str, north_image_name: str) -> RailPairData:
    """读取两侧原图及其原始尺寸的 Mask，供显示和数据返回复用。"""
    dataset_root = PROJECT_ROOT / load_dataset_relative_path()
    south_image, south_mask = load_rail_image_and_mask(
        _image_path(dataset_root / "south", south_image_name)
    )
    north_image, north_mask = load_rail_image_and_mask(
        _image_path(dataset_root / "north", north_image_name)
    )
    return {
        "south_image": south_image,
        "south_mask": south_mask,
        "north_image": north_image,
        "north_mask": north_mask,
    }


def _image_path(camera_dir: Path, image_name: str) -> Path:
    """在指定相机目录中定位图片，允许省略 PNG 后缀。"""
    filename = Path(image_name)
    if not image_name or filename.name != image_name or image_name == "..":
        raise ValueError(f"请输入单独的图片名称，收到 {image_name!r}")
    if not filename.suffix:
        filename = filename.with_suffix(".png")
    return camera_dir / filename


def _show_pair(
    result: RailPairData, window_title: str = _PREVIEW_WINDOW_TITLE
) -> bool:
    """最大化显示南北叠加图，返回是否继续预览。"""
    south_panel = _overlay_mask(result["south_image"], result["south_mask"])
    north_panel = _overlay_mask(result["north_image"], result["north_mask"])
    canvas = compose_image_pair(np.asarray(south_panel), np.asarray(north_panel))
    return review_rgb_image(canvas, window_title, should_maximize=True)


def _overlay_mask(
    image_data: NDArray[np.uint8], mask: NDArray[np.int32]
) -> Image.Image:
    """仅在实例像素上叠加颜色，保留背景和补边。"""
    max_instance_id = int(mask.max())
    palette = np.zeros((max_instance_id + 1, 4), dtype=np.uint8)
    palette[1:, 3] = round(_MASK_OPACITY * 255)
    # 让相邻实例 ID 的色相分散，便于区分紧邻的钢轨。
    for instance_id in range(1, max_instance_id + 1):
        hue = ((instance_id - 1) * _MASK_HUE_STEP) % 1.0
        color = hsv_to_rgb(hue, _MASK_SATURATION, _MASK_BRIGHTNESS)
        palette[instance_id, :3] = [round(channel * 255) for channel in color]

    preview = Image.fromarray(image_data)
    overlay = Image.fromarray(palette[mask])
    preview.paste(overlay, mask=overlay)
    return preview


def _preview_matched_pairs(appoint_image: str = "") -> None:
    """按采集时间预览配对，按空格或关闭窗口继续。"""
    pairs = match_pair_image(load_dataset_relative_path())
    pair_count = len(pairs["south"])
    if not pair_count:
        print("未找到可显示的南北配对。")
        return

    print(f"共 {pair_count} 对图片。关闭窗口或按空格查看下一对，按 Esc 退出。")
    # `--image` 允许带或不带扩展名，统一按文件名主干比较。
    wanted = Path(appoint_image).stem if appoint_image else ""
    image_pairs = zip(pairs["south"], pairs["north"], strict=True)
    for pair_index, (south_name, north_name) in enumerate(image_pairs, start=1):
        if wanted and north_name != wanted:
            continue
        title = (
            f"{_PREVIEW_WINDOW_TITLE} {pair_index}/{pair_count}"
            f" | {south_name} | {north_name}"
        )
        if not _show_pair(_load_pair(south_name, north_name), title):
            break


def main() -> None:
    """命令行入口：可选只预览某一对。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", help="只预览北侧文件名为该值的图对（可省略 .png）",
    )
    args = parser.parse_args()
    _preview_matched_pairs(args.image or "")


if __name__ == "__main__":
    main()
