"""拼接左右面板并对齐图像的水平中心线。"""

import numpy as np
from numpy.typing import NDArray


def compose_image_pair(
    left_image: NDArray[np.uint8],
    right_image: NDArray[np.uint8],
    *,
    background_value: int = 0,
) -> NDArray[np.uint8]:
    """在等宽面板中左右拼接图像，并对齐两图的中间像素行。

    Args:
        left_image: 形状为 (H, W, 3) 的左侧 uint8 RGB 图像。
        right_image: 形状为 (H, W, 3) 的右侧 uint8 RGB 图像。
        background_value: 空余区域的灰度值，取值范围为 0 至 255。

    Returns:
        上下居中的 RGB 画布。
    """
    left_height, left_width = map(int, left_image.shape[:2])
    right_height, right_width = map(int, right_image.shape[:2])
    canvas_height = max(left_height, right_height)
    panel_width = max(left_width, right_width)
    canvas = np.full(
        (canvas_height, 2 * panel_width, 3), background_value, dtype=np.uint8
    )
    center_row = canvas_height // 2
    for panel_index, image in enumerate((left_image, right_image)):
        height, width = image.shape[:2]
        top = center_row - height // 2
        left = panel_index * panel_width
        canvas[top:top + height, left:left + width] = image
    return canvas
