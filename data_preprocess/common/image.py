"""提供数据预处理共用的补边和图像预览。"""

from os import getpid
import sys
from threading import get_ident
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

_ArrayDType = TypeVar("_ArrayDType", bound=np.generic)
_PREVIEW_MAX_WIDTH = 1280
_PREVIEW_POLL_MS = 50
_ESCAPE_KEY = 27
_NEXT_IMAGE_KEY = ord(" ")
_SW_MAXIMIZE = 3


def pad_to_size(
    array: NDArray[_ArrayDType], target_height: int, target_width: int
) -> NDArray[_ArrayDType]:
    """在右侧和底部补零，保留原始像素坐标。

    Args:
        array: 前两个维度为高度和宽度的图像或 Mask 数组。
        target_height: 补边后的高度，单位为像素。
        target_width: 补边后的宽度，单位为像素。

    Returns:
        保留原数据类型和通道的数组。尺寸相同时返回原数组。

    Raises:
        ValueError: 目标高度或宽度小于输入尺寸。
    """
    height, width = array.shape[:2]
    if target_height < height or target_width < width:
        raise ValueError("目标高度和宽度必须至少等于输入尺寸")
    if (height, width) == (target_height, target_width):
        return array

    padded = np.zeros(
        (target_height, target_width, *array.shape[2:]), dtype=array.dtype
    )
    padded[:height, :width] = array
    return padded


def show_rgb_image(image_data: NDArray[np.uint8], window_title: str) -> None:
    """在可缩放的 OpenCV 窗口显示 RGB 图像，关闭后返回。

    Args:
        image_data: 形状为 (H, W, 3) 的 uint8 RGB 图像。
        window_title: OpenCV 窗口标题。
    """
    review_rgb_image(image_data, window_title)


def review_rgb_image(
    image_data: NDArray[np.uint8],
    window_title: str,
    *,
    use_fullscreen: bool = False,
    should_maximize: bool = False,
    background_value: int = 0,
    should_align_top: bool = False,
) -> bool:
    """显示一张检查图，并区分关闭窗口和按 Esc。

    Args:
        image_data: 形状为 (H, W, 3) 的 uint8 RGB 图像。
        window_title: OpenCV 窗口标题。
        use_fullscreen: 为 True 时全屏等比显示，并允许按空格继续。
        should_maximize: 为 True 时优先在 Windows 桌面工作区内最大化。
        background_value: 等比显示时补边的灰度值，取值范围为 0 至 255。
        should_align_top: 为 True 时将等比画布放在窗口内容区上缘。

    Returns:
        关闭窗口或在最大化、全屏预览中按空格返回 True，按 Esc 返回 False。
    """
    import cv2

    height, width = image_data.shape[:2]
    preview_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    use_maximized_window = should_maximize and sys.platform == "win32"
    window_name = window_title
    if use_maximized_window:
        # 按当前进程和线程定位预览窗口，用户标题可随图对更新。
        window_name = f"Rail preview {getpid()} {get_ident()}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        if use_maximized_window:
            maximize_window_workspace(window_name)
            cv2.setWindowTitle(window_name, window_title)
        elif use_fullscreen:
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

        if use_maximized_window or use_fullscreen:
            _, _, frame_width, frame_height = cv2.getWindowImageRect(window_name)
            scale = min(frame_width / width, frame_height / height)
            if scale > 0:
                scaled_width = max(1, round(width * scale))
                scaled_height = max(1, round(height * scale))
                scaled_size = scaled_width, scaled_height
                interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                preview_bgr = cv2.resize(
                    preview_bgr, scaled_size, interpolation=interpolation
                )
                # Win32 会拉伸到显示区域，先补边以保留原图比例。
                horizontal_pad = frame_width - scaled_width
                vertical_pad = frame_height - scaled_height
                top_pad = 0 if should_align_top else vertical_pad // 2
                preview_bgr = cv2.copyMakeBorder(
                    preview_bgr,
                    top_pad,
                    vertical_pad - top_pad,
                    horizontal_pad // 2,
                    (horizontal_pad + 1) // 2,
                    cv2.BORDER_CONSTANT,
                    value=(background_value,) * 3,
                )
        else:
            preview_width = min(width, _PREVIEW_MAX_WIDTH)
            preview_height = max(1, round(height * preview_width / width))
            cv2.resizeWindow(window_name, preview_width, preview_height)
        cv2.imshow(window_name, preview_bgr)
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            key = cv2.waitKey(_PREVIEW_POLL_MS)
            if key == _ESCAPE_KEY:
                return False
            if (should_maximize or use_fullscreen) and key == _NEXT_IMAGE_KEY:
                return True
    finally:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow(window_name)

    return True


def maximize_window_workspace(window_name: str) -> None:
    """通过 Windows 最大化 OpenCV 窗口，使用系统分配的桌面工作区。

    Args:
        window_name: 窗口创建时使用的名称。

    Raises:
        RuntimeError: 找不到同名窗口，或平台不是 Windows。
    """
    if sys.platform != "win32":
        raise RuntimeError("只有 Windows 支持最大化 OpenCV 窗口")
    from ctypes import WinDLL, c_int
    from ctypes.wintypes import BOOL, HWND, LPCWSTR

    user32 = WinDLL("user32")
    find_window = user32.FindWindowW
    find_window.argtypes = (LPCWSTR, LPCWSTR)
    find_window.restype = HWND
    handle = find_window(None, window_name)
    if not handle:
        raise RuntimeError(f"找不到 OpenCV 预览窗口 {window_name}")

    show_window = user32.ShowWindow
    show_window.argtypes = (HWND, c_int)
    show_window.restype = BOOL
    show_window(handle, _SW_MAXIMIZE)


def window_work_area() -> tuple[int, int]:
    """返回 Windows 桌面工作区的宽高，不含任务栏。

    Returns:
        (宽度, 高度)。非 Windows 或调用失败时返回常见屏幕尺寸的保守估计。
    """
    fallback = (1600, 900)
    if sys.platform != "win32":
        return fallback
    from ctypes import Structure, WinDLL, byref, c_uint, c_void_p
    from ctypes.wintypes import LONG

    class _WorkArea(Structure):
        _fields_ = [
            ("left", LONG), ("top", LONG), ("right", LONG), ("bottom", LONG),
        ]

    rect = _WorkArea()
    user32 = WinDLL("user32")
    user32.SystemParametersInfoW.argtypes = (c_uint, c_uint, c_void_p, c_uint)
    user32.SystemParametersInfoW.restype = c_uint
    try:
        success = user32.SystemParametersInfoW(0x0030, 0, byref(rect), 0)
    except OSError:
        return fallback
    if not success:
        return fallback
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width < 640 or height < 480:
        return fallback
    return width, height
