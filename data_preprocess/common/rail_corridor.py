"""从钢轨像素并集生成逐行有效区。"""

from math import ceil, floor

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from data_preprocess.common.roi_config import CorridorSettings


def make_corridor_mask(
    union: NDArray[np.uint8],
    settings: CorridorSettings,
    margins: tuple[int, int] | None = None,
) -> NDArray[np.uint8]:
    """计算逐行左右边界，处理短空缺并加入背景余量。

    Args:
        union: 原图尺寸 (H, W) 的钢轨像素并集，像素值为 0 或 1。
        settings: 有效区域计算参数。
        margins: 左右外扩的原图像素数，省略时使用通用边距。

    Returns:
        原图尺寸的 0/1 有效区，保留所有输入钢轨像素。
    """
    height, width = union.shape
    observed_rows = np.flatnonzero(union.any(axis=1))
    valid_mask = np.zeros_like(union)
    if not observed_rows.size:
        return valid_mask

    observed = union[observed_rows]
    raw_left = observed.argmax(axis=1)
    raw_right = width - 1 - observed[:, ::-1].argmax(axis=1)
    edges = np.full((height, 2), -1.0)
    edges[observed_rows, 0] = raw_left
    edges[observed_rows, 1] = raw_right

    row_distances = np.diff(observed_rows)
    gaps = np.flatnonzero(
        (row_distances > 1) & (row_distances <= settings.max_interp_gap + 1)
    )
    for gap_index in gaps:
        start, end = observed_rows[gap_index : gap_index + 2]
        edges[start + 1 : end] = np.linspace(edges[start], edges[end], end - start + 1)[
            1:-1
        ]

    valid_rows = np.flatnonzero(edges[:, 0] >= 0)
    if settings.smoothing_kernel > 1:
        _smooth_edges(edges, valid_rows, settings.smoothing_kernel)
        edges[observed_rows, 0] = np.minimum(edges[observed_rows, 0], raw_left)
        edges[observed_rows, 1] = np.maximum(edges[observed_rows, 1], raw_right)

    left_margin, right_margin = settings.margins if margins is None else margins
    for row in valid_rows:
        left = max(0, floor(edges[row, 0]) - left_margin)
        right = min(width - 1, ceil(edges[row, 1]) + right_margin)
        valid_mask[row, left : right + 1] = 1
    return valid_mask


def _smooth_edges(
    edges: NDArray[np.float64], valid_rows: NDArray[np.intp], kernel: int
) -> None:
    """分别平滑每段连续有效行，避免跨越长空缺。"""
    breaks = np.flatnonzero(np.diff(valid_rows) > 1) + 1
    half = kernel // 2
    for rows in np.split(valid_rows, breaks):
        padded = np.pad(edges[rows], ((half, half), (0, 0)), mode="edge")
        windows = sliding_window_view(padded, kernel, axis=0)
        edges[rows] = np.median(windows, axis=-1)
