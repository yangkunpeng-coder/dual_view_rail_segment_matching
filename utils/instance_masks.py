"""在原图坐标下读取 YOLO 实例掩码和 Polygon 标注。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.engine.results import Masks


def as_mask_stack(
    masks: np.ndarray | Tensor | Masks | None, shape: tuple[int, int]
) -> NDArray[np.bool_]:
    """将 YOLO 或原图掩码统一为原图尺寸的布尔数组。

    Args:
        masks: YOLO Results.masks、(N,H,W) 或 (H,W) 的 NumPy 数组或 Tensor。
            None 表示无检测实例。数组和 Tensor 须使用原图坐标。
        shape: 原图的 (高度, 宽度)。

    Returns:
        (N,H,W) 布尔掩码，保留各实例的重叠区域。

    Raises:
        ValueError: 掩码数据类型或记录的图像尺寸无效。
    """
    if masks is None:
        return np.zeros((0, *shape), dtype=np.bool_)

    if isinstance(masks, np.ndarray):
        array = masks
    else:
        from torch import Tensor

        data: np.ndarray | Tensor
        if isinstance(masks, Tensor):
            data = masks
        else:
            if tuple(masks.orig_shape) != shape:
                raise ValueError("YOLO 掩码记录的原图尺寸与输入图片不一致")
            data = masks.data
            if not len(data):
                return np.zeros((0, *shape), dtype=np.bool_)
            if data.shape[-2:] != shape:
                # 去除默认推理掩码的 letterbox 补边后恢复原图尺寸。
                from torch import as_tensor
                from ultralytics.utils.ops import scale_masks

                data = scale_masks(as_tensor(data)[None], shape, mode="nearest")[0] > 0.5
        array = data if isinstance(data, np.ndarray) else data.detach().cpu().numpy()
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[1:] != shape:
        raise ValueError(
            f"掩码必须为原图尺寸 (N, {shape[0]}, {shape[1]})，"
            f"实际为 {array.shape}。也可直接传入 YOLO Results.masks。"
        )
    if array.dtype.kind not in "buif":
        raise ValueError("实例掩码必须使用布尔或实数像素值")
    if array.dtype != np.bool_:
        array = np.greater(array, 0.5)
    return array.astype(np.bool_, copy=False)


def load_yolo_polygons(
    label_path: Path, image_size: tuple[int, int]
) -> list[NDArray[np.float64]]:
    """读取 YOLO 分割 TXT 并还原各实例的原图坐标。

    Args:
        label_path: YOLO 分割标签路径。空文件表示没有标注实例。
        image_size: 图片的 (宽度, 高度)，单位为像素。

    Returns:
        按标签行排列的多边形，每项形状为 (顶点数, 2)，列为 x、y。

    Raises:
        OSError: 无法读取标签文件。
        ValueError: 类别或归一化 Polygon 坐标无效。
    """
    polygons: list[NDArray[np.float64]] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        fields = line.split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
            coordinates = np.asarray(fields[1:], dtype=np.float64).reshape(-1, 2)
        except ValueError as error:
            location = f"{label_path} 第 {line_number} 行"
            raise ValueError(f"YOLO Polygon 无效 {location}") from error
        if (
            class_id < 0 or len(coordinates) < 3
            or not np.isfinite(coordinates).all()
            or np.any((coordinates < 0) | (coordinates > 1))
        ):
            raise ValueError(f"YOLO Polygon 无效 {label_path} 第 {line_number} 行")
        coordinates *= image_size
        polygons.append(coordinates)
    return polygons
