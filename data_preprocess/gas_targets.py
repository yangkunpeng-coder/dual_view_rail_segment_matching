"""把训练管线里的实例多边形转换成辅助监督目标张量。

本模块只做数据管线适配：按实例紧包围框栅格化、调用几何算法、把目标场打包成 batch 张量，
并把目标生成挂到 ``Format`` 阶段。中心线、方向与宽度的计算都在
``core_algorithm/common/rail_target_fields.py``，这里不重复实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset

from core_algorithm.common.rail_centerline import trace_instance_mask
from core_algorithm.common.rail_target_fields import (
    TargetFieldSpec,
    build_instance_fields,
    compose_fields,
)

GAS_TARGET_KEYS = (
    "gas_center",
    "gas_center_valid",
    "gas_direction",
    "gas_direction_valid",
    "gas_width",
)

_MIN_BOX_SIDE_PX = 2


@dataclass(frozen=True)
class GasTargetSpec:
    """数据管线侧的辅助目标参数。

    Attributes:
        mask_ratio: 目标网格相对输入画布的降采样比例，与数据管线的 mask_ratio 解耦；
            P2 下管线为 2、目标仍取 4，损失侧按面积平均对齐。
        sigma_cells: 中心线热力图的高斯标准差上限，单位是网格单元。
        direction_band_cells: 方向与宽度向两侧传播的带宽，单位是网格单元。
        tangent_window: 切向拟合的窗口长度（样本数）。
        tangent_max_residual_px: 切向拟合中心残差阈值，超过即判为不可估。
    """

    mask_ratio: int = 4
    sigma_cells: float = 0.75
    direction_band_cells: float = 2.0
    tangent_window: int = 5
    tangent_max_residual_px: float = 1.5

    def to_field_spec(self) -> TargetFieldSpec:
        """返回算法层使用的目标场参数。"""
        return TargetFieldSpec(
            grid_ratio=self.mask_ratio,
            sigma_cells=self.sigma_cells,
            direction_band_cells=self.direction_band_cells,
            tangent_window=self.tangent_window,
            tangent_max_residual_px=self.tangent_max_residual_px,
        )


def _instance_region(
    points: NDArray[np.float64], grid_shape: tuple[int, int], ratio: int
) -> NDArray[np.bool_]:
    """把实例多边形栅格化到网格上，返回该实例的前景区域。"""
    height, width = grid_shape
    region = np.zeros((height, width), dtype=np.uint8)
    scaled = np.round(points / ratio).astype(np.int32)
    cv2.fillPoly(region, [scaled], 1)
    return region.astype(bool)


def build_gas_targets(
    segments: list[NDArray[np.float32]],
    height: int,
    width: int,
    spec: GasTargetSpec,
) -> dict[str, torch.Tensor]:
    """生成一张训练图的五个辅助目标张量。

    Args:
        segments: 每个实例在画布像素坐标下的多边形，形状 (N_i, 2)。
        height: 增强并补边后的画布高度（像素）。
        width: 增强并补边后的画布宽度（像素）。
        spec: 生成参数。

    Returns:
        键见 ``GAS_TARGET_KEYS``，每项形状为 ``[C, h, w]`` 的 float32 张量，
        其中 ``h = height // mask_ratio``、``w = width // mask_ratio``。
    """
    ratio = max(spec.mask_ratio, 1)
    grid_shape = (max(height // ratio, 1), max(width // ratio, 1))
    field_spec = spec.to_field_spec()
    fields = []
    for segment in segments:
        points = np.asarray(segment, dtype=np.float64).reshape(-1, 2)
        if len(points) < 3:
            continue
        x_min, y_min = np.floor(points.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(points.max(axis=0)).astype(int)
        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, width), min(y_max, height)
        if x_max - x_min < _MIN_BOX_SIDE_PX or y_max - y_min < _MIN_BOX_SIDE_PX:
            continue
        local = points - np.array((x_min, y_min))
        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(local).astype(np.int32)], 1)
        trace = trace_instance_mask(mask, (x_min, y_min, x_max, y_max))
        if trace is None:
            continue
        region = _instance_region(points, grid_shape, ratio)
        instance = build_instance_fields(trace, region, field_spec)
        if instance is not None:
            fields.append(instance)

    arrays = compose_fields(fields, grid_shape)
    return {
        "gas_center": torch.from_numpy(arrays["center"]).unsqueeze(0),
        "gas_center_valid": torch.from_numpy(arrays["center_valid"]).unsqueeze(0),
        "gas_direction": torch.from_numpy(arrays["direction"]),
        "gas_direction_valid": torch.from_numpy(
            arrays["direction_valid"]
        ).unsqueeze(0),
        "gas_width": torch.from_numpy(arrays["width"]).unsqueeze(0),
    }


class GasFormat(Format):
    """在实例格式化之前追加辅助目标。

    目标必须在 ``_format_segments`` 之前生成：那里会按面积重排实例，并把 Polygon
    降采样成实例掩码，之后再取轮廓就丢失原始几何了。

    目标栅格由 ``target_spec.mask_ratio`` 决定，与数据管线的 ``mask_ratio`` 解耦：
    P2 下管线是 ``/2``、目标仍是 ``/4``，损失侧按面积平均对齐。
    """

    def __init__(self, *args, target_spec: GasTargetSpec, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if target_spec.mask_ratio < 1:
            raise ValueError("辅助目标的 mask_ratio 必须不小于 1")
        self.target_spec = target_spec

    def apply_instances(
        self, labels: dict, params: dict | None = None
    ) -> dict:
        """先用像素坐标多边形生成目标，再走原有实例格式化。"""
        if params is not None and params.get("instances") is not None:
            segments = list(params["instances"].segments)
            labels.update(
                build_gas_targets(
                    segments, params["h"], params["w"], self.target_spec
                )
            )
        return super().apply_instances(labels, params)


class GasDataset(YOLODataset):
    """带辅助目标的实例分割数据集。

    与 ``RailAugmentedDataset`` 同目录，保持数据管线组件集中。``format_class`` 必须在
    父类初始化之前设置，因为父类构造函数会调用 ``build_transforms`` 取它。
    """

    def __init__(self, *args, target_spec: GasTargetSpec, **kwargs) -> None:
        self.format_class = partial(GasFormat, target_spec=target_spec)
        super().__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """沿 batch 维堆叠五个目标键，其余键沿用基类规则。"""
        collated = YOLODataset.collate_fn(batch)
        for key in GAS_TARGET_KEYS:
            collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
        return collated
