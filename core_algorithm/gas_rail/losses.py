"""辅助监督的四项损失与权重调度。

四项都按图归一化再对有监督的图取均值：单张图内前景比例差异很大，按 batch 直接求和会让
钢轨密集的图主导梯度。

本模块只依赖 torch，不依赖 ultralytics，也不依赖数据集结构。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_EPSILON = 1e-6
_WIDTH_BETA = 0.25


def align_to_target(
    logits: torch.Tensor, target_shape: tuple[int, int]
) -> torch.Tensor:
    """把辅助输出按面积平均对齐到目标栅格。

    P2 结构下原型在 ``/2``，而辅助目标仍生成在 ``/4``，两者尺寸不一致；这里按面积
    平均降采样，数学上等价于 2×2 平均池化，尺寸相同时原样返回。

    Args:
        logits: ``[B, C, h, w]`` 辅助头输出。
        target_shape: 目标栅格的 ``(h, w)``。

    Returns:
        形状为 ``[B, C, target_h, target_w]`` 的张量。
    """
    if tuple(logits.shape[-2:]) == tuple(target_shape):
        return logits
    return F.interpolate(logits, size=tuple(target_shape), mode="area")



@dataclass(frozen=True)
class GasLossWeights:
    """辅助损失权重与 warmup。

    Attributes:
        center: 中心线热力图权重。
        direction: 方向场权重。
        smooth: 沿轨平滑权重，默认 0 表示不启用。
        width: 宽度权重，默认 0 表示不启用。
        dice: 原生实例掩码损失的 Dice 项系数，默认 0 表示不启用。
        warmup_epochs: 权重从 ``1/warmup_epochs`` 线性升到 1 所需轮数。
    """

    center: float = 0.1
    direction: float = 0.03
    smooth: float = 0.0
    width: float = 0.0
    dice: float = 0.0
    warmup_epochs: float = 3.0

    def warmup_gain(self, epoch_index: int) -> float:
        """返回第 ``epoch_index`` 轮（从 0 计）生效的权重增益。"""
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, (epoch_index + 1) / self.warmup_epochs)


def centerline_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """按正负目标质量分别归一化的中心线 BCE。

    Args:
        logits: ``[B, 1, h, w]`` 中心线 logits。
        target: ``[B, 1, h, w]`` 高斯软目标，[0, 1]。
        valid: ``[B, 1, h, w]`` 有效域，0/1。

    Returns:
        标量损失；正负质量都为零的图被跳过。
    """
    logits = logits.float()
    target = target.float()
    valid = valid.float()
    positive = valid * target
    negative = valid * (1.0 - target)
    pos_mass = positive.sum(dim=(1, 2, 3))
    neg_mass = negative.sum(dim=(1, 2, 3))
    pos_item = (positive * F.softplus(-logits)).sum(dim=(1, 2, 3))
    neg_item = (negative * F.softplus(logits)).sum(dim=(1, 2, 3))
    pos_item = pos_item / pos_mass.clamp(min=1.0)
    neg_item = neg_item / neg_mass.clamp(min=1.0)
    supervised = ((pos_mass > 0) & (neg_mass > 0)).float()
    total = (0.5 * (pos_item + neg_item) * supervised).sum()
    return total / supervised.sum().clamp(min=1.0)


def direction_loss(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """双角方向场的平方误差，只在有效带内计算。"""
    diff = (pred.float() - target.float()).pow(2).sum(dim=1, keepdim=True)
    weight = valid.float()
    mass = weight.sum(dim=(1, 2, 3))
    item = (diff * weight).sum(dim=(1, 2, 3)) / mass.clamp(min=1.0)
    supervised = (mass > 0).float()
    return (item * supervised).sum() / supervised.sum().clamp(min=1.0)


def smoothness_loss(
    pred: torch.Tensor, weight: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """沿轨切向的方向场变化率。

    在 ``x ± delta·t̂`` 处双线性采样方向向量，惩罚沿轨差分。只沿切向做差，避免把相邻两根
    钢轨的方向拉齐。

    Args:
        pred: ``[B, 2, h, w]`` 方向场。
        weight: ``[B, 1, h, w]`` 有效权重（通常取 ``V_d·T``，需 detach）。
        delta: 采样步长，单位是网格单元。

    Returns:
        标量损失；方向场恒零时为 0（此时梯度也为 0，属预期）。
    """
    pred = pred.float()
    weight = weight.float()
    batch, _, height, width = pred.shape
    if height < 2 or width < 2:
        return pred.sum() * 0.0
    norm = pred.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=_EPSILON)
    tangent = pred / norm
    ys = torch.linspace(-1.0, 1.0, height, device=pred.device, dtype=pred.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=pred.device, dtype=pred.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    scale = torch.tensor(
        (2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)),
        device=pred.device,
        dtype=pred.dtype,
    )
    offset = tangent.permute(0, 2, 3, 1) * delta * scale
    plus = F.grid_sample(
        pred, base + offset, mode="bilinear", align_corners=True,
        padding_mode="border",
    )
    minus = F.grid_sample(
        pred, base - offset, mode="bilinear", align_corners=True,
        padding_mode="border",
    )
    difference = (plus - minus).pow(2).sum(dim=1, keepdim=True)
    mass = weight.sum().clamp(min=1.0)
    return (difference * weight).sum() / (mass * (2.0 * delta) ** 2)


def width_loss(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """对数视宽的 Huber 损失，只在中心线有效处监督。"""
    difference = pred.float() - target.float()
    absolute = difference.abs()
    huber = torch.where(
        absolute < _WIDTH_BETA,
        0.5 * difference.pow(2) / _WIDTH_BETA,
        absolute - 0.5 * _WIDTH_BETA,
    )
    weight = valid.float()
    mass = weight.sum(dim=(1, 2, 3))
    item = (huber * weight).sum(dim=(1, 2, 3)) / mass.clamp(min=1.0)
    supervised = (mass > 0).float()
    return (item * supervised).sum() / supervised.sum().clamp(min=1.0)
