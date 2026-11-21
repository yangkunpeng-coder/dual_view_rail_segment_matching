"""辅助预测头：在 Mask 原型特征上回归中心线、方向与宽度。

辅助头只在训练期被损失调用，不进入 ``Segment26.forward`` 的推理分支，因此推理零开销。
"""

from __future__ import annotations

import math

import torch
from torch import nn

CENTER_CHANNEL = 0
DIRECTION_CHANNELS = slice(1, 3)
WIDTH_CHANNEL = 3

_SUPPORTED_CHANNELS = (1, 3, 4)


class GasHead(nn.Module):
    """在 Mask 原型上预测中心线热力图、方向场与对数视宽。

    Attributes:
        out_channels: 输出通道数，1=中心线，3=+方向，4=+宽度。
        trunk: 两层 3x3 卷积，共用给三组输出。
        head: 1x1 输出卷积，偏置按目标先验初始化。
    """

    def __init__(
        self,
        in_channels: int = 32,
        hidden_channels: int = 32,
        out_channels: int = 3,
        centerline_prior: float = 0.02,
        width_prior: float = 1.9,
    ) -> None:
        super().__init__()
        if out_channels not in _SUPPORTED_CHANNELS:
            raise ValueError(f"out_channels 只能是 {_SUPPORTED_CHANNELS} 之一")
        if not 0.0 < centerline_prior < 1.0:
            raise ValueError("centerline_prior 必须在 (0, 1) 内")
        self.out_channels = out_channels
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(hidden_channels, out_channels, 1)
        with torch.no_grad():
            # 末层权重置零，让初始 logits 完全由先验偏置决定：否则随机卷积会把 logits
            # 推到 ±10 量级，辅助项一开训就压过主任务。
            self.head.weight.zero_()
            self.head.bias.zero_()
            # 中心线取先验概率的 logit，方向从 0 起步，宽度取先验均值。
            self.head.bias[CENTER_CHANNEL] = math.log(
                centerline_prior / (1.0 - centerline_prior)
            )
            if out_channels >= 4:
                self.head.bias[WIDTH_CHANNEL] = width_prior

    def forward(self, proto: torch.Tensor) -> torch.Tensor:
        """输出辅助 logits。

        Args:
            proto: Mask 原型特征，形状 ``[B, in_channels, h, w]``。

        Returns:
            ``[B, out_channels, h, w]`` 的 logits。
        """
        return self.head(self.trunk(proto))
