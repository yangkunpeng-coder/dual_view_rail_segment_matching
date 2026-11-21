"""钢轨实例分割的辅助监督算法：辅助头、四项损失与损失接入。

三层依赖边界：

- ``head.py``、``losses.py`` 只依赖 torch；
- ``criterion.py`` 额外依赖 ultralytics（它是唯一的框架适配层）；
- 几何测量与目标场在 ``core_algorithm/common``，不依赖本包。

训练接入口在 ``experiments/yolo_seg/mask_seg_trainer.py``。
"""

from core_algorithm.gas_rail.criterion import (
    GAS_LOSS_KEYS,
    GasCriterion,
    GasSegmentLoss26,
    gas_channels,
    build_gas_criterion,
    ensure_gas_head,
)
from core_algorithm.gas_rail.head import GasHead
from core_algorithm.gas_rail.losses import (
    GasLossWeights,
    centerline_loss,
    direction_loss,
    smoothness_loss,
    width_loss,
)

__all__ = [
    "GAS_LOSS_KEYS",
    "GasCriterion",
    "GasHead",
    "GasLossWeights",
    "GasSegmentLoss26",
    "gas_channels",
    "build_gas_criterion",
    "centerline_loss",
    "direction_loss",
    "ensure_gas_head",
    "smoothness_loss",
    "width_loss",
]
