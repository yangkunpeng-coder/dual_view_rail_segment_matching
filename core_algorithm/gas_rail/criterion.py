"""把辅助几何损失接到 YOLO 的原生损失向量上。

``GasCriterion`` 包装 ``E2ELoss``：原生 5 项（box/seg/cls/dfl/sem）照常计算，四项辅助损失作为
新元素追加在损失向量末尾，由训练器统一求和反传。这是本方案唯一依赖 ultralytics 的文件，其余
算法文件只依赖 torch 或 numpy。
"""

from __future__ import annotations

import torch
from ultralytics.nn.modules.head import Segment26
from ultralytics.utils.loss import E2ELoss, SegmentLoss26, v8SegmentationLoss
from ultralytics.utils.ops import crop_mask

from core_algorithm.gas_rail.head import GasHead
from core_algorithm.gas_rail.losses import (
    GasLossWeights,
    align_to_target,
    centerline_loss,
    direction_loss,
    smoothness_loss,
    width_loss,
)

GAS_LOSS_KEYS = ("center_loss", "direction_loss", "smooth_loss", "width_loss")

_DEFAULT_HIDDEN_SINGLE = 16
_DEFAULT_HIDDEN_MULTI = 32


class GasSegmentLoss26(SegmentLoss26):
    """在原生 bbox 内 BCE 之外增加按实例的 soft Dice 重叠项。

    基类的 ``single_mask_loss`` 是 staticmethod，但调用点是 ``self.single_mask_loss(...)``，
    因此可以覆写成实例方法以读取 ``dice_weight``。
    """

    def __init__(
        self, model, tal_topk: int = 10, tal_topk2: int | None = None
    ) -> None:
        super().__init__(model, tal_topk, tal_topk2)
        self.dice_weight = 0.0

    def single_mask_loss(
        self,
        gt_mask: torch.Tensor,
        pred: torch.Tensor,
        proto: torch.Tensor,
        xyxy: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        """在原生 bbox 内 BCE 之上按实例叠加 soft Dice。"""
        base = v8SegmentationLoss.single_mask_loss(
            gt_mask, pred, proto, xyxy, area
        )
        if self.dice_weight <= 0:
            return base
        probability = torch.einsum("in,nhw->ihw", pred, proto).sigmoid()
        # crop_mask 会原地修改输入，必须传副本，否则会破坏 sigmoid 的反传缓冲区。
        cropped = crop_mask(probability.clone(), xyxy)
        target = crop_mask(gt_mask.float().clone(), xyxy)
        intersection = (cropped * target).sum(dim=(1, 2))
        union = cropped.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = (2.0 * intersection + 1.0) / (union + 1.0)
        return base + self.dice_weight * (1.0 - dice).mean()


def gas_channels(weights: GasLossWeights) -> int:
    """按权重决定辅助头输出通道数：1 中心线，3 加方向，4 加宽度。"""
    channels = 1
    if weights.direction > 0 or weights.smooth > 0:
        channels += 2
    if weights.width > 0:
        channels += 1
    return channels


def ensure_gas_head(
    model: torch.nn.Module,
    weights: GasLossWeights,
    hidden_channels: int | None = None,
    centerline_prior: float = 0.02,
    width_prior: float = 1.9,
) -> GasHead:
    """幂等挂载辅助头，返回已挂载的模块。

    Args:
        model: 已构建的检测/分割模型（非 DDP 包装层）。
        weights: 辅助权重，用来决定输出通道数。
        hidden_channels: 中间通道数；为 None 时单通道用 16、多通道用 32。
        centerline_prior: 中心线偏置先验（格子被中心线覆盖的比例）。
        width_prior: 宽度偏置先验（对数视宽的初值，单位原型格）。

    Returns:
        挂在 ``model.model[-1].gas`` 上的辅助头。

    Raises:
        TypeError: 分割头不是 ``Segment26``。
    """
    segment_head = model.model[-1]
    if not isinstance(segment_head, Segment26):
        raise TypeError("辅助头只能挂在 Segment26 分割头上")
    existing = getattr(segment_head, "gas", None)
    if existing is not None:
        return existing
    channels = gas_channels(weights)
    hidden = hidden_channels
    if hidden is None:
        hidden = (
            _DEFAULT_HIDDEN_SINGLE if channels == 1 else _DEFAULT_HIDDEN_MULTI
        )
    module = GasHead(
        in_channels=int(segment_head.nm),
        hidden_channels=hidden,
        out_channels=channels,
        centerline_prior=centerline_prior,
        width_prior=width_prior,
    )
    segment_head.gas = module
    return module


class GasCriterion:
    """包装 ``E2ELoss``，把四项辅助损失追加到损失向量末尾。"""

    def __init__(
        self,
        model: torch.nn.Module,
        head: GasHead,
        weights: GasLossWeights,
        target_keys: tuple[str, ...],
        smooth_delta: float = 1.0,
    ) -> None:
        self.model = model
        self.head = head
        self.weights = weights
        self.target_keys = target_keys
        self.smooth_delta = smooth_delta
        self.epoch_index = 0
        self.gain = 1.0
        self.base = E2ELoss(model, GasSegmentLoss26)
        self.base.one2many.dice_weight = weights.dice
        self.base.one2one.dice_weight = weights.dice

    def update(self) -> None:
        """每轮结束由训练器调用：推进分支权重与 warmup 增益。"""
        self.base.update()
        self.epoch_index += 1
        self.gain = self.weights.warmup_gain(self.epoch_index)

    def __call__(
        self, preds: dict, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算原生损失与辅助损失，返回拼接后的损失向量与日志项。"""
        loss, items = self.base(preds, batch)
        geometry_terms, geometry_items = self._geometry_loss(preds, batch)
        total = torch.cat((loss, geometry_terms))
        self._check_contract(total)
        return total, {**items, **geometry_items}

    def _geometry_loss(
        self, preds: dict, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算四个辅助项；没有目标的批（验证/测试）返回同键零值。"""
        device = batch["img"].device if "img" in batch else torch.device("cpu")
        if not all(key in batch for key in self.target_keys):
            zeros = torch.zeros(len(GAS_LOSS_KEYS), device=device)
            items = {
                key: zeros[index].detach()
                for index, key in enumerate(GAS_LOSS_KEYS)
            }
            return zeros, items

        proto = self.base.one2many.parse_output(preds)["one2many"]["proto"]
        if isinstance(proto, tuple):
            proto = proto[0]
        logits = self.head(proto)
        # P2 下原型在 /2、辅助目标在 /4，按面积平均降到目标栅格后再算损失。
        logits = align_to_target(
            logits, tuple(batch["gas_center"].shape[-2:])
        )
        channels = logits.shape[1]
        batch_size = logits.shape[0]
        center = batch["gas_center"].float()
        center_valid = batch["gas_center_valid"].float()
        direction_valid = batch["gas_direction_valid"].float()
        zero = logits.sum() * 0.0

        center_term = centerline_loss(
            logits[:, 0:1], center, center_valid
        )
        direction_term = zero
        smooth_term = zero
        width_term = zero
        if self.weights.direction > 0:
            if channels < 3:
                raise RuntimeError("方向权重非零但辅助头没有方向通道")
            direction_term = direction_loss(
                logits[:, 1:3], batch["gas_direction"].float(), direction_valid
            )
        if self.weights.smooth > 0:
            if channels < 3:
                raise RuntimeError("平滑权重非零但辅助头没有方向通道")
            smooth_term = smoothness_loss(
                logits[:, 1:3],
                (direction_valid * center).detach(),
                self.smooth_delta,
            )
        if self.weights.width > 0:
            if channels < 4:
                raise RuntimeError("宽度权重非零但辅助头没有宽度通道")
            width_term = width_loss(
                logits[:, 3:4], batch["gas_width"].float(), center_valid
            )
        terms = torch.stack(
            (center_term, direction_term, smooth_term, width_term)
        )
        weights = torch.tensor(
            (
                self.weights.center,
                self.weights.direction,
                self.weights.smooth,
                self.weights.width,
            ),
            device=terms.device,
            dtype=terms.dtype,
        )
        # 损失向量需要乘 batch 尺寸（与原生各分支一致）；日志项不乘，便于与 train/seg_loss 对比。
        scaled = terms * weights * self.gain
        vector = scaled * batch_size
        items = {
            key: scaled[index].detach()
            for index, key in enumerate(GAS_LOSS_KEYS)
        }
        return vector, items

    def _check_contract(self, loss: torch.Tensor) -> None:
        """断言上游契约，升级 Ultralytics 后第一时间报错而不是静默降级。"""
        expected = len(self.base.one2many.loss_names) + len(GAS_LOSS_KEYS)
        if loss.numel() != expected:
            raise RuntimeError(
                "YOLO 损失向量长度变化，见方案 §3.3 契约 2："
                f"期望 {expected}，实际 {loss.numel()}"
            )


def build_gas_criterion(
    model: torch.nn.Module,
    weights: GasLossWeights,
    target_keys: tuple[str, ...],
    smooth_delta: float = 1.0,
    hidden_channels: int | None = None,
    centerline_prior: float = 0.02,
    width_prior: float = 1.9,
) -> GasCriterion:
    """挂载辅助头并构建包装后的 criterion。"""
    head = ensure_gas_head(
        model,
        weights,
        hidden_channels=hidden_channels,
        centerline_prior=centerline_prior,
        width_prior=width_prior,
    )
    return GasCriterion(model, head, weights, target_keys, smooth_delta)
