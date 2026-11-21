"""为 YOLO26-seg 的 stride-4 检测头补充训练初期的几何正样本。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.tal import SegTaskAlignedAssigner

P2_WARMUP_BATCH_KEY = "p2_assignment_warmup"
_ASSIGNMENT_KEYS = {"enabled", "epochs", "max_positives_per_gt", "target_score"}
_DEFAULT_TARGET_SCORE = 0.5

_Assignment = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class P2AssignmentWarmupSettings:
    """已开启的 P2 分配热身设置。"""

    epochs: int
    max_positives_per_gt: int
    target_score: float


def parse_p2_assignment_warmup(
    block: object, config_file: Path
) -> P2AssignmentWarmupSettings | None:
    """校验热身配置，关闭时返回 None。

    Args:
        block: p2.assignment_warmup 配置块。
        config_file: 用于定位配置错误的文件路径。

    Returns:
        启用时返回生效设置，关闭时返回 None。

    Raises:
        ValueError: 配置字段或取值无效。
    """
    name = f"{config_file} 的 p2.assignment_warmup"
    if not isinstance(block, dict) or block.keys() - _ASSIGNMENT_KEYS:
        raise ValueError(f"{name} 必须是只包含已知字段的配置映射")
    is_enabled = block.get("enabled")
    if not isinstance(is_enabled, bool):
        raise ValueError(f"{name}.enabled 必须为布尔值")
    if not is_enabled:
        return None

    epochs = block.get("epochs")
    max_positives = block.get("max_positives_per_gt")
    for key, value in (("epochs", epochs), ("max_positives_per_gt", max_positives)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name}.{key} 必须为正整数")
    target_score = block.get("target_score", _DEFAULT_TARGET_SCORE)
    if type(target_score) not in (int, float) or not 0 < target_score <= 1:
        raise ValueError(f"{name}.target_score 必须位于 (0, 1]")
    return P2AssignmentWarmupSettings(epochs, max_positives, float(target_score))


def _count_per_gt(mask: Tensor, target_gt_idx: Tensor, gt_count: int) -> Tensor:
    """按图像及 GT 累加前景位置数。"""
    counts = target_gt_idx.new_zeros((len(mask), gt_count))
    return counts.scatter_add_(1, target_gt_idx, mask.long())


def _geometry_distances(
    centers: Tensor, gt_bboxes: Tensor, gt_masks: Tensor, mask_scale: Tensor
) -> Tensor:
    """计算框和 Mask 内的 P2 点到 GT 框中心的归一化距离平方。

    centers 与 gt_bboxes 使用输入图像像素坐标。返回 (B, G, A_p2)，
    不在对应 GT 框与掩码内的位置以 inf 表示。
    """
    height, width = gt_masks.shape[-2:]
    mask_centers = centers.unsqueeze(0) * mask_scale.reshape(-1, 1, 1)
    columns, rows = mask_centers.long().unbind(-1)
    in_grid = (
        (mask_centers >= 0).all(-1) & (columns < width) & (rows < height)
    )
    pixel_indices = rows.clamp(0, height - 1) * width + columns.clamp(0, width - 1)
    gt_count = gt_bboxes.shape[1]
    on_mask = gt_masks[:, :gt_count].flatten(2).gather(
        2, pixel_indices.unsqueeze(1).expand(-1, gt_count, -1)
    )

    left_top, right_bottom = gt_bboxes.unsqueeze(2).chunk(2, dim=-1)
    in_box = ((centers > left_top) & (centers < right_bottom)).all(-1)
    sizes = (right_bottom - left_top).clamp(min=1)
    center_offsets = (centers - (left_top + right_bottom) * 0.5) / sizes
    distances = center_offsets.square().sum(-1)
    return distances.masked_fill_(~(on_mask & in_box & in_grid[:, None]), torch.inf)


class _P2WarmupAssigner(SegTaskAlignedAssigner):
    """在原生分配上补足 P2 前景，保留同一检测头的正样本数量上限。"""

    def __init__(
        self, native: SegTaskAlignedAssigner, settings: P2AssignmentWarmupSettings
    ) -> None:
        super().__init__(
            topk=native.topk,
            topk2=native.topk2,
            num_classes=native.num_classes,
            alpha=native.alpha,
            beta=native.beta,
            stride=native.stride,
            eps=native.eps,
        )
        self.anchor_count = 0
        self.max_positives_per_gt = min(settings.max_positives_per_gt, self.topk2)
        self.target_score = settings.target_score

    def _forward(
        self,
        pd_scores: Tensor,
        pd_bboxes: Tensor,
        anc_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        mask_gt: Tensor,
        **extra: Tensor,
    ) -> _Assignment:
        """先做原生 Mask 感知分配，再向缺少有效 P2 正样本的 GT 补位。"""
        assignment = super()._forward(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **extra
        )
        labels, boxes, scores, foreground, gt_indices = assignment
        gt_count = gt_bboxes.shape[1]
        p2_slice = slice(0, self.anchor_count)
        p2_foreground = foreground[:, p2_slice]
        p2_effective = p2_foreground & (scores[:, p2_slice].sum(-1) > 0)
        existing = _count_per_gt(p2_effective, gt_indices[:, p2_slice], gt_count)
        need = (self.max_positives_per_gt - existing).clamp(min=0)
        need = need * mask_gt.squeeze(-1).bool()
        selected, is_selected = self._select_candidates(
            anc_points[p2_slice],
            gt_bboxes,
            extra["gt_masks"],
            extra["mask_scale"],
            p2_foreground,
            need,
        )
        batch_indices, selected_gt, slots = is_selected.nonzero(as_tuple=True)
        if not batch_indices.numel():
            return assignment

        added = is_selected.sum(-1)
        counts = _count_per_gt(foreground, gt_indices, gt_count)
        # 只有新增 P2 导致超额时才替换旧位置。保留已经有效的 P2 正样本。
        drop_counts = (counts + added - self.topk2).clamp(min=0).minimum(added)
        droppable = foreground.clone()
        droppable[:, p2_slice] &= ~p2_effective
        dropped = self._weakest_positions(
            droppable, gt_indices, scores, drop_counts, selected.shape[-1]
        )
        scores[dropped] = 0
        foreground[dropped] = False

        anchor_indices = selected[batch_indices, selected_gt, slots]
        indices = (batch_indices, anchor_indices)
        classes = gt_labels[batch_indices, selected_gt, 0].long()
        foreground[indices] = True
        gt_indices[indices] = selected_gt
        labels[indices] = classes
        boxes[indices] = gt_bboxes[batch_indices, selected_gt]
        scores[indices] = 0
        scores[batch_indices, anchor_indices, classes] = self.target_score
        return assignment

    def _select_candidates(
        self,
        centers: Tensor,
        gt_bboxes: Tensor,
        gt_masks: Tensor,
        mask_scale: Tensor,
        foreground: Tensor,
        need: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """从空闲 P2 位置选择离 GT 中心最近的点，冲突时只保留一个 GT。"""
        distances = _geometry_distances(
            centers.float(), gt_bboxes.float(), gt_masks, mask_scale.float()
        )
        distances.masked_fill_(
            foreground[:, None] | (need == 0).unsqueeze(-1), torch.inf
        )
        gt_ids = torch.arange(distances.shape[1], device=distances.device)
        closest_gt = distances.argmin(dim=1, keepdim=True)
        distances.masked_fill_(closest_gt != gt_ids.view(1, -1, 1), torch.inf)
        count = min(self.max_positives_per_gt, len(centers))
        nearest, selected = distances.topk(count, dim=-1, largest=False)
        slots = torch.arange(count, device=distances.device)
        return selected, nearest.isfinite() & (slots < need.unsqueeze(-1))

    @staticmethod
    def _weakest_positions(
        droppable: Tensor,
        gt_indices: Tensor,
        scores: Tensor,
        drop_counts: Tensor,
        count: int,
    ) -> tuple[Tensor, Tensor]:
        """在超额的 GT 中挑出质量最低的原生正样本位置。"""
        gt_ids = torch.arange(drop_counts.shape[1], device=scores.device)
        belongs = gt_indices[:, None] == gt_ids.view(1, -1, 1)
        candidates = droppable[:, None] & belongs
        weights = scores.sum(-1).unsqueeze(1).expand_as(candidates)
        weakest, positions = weights.masked_fill(~candidates, torch.inf).topk(
            count, dim=-1, largest=False
        )
        slots = torch.arange(count, device=scores.device)
        is_dropped = weakest.isfinite() & (slots < drop_counts.unsqueeze(-1))
        batch, gt, slot = is_dropped.nonzero(as_tuple=True)
        return batch, positions[batch, gt, slot]


class P2AssignmentWarmupCriterion:
    """仅替换标记批次的分配器，复用原生 E2E 损失和权重调度。"""

    def __init__(
        self, native: E2ELoss, settings: P2AssignmentWarmupSettings
    ) -> None:
        self._native = native
        self._heads = (native.one2many, native.one2one)
        if not all(
            isinstance(head.assigner, SegTaskAlignedAssigner) for head in self._heads
        ):
            raise ValueError("P2 分配热身需要 YOLO26-seg 的 Mask 感知分配器")
        self._warmup_assigners = tuple(
            _P2WarmupAssigner(head.assigner, settings) for head in self._heads
        )

    def __call__(
        self,
        preds: dict | tuple,
        batch: dict[str, Tensor | bool],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """对热身训练批次补充分配，其余批次直接计算原生损失。"""
        if batch.get(P2_WARMUP_BATCH_KEY) is not True:
            return self._native(preds, batch)

        raw = self._native.one2many.parse_output(preds)
        p2 = raw["one2many"]["feats"][0]
        anchor_count = p2.shape[-2] * p2.shape[-1]
        native_assigners = tuple(head.assigner for head in self._heads)
        try:
            for head, warmup in zip(self._heads, self._warmup_assigners, strict=True):
                warmup.anchor_count = anchor_count
                head.assigner = warmup
            return self._native(preds, batch)
        finally:
            for head, native in zip(self._heads, native_assigners, strict=True):
                head.assigner = native

    def update(self) -> None:
        """保留原生 one-to-many 与 one-to-one 损失权重调度。"""
        self._native.update()
