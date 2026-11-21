"""将 P2 分配热身接入 YOLO26 分割训练的前几轮。"""

from pathlib import Path

from torch import Tensor
from ultralytics.utils import YAML
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import unwrap_model

from core_algorithm.p2_assignment_warmup import (
    P2_WARMUP_BATCH_KEY,
    P2AssignmentWarmupCriterion,
    parse_p2_assignment_warmup,
)
from experiments.yolo_seg.mask_seg_trainer import MaskSegmentationTrainer

P2_ASSIGNMENT_WARMUP_SNAPSHOT = "p2_assignment_warmup.yaml"
_P2_STRIDES = [4, 8, 16, 32]


class P2AssignmentWarmupTrainer(MaskSegmentationTrainer):
    """使用当前实验快照标记热身批次，保留原生验证和权重选择。"""

    def _setup_train(self) -> None:
        """从快照加载设置，在上游初始化 EMA 和恢复状态后包装训练损失。"""
        snapshot = Path(self.save_dir) / "configs" / P2_ASSIGNMENT_WARMUP_SNAPSHOT
        settings = parse_p2_assignment_warmup(YAML.load(snapshot), snapshot)
        if settings is None:
            raise ValueError(f"P2 热身训练器要求开启热身配置 {snapshot}")

        super()._setup_train()
        model = unwrap_model(self.model)
        if model.stride.tolist() != _P2_STRIDES:
            raise ValueError("P2 分配热身要求模型的检测 stride 依次为 4、8、16、32")
        criterion = getattr(model, "criterion", None)
        if criterion is None:
            criterion = model.init_criterion()
        if not isinstance(criterion, E2ELoss):
            raise ValueError("P2 分配热身要求 YOLO26-seg 的 E2E 损失")
        model.criterion = P2AssignmentWarmupCriterion(criterion, settings)
        self._warmup_epochs = settings.epochs

    def preprocess_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor | bool]:
        """预处理训练批次，并按本次运行的全局轮次标记热身。

        Args:
            batch: 上游数据加载器输出的实例分割训练批次。

        Returns:
            原生预处理后的批次与热身开关。验证器使用自己的预处理。
        """
        batch = super().preprocess_batch(batch)
        batch[P2_WARMUP_BATCH_KEY] = self.model.training and (
            0 <= self.epoch < self._warmup_epochs
        )
        return batch
