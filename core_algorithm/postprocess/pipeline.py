"""分割后处理管线：Mask 去重、质量元数据与耗时统计。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from core_algorithm.postprocess.mask_nms import suppress_duplicate_masks
from core_algorithm.postprocess.mask_quality import (
    MaskQuality,
    measure_mask_qualities,
)


@dataclass(frozen=True)
class PostprocessConfig:
    """后处理开关与参数，默认只用 Mask 去重。"""

    enabled: bool = True
    mask_nms_iou: float = 0.65
    min_area_px: int = 8
    quality_enabled: bool = False


@dataclass(frozen=True)
class PostprocessResult:
    """一张图的后处理结果，Mask 与分数已按保留实例重排。"""

    masks: NDArray[np.bool_]
    scores: NDArray[np.float32]
    keep_indices: NDArray[np.intp]
    suppressed_count: int
    qualities: tuple[MaskQuality, ...]
    latency_ms: float


def run_postprocess(
    masks: NDArray[np.bool_],
    scores: NDArray[np.floating],
    config: PostprocessConfig | None = None,
    *,
    boxes: NDArray[np.floating] | None = None,
) -> PostprocessResult:
    """对单张图的实例做去重并生成质量元数据。

    Args:
        masks: 原图尺寸的 (N, H, W) 布尔实例掩码。
        scores: 各实例置信度，形状为 (N,)。
        config: 后处理参数；省略时使用默认配置。
        boxes: 可选的实例检测框，形状为 (N, 4)，xyxy 格式，用于加速包围盒预筛。

    Returns:
        去重后的 Mask、分数、原始索引、抑制数量、质量特征和处理耗时。

    Raises:
        ValueError: 输入数组形状、类型或分数无效。
    """
    config = config or PostprocessConfig()
    started = perf_counter()
    score_array = np.asarray(scores, dtype=np.float32)
    _validate_arrays(masks, score_array, boxes)
    if config.enabled:
        result = suppress_duplicate_masks(
            masks,
            score_array,
            iou_threshold=config.mask_nms_iou,
            min_area_px=config.min_area_px,
            boxes=boxes,
        )
        keep_indices = result.keep_indices
        suppressed_count = int(len(result.suppressed_indices))
    else:
        keep_indices = np.arange(len(masks), dtype=np.intp)
        suppressed_count = 0
    if len(keep_indices) == len(masks) and np.array_equal(
        keep_indices, np.arange(len(masks), dtype=np.intp)
    ):
        # 没有实例被抑制且顺序未变时直接复用输入，避免复制整图尺寸的 Mask 栈。
        kept_masks = masks
        kept_scores = score_array
    else:
        kept_masks = np.ascontiguousarray(masks[keep_indices])
        kept_scores = np.ascontiguousarray(score_array[keep_indices])
    qualities = (
        measure_mask_qualities(
            kept_masks,
            None if boxes is None else np.asarray(boxes)[keep_indices],
        )
        if config.quality_enabled
        else ()
    )
    return PostprocessResult(
        masks=kept_masks,
        scores=kept_scores,
        keep_indices=keep_indices,
        suppressed_count=suppressed_count,
        qualities=qualities,
        latency_ms=(perf_counter() - started) * 1000.0,
    )


def _validate_arrays(
    masks: NDArray[np.bool_],
    scores: NDArray[np.floating],
    boxes: NDArray[np.floating] | None,
) -> None:
    """在进入算法前统一校验输入，避免错误传播到评测产物。"""
    if masks.ndim != 3:
        raise ValueError("masks 必须为 (N, H, W) 的三维数组")
    if masks.dtype != np.bool_:
        raise ValueError("masks 必须为布尔数组")
    if scores.shape != (len(masks),):
        raise ValueError("scores 长度必须与 masks 实例数一致")
    if not np.isfinite(scores).all():
        raise ValueError("scores 必须为有限值")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores 必须位于 [0, 1]")
    if boxes is not None and boxes.shape != (len(masks), 4):
        raise ValueError("boxes 必须为 (N, 4) 的 xyxy 数组")
