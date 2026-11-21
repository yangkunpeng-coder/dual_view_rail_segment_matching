"""钢轨实例分割后处理：Mask 去重与只读质量元数据。"""

from __future__ import annotations

from core_algorithm.postprocess.mask_nms import (
    MaskNmsResult,
    suppress_duplicate_masks,
)
from core_algorithm.postprocess.mask_quality import (
    MaskQuality,
    measure_mask_qualities,
    measure_mask_quality,
)
from core_algorithm.postprocess.pipeline import (
    PostprocessConfig,
    PostprocessResult,
    run_postprocess,
)

__all__ = [
    "MaskNmsResult",
    "MaskQuality",
    "PostprocessConfig",
    "PostprocessResult",
    "measure_mask_qualities",
    "measure_mask_quality",
    "run_postprocess",
    "suppress_duplicate_masks",
]
