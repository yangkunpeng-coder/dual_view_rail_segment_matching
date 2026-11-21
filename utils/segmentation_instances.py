"""把模型预测或归档 NPZ 转成匹配模块可直接消费的实例视图。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from core_algorithm.postprocess import (
    MaskQuality,
    PostprocessConfig,
    run_postprocess,
)
from data_preprocess.common.rail_corridor import make_corridor_mask
from data_preprocess.common.roi_config import CorridorSettings
from utils.instance_masks import as_mask_stack
from utils.segmentation_postprocess import load_postprocess_config
from utils.segmentation_predict import PREDICT_CONFIG, load_predict_options

RoiBox = tuple[int, int, int, int]
RoiStatus = Literal["raw", "cropped", "no_corridor", "empty_crop"]


@dataclass(frozen=True)
class PredictedView:
    """单侧原图和去重后实例，mask_index 即 masks 数组下标。

    roi_bbox 使用原图像素 xyxy 坐标，右下为开区间。
    """

    image_rgb: NDArray[np.uint8]
    masks: NDArray[np.bool_]
    scores: NDArray[np.float32]
    raw_count: int
    suppressed_count: int
    postprocess_ms: float
    qualities: tuple[MaskQuality, ...]
    roi_bbox: RoiBox | None = None
    roi_status: RoiStatus = "raw"


class PredictedViewLoader:
    """加载一次模型，对单张图片预测并去重。"""

    def __init__(
        self,
        weights_path: str | Path,
        predict_config: str | Path = PREDICT_CONFIG,
        postprocess_config: PostprocessConfig | None = None,
        *,
        corridor: CorridorSettings | None = None,
    ) -> None:
        """初始化模型、预测参数和后处理配置。

        Args:
            weights_path: 本地实例分割权重。
            predict_config: YOLO 预测配置。
            postprocess_config: 后处理参数，省略时读取 seg_postprocess.yaml。
            corridor: 预测走廊参数，省略时直接在原图上预测一次。

        Raises:
            FileNotFoundError: 本地权重文件不存在。
            ValueError: 权重扩展名或模型任务无效。
        """
        from ultralytics import YOLO

        weights_file = Path(weights_path).resolve()
        if weights_file.suffix.lower() != ".pt":
            raise ValueError(f"预测需要本地 .pt 权重 {weights_file}")
        if not weights_file.is_file():
            raise FileNotFoundError(f"找不到实例分割权重 {weights_file}")
        self.model = YOLO(str(weights_file))
        if self.model.task != "segment":
            raise ValueError(f"预测需要实例分割权重 {weights_file}")
        self.options = load_predict_options(Path(predict_config).resolve())
        self.postprocess_config = (
            postprocess_config or load_postprocess_config()
        )
        self.corridor = corridor

    def load(self, image_path: str | Path) -> PredictedView:
        """预测单张图片，返回原图坐标下去重后的实例。

        Args:
            image_path: 原始相机图像的路径。

        Returns:
            原图、Mask、置信度和 ROI 状态。走廊或二次结果为空时保留首遍结果。
        """
        image_bgr, masks, scores, boxes = self._predict(
            str(Path(image_path).resolve())
        )
        roi_bbox = None
        roi_status: RoiStatus = "raw"
        if self.corridor is not None:
            crop = _crop_predicted_corridor(image_bgr, masks, self.corridor)
            if crop is None:
                roi_status = "no_corridor"
            else:
                crop_bgr, roi_bbox = crop
                _, crop_masks, crop_scores, crop_boxes = self._predict(crop_bgr)
                nonempty = crop_masks.any(axis=(1, 2))
                if nonempty.any():
                    x0, y0, x1, y1 = roi_bbox
                    masks = np.zeros(
                        (int(nonempty.sum()), *image_bgr.shape[:2]), dtype=np.bool_
                    )
                    masks[:, y0:y1, x0:x1] = crop_masks[nonempty]
                    scores = crop_scores[nonempty]
                    boxes = None if crop_boxes is None else crop_boxes[nonempty]
                    if boxes is not None:
                        boxes[:, 0::2] += x0
                        boxes[:, 1::2] += y0
                    roi_status = "cropped"
                else:
                    roi_status = "empty_crop"

        postprocessed = run_postprocess(
            masks, scores, self.postprocess_config, boxes=boxes
        )
        return PredictedView(
            image_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            masks=postprocessed.masks,
            scores=postprocessed.scores,
            raw_count=len(masks),
            suppressed_count=postprocessed.suppressed_count,
            postprocess_ms=postprocessed.latency_ms,
            qualities=postprocessed.qualities,
            roi_bbox=roi_bbox,
            roi_status=roi_status,
        )

    def _predict(
        self, source: str | NDArray[np.uint8]
    ) -> tuple[
        NDArray[np.uint8], NDArray[np.bool_], NDArray[np.float32],
        NDArray[np.float32] | None,
    ]:
        """返回输入画布的 BGR、实例掩码、置信度和检测框。"""
        result = self.model.predict(
            source=source,
            stream=False,
            save=False,
            verbose=False,
            **self.options,
        )[0]
        masks = as_mask_stack(result.masks, result.orig_shape)
        scores = (
            result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            if result.boxes is not None
            else np.empty(0, dtype=np.float32)
        )
        boxes = None
        if result.boxes is not None and len(result.boxes) == len(masks):
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        return result.orig_img, masks, scores, boxes


def _crop_predicted_corridor(
    image_bgr: NDArray[np.uint8],
    masks: NDArray[np.bool_],
    settings: CorridorSettings,
) -> tuple[NDArray[np.uint8], RoiBox] | None:
    """把首遍预测的逐行有效区外填灰，并裁出外接矩形。"""
    union = masks.any(axis=0).view(np.uint8)
    corridor = make_corridor_mask(union, settings)
    x0, y0, width, height = cv2.boundingRect(corridor)
    if not width or not height:
        return None
    x1, y1 = x0 + width, y0 + height
    crop = image_bgr[y0:y1, x0:x1].copy()
    crop[corridor[y0:y1, x0:x1] == 0] = settings.outside_fill_rgb[::-1]
    return crop, (x0, y0, x1, y1)


def load_archived_view(
    npz_path: str | Path,
    image_path: str | Path,
    postprocess_config: PostprocessConfig | None = None,
) -> PredictedView:
    """读取归档 NPZ 与原图，去重后返回实例视图，不加载 YOLO。"""
    config = postprocess_config or load_postprocess_config()
    data = np.load(Path(npz_path))
    masks = np.asarray(data["masks"], dtype=np.bool_)
    scores = np.asarray(data["scores"], dtype=np.float32)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"无法读取原图 {image_path}")
    if masks.shape[1:] != image_bgr.shape[:2]:
        raise ValueError(f"Mask 尺寸与原图不一致 {image_path}")
    postprocessed = run_postprocess(masks, scores, config)
    return PredictedView(
        image_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        masks=postprocessed.masks,
        scores=postprocessed.scores,
        raw_count=len(masks),
        suppressed_count=postprocessed.suppressed_count,
        postprocess_ms=postprocessed.latency_ms,
        qualities=postprocessed.qualities,
    )
