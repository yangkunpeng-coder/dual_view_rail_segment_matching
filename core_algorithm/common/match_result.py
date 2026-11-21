"""南北钢轨匹配共用的结果类型。"""

from __future__ import annotations

from typing import Literal, TypedDict

Side = Literal["south", "north"]


class RailInstance(TypedDict):
    """图内实例编号、输入 Mask 索引及原图测量点 (x, y)。"""

    instance_id: str
    mask_index: int
    center_xy: tuple[float, int] | None


class RailMatch(TypedDict):
    """图对内共享 ID、两侧实例和归一化横向位置残差。"""

    shared_id: str
    south: RailInstance
    north: RailInstance
    position_error: float


class UnmatchedRail(RailInstance):
    """保留图内编号和未能建立可靠对应的原因。"""

    reason: str


class RailMatchingResult(TypedDict):
    """同一图对的匹配与单侧未匹配实例，可序列化为 JSON。"""

    south_image_path: str
    north_image_path: str
    matches: list[RailMatch]
    unmatched_south: list[UnmatchedRail]
    unmatched_north: list[UnmatchedRail]


def make_unmatched_rail(instance: RailInstance, reason: str) -> UnmatchedRail:
    """在实例的原有字段上附加未匹配原因。

    Args:
        instance: 图内实例编号、Mask 索引及测量点。
        reason: 本次配对未使用该实例的原因。

    Returns:
        带未匹配原因的实例记录。
    """
    return {
        "instance_id": instance["instance_id"],
        "mask_index": instance["mask_index"],
        "center_xy": instance["center_xy"],
        "reason": reason,
    }
