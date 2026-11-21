"""南北钢轨匹配真值的逐条确认标注工具。

用法：

    python -m annotation.rail_match_gt.annotate --split valid
    python -m annotation.rail_match_gt.annotate --pair-id P0084 --snapshot out.png

工具把一对图拆成"一根一步"的顺序：先按南侧从左到右，再走北侧多出来的实例。
每步只在图上高亮当前这一对（其余压淡），底部给出编号选项，按数字作答。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from colorsys import hsv_to_rgb
from PIL import Image, ImageDraw, ImageFont

from annotation.rail_match_gt.prelabel import (
    PairView,
    SideView,
    build_prelabel,
    load_pair,
    load_settings,
)
from data_preprocess.common.config import PROJECT_ROOT
from data_preprocess.common.image import maximize_window_workspace, window_work_area
from utils.match_gt import (
    SCHEMA_VERSION,
    read_pair_annotation,
    validate_pair_annotation,
    write_pair_annotation,
)

_PANEL_WIDTH = 700
_MIN_PANEL_WIDTH = 520
_FRAME_MARGIN = 8
_HEADER_BAR = 62
_NOTE_BAR = 34
_FOOTER_BAR = 96
_TEXT_COLOR = (0, 0, 0)
_OTHER_OPACITY = 0.28
_CURRENT_OPACITY = 0.62
_CURRENT_COLOR = (225, 30, 30)
_DIM_COLOR = (170, 170, 170)
_UNCERTAIN_COLOR = (168, 78, 200)
_DIGIT_LIMIT = 9
_WINDOW = "rail match ground truth"
_DEFAULT_CONFIG = "config/eval/rail_match_gt.yaml"
_FONT_FILE = "msyhbd.ttc"
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    """按像素字号取中文字体，失败时退回 Pillow 默认字体。"""
    size = max(10, int(size))
    cached = _FONT_CACHE.get(size)
    if cached is not None:
        return cached
    try:
        font = ImageFont.truetype(_FONT_FILE, size)
    except OSError:
        font = ImageFont.load_default(size=size)
    _FONT_CACHE[size] = font
    return font


def _text_width(text: str, size: int) -> float:
    """返回文本在给定字号下的像素宽度。"""
    return _font(size).getlength(text)


@dataclass
class Decision:
    """一对图的人工判定，全部以显示顺序索引。"""

    pairs: dict[int, int] = field(default_factory=dict)
    single: dict[str, int] = field(default_factory=dict)
    uncertain: set[str] = field(default_factory=set)

    def copy(self) -> "Decision":
        """返回可独立修改的副本，供撤销使用。"""
        return Decision(dict(self.pairs), dict(self.single), set(self.uncertain))

    def assigned(self) -> set[str]:
        """返回已有状态的实例键，形如 `south:3`。"""
        used = {f"south:{south}" for south in self.pairs}
        used.update(f"north:{north}" for north in self.pairs.values())
        used.update(self.single)
        used.update(self.uncertain)
        return used


@dataclass(frozen=True)
class Step:
    """一个待确认对象：一根南侧实例，或一根没有对应方的北侧实例。"""

    kind: Literal["south", "north"]
    order: int


@dataclass(frozen=True)
class Option:
    """底部的一个编号选项。"""

    label: str
    action: tuple[str, int]


@dataclass
class Panel:
    """已缩放到面板尺寸的一侧图像与 Mask。"""

    key: str
    image: np.ndarray
    masks: np.ndarray
    identifiers: tuple[str, ...]
    centers: tuple[tuple[float, float] | None, ...]
    scale: float
    top: int = 0
    left: int = 0


@dataclass
class Session:
    """当前图对的输入、骨架、判定与步骤进度。"""

    row: dict[str, object]
    pair: PairView
    south_panel: Panel
    north_panel: Panel
    skeleton: dict[int, int]
    decision: Decision
    history: list[Decision]
    path: Path
    steps: list[Step]
    south_visual: tuple[int, ...] = ()
    north_visual: tuple[int, ...] = ()
    override: int | None = None
    recovered: int = 0
    skipped: bool = False
    step_index: int = 0
    saved: bool = True

    @property
    def pair_id(self) -> str:
        """采集事件编号。"""
        return str(self.row["pair_id"])

    @property
    def split(self) -> str:
        """所属划分。"""
        return str(self.row["split"])

    def current(self) -> Step | None:
        """返回当前步骤，走完时返回 None。"""
        if 0 <= self.step_index < len(self.steps):
            return self.steps[self.step_index]
        return None


class Annotator:
    """单窗口逐条确认流程：渲染、按键与保存。"""

    def __init__(
        self, rows: list[dict[str, object]], settings: dict[str, object],
        annotator: str, start: int = 0, mirror_north: bool = True,
    ) -> None:
        self.rows = rows
        self.settings = settings
        self.annotator = annotator
        self.index = start
        self.mirror_north = mirror_north
        self.gt_root = PROJECT_ROOT / str(settings["root"])
        width, _ = window_work_area()
        self.panel_width = max(_MIN_PANEL_WIDTH, (width - 2 * _FRAME_MARGIN) // 2)
        self.font_scale = max(1.0, self.panel_width / _PANEL_WIDTH)
        self.header_bar = round(_HEADER_BAR * self.font_scale)
        self.note_bar = round(_NOTE_BAR * self.font_scale)
        self.footer_bar = round(_FOOTER_BAR * self.font_scale)
        self.display_scale = 1.0
        self.display_pad = (0, 0)
        self.frame_size = (0, 0)
        self._texts: list[tuple[tuple[int, int], str, int, tuple[int, int, int]]] = []
        first_open = self._first_unfinished(start)
        if first_open >= len(self.rows):
            print("清单内的图对都已完成，按 r 可重标某一对", flush=True)
            first_open = max(0, len(self.rows) - 1)
        elif first_open > start:
            print(
                f"跳过已完成的 {first_open - start} 对，从 "
                f"{self.rows[first_open]['pair_id']} 继续",
                flush=True,
            )
        self.index = first_open
        self.session = self._open(self.index)
        self.finished = False
        self.running = True

    def _first_unfinished(self, start: int) -> int:
        """从 start 起找第一个没有完成标记的图对。"""
        for index in range(start, len(self.rows)):
            row = self.rows[index]
            path = (
                self.gt_root / str(row["split"])
                / f"{row['pair_id']}_association.json"
            )
            if not path.is_file():
                return index
            try:
                payload = read_pair_annotation(path)
            except (OSError, ValueError):
                return index
            if payload.get("schema_version") != SCHEMA_VERSION:
                return index
            review = payload.get("review") or {}
            if review.get("status") not in ("done", "skipped"):
                return index
        return len(self.rows)

    # ---------- 载入 ----------

    def _open(self, index: int) -> Session:
        """载入图对，恢复已保存的判定，并把进度定位到第一个未确认的实例。"""
        row = self.rows[index]
        pair = load_pair(row)
        prelabel = build_prelabel(pair, self.settings)
        skeleton = {
            int(entry["south"]): int(entry["north"]) for entry in prelabel["pairs"]
        }
        path = self.gt_root / str(row["split"]) / f"{row['pair_id']}_association.json"
        restored = self._restore(path)
        status = restored[1] if restored else ""
        decision = restored[0] if restored else Decision()
        south_panel = _build_panel(pair.south, self.panel_width)
        north_panel = _build_panel(pair.north, self.panel_width, self.mirror_north)
        south_visual = _visual_order(south_panel)
        north_visual = _visual_order(north_panel)
        steps = _build_steps(decision, skeleton, south_visual, north_visual)
        session = Session(
            row=row, pair=pair,
            south_panel=south_panel, north_panel=north_panel,
            skeleton=skeleton, decision=decision, history=[], path=path,
            steps=steps, south_visual=south_visual, north_visual=north_visual,
            skipped=status == "skipped",
            step_index=_resume_index(steps, decision),
        )
        if restored is not None:
            done = len(decision.pairs) + len(decision.single) + len(
                decision.uncertain
            )
            session.recovered = done
            print(
                f"{session.pair_id} 已恢复 {done} 条旧判定，从第 "
                f"{min(session.step_index + 1, len(steps))}/{len(steps)} 根继续；"
                "按 r 可重置本对",
                flush=True,
            )
        return session

    def _restore(self, path: Path) -> tuple[Decision, str] | None:
        """读取已保存的真值，返回判定与状态；版本不一致时忽略旧文件。"""
        if not path.is_file():
            return None
        payload = read_pair_annotation(path)
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            print(
                f"{path.name} 是旧版本真值（schema {version}），已忽略，"
                "本对从头开始；保存时会覆盖该文件",
                flush=True,
            )
            return None
        orders = {
            side: {
                f"{side[0].upper()}#{int(record['group_id'])}": order
                for order, record in enumerate(payload["instances"][side])
            }
            for side in ("south", "north")
        }
        decision = Decision()
        for match in payload.get("matches", []):
            south = orders["south"].get(str(match["south_instance_id"]))
            north = orders["north"].get(str(match["north_instance_id"]))
            if south is not None and north is not None:
                decision.pairs[south] = north
        for entry in payload.get("unmatched", []):
            side = str(entry["camera"])
            order = orders[side].get(str(entry["instance_id"]))
            if order is not None:
                decision.single[f"{side}:{order}"] = str(entry["reason"])
        for entry in payload.get("ambiguous", []):
            side = str(entry["camera"])
            order = orders[side].get(str(entry["instance_id"]))
            if order is not None:
                decision.uncertain.add(f"{side}:{order}")
        status = str((payload.get("review") or {}).get("status", ""))
        return decision, status

    # ---------- 步骤与选项 ----------

    def _target(self, step: Step) -> int | None:
        """当前步骤要对齐到的对侧序号：微调优先，其次骨架建议，最后首个候选。"""
        if self.session.override is not None:
            return self.session.override
        found = self._suggestion(step)
        if found is not None:
            return found
        candidates = self._candidates(step)
        return candidates[0] if candidates else None

    def _nudge(self, delta: int) -> None:
        """把对侧选择按视觉左右移动一根，红色高亮随之更新。

        南侧序号随屏幕向左增大，北侧序号随屏幕向右增大；北图又被水平镜像，
        因此两边的序号方向都要换算成屏幕方向。
        """
        session = self.session
        step = session.current()
        if step is None:
            return
        moved = _opposite(step.kind)
        total = len(self._panel(moved).identifiers)
        if total == 0:
            return
        if moved == "south":
            delta = -delta
        elif self.mirror_north:
            delta = -delta
        current = self._target(step)
        if current is None:
            base = max(0, min(total - 1, self._anchor(step)))
            current = max(0, min(total - 1, base + delta))
        else:
            current = max(0, min(total - 1, current + delta))
        session.override = current
        print(
            f"{self._panel(moved).identifiers[current]} 待确认",
            flush=True,
        )

    def _options(self, step: Step) -> list[Option]:
        """生成当前步骤的编号选项，第一项永远是红色高亮的那一根。"""
        target = self._target(step)
        candidates = self._candidates(step)
        labels: list[Option] = []
        if target is not None:
            labels.append(Option("是", ("pair", target)))
        else:
            labels.append(Option("没有对侧", ("single", step.order)))
        for other in candidates:
            if other == target:
                continue
            name = self._panel(_opposite(step.kind)).identifiers[other]
            labels.append(Option(name, ("pair", other)))
        if target is not None:
            labels.append(Option("没有对侧", ("single", step.order)))
        labels.append(Option("拿不准", ("uncertain", step.order)))
        return labels[: _DIGIT_LIMIT]

    def _suggestion(self, step: Step) -> int | None:
        """当前步骤的骨架建议；只采纳尚未被占用的对侧实例。"""
        session = self.session
        if step.kind == "south":
            found = session.decision.pairs.get(step.order)
            if found is None:
                found = session.skeleton.get(step.order)
        else:
            found = next(
                (
                    south for south, north in session.decision.pairs.items()
                    if north == step.order
                ),
                None,
            )
            if found is None:
                found = next(
                    (
                        south for south, north in session.skeleton.items()
                        if north == step.order
                    ),
                    None,
                )
        if found is None or found not in self._free_partners(step):
            return None
        return found

    def _free_partners(self, step: Step) -> list[int]:
        """返回当前步骤可用的对侧序号。

        已确认配对的对象不能再选；尚未走到的实例只在"骨架已给它建议"时保留，
        一旦该实例已经答过（配对、单侧或不确定），它就不再被预留。
        """
        session = self.session
        answered = session.decision.assigned()
        if step.kind == "south":
            total = len(session.north_panel.identifiers)
            used = {
                north for south, north in session.decision.pairs.items()
                if south != step.order
            }
            for south, north in session.skeleton.items():
                if south == step.order or f"south:{south}" in answered:
                    continue
                used.add(north)
            return [index for index in range(total) if index not in used]
        total = len(session.south_panel.identifiers)
        used = {
            south for south, north in session.decision.pairs.items()
            if north != step.order
        }
        for south, north in session.skeleton.items():
            if north == step.order or f"south:{south}" in answered:
                continue
            used.add(south)
        return [index for index in range(total) if index not in used]

    def _candidates(self, step: Step) -> list[int]:
        """只给出锚点附近两根以内的空闲候选，避免把远处实例硬配过来。"""
        session = self.session
        free = self._free_partners(step)
        if not free:
            return []
        anchor = self._anchor(step)
        ordered = sorted(free, key=lambda value: (abs(value - anchor), value))
        return [value for value in ordered if abs(value - anchor) <= 2][:3]

    def _anchor(self, step: Step) -> int:
        """估计当前步骤对侧应有的序号。"""
        session = self.session
        offset = _median_offset(session.skeleton)
        if step.kind == "south":
            found = session.skeleton.get(step.order)
            return found if found is not None else step.order + offset
        found = next(
            (
                south for south, north in session.skeleton.items()
                if north == step.order
            ),
            None,
        )
        return found if found is not None else step.order - offset

    def _answer(self, option: Option, *, auto_next: bool = True) -> None:
        """记录当前步骤的答案并前进；本对走完时自动进入下一对。"""
        session = self.session
        step = session.current()
        if step is None:
            return
        self._push()
        action, value = option.action
        decision = session.decision
        session.skipped = False
        if action == "pair":
            if step.kind == "south":
                south, north = step.order, value
            else:
                south, north = value, step.order
            _drop_north(decision, north)
            _drop_south(decision, south)
            decision.pairs[south] = north
            decision.single.pop(f"south:{south}", None)
            decision.single.pop(f"north:{north}", None)
            decision.uncertain.discard(f"south:{south}")
            decision.uncertain.discard(f"north:{north}")
        elif action == "single":
            _clear_step(decision, step)
            decision.single[f"{step.kind}:{step.order}"] = "unknown"
        else:
            _clear_step(decision, step)
            decision.uncertain.add(f"{step.kind}:{step.order}")
        session.saved = False
        self._advance()
        if auto_next and session.current() is None:
            if self.index + 1 < len(self.rows):
                print(f"{session.pair_id} 已走完，进入下一对南北图片", flush=True)
                self._goto(self.index + 1)
            else:
                print(
                    f"{session.pair_id} 已走完，这是本次清单的最后一对", flush=True
                )
                self.save()
                self._mark_finished()

    def _advance(self) -> None:
        """推进到下一个未确认的步骤。

        走完一轮后重新生成步骤队列：改配留下的空位会补到末尾。
        """
        session = self.session
        session.override = None
        session.steps = _build_steps(
            session.decision, session.skeleton,
            session.south_visual, session.north_visual,
        )
        session.step_index = _resume_index(session.steps, session.decision)

    def _push(self) -> None:
        """保存撤销快照。"""
        session = self.session
        session.history.append(session.decision.copy())
        del session.history[:-50]

    def _back(self) -> None:
        """撤销上一步。"""
        session = self.session
        if not session.history:
            return
        session.decision = session.history.pop()
        session.saved = False
        session.override = None
        session.steps = _build_steps(
            session.decision, session.skeleton,
            session.south_visual, session.north_visual,
        )
        session.step_index = _resume_index(session.steps, session.decision)

    def _mark_skipped(self) -> None:
        """把本对标记为无法标注，评测时整对排除，然后进入下一对。"""
        session = self.session
        session.decision = Decision()
        session.override = None
        session.skipped = True
        self.save()
        print(f"{session.pair_id} 已标记为无法标注，评测时整对排除", flush=True)
        if self.index + 1 < len(self.rows):
            self._goto(self.index + 1)
        else:
            self._mark_finished()

    def _reset_pair(self) -> None:
        """清空本对已保存的判定，从第一根重新确认。"""
        session = self.session
        self._push()
        session.decision = Decision()
        session.override = None
        session.skipped = False
        session.saved = False
        session.steps = _build_steps(
            session.decision, session.skeleton,
            session.south_visual, session.north_visual,
        )
        session.step_index = 0
        print(f"{session.pair_id} 已重置，从第一根重新确认", flush=True)

    def _toggle_mirror(self) -> None:
        """切换北图的水平镜像显示，只重建面板，不动判定与步骤。"""
        session = self.session
        self.mirror_north = not self.mirror_north
        session.north_panel = _build_panel(
            session.pair.north, self.panel_width, self.mirror_north
        )
        print(f"北图镜像显示：{'开' if self.mirror_north else '关'}", flush=True)

    def _accept_rest(self) -> None:
        """按骨架接受本对剩余步骤。"""
        session = self.session
        limit = len(session.steps) * 2 + 4
        for _ in range(limit):
            step = session.current()
            if step is None:
                return
            options = self._options(step)
            if not options:
                return
            self._answer(options[0], auto_next=False)

    # ---------- 渲染 ----------

    def _current_pair(self) -> tuple[int | None, int | None]:
        """返回当前步骤要高亮的南侧与北侧序号，没有的一侧为 None。

        没有骨架建议时，用最近的空闲候选来高亮，保证两侧都有一颗红色掩码。
        """
        step = self.session.current()
        if step is None:
            return None, None
        target = self._target(step)
        if step.kind == "south":
            return step.order, target
        return target, step.order

    def render(self) -> np.ndarray:
        """把当前状态绘制成 BGR 画布。"""
        session = self.session
        south, north = session.south_panel, session.north_panel
        area = max(south.image.shape[0], north.image.shape[0])
        height = area + self.header_bar + self.note_bar + self.footer_bar
        canvas = np.full((height, 2 * self.panel_width, 3), 255, dtype=np.uint8)
        south.left, north.left = 0, self.panel_width
        south.top = self.header_bar + (area - south.image.shape[0]) // 2
        north.top = self.header_bar + (area - north.image.shape[0]) // 2
        south_order, north_order = self._current_pair()
        south_colors, north_colors = self._colors()
        for panel, current, colors in (
            (south, south_order, south_colors),
            (north, north_order, north_colors),
        ):
            canvas[
                panel.top:panel.top + panel.image.shape[0],
                panel.left:panel.left + panel.image.shape[1],
            ] = self._panel_pixels(panel, current, colors)
        self._texts = []
        self._draw_bars(canvas)
        self._draw_mirror_note(canvas)
        if self.finished:
            self._draw_finished(canvas)
        if self._texts:
            image = Image.fromarray(canvas)
            drawer = ImageDraw.Draw(image)
            for position, text, size, color in self._texts:
                drawer.text(position, text, fill=color, font=_font(size))
            canvas = np.asarray(image)
        return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    def _draw_mirror_note(self, canvas: np.ndarray) -> None:
        """在图像下方的空白条里标注北图已水平镜像，不覆盖图像内容。"""
        if not self.mirror_north:
            return
        panel = self.session.north_panel
        area = max(
            self.session.south_panel.image.shape[0], panel.image.shape[0]
        )
        text = "注：北图已水平镜像显示（只影响显示，数据与坐标不变）"
        size = round(24 * self.font_scale)
        top = self.header_bar + area + round(4 * self.font_scale)
        self._texts.append(
            ((panel.left + round(12 * self.font_scale), top), text, size, _TEXT_COLOR)
        )

    def _colors(self) -> tuple[dict[int, tuple[int, int, int]], dict[int, tuple[int, int, int]]]:
        """按"已确认优先、否则按骨架"给全部实例分配颜色。

        同一根配对在两侧同色；单侧为灰，不确定为紫。
        """
        session = self.session
        effective = dict(session.skeleton)
        effective.update(session.decision.pairs)
        south_colors: dict[int, tuple[int, int, int]] = {}
        north_colors: dict[int, tuple[int, int, int]] = {}
        for south, north in effective.items():
            color = _pair_color(south + 1)
            south_colors[south] = color
            north_colors[north] = color
        for key in session.decision.single:
            side, _, order = key.partition(":")
            target = south_colors if side == "south" else north_colors
            target[int(order)] = _DIM_COLOR
        for key in session.decision.uncertain:
            side, _, order = key.partition(":")
            target = south_colors if side == "south" else north_colors
            target[int(order)] = _UNCERTAIN_COLOR
        return south_colors, north_colors

    def _panel_pixels(
        self, panel: Panel, current: int | None, colors: dict[int, tuple[int, int, int]]
    ) -> np.ndarray:
        """全部实例按配对颜色填充，当前实例固定用红色并加深，不加描边。"""
        base = panel.image.copy()
        for position in range(len(panel.identifiers)):
            mask = panel.masks[position]
            is_current = current is not None and position == current
            color = _CURRENT_COLOR if is_current else colors.get(position, _DIM_COLOR)
            opacity = _CURRENT_OPACITY if is_current else _OTHER_OPACITY
            base[mask] = (
                (1 - opacity) * base[mask] + opacity * np.asarray(color)
            ).astype(np.uint8)
        return base

    def _anchor_point(self, side: str, order: int) -> tuple[int, int] | None:
        """取实例 50% 截面中心作为落点。"""
        panel = self._panel(side)
        if not 0 <= order < len(panel.centers):
            return None
        center = panel.centers[order]
        if center is None:
            return None
        return panel.left + round(center[0]), panel.top + round(center[1])

    def _panel(self, side: str) -> Panel:
        """返回指定侧的面板。"""
        return (
            self.session.south_panel if side == "south"
            else self.session.north_panel
        )

    def _draw_bars(self, canvas: np.ndarray) -> None:
        """排入标题行与编号选项行，文字统一在 render 里用中文字体绘制。"""
        session = self.session
        step = session.current()
        total = len(session.steps)
        confirmed = len(session.decision.pairs) + len(session.decision.single)
        confirmed += len(session.decision.uncertain)
        if self.finished:
            header = (
                f"{session.pair_id}  [{session.split}]  本次清单已全部标完"
                "   按 Esc 退出"
            )
        elif step is None:
            header = (
                f"{session.pair_id}  [{session.split}]  本对已走完"
                f"   已确认 {confirmed}   按 Space 进入下一对南北图片，r 可重来"
            )
        else:
            index = min(session.step_index + 1, total)
            name = self._panel(step.kind).identifiers[step.order]
            suffix = "" if step.kind == "south" else "（北侧多出）"
            header = (
                f"{session.pair_id}  [{session.split}]  第 {index}/{total} 根"
                f"   {name}{suffix}   已确认 {confirmed}"
            )
        if session.recovered:
            header += f"   （已恢复 {session.recovered} 条旧判定，r 重置）"
        if session.skipped:
            header += "   已标记无法标注（Space 保持，r 重新标注）"
        header_size = round(30 * self.font_scale)
        self._texts.append(
            ((round(12 * self.font_scale), round(10 * self.font_scale)),
             header, header_size, _TEXT_COLOR)
        )
        if step is None:
            return
        options = self._options(step)
        option_size = round(30 * self.font_scale)
        x = round(12 * self.font_scale)
        top = canvas.shape[0] - self.footer_bar + round(10 * self.font_scale)
        for index, option in enumerate(options, start=1):
            text = f"{index} {option.label}"
            self._texts.append(((x, top), text, option_size, _TEXT_COLOR))
            x += round(_text_width(text, option_size)) + round(30 * self.font_scale)
        hint = (
            "a/d 换对侧一根    b 上一步    p 上一对    r 重置本对    m 镜像开关    "
            "Space 整对标为无法标注    Esc 退出"
        )
        self._texts.append(
            ((round(12 * self.font_scale),
              canvas.shape[0] - round(26 * self.font_scale)),
             hint, round(20 * self.font_scale), _TEXT_COLOR)
        )

    # ---------- 保存 ----------

    def _pending_count(self) -> int:
        """返回还没有状态的实例数。"""
        session = self.session
        assigned = session.decision.assigned()
        total = len(session.south_panel.identifiers) + len(
            session.north_panel.identifiers
        )
        return total - len(assigned)

    def _shift_value(self) -> int | None:
        """按骨架估计整数错移。"""
        pairs = self.session.skeleton
        if not pairs:
            return None
        values = sorted(north - south for south, north in pairs.items())
        return values[len(values) // 2]

    def payload(self) -> dict[str, object]:
        """把当前判定组装成逐对 JSON。"""
        session = self.session
        decision = session.decision
        instances = {
            side: [
                _instance_record(session, side, order)
                for order in range(len(self._panel(side).identifiers))
            ]
            for side in ("south", "north")
        }
        matches = []
        for index, (south, north) in enumerate(sorted(decision.pairs.items()), start=1):
            matches.append({
                "gt_rail_id": f"{session.pair_id}_R{index:02d}",
                "south_instance_id": _identifier(session, "south", south),
                "north_instance_id": _identifier(session, "north", north),
                "source_algorithm": "human",
                "source_shared_id": "",
                "note": "",
            })
        unmatched = [
            {
                "camera": key.split(":")[0],
                "instance_id": _identifier(
                    session, key.split(":")[0], int(key.split(":")[1])
                ),
                "reason": reason,
                "note": "",
            }
            for key, reason in sorted(decision.single.items())
        ]
        ambiguous = [
            {
                "camera": key.split(":")[0],
                "instance_id": _identifier(
                    session, key.split(":")[0], int(key.split(":")[1])
                ),
                "candidate_instance_ids": [],
                "reason": "insufficient_geometry",
                "note": "",
            }
            for key in sorted(decision.uncertain)
        ]
        pending = self._pending_count()
        if session.skipped:
            status = "skipped"
            note = "无法标注，评测时整对排除"
        elif pending:
            status = "in_progress"
            note = f"pending={pending}"
        else:
            status = "done"
            note = ""
        review = {
            "annotator": self.annotator,
            "reviewer": "",
            "status": status,
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            "note": note,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "gt_version": str(self.settings.get("gt_version", "v1")),
            "pair_id": session.pair_id,
            "split": session.split,
            "pair_shift": self._shift_value(),
            "source": {
                "south_image": str(session.row["south_image_path"]),
                "north_image": str(session.row["north_image_path"]),
                "south_image_sha256": str(session.row["south_image_sha256"]),
                "north_image_sha256": str(session.row["north_image_sha256"]),
                "south_label_sha256": session.pair.south.label_sha256,
                "north_label_sha256": session.pair.north.label_sha256,
            },
            "instances": instances,
            "matches": matches,
            "unmatched": unmatched,
            "ambiguous": ambiguous,
            "review": review,
        }

    def save(self) -> bool:
        """校验并保存当前图对，返回是否已完整。"""
        session = self.session
        decision = session.decision
        if not session.skipped and not (
            decision.pairs or decision.single or decision.uncertain
        ):
            if not session.path.is_file():
                session.saved = True
                return False
        payload = self.payload()
        expected = {
            side: list(self._panel(side).identifiers)
            for side in ("south", "north")
        }
        errors = validate_pair_annotation(payload, expected)
        blocking = [error for error in errors if not error.endswith("没有状态")]
        if blocking:
            print("校验未通过：" + "；".join(blocking[:4]), flush=True)
            return False
        write_pair_annotation(session.path, payload)
        session.saved = True
        pending = self._pending_count()
        print(
            f"{session.pair_id} 已保存 配对 {len(payload['matches'])} "
            f"单侧 {len(payload['unmatched'])} 不确定 {len(payload['ambiguous'])} "
            f"待处理 {pending}",
            flush=True,
        )
        return not pending

    # ---------- 交互 ----------

    def run(self) -> None:
        """进入交互循环，窗口最大化到桌面工作区。"""
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        try:
            maximize_window_workspace(_WINDOW)
        except RuntimeError as error:
            print(f"窗口未能最大化，按普通窗口显示：{error}", flush=True)
        while self.running:
            cv2.imshow(_WINDOW, self._fit_display(self.render()))
            key = cv2.waitKey(20) & 0xFF
            if key != 255:
                self._on_key(key)
            if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
        self.save()
        cv2.destroyAllWindows()

    def _fit_display(self, canvas: np.ndarray) -> np.ndarray:
        """把画布等比缩放并补边到窗口内容区。"""
        rect = cv2.getWindowImageRect(_WINDOW)
        frame_width, frame_height = int(rect[2]), int(rect[3])
        if frame_width <= 0 or frame_height <= 0:
            return canvas
        self.frame_size = (frame_width, frame_height)
        height, width = canvas.shape[:2]
        scale = min(frame_width / width, frame_height / height)
        if scale <= 0:
            return canvas
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        if target == (width, height):
            resized = canvas
        else:
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            resized = cv2.resize(canvas, target, interpolation=interpolation)
        pad_x = (frame_width - target[0]) // 2
        pad_y = (frame_height - target[1]) // 2
        self.display_scale = scale
        self.display_pad = (pad_x, pad_y)
        return cv2.copyMakeBorder(
            resized, pad_y, frame_height - target[1] - pad_y,
            pad_x, frame_width - target[0] - pad_x,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )

    def _on_key(self, key: int) -> None:
        """处理键盘操作。"""
        session = self.session
        if key == 27:
            self.running = False
        elif key == 13:
            options = self._options(session.current()) if session.current() else []
            if options:
                self._answer(options[0])
        elif ord("1") <= key <= ord("9"):
            index = key - ord("1")
            step = session.current()
            options = self._options(step) if step else []
            if index < len(options):
                self._answer(options[index])
        elif key == 32:
            self._mark_skipped()
        elif key == ord("p"):
            self._goto(self.index - 1)
        elif key == ord("b"):
            self._back()
        elif key == ord("r"):
            self._reset_pair()
        elif key == ord("m"):
            self._toggle_mirror()
        elif key == ord("f"):
            self._accept_rest()
        elif key in (81, ord("a"), ord("A")):
            self._nudge(-1)
        elif key in (83, ord("d"), ord("D")):
            self._nudge(1)
        elif key == ord("s"):
            self.save()

    def _print_end_hint(self) -> None:
        """提示清单已到末尾，并说明如何继续另一批图对。"""
        scope = f"（{self.session.split}，共 {len(self.rows)} 对）"
        print(
            f"本次清单已全部走完{scope}。按 Esc 退出；"
            "想一次标完所有划分，启动时不要加 --split",
            flush=True,
        )

    def _mark_finished(self) -> None:
        """清单全部走完：在界面上显示完成提示。"""
        self.finished = True
        self._print_end_hint()

    def _status_counts(self) -> tuple[int, int, int]:
        """统计本次清单里已完成、无法标注与未完成的对数。"""
        done = skipped = pending = 0
        for row in self.rows:
            path = (
                self.gt_root / str(row["split"])
                / f"{row['pair_id']}_association.json"
            )
            if not path.is_file():
                pending += 1
                continue
            try:
                payload = read_pair_annotation(path)
            except (OSError, ValueError):
                pending += 1
                continue
            status = (payload.get("review") or {}).get("status")
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
            else:
                pending += 1
        return done, skipped, pending

    def _draw_finished(self, canvas: np.ndarray) -> None:
        """在画面中央盖一层完成提示。"""
        done, skipped, pending = self._status_counts()
        scope = "、".join(sorted({str(row["split"]) for row in self.rows}))
        lines = [
            f"{scope} 已全部标完",
            f"已确认 {done} 对　无法标注 {skipped} 对　未完成 {pending} 对",
            "按 Esc 退出；要标另一个划分，请重新运行并指定 --split",
        ]
        sizes = [round(40 * self.font_scale)] * 1 + [
            round(26 * self.font_scale)
        ] * 2
        widths = [_text_width(text, size) for text, size in zip(lines, sizes)]
        box_width = round(max(widths)) + round(80 * self.font_scale)
        line_gap = round(18 * self.font_scale)
        box_height = sum(
            round(size * 1.5) for size in sizes
        ) + line_gap * (len(lines) - 1) + round(60 * self.font_scale)
        left = (canvas.shape[1] - box_width) // 2
        top = (canvas.shape[0] - box_height) // 2
        cv2.rectangle(
            canvas, (left, top), (left + box_width, top + box_height),
            (255, 255, 255), -1,
        )
        cv2.rectangle(
            canvas, (left, top), (left + box_width, top + box_height),
            (30, 30, 30), max(2, round(3 * self.font_scale)),
        )
        cursor = top + round(30 * self.font_scale)
        for text, size in zip(lines, sizes):
            self._texts.append(
                ((left + round(40 * self.font_scale), cursor), text, size,
                 _TEXT_COLOR)
            )
            cursor += round(size * 1.5) + line_gap

    def _goto(self, index: int) -> None:
        """切换图对，切换前保存；越界时给出明确提示。"""
        if not 0 <= index < len(self.rows):
            if index >= len(self.rows):
                self.save()
                self._mark_finished()
            return
        self.save()
        self.index = index
        self.finished = False
        self.session = self._open(index)


def _pair_color(order: int) -> tuple[int, int, int]:
    """按南侧序号生成稳定的配对颜色，两侧同色。"""
    hue = (order * 0.61803398875) % 1
    red, green, blue = hsv_to_rgb(hue, 0.72, 1)
    return round(red * 255), round(green * 255), round(blue * 255)


def _visual_order(panel: Panel) -> tuple[int, ...]:
    """按画面从左到右的横向位置排序，缺失中心的实例排在最后。"""
    def key(order: int) -> float:
        center = panel.centers[order]
        return float("inf") if center is None else center[0]

    return tuple(sorted(range(len(panel.identifiers)), key=key))


def _opposite(kind: str) -> str:
    """返回另一侧的侧别名。"""
    return "north" if kind == "south" else "south"


def _median_offset(skeleton: dict[int, int]) -> int:
    """按骨架估计整数错移，缺少样本时返回 0。"""
    if not skeleton:
        return 0
    values = sorted(north - south for south, north in skeleton.items())
    return values[len(values) // 2]


def _build_steps(
    decision: Decision, skeleton: dict[int, int],
    south_visual: tuple[int, ...], north_visual: tuple[int, ...],
) -> list[Step]:
    """生成"一根一步"的队列：南侧严格按画面从左到右，最后补北侧多出的实例。"""
    answered = {south for south in skeleton if f"south:{south}" in decision.assigned()}
    reserved = {
        north for south, north in skeleton.items() if south not in answered
    }
    claimed = set(decision.pairs.values()) | reserved
    answered_keys = decision.assigned()
    steps = [Step("south", order) for order in south_visual]
    steps.extend(
        Step("north", order)
        for order in north_visual
        if order not in claimed and f"north:{order}" not in answered_keys
    )
    return steps


def _resume_index(steps: list[Step], decision: Decision) -> int:
    """返回第一个还没确认的步骤下标。"""
    assigned = decision.assigned()
    for index, step in enumerate(steps):
        if f"{step.kind}:{step.order}" not in assigned:
            return index
    return len(steps)


def _drop_south(decision: Decision, order: int) -> None:
    """移除该南侧实例已有的配对。"""
    decision.pairs.pop(order, None)


def _drop_north(decision: Decision, order: int) -> None:
    """移除所有指向该北侧实例的配对。"""
    for south, north in list(decision.pairs.items()):
        if north == order:
            decision.pairs.pop(south)


def _clear_step(decision: Decision, step: Step) -> None:
    """清空该步骤涉及的配对与其它状态。"""
    if step.kind == "south":
        _drop_south(decision, step.order)
    else:
        _drop_north(decision, step.order)
    decision.single.pop(f"{step.kind}:{step.order}", None)
    decision.uncertain.discard(f"{step.kind}:{step.order}")


def _build_panel(view: SideView, width: int, mirror: bool = False) -> Panel:
    """把一侧原图与 Mask 缩放到面板尺寸，Mask 顺序与编号、中心一致。

    Args:
        view: 一侧视图。
        width: 面板宽度，单位为像素。
        mirror: 为 True 时水平翻转图像与 Mask（只影响显示坐标）。
    """
    height, source_width = view.image.shape[:2]
    scale = width / source_width
    size = (width, max(1, round(height * scale)))
    image = cv2.resize(view.image, size, interpolation=cv2.INTER_AREA)
    masks = np.zeros((len(view.masks), size[1], size[0]), dtype=np.bool_)
    for position, mask_index in enumerate(view.order):
        masks[position] = cv2.resize(
            view.masks[mask_index].astype(np.uint8), size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.bool_)
    centers = tuple(
        None if center is None else (center[0] * scale, center[1] * scale)
        for center in view.centers
    )
    if mirror:
        image = cv2.flip(image, 1)
        masks = masks[:, :, ::-1]
        last = size[0] - 1
        centers = tuple(
            None if center is None else (last - center[0], center[1])
            for center in centers
        )
    return Panel(view.key, image, np.ascontiguousarray(masks),
                 view.identifiers, centers, scale)


def _instance_record(session: Session, side: str, order: int) -> dict[str, object]:
    """组装实例记录，保留 group_id、shape_index 与 mask_index 的映射。"""
    view = session.pair.south if side == "south" else session.pair.north
    mask_index = view.order[order]
    reference = view.references[mask_index]
    return {
        "instance_id": _identifier(session, side, order),
        "group_id": reference.group_id,
        "shape_index": reference.shape_index,
        "mask_index": int(mask_index),
    }


def _identifier(session: Session, side: str, order: int) -> str:
    """按显示顺序返回实例编号。"""
    panel = session.south_panel if side == "south" else session.north_panel
    return panel.identifiers[order]


def _read_manifest(path: Path) -> list[dict[str, object]]:
    """读取图对清单。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少清单 {path}，请先运行清单脚本")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def annotate(
    split: str | None = None, config_path: str | Path = _DEFAULT_CONFIG,
    pair_id: str | None = None, annotator: str = "annotator",
    snapshot: Path | None = None, mirror_north: bool = True,
) -> None:
    """启动标注工具或输出单对快照。"""
    settings = load_settings(config_path)
    rows = _read_manifest(PROJECT_ROOT / str(settings["root"]) / "manifest.csv")
    if split is not None:
        rows = [row for row in rows if str(row["split"]) == split]
    if not rows:
        raise ValueError(f"清单中没有 split={split} 的图对")
    start = 0
    if pair_id is not None:
        matches = [
            index for index, row in enumerate(rows) if str(row["pair_id"]) == pair_id
        ]
        if not matches:
            raise ValueError(f"清单中没有图对 {pair_id}")
        start = matches[0]
    if snapshot is not None:
        app = Annotator(rows, settings, annotator, start, mirror_north)
        destination = Path(snapshot)
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), app.render())
        print(f"已写出快照 {destination}", flush=True)
        return
    Annotator(rows, settings, annotator, start, mirror_north).run()


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="南北钢轨匹配真值标注工具")
    parser.add_argument("--split", help="只标注 valid 或 test")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="真值配置路径")
    parser.add_argument("--pair-id", help="从指定图对开始")
    parser.add_argument("--annotator", default="annotator", help="标注人")
    parser.add_argument("--snapshot", type=Path, help="只渲染该路径的 PNG 后退出")
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="北图不做水平镜像显示（默认镜像，便于与南图左右对照）",
    )
    return parser.parse_args()


def _main() -> None:
    """命令行入口。"""
    args = _parse_args()
    annotate(
        split=args.split, config_path=args.config, pair_id=args.pair_id,
        annotator=args.annotator, snapshot=args.snapshot,
        mirror_north=not args.no_mirror,
    )


if __name__ == "__main__":
    _main()
