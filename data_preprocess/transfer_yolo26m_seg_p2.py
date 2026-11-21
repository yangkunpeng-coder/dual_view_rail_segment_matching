"""把三尺度 YOLO26m-seg 权重迁移到新增 P2 分支的四尺度模型。

迁移规则：

1. Backbone 与 Neck（层 0 到 22）按原索引直接复制。
2. Segment26 的检测、分类、Mask 系数与 one2one 分支整体后移一位：
   旧 P3/P4/P5（索引 0/1/2）对应新 P3/P4/P5（索引 1/2/3）。
3. Proto26 的 feat_refine 同样后移一位，新索引 0 没有旧权重，随机初始化。
4. 新 P2 分支用基线 P3 分支权重热启动：两者输入通道同为 256，形状一致。若保持随机
   初始化，P2 尺度在 TAL 分配中永远进不了 top-k，框与掩码分支拿不到梯度。
5. 新增 P2 融合层（C3k2）与 proto.feat_refine.0 没有对应结构，保持随机初始化。
6. 形状不一致的参数跳过并记录，禁止 reshape 强行加载。

用法（项目根目录，或在 PyCharm 中直接运行本文件）：

    python data_preprocess/transfer_yolo26m_seg_p2.py

产物：初始化权重 .pt 与逐层明细 .csv。迁移说明与结果见
doc/readme/transfer_yolo26m_seg_p2_readme.md。
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.tasks import SegmentationModel, yaml_model_load  # noqa: E402

# 新增 P2 融合层数：Upsample、Concat、C3k2，共 3 层。
_ADDED_LAYERS = 3
_BRANCHES = (
    "cv2",
    "cv3",
    "cv4",
    "one2one_cv2",
    "one2one_cv3",
    "one2one_cv4",
)
_BRANCH_PATTERN = re.compile(
    r"^model\.(?P<layer>\d+)\.(?P<branch>"
    + "|".join(_BRANCHES)
    + r")\.(?P<index>\d+)\.(?P<rest>.+)$"
)
_REFINE_PATTERN = re.compile(
    r"^model\.(?P<layer>\d+)\.proto\.feat_refine\.(?P<index>\d+)\.(?P<rest>.+)$"
)
_LAYER_PATTERN = re.compile(r"^model\.(?P<layer>\d+)\.")


@dataclass(frozen=True)
class TransferRow:
    """一层权重的迁移结果。"""

    source_key: str
    target_key: str
    shape: str
    kind: str
    status: str


def _shape(value: torch.Tensor) -> str:
    """返回形如 256x256x3x3 的形状描述。"""
    return "x".join(str(size) for size in value.shape) or "scalar"


def build_target_model(model_yaml: Path, nc: int) -> SegmentationModel:
    """按模型 YAML 构建四尺度实例分割模型。

    模型尺度（n/s/m/l/x）从文件名解析，因此文件名必须保留尺度字母，
    例如 yolo26m-seg-p2.yaml。否则 parse_model 会默认按 'n' 处理并报通道错误。

    Args:
        model_yaml: 含 P2 分支的模型结构 YAML。
        nc: 类别数，训练数据集只有 rail。

    Returns:
        随机初始化的四尺度模型。

    Raises:
        ValueError: 文件名缺少可识别的模型尺度。
    """
    cfg = yaml_model_load(model_yaml)
    scale = cfg.get("scale")
    if not scale or scale not in cfg.get("scales", {}):
        raise ValueError(
            f"{model_yaml.name} 未能从文件名解析模型尺度，"
            "请使用形如 yolo26m-seg-p2.yaml 的文件名"
        )
    cfg["nc"] = nc
    return SegmentationModel(cfg, nc=nc, verbose=False).float()


def load_source_state(weights: Path) -> dict[str, torch.Tensor]:
    """读取基线权重并返回带 model. 前缀的状态字典。

    Args:
        weights: 三尺度基线权重 .pt。

    Returns:
        参数名到张量的映射。

    Raises:
        FileNotFoundError: 权重不存在。
        KeyError: 检查点既没有 ema 也没有 model。
    """
    if not weights.is_file():
        raise FileNotFoundError(f"缺少基线权重 {weights}")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    module = checkpoint.get("ema")
    if module is None:
        module = checkpoint.get("model")
    if module is None:
        raise KeyError(f"{weights} 既没有 ema 也没有 model")
    state = module.float().state_dict()
    if not any(key.startswith("model.") for key in state):
        state = {f"model.{key}": value for key, value in state.items()}
    return state


def _source_key_for(
    target_key: str, target_layer: int, source_layer: int
) -> tuple[str | None, str]:
    """给出目标参数在基线中的对应参数名及迁移方式。"""
    layer = _LAYER_PATTERN.match(target_key)
    first_new_layer = target_layer - _ADDED_LAYERS
    if layer and first_new_layer <= int(layer.group("layer")) < target_layer:
        # 新增 P2 融合层的参数（Upsample/Concat 无参数，C3k2 需要随机初始化）。
        return None, "random_init_p2"
    head_prefix = f"model.{target_layer}."
    branch = _BRANCH_PATTERN.match(target_key)
    if branch and branch.group("layer") == str(target_layer):
        index = int(branch.group("index"))
        if index == 0:
            # 新 P2 分支用基线 P3 分支权重热启动：两者输入通道同为 256，形状一致。
            # 若保持随机初始化，P2 尺度在 TAL 分配中永远进不了 top-k，框与掩码分支得不到梯度。
            fields = [branch.group("branch"), "0", branch.group("rest")]
            return f"model.{source_layer}.{'.'.join(fields)}", "warm_start_p3"
        fields = [branch.group("branch"), str(index - 1), branch.group("rest")]
        return f"model.{source_layer}.{'.'.join(fields)}", "shifted"
    refine = _REFINE_PATTERN.match(target_key)
    if refine and refine.group("layer") == str(target_layer):
        index = int(refine.group("index"))
        if index == 0:
            return None, "random_init_p2"
        return (
            f"model.{source_layer}.proto.feat_refine."
            f"{index - 1}.{refine.group('rest')}",
            "shifted",
        )
    if target_key.startswith(head_prefix):
        return f"model.{source_layer}.{target_key[len(head_prefix):]}", "head_direct"
    return target_key, "direct"


def plan_transfer(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    target_layer: int,
    source_layer: int,
) -> list[TransferRow]:
    """为每个目标参数确定来源、迁移方式和最终状态。"""
    rows: list[TransferRow] = []
    for target_key, target_value in target_state.items():
        source_key, kind = _source_key_for(target_key, target_layer, source_layer)
        shape = _shape(target_value)
        if source_key is None:
            rows.append(TransferRow("", target_key, shape, kind, kind))
            continue
        source_value = source_state.get(source_key)
        if source_value is None:
            rows.append(
                TransferRow(source_key, target_key, shape, kind, "missing_in_source")
            )
            continue
        if source_value.shape != target_value.shape:
            shape = f"{_shape(source_value)} -> {shape}"
            rows.append(
                TransferRow(source_key, target_key, shape, kind, "shape_mismatch")
            )
            continue
        rows.append(TransferRow(source_key, target_key, shape, kind, "transferred"))
    return rows


def apply_transfer(
    model: SegmentationModel,
    source_state: dict[str, torch.Tensor],
    rows: Iterable[TransferRow],
) -> Counter[str]:
    """把可迁移的权重写入模型，返回按状态统计的数量。"""
    merged = dict(model.state_dict())
    counts: Counter[str] = Counter()
    for row in rows:
        counts[row.status] += 1
        if row.status == "transferred":
            merged[row.target_key] = source_state[row.source_key].clone()
    model.load_state_dict(merged, strict=True)
    return counts


def verify_transfer(
    model: SegmentationModel,
    source_state: dict[str, torch.Tensor],
    rows: Iterable[TransferRow],
) -> list[str]:
    """逐层比对已迁移参数，返回与基线不一致的参数名。"""
    target_state = model.state_dict()
    return [
        row.target_key
        for row in rows
        if row.status == "transferred"
        and not torch.equal(target_state[row.target_key], source_state[row.source_key])
    ]


def run_smoke(model: SegmentationModel, imgsz: int) -> None:
    """前向一次零输入，打印四个尺度的输入尺寸与原型尺寸。"""
    records: dict[str, Any] = {}
    head = model.model[-1]

    def _head_hook(_module: Any, inputs: Any) -> None:
        records["head_inputs"] = [tuple(feature.shape) for feature in inputs[0]]

    def _proto_hook(_module: Any, _inputs: Any, output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        records["proto"] = tuple(value.shape)

    handles = [
        head.register_forward_pre_hook(_head_hook),
        head.proto.register_forward_hook(_proto_hook),
    ]
    model.eval()
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, imgsz, imgsz))
    finally:
        for handle in handles:
            handle.remove()
    print(f"输入 {imgsz}：四尺度特征 {records.get('head_inputs')}")
    print(f"Proto 尺寸 {records.get('proto')}")
    print(f"模型 stride {getattr(head, 'stride', None)}")


def _write_csv(path: Path, rows: Iterable[TransferRow]) -> None:
    lines = ["status,kind,source_key,target_key,shape"]
    for row in rows:
        lines.append(
            f"{row.status},{row.kind},{row.source_key},{row.target_key},{row.shape}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    """执行迁移、抽查一致性和前向自检，并写出权重与明细。"""
    parser = ArgumentParser(description="YOLO26m-seg P2 权重迁移")
    parser.add_argument("--source", default="pretrain_pt/yolo26m-seg.pt")
    parser.add_argument("--model", default="config/model/yolo26m-seg-p2.yaml")
    parser.add_argument("--output", default="pretrain_pt/yolo26m-seg-p2-init-warm.pt")
    parser.add_argument("--report", default="pretrain_pt/yolo26m-seg-p2-transfer-warm")
    parser.add_argument("--nc", type=int, default=1)
    parser.add_argument("--smoke-imgsz", type=int, default=640)
    args = parser.parse_args()

    source = (PROJECT_ROOT / args.source).resolve()
    model_yaml = (PROJECT_ROOT / args.model).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    report = (PROJECT_ROOT / args.report).resolve()

    model = build_target_model(model_yaml, args.nc)
    source_state = load_source_state(source)
    target_layer = len(model.model) - 1
    source_layer = target_layer - _ADDED_LAYERS
    rows = plan_transfer(model.state_dict(), source_state, target_layer, source_layer)
    counts = apply_transfer(model, source_state, rows)
    mismatches = verify_transfer(model, source_state, rows)
    print(f"目标头层 {target_layer}，基线头层 {source_layer}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    if mismatches:
        print(f"一致性复核不一致 {len(mismatches)} 项：{mismatches[:5]}")

    torch.save(
        {
            "model": model,
            "epoch": -1,
            "train_args": {},
            "date": datetime.now().isoformat(timespec="seconds"),
        },
        output,
    )
    _write_csv(report.with_suffix(".csv"), rows)
    print(f"已写入 {output}")

    if args.smoke_imgsz:
        run_smoke(model, args.smoke_imgsz)


if __name__ == "__main__":
    main()
