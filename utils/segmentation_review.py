"""保存验证集预测图的路径及高漏检样本对照。"""

import csv
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

from utils.segmentation_predict import prediction_image_path

_MASK_MATCH_IOU = 0.50
_MISSED_SAMPLE_LIMIT = 10
_PANEL_MAX_WIDTH = 960
_HEADER_HEIGHT = 56
_GROUND_TRUTH_FILL = (40, 230, 105, 95)
_GROUND_TRUTH_OUTLINE = (40, 230, 105, 255)
_REVIEW_COLUMNS = (
    "rank", "camera", "image_name", "target_count", "matched_count", "missed_count",
    "recall", "confidence_threshold", "mask_iou_threshold", "comparison",
)


class ImageMaskCounts(NamedTuple):
    """一张图的标注实例数和通过 Mask IoU 门限的配对数。"""

    target_count: int
    matched_count: int

    @property
    def missed_count(self) -> int:
        """返回未获得有效对应的标注实例数。"""
        return self.target_count - self.matched_count


def prediction_path(
    run_dir: Path, image_name: str, *, prediction_dir: Path | None = None
) -> Path:
    """返回带相机前缀的验证图对应的预测图路径。

    Args:
        run_dir: 本次训练输出目录。
        image_name: 以 north_ 或 south_ 开头的验证图片文件名。
        prediction_dir: 叠加图根目录。训练归档时传入临时渲染目录；省略时
            沿用旧实验的 ``run_dir/predictions``。

    Returns:
        按相机分组归档的原名图片路径。

    Raises:
        ValueError: 图片名缺少有效的相机前缀。
    """
    predictions_dir = (
        prediction_dir if prediction_dir is not None else run_dir / "predictions"
    )
    path = prediction_image_path(predictions_dir, image_name)
    if path.parent == predictions_dir:
        raise ValueError(f"预测图像缺少南北相机前缀 {image_name}")
    return path


def save_missed_samples(
    run_dir: Path,
    image_paths: dict[str, Path],
    mask_counts: dict[str, ImageMaskCounts],
    confidence: float,
    *,
    prediction_dir: Path | None = None,
) -> None:
    """按漏检数选出前十张，保存标注和预测对照图及统计表。

    Args:
        run_dir: 已保存验证集预测图的实验目录。
        image_paths: 验证图片文件名及其实际文件路径。
        mask_counts: 本次训练验证或独立评价统计的逐图实例数量。
        confidence: 验证计数采用的置信度门限。
        prediction_dir: 叠加图根目录，省略时沿用旧实验的 run_dir/predictions。

    Raises:
        OSError: 缺少图像、标签或预测图，或无法写入报告。
        ValueError: 样本文件名或 YOLO 分割标签无效。
    """
    ranked = sorted(
        (
            (name, counts)
            for name, counts in mask_counts.items()
            if counts.missed_count > 0
        ),
        key=lambda item: (
            -item[1].missed_count,
            item[1].matched_count / item[1].target_count,
            item[0],
        ),
    )[:_MISSED_SAMPLE_LIMIT]
    if not ranked:
        return

    report_dir = run_dir / "missed_samples"
    report_dir.mkdir()
    rows = []
    for rank, (image_name, counts) in enumerate(ranked, start=1):
        if image_name not in image_paths:
            raise FileNotFoundError(f"漏检样本缺少对应的预测图像 {image_name}")
        prediction = prediction_path(
            run_dir, image_name, prediction_dir=prediction_dir
        )
        comparison = report_dir / f"{rank:02d}_{Path(image_name).stem}.jpg"
        preview = _draw_comparison(image_paths[image_name], prediction, counts)
        preview.save(comparison, quality=90)
        rows.append(
            (
                rank,
                prediction.parent.name,
                prediction.name,
                counts.target_count,
                counts.matched_count,
                counts.missed_count,
                f"{counts.matched_count / counts.target_count:.6f}",
                confidence,
                _MASK_MATCH_IOU,
                comparison.name,
            )
        )
    with (report_dir / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_REVIEW_COLUMNS)
        writer.writerows(rows)


def _draw_comparison(
    image_path: Path, prediction: Path, counts: ImageMaskCounts
) -> Image.Image:
    """左侧绘制 YOLO 标注掩码，右侧复用归档的预测叠加图。"""
    from ultralytics.data.utils import img2label_paths

    with Image.open(image_path) as source:
        width = min(source.width, _PANEL_MAX_WIDTH)
        height = max(1, round(source.height * width / source.width))
        size = width, height
        ground_truth = source.convert("RGB").resize(size)
    with Image.open(prediction) as source:
        predicted = source.convert("RGB").resize(size)

    label_path = Path(img2label_paths([image_path])[0])
    draw = ImageDraw.Draw(ground_truth, "RGBA")
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        coordinates = [float(value) for value in fields[1:]]
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"YOLO 分割多边形坐标无效 {label_path}")
        points = [
            (x * width, y * height)
            for x, y in zip(coordinates[::2], coordinates[1::2], strict=True)
        ]
        draw.polygon(points, fill=_GROUND_TRUTH_FILL, outline=_GROUND_TRUTH_OUTLINE)

    comparison = Image.new("RGB", (2 * width, height + _HEADER_HEIGHT), (24, 24, 24))
    comparison.paste(ground_truth, (0, _HEADER_HEIGHT))
    comparison.paste(predicted, (width, _HEADER_HEIGHT))
    header = ImageDraw.Draw(comparison)
    font = ImageFont.load_default(size=18)
    title = (
        f"{image_path.name} | GT {counts.target_count} | "
        f"Matched {counts.matched_count} | Missed {counts.missed_count}"
    )
    header.text((8, 5), title, fill="white", font=font)
    header.text((8, 29), "Ground truth masks", fill="white", font=font)
    header.text((width + 8, 29), "Predicted masks", fill="white", font=font)
    return comparison
