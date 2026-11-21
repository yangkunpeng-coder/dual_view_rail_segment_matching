"""打包分析报告中的数值文件和配套说明。"""

from fnmatch import fnmatchcase
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

_REPORT_SUFFIXES = {".csv", ".json", ".yaml", ".yml", ".npz"}


def zip_analysis_reports(
    output_dir: Path,
    archive_name: str,
    readme_path: Path,
    *,
    include_all_files: bool = False,
    exclude_patterns: tuple[str, ...] = (),
) -> Path:
    """在报告目录中创建 ZIP，保留 CSV 等数值文件的相对目录。

    Args:
        output_dir: 已生成报告的目录。
        archive_name: ZIP 文件名。
        readme_path: 可随压缩包归档的说明文档。精简交付包中缺失时仅归档结果。
        include_all_files: 为 True 时归档所有符合筛选的文件。
        exclude_patterns: 排除文件的通配模式，匹配以 / 分隔的相对路径。

    Returns:
        创建的 ZIP 路径。

    Raises:
        ValueError: ZIP 文件名无效。
        OSError: 无法读取结果文件或已有说明，或无法创建压缩包。
    """
    name = Path(archive_name)
    if name.name != archive_name or name.suffix.lower() != ".zip":
        raise ValueError(f"需要单独的 ZIP 文件名 {archive_name}")
    archive_path = output_dir / name
    if archive_path.exists():
        raise FileExistsError(f"分析压缩包已存在 {archive_path}")

    pending_path = archive_path.with_suffix(".zip.part")
    archive_stream = pending_path.open("xb")
    try:
        with archive_stream, ZipFile(archive_stream, "w", ZIP_DEFLATED) as archive:
            for path in sorted(output_dir.rglob("*")):
                if path in (pending_path, archive_path) or not path.is_file():
                    continue
                relative_name = path.relative_to(output_dir).as_posix()
                if any(
                    fnmatchcase(relative_name, pattern) for pattern in exclude_patterns
                ):
                    continue
                if include_all_files or path.suffix.lower() in _REPORT_SUFFIXES:
                    archive.write(path, relative_name)
            if readme_path.is_file():
                archive.write(readme_path, readme_path.name)
        pending_path.rename(archive_path)
    finally:
        pending_path.unlink(missing_ok=True)
    return archive_path
