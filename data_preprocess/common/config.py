"""读取钢轨数据集的项目配置。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config"
# 配置按用途分到一级子目录后，旧的平铺路径仍要能找到同名文件。
_CONFIG_SUBDIRS = ("data", "model", "train", "predict", "eval", "experiments")


def config_path(name: str | Path) -> Path:
    """解析配置文件名或相对路径，兼容旧版平铺结构。

    查找顺序依次为：

    1. 绝对路径，或相对项目根目录的完整路径；
    2. `config/<文件名>`；
    3. `config/<一级子目录>/<文件名>`。

    Args:
        name: 配置文件名（例如 ``clop_seg.yaml``）或相对项目根目录的路径
            （例如 ``config/data/clop_seg.yaml``）。

    Returns:
        实际存在的配置文件绝对路径。

    Raises:
        FileNotFoundError: 上述位置都找不到该文件。
    """
    value = Path(name)
    basename = value.name
    candidates: list[Path] = []
    if value.is_absolute():
        candidates.append(value)
    else:
        candidates.append(PROJECT_ROOT / value)
        if len(value.parts) == 1:
            candidates.append(CONFIG_ROOT / value)
    if basename:
        candidates.append(CONFIG_ROOT / basename)
        candidates.extend(
            CONFIG_ROOT / subdir / basename for subdir in _CONFIG_SUBDIRS
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"找不到配置文件 {name}（已查找 config/ 和 {', '.join(_CONFIG_SUBDIRS)}）"
    )


def load_yaml_mapping(config_path: Path) -> dict[str, object]:
    """读取 YAML 配置，并要求顶层为字符串键的映射。

    Args:
        config_path: YAML 文件路径。

    Returns:
        YAML 配置字典。

    Raises:
        OSError: 无法读取配置文件。
        ValueError: YAML 格式或顶层类型无效。
    """
    import yaml

    with config_path.open(encoding="utf-8-sig") as config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as error:
            raise ValueError(f"无法解析配置文件 {config_path}") from error
    if not isinstance(config, dict) or not all(
        isinstance(key, str) for key in config
    ):
        raise ValueError(f"{config_path} 必须使用字符串作为顶层配置键")
    return config


def load_dataset_relative_path() -> Path:
    """读取配置中相对于项目根目录的数据集路径。

    Returns:
        配置的相对路径。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: YAML 无法解析或 dataset_path 不是非空相对路径。
    """
    source = config_path("rail_data.yaml")
    dataset_path = load_yaml_mapping(source).get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        raise ValueError(f"{source} 的 dataset_path 必须是非空字符串")

    relative_path = Path(dataset_path)
    if relative_path.anchor:
        raise ValueError(f"{source} 的 dataset_path 必须是相对路径")
    return relative_path
