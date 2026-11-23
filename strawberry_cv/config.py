"""配置加载与路径解析。

配置文件使用 JSON（标准库即可读取，不依赖 PyYAML）；如果安装了 PyYAML，
也可以直接传 .yaml / .yml 文件。

路径规则：
- 相对路径相对于项目根目录（本包的上一级，即 D:\\cv课设）；
- 绝对路径原样使用；
- 训练/导出产物默认写到 ASCII 工作目录，避免中文路径在 Ultralytics 下出问题。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    """配置缺项或类型错误。"""


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ConfigError(
                "读取 YAML 配置需要 PyYAML：pip install pyyaml；"
                "或改用 configs/base.json"
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise ConfigError(f"不支持的配置文件类型：{path.suffix}")
    if not isinstance(data, dict):
        raise ConfigError("配置文件顶层必须是对象/字典")
    return data


class Config:
    """轻量配置对象，支持点号取值。"""

    def __init__(self, data: dict[str, Any], root: Path | str = PROJECT_ROOT):
        self._data = data
        self.root = Path(root).resolve()

    # ---------- 构造 ----------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            path = PROJECT_ROOT / "configs" / "base.json"
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise ConfigError(f"配置文件不存在：{path}")
        return cls(_load_mapping(path))

    # ---------- 取值 ----------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, None)
        if value is None:
            raise ConfigError(f"配置缺项：{dotted}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))

    # ---------- 路径 ----------
    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def path(self, dotted: str, default: str | Path | None = None) -> Path:
        value = self.get(dotted, default)
        if value is None:
            raise ConfigError(f"配置缺项：{dotted}")
        return self.resolve_path(value)

    def derived(self, name: str) -> Path:
        """派生工作路径。"""
        work = self.path("paths.work_dir")
        mapping = {
            "yolo_1cls": work / "yolo_detection_1cls",
            "yolo_3cls": work / "yolo_detection_3cls",
            "roi": work / "roi_classification",
            "splits": work / "splits",
            "runs": work / "runs",
            "inference": work / "inference",
            "reports": work / "reports",
        }
        if name not in mapping:
            raise KeyError(f"未知派生路径：{name}，可选：{sorted(mapping)}")
        return mapping[name]

    # ---------- 校验 ----------
    def validate(self, require_data: bool = False) -> list[str]:
        """返回问题列表；require_data=True 时检查数据路径是否存在。"""
        issues: list[str] = []
        for key in ("paths.images", "paths.labels", "paths.scene_manifest"):
            try:
                p = self.path(key)
            except ConfigError as exc:
                issues.append(str(exc))
                continue
            if require_data and not p.exists():
                issues.append(f"路径不存在：{key} -> {p}")

        ratios = self.get("split.ratios")
        if not isinstance(ratios, (list, tuple)) or len(ratios) != 3:
            issues.append("split.ratios 必须是长度为 3 的数组")
        elif abs(sum(float(x) for x in ratios) - 1.0) > 1e-6:
            issues.append("split.ratios 之和必须为 1")

        num_classes = self.get("classifier.num_classes", len(self.class_names()))
        if int(num_classes) != len(self.class_names()):
            issues.append(
                f"classifier.num_classes={num_classes} 与类别数 {len(self.class_names())} 不一致"
            )

        if int(self.get("detector.num_classes", 1)) not in (1, 3):
            issues.append("detector.num_classes 只支持 1 或 3")
        return issues

    def class_names(self) -> tuple[str, ...]:
        from .constants import CLASS_NAMES

        return tuple(self.get("classifier.class_names", CLASS_NAMES))

    def describe(self) -> str:
        lines = [
            f"project root : {self.root}",
            f"data images  : {self.path('paths.images', '?')}",
            f"data labels  : {self.path('paths.labels', '?')}",
            f"work dir     : {self.path('paths.work_dir', '?')}",
        ]
        return "\n".join(lines)


def resolve_device(value: str | int | None = None) -> str:
    """把 auto / None / 0 / cpu 统一成 Ultralytics 可用的设备字符串。"""
    if value in (None, "", "auto"):
        try:
            import torch  # type: ignore

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return str(value)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
