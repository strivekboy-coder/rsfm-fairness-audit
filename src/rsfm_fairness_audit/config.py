from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when a YAML configuration file cannot be loaded."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "PyYAML is required to read YAML config files. Install the base project dependencies "
            "or pass equivalent CLI options explicitly."
        ) from exc
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a YAML mapping: {config_path}")
    return data
