"""Configuration loader with YAML support and caching."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class ConfigLoader:
    """Loads and merges YAML configuration files.

    Provides a unified interface to access all configuration values
    with dot-notation style access and type safety.
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        """Initialize configuration loader.

        Args:
            config_dir: Path to the configuration directory.
                        Defaults to project_root/config.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "config"
        self._config_dir = config_dir
        self._configs: dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all YAML files from the config directory."""
        for yaml_file in self._config_dir.glob("*.yaml"):
            key = yaml_file.stem
            with open(yaml_file, "r", encoding="utf-8") as f:
                self._configs[key] = yaml.safe_load(f) or {}
            logger.debug(f"Loaded config: {key}")

    def get(self, section: str, key: str | None = None, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            section: Top-level config file name (without .yaml).
            key: Dot-separated key path within the section.
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        config = self._configs.get(section, {})
        if key is None:
            return config

        keys = key.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def get_path(self, key: str) -> Path:
        """Get a path from paths config, resolved relative to project root.

        Args:
            key: Dot-separated key path in paths.yaml.

        Returns:
            Resolved Path object.
        """
        project_root = Path(__file__).parent.parent.parent
        raw_path = self.get("paths", key, "")
        return project_root / raw_path

    @property
    def weights(self) -> dict[str, Any]:
        """Get weights configuration."""
        return self._configs.get("weights", {})

    @property
    def ranking(self) -> dict[str, Any]:
        """Get ranking configuration."""
        return self._configs.get("ranking", {})

    @property
    def features(self) -> dict[str, Any]:
        """Get features configuration."""
        return self._configs.get("features", {})

    @property
    def paths(self) -> dict[str, Any]:
        """Get paths configuration."""
        return self._configs.get("paths", {})


@lru_cache(maxsize=1)
def get_config(config_dir: str | None = None) -> ConfigLoader:
    """Get or create the singleton ConfigLoader instance.

    Args:
        config_dir: Optional path to config directory.

    Returns:
        ConfigLoader instance.
    """
    path = Path(config_dir) if config_dir else None
    return ConfigLoader(config_dir=path)
