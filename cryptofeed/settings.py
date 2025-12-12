"""
Pydantic Settings wrapper for container / 12-factor configuration.

Precedence (highest last):
1. Code defaults
2. YAML file (config_path, if present)
3. Environment variables `CRYPTOFEED_*` using nested `__` for hierarchy
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogSettings(BaseModel):
    level: str = Field(default="INFO", description="Log level")
    filename: Optional[str] = Field(default=None, description="Optional log file path")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CRYPTOFEED_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="allow",
    )

    # Source of truth for YAML config
    config_path: Optional[str] = Field(
        default=None, description="Path to config YAML (optional)"
    )

    uvloop: bool = Field(default=True, description="Enable uvloop event policy")
    log: LogSettings = Field(default_factory=LogSettings)

    # Loosely typed payloads to avoid rigid schema here
    kafka: Dict[str, Any] = Field(default_factory=dict)
    exchanges: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, file_secret_settings):
        def yaml_settings_source(settings: BaseSettings) -> Dict[str, Any]:
            config_path = settings.__dict__.get("config_path")
            if not config_path:
                return {}
            path = Path(config_path)
            if not path.exists():
                return {}
            data = yaml.safe_load(path.read_text()) or {}
            if not isinstance(data, dict):
                return {}
            return data

        return (
            init_settings,
            yaml_settings_source,
            env_settings,
            file_secret_settings,
        )

    def to_feed_config(self) -> Dict[str, Any]:
        """Convert settings into the dict expected by Config / FeedHandler."""
        cfg: Dict[str, Any] = {
            "uvloop": self.uvloop,
            "log": {
                "level": self.log.level,
                "filename": self.log.filename,
            },
        }
        if self.kafka:
            cfg["kafka"] = self.kafka
        # Merge arbitrary exchanges mapping as-is
        if self.exchanges:
            cfg.update(self.exchanges)
        return cfg
