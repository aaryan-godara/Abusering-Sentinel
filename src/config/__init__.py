"""Type-safe project configuration.

Configuration is resolved from, in order of precedence:

1. Explicit keyword arguments passed to :class:`Settings`.
2. Process environment variables.
3. A ``.env`` file at the project root.
4. The defaults declared on :class:`Settings`.

Import :func:`get_settings` rather than reading ``os.environ`` directly so that
values such as the random seed have a single source of truth.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""Absolute path to the repository root (the directory holding ``pyproject.toml``)."""


class Environment(str, Enum):
    """Deployment context the code is running in."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Validated project settings."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    random_seed: int = Field(
        default=42,
        ge=0,
        description="Seed for every stochastic component, ensuring reproducible runs.",
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment context.",
    )
    data_directory: Path = Field(
        default=Path("data"),
        description="Dataset root, relative to the project root unless absolute.",
    )

    @field_validator("data_directory")
    @classmethod
    def _resolve_against_project_root(cls, value: Path) -> Path:
        """Make ``data_directory`` absolute so callers are CWD-independent."""
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def raw_data_directory(self) -> Path:
        """Directory for generated, unmodified source datasets."""
        return self.data_directory / "raw"

    @property
    def processed_data_directory(self) -> Path:
        """Directory for cleaned and split datasets."""
        return self.data_directory / "processed"

    @property
    def manifests_directory(self) -> Path:
        """Directory for dataset manifests describing each generation run."""
        return self.data_directory / "manifests"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached project settings.

    The cache keeps configuration consistent within a process. Call
    ``get_settings.cache_clear()`` in tests that mutate the environment.
    """
    return Settings()


__all__ = ["Environment", "PROJECT_ROOT", "Settings", "get_settings"]
