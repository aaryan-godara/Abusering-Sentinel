"""Tests for the project configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, Environment, Settings, get_settings


def test_defaults_are_applied_without_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no overrides, settings resolve to the documented defaults."""
    for key in ("RANDOM_SEED", "ENVIRONMENT", "DATA_DIRECTORY"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    assert settings.random_seed == 42
    assert settings.environment is Environment.DEVELOPMENT
    assert settings.data_directory == PROJECT_ROOT / "data"


def test_get_settings_loads_and_caches_settings() -> None:
    """The cached accessor returns a usable Settings instance."""
    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings is get_settings()


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Environment variables take precedence over declared defaults."""
    monkeypatch.setenv("RANDOM_SEED", "7")
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("DATA_DIRECTORY", str(tmp_path))

    settings = Settings(_env_file=None)

    assert settings.random_seed == 7
    assert settings.environment is Environment.TESTING
    assert settings.data_directory == tmp_path


def test_derived_data_paths_sit_under_the_data_directory(tmp_path: Path) -> None:
    """Raw, processed, and manifest paths derive from the data directory."""
    settings = Settings(_env_file=None, data_directory=tmp_path)

    assert settings.raw_data_directory == tmp_path / "raw"
    assert settings.processed_data_directory == tmp_path / "processed"
    assert settings.manifests_directory == tmp_path / "manifests"


def test_invalid_values_are_rejected() -> None:
    """Validation rejects a negative seed and an unknown environment name."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, random_seed=-1)

    with pytest.raises(ValueError):
        Settings(_env_file=None, environment="staging")
