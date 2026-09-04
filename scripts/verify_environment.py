"""Report the state of the local development environment.

Run with:

    python scripts/verify_environment.py

Exits with status 1 if a required dependency is missing.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_PACKAGES: tuple[str, ...] = (
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "networkx",
    "faker",
    "pydantic",
    "pydantic-settings",
    "pytest",
)

MINIMUM_PYTHON: tuple[int, int] = (3, 11)


def _in_virtual_environment() -> bool:
    """Return True when running inside a virtual environment."""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _report_interpreter() -> list[str]:
    """Describe the interpreter, platform, and project location."""
    problems: list[str] = []
    current = sys.version_info[:2]

    print("Interpreter")
    print(f"  Python version    : {platform.python_version()}")
    print(f"  Implementation    : {platform.python_implementation()}")
    print(f"  Executable        : {sys.executable}")
    if current < MINIMUM_PYTHON:
        problems.append(
            f"Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]}+ required, "
            f"found {current[0]}.{current[1]}."
        )

    print("\nPlatform")
    print(f"  Operating system  : {platform.system()} {platform.release()}")
    print(f"  Machine           : {platform.machine()}")

    print("\nProject")
    print(f"  Project root      : {PROJECT_ROOT}")

    active = _in_virtual_environment()
    print("\nVirtual environment")
    print(f"  Active            : {'yes' if active else 'no'}")
    print(f"  Prefix            : {sys.prefix}")
    if not active:
        problems.append(
            "Not running inside a virtual environment; "
            "activate .venv before installing packages."
        )

    return problems


def _report_dependencies() -> list[str]:
    """Print the installed version of each required package."""
    problems: list[str] = []
    print("\nDependencies")
    width = max(len(name) for name in REQUIRED_PACKAGES)
    for name in REQUIRED_PACKAGES:
        try:
            print(f"  {name:<{width}} : {version(name)}")
        except PackageNotFoundError:
            print(f"  {name:<{width}} : NOT INSTALLED")
            problems.append(f"Missing dependency: {name}")
    return problems


def _report_configuration() -> list[str]:
    """Load project settings and print the resolved values."""
    print("\nConfiguration")
    try:
        from src.config import get_settings
    except ImportError as error:
        print(f"  Could not import src.config: {error}")
        return [f"Configuration import failed: {error}"]

    settings = get_settings()
    print(f"  Random seed       : {settings.random_seed}")
    print(f"  Environment       : {settings.environment.value}")
    print(f"  Data directory    : {settings.data_directory}")
    return []


def main() -> int:
    """Print the full environment report and return a process exit code."""
    print("AbuseRing Sentinel - environment verification")
    print("=" * 60)

    problems = _report_interpreter()
    problems += _report_dependencies()
    problems += _report_configuration()

    print("\n" + "=" * 60)
    if problems:
        print(f"{len(problems)} issue(s) found:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Environment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
