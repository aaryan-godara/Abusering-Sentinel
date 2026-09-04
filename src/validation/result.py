"""Shared types for the validation subsystem.

A :class:`ValidationIssue` is a single finding; a :class:`ValidationResult`
aggregates findings for one check module. Severity levels:

* ``error``   - a hard integrity violation; the dataset is not usable as-is.
* ``warning`` - something worth flagging (e.g. distributions look too separable)
                but not a correctness failure.
* ``info``    - informational context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["error", "warning", "info"]


@dataclass
class ValidationIssue:
    """One validation finding."""

    check: str
    severity: Severity
    message: str
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class ValidationResult:
    """Findings for one validation module."""

    module: str
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(
        self, check: str, severity: Severity, message: str, **context
    ) -> None:
        self.issues.append(ValidationIssue(check, severity, message, context))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when there are no error-severity issues."""
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


__all__ = ["Severity", "ValidationIssue", "ValidationResult"]
