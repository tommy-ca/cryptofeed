"""Lightweight maintenance stubs for Kafka backend.

The full maintenance toolkit (doc updater, markers, integration bridges) is
not required for runtime operation of the Kafka callbacks. To avoid hard
dependencies during tests, this module provides minimal no-op implementations
of the maintenance interfaces referenced by the Kafka package.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


# Deprecation helpers -----------------------------------------------------


def emit_class_deprecation_warning(old_name: str, new_name: str, *, stacklevel: int = 2) -> None:
    warnings.warn(
        f"{old_name} is deprecated; use {new_name} instead",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def emit_import_deprecation_warning(old_path: str, new_path: str, *, stacklevel: int = 2) -> None:
    warnings.warn(
        f"Importing {old_path} is deprecated; use {new_path} instead",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _resolve_user_stacklevel(default: int = 2) -> int:  # pragma: no cover - trivial
    return default


def track_kafka_usage(event: str, **metadata: Any) -> None:  # pragma: no cover - telemetry stub
    return None


class DeprecationWarningSystem:
    def emit(self, message: str, *, stacklevel: int = 2) -> None:
        warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def get_deprecation_warning_system() -> DeprecationWarningSystem:
    return DeprecationWarningSystem()


# Documentation/maintenance placeholders ---------------------------------


class DocumentationAutoUpdater:
    def run(self) -> None:
        return None


class DeprecationMarkerManager:
    def apply(self) -> None:
        return None


class DocumentationVersionManager:
    def snapshot(self) -> None:
        return None


class CodeExampleValidator:
    def validate(self) -> None:
        return None


# Integration placeholders ------------------------------------------------


@dataclass
class MaintenanceEvent:
    name: str
    payload: dict[str, Any] | None = None


@dataclass
class IntegrationResult:
    success: bool = True
    detail: str | None = None


class DeprecationMonitoringBridge:
    def publish(self, event: MaintenanceEvent) -> IntegrationResult:
        return IntegrationResult()


class MigrationDocumentationBridge:
    def publish(self, event: MaintenanceEvent) -> IntegrationResult:
        return IntegrationResult()


class HealthAlertingBridge:
    def publish(self, event: MaintenanceEvent) -> IntegrationResult:
        return IntegrationResult()


class MaintenanceCoordinator:
    def __init__(self) -> None:
        self.bridges = []

    def register(self, bridge: Any) -> None:
        self.bridges.append(bridge)

    def publish(self, event: MaintenanceEvent) -> IntegrationResult:
        for bridge in self.bridges:
            bridge.publish(event)
        return IntegrationResult()


__all__ = [
    "DocumentationAutoUpdater",
    "DeprecationMarkerManager",
    "DocumentationVersionManager",
    "CodeExampleValidator",
    "DeprecationWarningSystem",
    "get_deprecation_warning_system",
    "emit_class_deprecation_warning",
    "emit_import_deprecation_warning",
    "track_kafka_usage",
    "_resolve_user_stacklevel",
    "MaintenanceEvent",
    "IntegrationResult",
    "DeprecationMonitoringBridge",
    "MigrationDocumentationBridge",
    "HealthAlertingBridge",
    "MaintenanceCoordinator",
]
