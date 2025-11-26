"""
Documentation maintenance and automation tools for Kafka backend.

This package provides automated documentation maintenance capabilities:
- doc_updater: Automated documentation updates for component changes
- deprecation_markers: Deprecation marker management in documentation
- doc_versioning: Documentation versioning and rollback capabilities
- code_validator: Code example validation for accuracy
- deprecation_system: Centralized deprecation warning and tracking
- integration: Unified integration layer for all maintenance components

Requirements: 3.3, 6.1, 7.2
"""

from cryptofeed.backends.kafka.maintenance.doc_updater import DocumentationAutoUpdater
from cryptofeed.backends.kafka.maintenance.deprecation_markers import DeprecationMarkerManager
from cryptofeed.backends.kafka.maintenance.doc_versioning import DocumentationVersionManager
from cryptofeed.backends.kafka.maintenance.code_validator import CodeExampleValidator
from cryptofeed.backends.kafka.maintenance.deprecation_system import (
    DeprecationWarningSystem,
    get_deprecation_warning_system,
    emit_class_deprecation_warning,
    emit_import_deprecation_warning,
    track_kafka_usage,
    _resolve_user_stacklevel,
)
from cryptofeed.backends.kafka.maintenance.integration import (
    MaintenanceEvent,
    IntegrationResult,
    DeprecationMonitoringBridge,
    MigrationDocumentationBridge,
    HealthAlertingBridge,
    MaintenanceCoordinator,
)

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
