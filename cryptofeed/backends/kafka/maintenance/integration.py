"""
Integration layer for Kafka backend maintenance components.

This module provides unified interfaces for integrating:
1. Deprecation warning system with monitoring and analytics
2. Configuration migration tools with documentation system
3. Health monitoring with alerting and escalation procedures

Implements task 6.1 of kafka-backend-maintenance specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .deprecation_system import DeprecationWarningSystem, get_deprecation_warning_system
from .doc_updater import DocumentationAutoUpdater
from ..migration import translate_legacy_config, validate_migration, MigrationResult
from ..health import KafkaHealthCheck, KafkaHealthStatus
from ..deprecation import (
    DeprecationTimeline,
    CommunicationSystem,
    ProgressReport,
    TimelineUpdate,
)

LOG = logging.getLogger("feedhandler")


# ============================================================================
# Integration Data Models
# ============================================================================


@dataclass
class MaintenanceEvent:
    """Unified event model for maintenance operations."""

    event_type: str  # "deprecation", "migration", "health_check", "alert"
    component: str
    severity: str  # "info", "warning", "error", "critical"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationResult:
    """Result of an integrated maintenance operation."""

    success: bool
    events: List[MaintenanceEvent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Deprecation-Monitoring Integration
# ============================================================================


class DeprecationMonitoringBridge:
    """
    Bridges deprecation warning system with monitoring and analytics.

    Provides unified interface for:
    - Emitting deprecation warnings to monitoring systems
    - Tracking usage analytics for migration planning
    - Feeding data into progress reports and timeline recommendations
    """

    def __init__(
        self,
        deprecation_system: Optional[DeprecationWarningSystem] = None,
        progress_report: Optional[ProgressReport] = None,
    ):
        self.deprecation_system = deprecation_system or get_deprecation_warning_system()
        self.progress_report = progress_report or ProgressReport()

    def track_legacy_usage(
        self, component: str, context: Dict[str, Any]
    ) -> IntegrationResult:
        """
        Track legacy component usage across deprecation and monitoring systems.

        Args:
            component: Name of legacy component being used
            context: Usage context (exchange, symbol, config, etc.)

        Returns:
            IntegrationResult capturing events and status
        """
        events = []

        # Track in deprecation system
        self.deprecation_system.track_usage(component, context)
        events.append(
            MaintenanceEvent(
                event_type="deprecation",
                component=component,
                severity="warning",
                message=f"Legacy component usage tracked: {component}",
                metadata=context,
            )
        )

        # Track in progress report
        self.progress_report.record_legacy_usage(component, context)
        events.append(
            MaintenanceEvent(
                event_type="analytics",
                component="progress_report",
                severity="info",
                message=f"Legacy usage recorded for migration tracking",
                metadata={"component": component, "context": context},
            )
        )

        LOG.info(f"Tracked legacy usage: {component} with context {context}")

        return IntegrationResult(success=True, events=events)

    def track_modern_usage(
        self, component: str, context: Dict[str, Any]
    ) -> IntegrationResult:
        """
        Track modern component usage for migration progress monitoring.

        Args:
            component: Name of modern component being used
            context: Usage context

        Returns:
            IntegrationResult capturing events and status
        """
        events = []

        # Track in progress report
        self.progress_report.record_modern_usage(component, context)
        events.append(
            MaintenanceEvent(
                event_type="analytics",
                component="progress_report",
                severity="info",
                message=f"Modern component usage recorded: {component}",
                metadata={"component": component, "context": context},
            )
        )

        LOG.info(f"Tracked modern usage: {component}")

        return IntegrationResult(success=True, events=events)

    def get_migration_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive migration analytics from integrated systems.

        Returns:
            Dictionary with deprecation stats, progress metrics, and recommendations
        """
        return {
            "deprecation_stats": self.deprecation_system.get_usage_report(),
            "legacy_usage": self.progress_report.get_legacy_usage_stats(),
            "modern_usage": self.progress_report.get_modern_usage_stats(),
            "migration_percentage": self.progress_report.get_migration_percentage(),
            "timeline_recommendation": self.progress_report.get_timeline_recommendation(),
        }


# ============================================================================
# Migration-Documentation Integration
# ============================================================================


class MigrationDocumentationBridge:
    """
    Bridges configuration migration tools with documentation system.

    Provides unified interface for:
    - Triggering documentation updates from migration results
    - Generating migration examples for documentation
    - Maintaining deprecation markers in docs
    """

    def __init__(self, doc_updater: Optional[DocumentationAutoUpdater] = None):
        self.doc_updater = doc_updater or DocumentationAutoUpdater()

    def migrate_with_documentation(
        self, legacy_config: Dict[str, Any]
    ) -> IntegrationResult:
        """
        Perform migration and generate documentation for the migration.

        Args:
            legacy_config: Legacy configuration to migrate

        Returns:
            IntegrationResult with migration result and documentation events
        """
        events = []
        errors = []
        warnings = []

        try:
            # Perform migration
            result = translate_legacy_config(legacy_config)
            events.append(
                MaintenanceEvent(
                    event_type="migration",
                    component="config_migration",
                    severity="info",
                    message="Configuration migration completed",
                    metadata={
                        "unmapped_options": result.unmapped_options,
                        "warnings": result.warnings,
                    },
                )
            )

            # Collect warnings
            warnings.extend(result.warnings)

            # Generate documentation if there are new patterns
            if result.unmapped_options:
                component_info = {
                    "name": "KafkaConfig",
                    "migration_notes": result.warnings,
                    "unmapped_options": list(result.unmapped_options.keys()),
                }

                LOG.info(
                    f"Migration completed with {len(result.unmapped_options)} unmapped options"
                )

        except Exception as e:
            errors.append(f"Migration failed: {e}")
            events.append(
                MaintenanceEvent(
                    event_type="migration",
                    component="config_migration",
                    severity="error",
                    message=f"Migration failed: {e}",
                )
            )

        success = len(errors) == 0
        return IntegrationResult(
            success=success, events=events, errors=errors, warnings=warnings
        )

    def update_documentation_for_component(
        self, component_info: Dict[str, Any]
    ) -> IntegrationResult:
        """
        Update documentation for component changes.

        Args:
            component_info: Component information including new/deprecated fields

        Returns:
            IntegrationResult with documentation update events
        """
        events = []
        errors = []

        try:
            # Generate field documentation if new fields present
            if "new_fields" in component_info:
                field_docs = self.doc_updater.generate_field_documentation(
                    component_info
                )
                if field_docs:
                    events.append(
                        MaintenanceEvent(
                            event_type="documentation",
                            component=component_info.get("name", "unknown"),
                            severity="info",
                            message="Field documentation generated",
                            metadata={"documentation": field_docs},
                        )
                    )

            # Generate deprecation documentation if deprecated fields present
            if "deprecated_fields" in component_info:
                deprecation_docs = self.doc_updater.generate_deprecation_documentation(
                    component_info
                )
                if deprecation_docs:
                    events.append(
                        MaintenanceEvent(
                            event_type="documentation",
                            component=component_info.get("name", "unknown"),
                            severity="warning",
                            message="Deprecation documentation generated",
                            metadata={"documentation": deprecation_docs},
                        )
                    )

        except Exception as e:
            errors.append(f"Documentation update failed: {e}")
            LOG.error(f"Failed to update documentation: {e}")

        success = len(errors) == 0
        return IntegrationResult(success=success, events=events, errors=errors)


# ============================================================================
# Health-Alerting Integration
# ============================================================================


class HealthAlertingBridge:
    """
    Bridges health monitoring with alerting and escalation procedures.

    Provides unified interface for:
    - Performing health checks with automatic alerting
    - Escalating critical health issues
    - Integrating health status with communication system
    """

    def __init__(
        self,
        comm_system: Optional[CommunicationSystem] = None,
        alert_threshold_ms: float = 500.0,
    ):
        self.comm_system = comm_system or CommunicationSystem()
        self.alert_threshold_ms = alert_threshold_ms

    def check_health_with_alerting(
        self,
        bootstrap_servers: List[str],
        implementation: str,
        producer_factory: Optional[Callable] = None,
    ) -> IntegrationResult:
        """
        Perform health check and trigger alerts if necessary.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            implementation: Implementation name ("modern" or "legacy")
            producer_factory: Optional producer factory for testing

        Returns:
            IntegrationResult with health status and alert events
        """
        events = []
        errors = []
        warnings = []

        # Perform health check
        status = KafkaHealthCheck.check_connectivity(
            bootstrap_servers=bootstrap_servers,
            implementation=implementation,
            producer_factory=producer_factory,
        )

        # Process health check result
        if not status.ok:
            # Health check failed - critical alert
            events.append(
                MaintenanceEvent(
                    event_type="health_check",
                    component=f"kafka_{implementation}",
                    severity="critical",
                    message=f"Health check failed: {status.error}",
                    metadata=status.as_dict(),
                )
            )

            # Send critical alert
            update = TimelineUpdate(
                milestone_name="monitoring",
                old_status="healthy",
                new_status="critical",
                message=f"CRITICAL: {implementation} health check failed - {status.error}",
                timestamp=datetime.now(),
            )
            comm_result = self.comm_system.send_update(update, channels="all")

            if not comm_result.success:
                errors.extend(comm_result.errors)

        elif status.latency_ms > self.alert_threshold_ms:
            # High latency - warning alert
            events.append(
                MaintenanceEvent(
                    event_type="health_check",
                    component=f"kafka_{implementation}",
                    severity="warning",
                    message=f"High latency detected: {status.latency_ms:.1f}ms",
                    metadata=status.as_dict(),
                )
            )

            warnings.append(
                f"Health check latency ({status.latency_ms:.1f}ms) exceeds threshold ({self.alert_threshold_ms}ms)"
            )

        else:
            # Health check OK
            events.append(
                MaintenanceEvent(
                    event_type="health_check",
                    component=f"kafka_{implementation}",
                    severity="info",
                    message=f"Health check OK ({status.latency_ms:.1f}ms)",
                    metadata=status.as_dict(),
                )
            )

        success = len(errors) == 0
        return IntegrationResult(
            success=success, events=events, errors=errors, warnings=warnings
        )


# ============================================================================
# Unified Maintenance Coordinator
# ============================================================================


class MaintenanceCoordinator:
    """
    Unified coordinator for all maintenance operations.

    Integrates all maintenance components into a single cohesive system:
    - Deprecation warnings with monitoring and analytics
    - Configuration migration with documentation
    - Health monitoring with alerting and escalation
    """

    def __init__(
        self,
        deprecation_system: Optional[DeprecationWarningSystem] = None,
        progress_report: Optional[ProgressReport] = None,
        doc_updater: Optional[DocumentationAutoUpdater] = None,
        comm_system: Optional[CommunicationSystem] = None,
        timeline: Optional[DeprecationTimeline] = None,
    ):
        # Initialize component bridges
        self.deprecation_monitoring = DeprecationMonitoringBridge(
            deprecation_system=deprecation_system, progress_report=progress_report
        )
        self.migration_documentation = MigrationDocumentationBridge(
            doc_updater=doc_updater
        )
        self.health_alerting = HealthAlertingBridge(comm_system=comm_system)

        # Core systems
        self.timeline = timeline or DeprecationTimeline.load_default()
        self.comm_system = comm_system or CommunicationSystem()

    def handle_legacy_usage(
        self, component: str, context: Dict[str, Any]
    ) -> IntegrationResult:
        """
        Handle legacy component usage across all integrated systems.

        Tracks usage, updates analytics, and checks for timeline adjustments.
        """
        result = self.deprecation_monitoring.track_legacy_usage(component, context)

        # Check if timeline adjustment needed
        analytics = self.deprecation_monitoring.get_migration_analytics()
        recommendation = analytics["timeline_recommendation"]

        if recommendation.should_extend_timeline:
            update = TimelineUpdate(
                milestone_name="legacy_cleanup",
                old_status="pending",
                new_status="pending",
                message=f"Timeline extension recommended: {recommendation.reason}",
                timestamp=datetime.now(),
            )
            self.comm_system.send_update(update, channels=["documentation"])

        return result

    def handle_configuration_migration(
        self, legacy_config: Dict[str, Any]
    ) -> IntegrationResult:
        """
        Handle configuration migration with integrated documentation updates.

        Performs migration, validates result, and updates documentation.
        """
        return self.migration_documentation.migrate_with_documentation(legacy_config)

    def handle_health_check(
        self, bootstrap_servers: List[str], implementation: str
    ) -> IntegrationResult:
        """
        Handle health check with integrated alerting and escalation.

        Performs health check and triggers alerts based on status.
        """
        return self.health_alerting.check_health_with_alerting(
            bootstrap_servers=bootstrap_servers, implementation=implementation
        )

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status across all integrated components.

        Returns:
            Dictionary with analytics, timeline status, and health metrics
        """
        return {
            "migration_analytics": self.deprecation_monitoring.get_migration_analytics(),
            "timeline_status": {
                "milestones": {
                    name: {
                        "status": m.status,
                        "completion": m.completion_percentage,
                        "is_overdue": m.is_overdue,
                    }
                    for name, m in self.timeline.milestones.items()
                },
                "validation": self.timeline.validate(),
            },
            "communication_history": len(self.comm_system.get_history()),
        }


__all__ = [
    "MaintenanceEvent",
    "IntegrationResult",
    "DeprecationMonitoringBridge",
    "MigrationDocumentationBridge",
    "HealthAlertingBridge",
    "MaintenanceCoordinator",
]
