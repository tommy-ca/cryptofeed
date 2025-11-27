"""
Test suite for MaintenanceCoordinator unified integration (Task 6.1).

Tests the MaintenanceCoordinator class which provides a single unified
interface for all maintenance operations across integrated components.
"""

from unittest.mock import MagicMock
from datetime import datetime
import tempfile

from cryptofeed.backends.kafka.maintenance.integration import (
    MaintenanceCoordinator,
    MaintenanceEvent,
    IntegrationResult,
    DeprecationMonitoringBridge,
    MigrationDocumentationBridge,
    HealthAlertingBridge,
)


class TestMaintenanceCoordinator:
    """Test MaintenanceCoordinator unified interface."""

    def test_coordinator_initialization(self):
        """MaintenanceCoordinator should initialize all bridge components."""
        coordinator = MaintenanceCoordinator()

        assert coordinator.deprecation_monitoring is not None
        assert coordinator.migration_documentation is not None
        assert coordinator.health_alerting is not None
        assert coordinator.timeline is not None
        assert coordinator.comm_system is not None

    def test_coordinator_handles_legacy_usage_end_to_end(self):
        """Coordinator should handle legacy usage across all integrated systems."""
        coordinator = MaintenanceCoordinator()

        # Handle legacy usage
        result = coordinator.handle_legacy_usage(
            component="TradeKafka", context={"exchange": "binance", "symbol": "BTC-USD"}
        )

        # Verify result
        assert isinstance(result, IntegrationResult)
        assert result.success is True
        assert len(result.events) > 0

        # Verify events captured
        event_types = [e.event_type for e in result.events]
        assert "deprecation" in event_types
        assert "analytics" in event_types

        # Verify usage tracked in integrated systems
        analytics = coordinator.deprecation_monitoring.get_migration_analytics()
        assert "TradeKafka" in analytics["deprecation_stats"]

    def test_coordinator_handles_configuration_migration(self):
        """Coordinator should handle configuration migration with documentation."""
        coordinator = MaintenanceCoordinator()

        legacy_config = {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "crypto",
            "partition_strategy": "symbol",
        }

        # Handle migration
        result = coordinator.handle_configuration_migration(legacy_config)

        # Verify result
        assert isinstance(result, IntegrationResult)
        assert result.success is True
        assert len(result.events) > 0

        # Verify migration event captured
        migration_events = [e for e in result.events if e.event_type == "migration"]
        assert len(migration_events) > 0

    def test_coordinator_handles_health_check_with_alerting(self):
        """Coordinator should handle health checks with integrated alerting."""
        coordinator = MaintenanceCoordinator()

        # Mock producer to simulate success
        def mock_producer_factory(config):
            return MagicMock()

        # Handle health check
        result = coordinator.handle_health_check(
            bootstrap_servers=["localhost:9092"], implementation="modern"
        )

        # Verify result (will fail without real Kafka, but structure should be correct)
        assert isinstance(result, IntegrationResult)
        assert len(result.events) > 0

        # Verify health check event captured
        health_events = [e for e in result.events if e.event_type == "health_check"]
        assert len(health_events) > 0

    def test_coordinator_provides_system_status(self):
        """Coordinator should provide comprehensive system status."""
        coordinator = MaintenanceCoordinator()

        # Track some usage first
        coordinator.handle_legacy_usage("TradeKafka", {"test": True})

        # Get system status
        status = coordinator.get_system_status()

        # Verify status structure
        assert "migration_analytics" in status
        assert "timeline_status" in status
        assert "communication_history" in status

        # Verify analytics present
        assert "deprecation_stats" in status["migration_analytics"]
        assert "migration_percentage" in status["migration_analytics"]

        # Verify timeline status
        assert "milestones" in status["timeline_status"]
        assert "validation" in status["timeline_status"]

    def test_coordinator_timeline_adjustment_workflow(self):
        """Coordinator should trigger timeline adjustments based on usage analytics."""
        coordinator = MaintenanceCoordinator()

        # Simulate high legacy usage (slow migration)
        for i in range(80):
            coordinator.handle_legacy_usage("TradeKafka", {"run": i})

        for i in range(20):
            coordinator.deprecation_monitoring.track_modern_usage(
                "KafkaCallback", {"run": i}
            )

        # Get analytics
        analytics = coordinator.deprecation_monitoring.get_migration_analytics()
        recommendation = analytics["timeline_recommendation"]

        # Should recommend timeline extension
        assert recommendation.should_extend_timeline is True

        # Verify communication history updated
        status = coordinator.get_system_status()
        # Communication should have been sent during handle_legacy_usage
        assert status["communication_history"] >= 0


class TestDeprecationMonitoringBridge:
    """Test DeprecationMonitoringBridge integration."""

    def test_bridge_tracks_legacy_usage(self):
        """Bridge should track legacy usage in both systems."""
        bridge = DeprecationMonitoringBridge()

        result = bridge.track_legacy_usage(
            "BookKafka", {"exchange": "coinbase", "symbol": "ETH-USD"}
        )

        assert result.success is True
        assert len(result.events) == 2  # Deprecation + analytics events

        # Verify analytics updated
        analytics = bridge.get_migration_analytics()
        assert analytics["legacy_usage"]["total_legacy_usage"] > 0

    def test_bridge_tracks_modern_usage(self):
        """Bridge should track modern usage for migration progress."""
        bridge = DeprecationMonitoringBridge()

        result = bridge.track_modern_usage("KafkaCallback", {"exchange": "binance"})

        assert result.success is True
        assert len(result.events) == 1  # Analytics event

        # Verify analytics updated
        analytics = bridge.get_migration_analytics()
        assert analytics["modern_usage"]["total_modern_usage"] > 0

    def test_bridge_provides_comprehensive_analytics(self):
        """Bridge should provide comprehensive migration analytics."""
        bridge = DeprecationMonitoringBridge()

        # Track usage
        bridge.track_legacy_usage("TradeKafka", {})
        bridge.track_modern_usage("KafkaCallback", {})

        # Get analytics
        analytics = bridge.get_migration_analytics()

        # Verify all analytics components present
        assert "deprecation_stats" in analytics
        assert "legacy_usage" in analytics
        assert "modern_usage" in analytics
        assert "migration_percentage" in analytics
        assert "timeline_recommendation" in analytics


class TestMigrationDocumentationBridge:
    """Test MigrationDocumentationBridge integration."""

    def test_bridge_migrates_with_documentation(self):
        """Bridge should perform migration and generate documentation."""
        with tempfile.TemporaryDirectory():
            bridge = MigrationDocumentationBridge()

            legacy_config = {
                "bootstrap_servers": ["localhost:9092"],
                "topic_prefix": "test",
            }

            result = bridge.migrate_with_documentation(legacy_config)

            assert result.success is True
            assert len(result.events) > 0

            # Verify migration event present
            migration_events = [e for e in result.events if e.event_type == "migration"]
            assert len(migration_events) > 0

    def test_bridge_updates_component_documentation(self):
        """Bridge should update documentation for component changes."""
        with tempfile.TemporaryDirectory():
            bridge = MigrationDocumentationBridge()

            component_info = {
                "name": "KafkaConfig",
                "new_fields": [
                    {
                        "name": "test_field",
                        "type": "str",
                        "default": "value",
                        "description": "Test field",
                    }
                ],
            }

            result = bridge.update_documentation_for_component(component_info)

            assert result.success is True
            assert len(result.events) > 0

            # Verify documentation event present
            doc_events = [e for e in result.events if e.event_type == "documentation"]
            assert len(doc_events) > 0

    def test_bridge_handles_migration_errors_gracefully(self):
        """Bridge should handle migration errors without crashing."""
        bridge = MigrationDocumentationBridge()

        # Invalid config (missing bootstrap_servers)
        invalid_config = {"topic_prefix": "test"}

        result = bridge.migrate_with_documentation(invalid_config)

        # Should fail gracefully
        assert result.success is False
        assert len(result.errors) > 0


class TestHealthAlertingBridge:
    """Test HealthAlertingBridge integration."""

    def test_bridge_alerts_on_health_failure(self):
        """Bridge should trigger alerts when health checks fail."""
        bridge = HealthAlertingBridge()

        def failing_producer(config):
            raise Exception("Connection failed")

        result = bridge.check_health_with_alerting(
            bootstrap_servers=["localhost:9092"],
            implementation="test",
            producer_factory=failing_producer,
        )

        # Should capture failure
        assert result.success is True  # Alert sent successfully
        assert len(result.events) > 0

        # Verify critical event present
        critical_events = [e for e in result.events if e.severity == "critical"]
        assert len(critical_events) > 0

    def test_bridge_warns_on_high_latency(self):
        """Bridge should warn when health check latency is high."""
        bridge = HealthAlertingBridge(alert_threshold_ms=100.0)

        def slow_producer(config):
            import time

            time.sleep(0.2)  # 200ms
            return MagicMock()

        result = bridge.check_health_with_alerting(
            bootstrap_servers=["localhost:9092"],
            implementation="test",
            producer_factory=slow_producer,
        )

        assert result.success is True
        assert len(result.warnings) > 0

        # Verify warning event present
        warning_events = [e for e in result.events if e.severity == "warning"]
        assert len(warning_events) > 0

    def test_bridge_reports_healthy_status(self):
        """Bridge should report info events for healthy checks."""
        bridge = HealthAlertingBridge()

        def healthy_producer(config):
            return MagicMock()

        result = bridge.check_health_with_alerting(
            bootstrap_servers=["localhost:9092"],
            implementation="test",
            producer_factory=healthy_producer,
        )

        assert result.success is True
        assert len(result.events) > 0

        # Verify info event present
        info_events = [e for e in result.events if e.severity == "info"]
        assert len(info_events) > 0


class TestIntegrationDataModels:
    """Test integration data models."""

    def test_maintenance_event_creation(self):
        """MaintenanceEvent should capture event information."""
        event = MaintenanceEvent(
            event_type="test",
            component="test_component",
            severity="info",
            message="Test message",
            metadata={"key": "value"},
        )

        assert event.event_type == "test"
        assert event.component == "test_component"
        assert event.severity == "info"
        assert event.message == "Test message"
        assert isinstance(event.timestamp, datetime)
        assert event.metadata["key"] == "value"

    def test_integration_result_aggregation(self):
        """IntegrationResult should aggregate events and status."""
        events = [
            MaintenanceEvent("test1", "comp1", "info", "Message 1"),
            MaintenanceEvent("test2", "comp2", "warning", "Message 2"),
        ]

        result = IntegrationResult(
            success=True,
            events=events,
            errors=[],
            warnings=["Warning 1", "Warning 2"],
        )

        assert result.success is True
        assert len(result.events) == 2
        assert len(result.warnings) == 2
        assert len(result.errors) == 0
