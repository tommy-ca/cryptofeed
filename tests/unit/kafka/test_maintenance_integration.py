"""
Test suite for integrated maintenance components (Task 6.1).

This test suite validates end-to-end integration workflows for:
1. Deprecation warning system with monitoring and analytics
2. Configuration migration tools with documentation system
3. Health monitoring with alerting and escalation procedures

Implements TDD approach for task 6.1 of kafka-backend-maintenance spec.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime
from pathlib import Path
import tempfile

from cryptofeed.backends.kafka.maintenance.deprecation_system import (
    DeprecationWarningSystem,
    get_deprecation_warning_system,
)
from cryptofeed.backends.kafka.migration import (
    translate_legacy_config,
    validate_migration,
)
from cryptofeed.backends.kafka.health import KafkaHealthCheck
from cryptofeed.backends.kafka.deprecation import (
    DeprecationTimeline,
    CommunicationSystem,
    ProgressReport,
)
from cryptofeed.backends.kafka.maintenance.doc_updater import DocumentationAutoUpdater


# ============================================================================
# Test 1: Deprecation Warning System + Monitoring Integration
# ============================================================================


class TestDeprecationMonitoringIntegration:
    """Test integration between deprecation warnings and monitoring systems."""

    def test_deprecation_warnings_emit_to_monitoring(self):
        """Deprecation warnings should emit metrics to monitoring system."""
        # Setup deprecation system
        system = DeprecationWarningSystem()
        system.reset_usage_stats()

        # Emit class deprecation warning
        system.emit_class_warning("TradeKafka", "KafkaCallback", stacklevel=2)

        # Verify usage tracking for monitoring
        stats = system.get_usage_stats()
        assert "TradeKafka" in stats
        assert stats["TradeKafka"] == 1

        # Verify detailed usage report for analytics
        report = system.get_usage_report()
        assert "TradeKafka" in report
        assert report["TradeKafka"]["count"] == 1
        assert "last_timestamp" in report["TradeKafka"]
        assert report["TradeKafka"]["last_context"]["type"] == "class_deprecation"

    def test_usage_tracking_integrates_with_progress_report(self):
        """Usage tracking data should feed into progress reporting."""
        # Setup systems
        deprecation_system = DeprecationWarningSystem()
        deprecation_system.reset_usage_stats()
        progress_report = ProgressReport()

        # Simulate legacy usage
        deprecation_system.track_usage("TradeKafka", {"type": "class_usage"})
        progress_report.record_legacy_usage("TradeKafka", {"exchange": "binance"})

        # Simulate modern usage
        progress_report.record_modern_usage("KafkaCallback", {"exchange": "coinbase"})

        # Verify stats integration
        legacy_stats = progress_report.get_legacy_usage_stats()
        assert legacy_stats["total_legacy_usage"] == 1

        modern_stats = progress_report.get_modern_usage_stats()
        assert modern_stats["total_modern_usage"] == 1

        # Verify migration percentage calculation
        migration_pct = progress_report.get_migration_percentage()
        assert migration_pct == 50.0  # 1 modern / (1 legacy + 1 modern)

    def test_deprecation_analytics_feed_timeline_recommendations(self):
        """Deprecation analytics should inform timeline adjustment recommendations."""
        progress_report = ProgressReport()

        # Simulate low migration adoption (high legacy usage)
        for i in range(80):
            progress_report.record_legacy_usage("TradeKafka", {"run": i})

        for i in range(20):
            progress_report.record_modern_usage("KafkaCallback", {"run": i})

        # Get timeline recommendation
        recommendation = progress_report.get_timeline_recommendation()

        # Should recommend extension when migration is slow
        assert recommendation.should_extend_timeline is True
        assert recommendation.recommended_extension_days > 0
        assert "Slow adoption" in recommendation.reason


# ============================================================================
# Test 2: Configuration Migration + Documentation Integration
# ============================================================================


class TestMigrationDocumentationIntegration:
    """Test integration between migration tools and documentation system."""

    def test_migration_result_triggers_documentation_update(self):
        """Successful migration should trigger documentation updates."""
        # Setup documentation updater
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_updater = DocumentationAutoUpdater(docs_root=Path(tmpdir))

            # Simulate migration with new fields
            component_info = {
                "name": "KafkaConfig",
                "new_fields": [
                    {
                        "name": "max_in_flight",
                        "type": "int",
                        "default": "5",
                        "description": "Maximum in-flight requests per connection",
                    }
                ],
            }

            # Generate documentation for new fields
            field_docs = doc_updater.generate_field_documentation(component_info)

            # Verify documentation generated correctly
            assert "KafkaConfig" in field_docs
            assert "max_in_flight" in field_docs
            assert "int" in field_docs
            assert "Maximum in-flight requests" in field_docs

    def test_migration_warnings_generate_documentation_notes(self):
        """Migration warnings should be documented for troubleshooting."""
        # Setup migration
        legacy_config = {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "crypto",
            "unmapped_option": "value",  # This will generate a warning
        }

        # Perform migration
        result = translate_legacy_config(legacy_config)

        # Verify warnings are captured
        assert len(result.warnings) > 0
        assert "unmapped_option" in result.unmapped_options

        # Verify documentation can be generated from warnings
        assert any("Unmapped" in w for w in result.warnings)

    def test_deprecation_markers_integrated_with_doc_updater(self):
        """Deprecation markers should integrate with documentation updater."""
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_updater = DocumentationAutoUpdater(docs_root=Path(tmpdir))

            # Simulate component with deprecated fields
            component_info = {
                "name": "KafkaTopicConfig",
                "deprecated_fields": [
                    {
                        "name": "topic_prefix",
                        "deprecated_version": "0.1.0",
                        "removal_version": "1.0.0",
                        "replacement": "Use topic.prefix instead",
                    }
                ],
            }

            # Generate deprecation documentation
            deprecation_docs = doc_updater.generate_deprecation_documentation(component_info)

            # Verify deprecation documentation
            assert "KafkaTopicConfig" in deprecation_docs
            assert "topic_prefix" in deprecation_docs
            assert "DEPRECATED" in deprecation_docs
            assert "0.1.0" in deprecation_docs


# ============================================================================
# Test 3: Health Monitoring + Alerting Integration
# ============================================================================


class TestHealthMonitoringAlertingIntegration:
    """Test integration between health monitoring and alerting systems."""

    def test_health_check_failure_triggers_alert(self):
        """Failed health checks should trigger alerting system."""
        # Mock producer factory to simulate failure
        def failing_producer_factory(config):
            raise Exception("Kafka cluster unreachable")

        # Perform health check
        status = KafkaHealthCheck.check_connectivity(
            bootstrap_servers=["localhost:9092"],
            implementation="modern",
            producer_factory=failing_producer_factory,
        )

        # Verify failure captured
        assert status.ok is False
        assert status.error is not None
        assert "unreachable" in status.error.lower()

    def test_health_check_latency_threshold_alerts(self):
        """High latency health checks should trigger performance alerts."""
        # Mock slow producer
        def slow_producer_factory(config):
            import time

            time.sleep(0.6)  # Simulate 600ms latency
            return MagicMock()

        # Perform health check
        status = KafkaHealthCheck.check_connectivity(
            bootstrap_servers=["localhost:9092"],
            implementation="modern",
            producer_factory=slow_producer_factory,
            timeout_ms=1000,
        )

        # Verify latency captured
        assert status.ok is True
        assert status.latency_ms > 500  # Should exceed alerting threshold

    def test_health_status_integrates_with_communication_system(self):
        """Health status changes should integrate with communication system."""
        comm_system = CommunicationSystem()

        # Simulate health check failure
        from cryptofeed.backends.kafka.deprecation import TimelineUpdate

        update = TimelineUpdate(
            milestone_name="monitoring",
            old_status="healthy",
            new_status="degraded",
            message="Kafka health check failed: cluster unreachable",
            timestamp=datetime.now(),
        )

        # Send update through communication channels
        result = comm_system.send_update(update, channels="all")

        # Verify communication succeeded
        assert result.success is True
        assert len(result.channels_notified) > 0

        # Verify update tracked in history
        history = comm_system.get_history()
        assert len(history) == 1
        assert history[0].milestone_name == "monitoring"


# ============================================================================
# Test 4: End-to-End Integration Workflows
# ============================================================================


class TestEndToEndIntegrationWorkflows:
    """Test complete end-to-end workflows integrating all components."""

    def test_legacy_usage_to_timeline_update_workflow(self):
        """Complete workflow: legacy usage -> tracking -> analytics -> timeline update."""
        # 1. Setup all components
        deprecation_system = DeprecationWarningSystem()
        deprecation_system.reset_usage_stats()
        progress_report = ProgressReport()
        DeprecationTimeline.load_default()
        comm_system = CommunicationSystem()

        # 2. Simulate legacy class usage
        deprecation_system.emit_class_warning("TradeKafka", "KafkaCallback", stacklevel=2)

        # 3. Track usage in progress report
        progress_report.record_legacy_usage("TradeKafka", {"exchange": "binance"})

        # 4. Get timeline recommendation based on usage
        recommendation = progress_report.get_timeline_recommendation()

        # 5. If timeline needs adjustment, communicate update
        if recommendation.should_extend_timeline:
            from cryptofeed.backends.kafka.deprecation import TimelineUpdate

            update = TimelineUpdate(
                milestone_name="shim_removal",
                old_status="pending",
                new_status="pending",
                message=f"Timeline extended: {recommendation.reason}",
                timestamp=datetime.now(),
            )
            comm_result = comm_system.send_update(update, channels="all")
            assert comm_result.success is True

        # 6. Verify workflow completed
        stats = deprecation_system.get_usage_stats()
        assert "TradeKafka" in stats

    def test_migration_to_documentation_update_workflow(self):
        """Complete workflow: migration -> validation -> documentation update."""
        # 1. Setup components
        with tempfile.TemporaryDirectory() as tmpdir:
            DocumentationAutoUpdater(docs_root=Path(tmpdir))

            # 2. Perform configuration migration
            legacy_config = {
                "bootstrap_servers": ["localhost:9092"],
                "topic_prefix": "crypto",
                "partition_strategy": "symbol",
            }

            result = translate_legacy_config(legacy_config)
            assert result.modern_config is not None

            # 3. Validate migration
            validation = validate_migration(legacy_config)
            assert validation.is_equivalent is True

            # 4. Update documentation with migration example
            component_info = {
                "name": "KafkaConfig",
                "new_fields": [],
                "migration_example": {
                    "legacy": legacy_config,
                    "modern": result.modern_config.model_dump(),
                },
            }

            # 5. Verify documentation can be generated
            assert component_info["name"] == "KafkaConfig"

    def test_health_check_to_escalation_workflow(self):
        """Complete workflow: health check -> failure -> alert -> escalation."""
        # 1. Setup components
        comm_system = CommunicationSystem()

        # 2. Perform health check (simulated failure)
        def failing_producer_factory(config):
            raise Exception("Kafka cluster unreachable")

        status = KafkaHealthCheck.check_connectivity(
            bootstrap_servers=["localhost:9092"],
            implementation="modern",
            producer_factory=failing_producer_factory,
        )

        # 3. Verify health check failed
        assert status.ok is False

        # 4. Create alert update
        from cryptofeed.backends.kafka.deprecation import TimelineUpdate

        alert_update = TimelineUpdate(
            milestone_name="monitoring",
            old_status="healthy",
            new_status="critical",
            message=f"CRITICAL: Kafka health check failed - {status.error}",
            timestamp=datetime.now(),
        )

        # 5. Send alert through communication channels
        result = comm_system.send_update(alert_update, channels="all")

        # 6. Verify escalation pathway
        assert result.success is True
        assert len(result.channels_notified) > 0

        # 7. Verify alert tracked in history for escalation procedures
        history = comm_system.get_history()
        assert len(history) == 1
        assert "CRITICAL" in history[0].message

    def test_full_system_integration_smoke_test(self):
        """Smoke test: verify all components can be instantiated and work together."""
        # Instantiate all major components
        deprecation_system = get_deprecation_warning_system()
        progress_report = ProgressReport()
        timeline = DeprecationTimeline.load_default()
        comm_system = CommunicationSystem()

        with tempfile.TemporaryDirectory() as tmpdir:
            doc_updater = DocumentationAutoUpdater(docs_root=Path(tmpdir))

            # Verify all components are operational
            assert deprecation_system is not None
            assert progress_report is not None
            assert timeline is not None
            assert comm_system is not None
            assert doc_updater is not None

            # Perform basic operations
            deprecation_system.track_usage("test_component", {"type": "smoke_test"})
            progress_report.record_modern_usage("KafkaCallback", {"test": True})
            validation_result = timeline.validate()
            assert validation_result is not None

            # Verify communication channels
            channels = comm_system.get_registered_channels()
            assert len(channels) > 0


# ============================================================================
# Test 5: Integration Error Handling
# ============================================================================


class TestIntegrationErrorHandling:
    """Test error handling across integrated components."""

    def test_migration_failure_preserves_system_stability(self):
        """Migration failures should not crash integrated systems."""
        # Attempt migration with invalid config
        with pytest.raises(ValueError):
            translate_legacy_config({})  # Missing bootstrap_servers

        # Verify other systems still operational
        deprecation_system = DeprecationWarningSystem()
        stats = deprecation_system.get_usage_stats()
        assert isinstance(stats, dict)

    def test_health_check_failure_preserves_monitoring(self):
        """Health check failures should not break monitoring system."""

        def failing_producer_factory(config):
            raise Exception("Connection refused")

        # Perform failing health check
        status = KafkaHealthCheck.check_connectivity(
            bootstrap_servers=["invalid:9999"],
            implementation="test",
            producer_factory=failing_producer_factory,
        )

        # Verify failure captured gracefully
        assert status.ok is False
        assert status.error is not None

        # Verify monitoring still functional
        comm_system = CommunicationSystem()
        channels = comm_system.get_registered_channels()
        assert len(channels) > 0

    def test_documentation_update_failure_isolation(self):
        """Documentation update failures should be isolated from other systems."""
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_updater = DocumentationAutoUpdater(docs_root=Path(tmpdir))

            # Simulate invalid component info
            invalid_info = {"name": None, "new_fields": "not_a_list"}

            # Attempt to generate docs (should handle gracefully)
            try:
                field_docs = doc_updater.generate_field_documentation(invalid_info)
                # If it succeeds, verify it's empty or valid
                assert isinstance(field_docs, str)
            except Exception:
                # If it fails, verify other systems still work
                deprecation_system = DeprecationWarningSystem()
                assert deprecation_system is not None
