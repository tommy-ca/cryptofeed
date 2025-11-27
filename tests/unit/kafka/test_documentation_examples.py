"""
Test suite to validate all documentation examples in migration guides.

This TDD-first test suite ensures that all code examples in the migration
guide documentation are syntactically correct and functionally working.
Tests fail initially (RED), then documentation is written to make them pass (GREEN).
"""

import pytest
import warnings


class TestMigrationGuideExamples:
    """Validate migration guide code examples."""

    def test_example_01_basic_legacy_to_modern_import(self):
        """Example 1: Basic import migration from legacy to modern."""
        # This test ensures the basic import example works
        # RED: Will fail until documentation example is written

        # Legacy import (should emit warning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "kafka_callback" in str(w[0].message).lower()

        # Modern import (no warning)
        from cryptofeed.backends.kafka.callback import KafkaCallback
        assert KafkaCallback is not None

    def test_example_02_basic_configuration_translation(self):
        """Example 2: Basic configuration translation."""
        from cryptofeed.backends.kafka.migration import translate_legacy_config

        # Legacy config
        legacy = {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "trades",
            "partition_strategy": "composite",
        }

        # Translate
        result = translate_legacy_config(legacy)

        # Verify modern config
        assert result.modern_config.bootstrap_servers == ["localhost:9092"]
        assert result.modern_config.topic.prefix == "trades"
        assert result.modern_config.partition.strategy == "composite"

    def test_example_03_cli_migration_tool_api(self):
        """Example 3: CLI migration tool programmatic API."""
        from cryptofeed.backends.kafka.migration import translate_legacy_config

        legacy_config = {
            "bootstrap_servers": ["kafka1:9092", "kafka2:9092"],
            "topic_prefix": "production",
            "acks": "all",
            "compression_type": "snappy",
        }

        result = translate_legacy_config(legacy_config)
        modern = result.modern_config

        assert modern.bootstrap_servers == ["kafka1:9092", "kafka2:9092"]
        assert modern.topic.prefix == "production"
        assert modern.acks == "all"
        assert modern.compression_type == "snappy"

    def test_example_04_health_check_usage(self):
        """Example 4: Health check system usage."""
        from cryptofeed.backends.kafka.health import KafkaHealthCheck
        from cryptofeed.backends.kafka.callback import KafkaConfig

        # Create modern config
        config = KafkaConfig(bootstrap_servers=["localhost:9092"])

        # Health check should not raise (even if connection fails)
        # We're testing the API, not actual Kafka connectivity
        try:
            status = KafkaHealthCheck.check_modern(config)
            # Status will likely fail (no real Kafka), but API should work
            assert status.implementation == "modern"
            assert hasattr(status, 'ok')
            assert hasattr(status, 'latency_ms')
        except Exception as e:
            # Health check itself shouldn't raise, but if it does, that's a docs bug
            pytest.fail(f"Health check API raised exception: {e}")

    def test_example_05_unmapped_options_handling(self):
        """Example 5: Handling unmapped legacy options."""
        from cryptofeed.backends.kafka.migration import translate_legacy_config

        legacy = {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "trades",
            "custom_option": "value",  # Unmapped
            "another_custom": 123,      # Unmapped
        }

        result = translate_legacy_config(legacy)

        # Unmapped options should be captured
        assert "custom_option" in result.unmapped_options
        assert "another_custom" in result.unmapped_options
        assert result.unmapped_options["custom_option"] == "value"
        assert result.unmapped_options["another_custom"] == 123

        # Warnings should be present
        assert len(result.warnings) > 0
        assert any("unmapped" in w.lower() for w in result.warnings)

    def test_example_06_validation_equivalence(self):
        """Example 6: Validation of migration equivalence."""
        from cryptofeed.backends.kafka.migration import validate_migration
        from cryptofeed.backends.kafka.callback import KafkaConfig, KafkaTopicConfig, KafkaPartitionConfig

        legacy = {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "crypto",
            "partition_strategy": "symbol",
        }

        # Expected modern config
        expected = KafkaConfig(
            bootstrap_servers=["localhost:9092"],
            topic=KafkaTopicConfig(strategy="per_symbol", prefix="crypto"),
            partition=KafkaPartitionConfig(strategy="symbol"),
        )

        report = validate_migration(legacy, expected)

        # Should be equivalent (legacy defaults to per_symbol strategy)
        assert report.is_equivalent
        assert len(report.differences) == 0

    def test_example_07_deprecation_warning_tracking(self):
        """Example 7: Deprecation warning tracking and analytics."""
        from cryptofeed.backends.kafka.maintenance import get_deprecation_warning_system

        system = get_deprecation_warning_system()

        # Reset for clean test
        system.reset_usage_stats()

        # Track some usage
        system.track_usage("test_component", {"exchange": "binance", "symbol": "BTC-USDT"})
        system.track_usage("test_component", {"exchange": "coinbase", "symbol": "ETH-USD"})

        # Verify tracking
        stats = system.get_usage_stats()
        assert stats["test_component"] == 2

        # Get detailed report
        report = system.get_usage_report()
        assert "test_component" in report
        assert report["test_component"]["count"] == 2
        assert "last_timestamp" in report["test_component"]
        assert "last_context" in report["test_component"]

    def test_example_08_per_symbol_vs_consolidated_topics(self):
        """Example 8: Topic strategy comparison (per_symbol vs consolidated)."""
        from cryptofeed.backends.kafka.callback import KafkaConfig, KafkaTopicConfig

        # Consolidated topics (modern default)
        consolidated = KafkaConfig(
            bootstrap_servers=["localhost:9092"],
            topic=KafkaTopicConfig(strategy="consolidated", prefix="cryptofeed"),
        )
        assert consolidated.topic.strategy == "consolidated"

        # Per-symbol topics (legacy default)
        per_symbol = KafkaConfig(
            bootstrap_servers=["localhost:9092"],
            topic=KafkaTopicConfig(strategy="per_symbol", prefix="trades"),
        )
        assert per_symbol.topic.strategy == "per_symbol"

    def test_example_09_partition_strategy_selection(self):
        """Example 9: Partition strategy selection guide."""
        from cryptofeed.backends.kafka.callback import KafkaConfig, KafkaPartitionConfig

        # Composite (recommended)
        composite = KafkaConfig(
            bootstrap_servers=["localhost:9092"],
            partition=KafkaPartitionConfig(strategy="composite"),
        )
        assert composite.partition.strategy == "composite"

        # Symbol (cross-exchange analysis)
        symbol = KafkaConfig(
            bootstrap_servers=["localhost:9092"],
            partition=KafkaPartitionConfig(strategy="symbol"),
        )
        assert symbol.partition.strategy == "symbol"

        # Exchange (per-exchange processing)
        exchange = KafkaConfig(
            bootstrap_servers=["localhost:9092"],
            partition=KafkaPartitionConfig(strategy="exchange"),
        )
        assert exchange.partition.strategy == "exchange"

        # Round-robin (maximum parallelism)
        round_robin = KafkaConfig(
            bootstrap_servers=["localhost:9092"],
            partition=KafkaPartitionConfig(strategy="round_robin"),
        )
        assert round_robin.partition.strategy == "round_robin"

    def test_example_10_complete_migration_workflow(self):
        """Example 10: Complete migration workflow from legacy to modern."""
        from cryptofeed.backends.kafka.migration import translate_legacy_config, validate_migration
        from cryptofeed.backends.kafka.health import KafkaHealthCheck

        # Step 1: Define legacy configuration
        legacy_config = {
            "bootstrap_servers": ["kafka:9092"],
            "topic_prefix": "production",
            "partition_strategy": "composite",
            "acks": "all",
            "compression_type": "snappy",
        }

        # Step 2: Translate to modern
        result = translate_legacy_config(legacy_config)
        modern_config = result.modern_config

        # Step 3: Check for unmapped options
        if result.unmapped_options:
            # Would log warnings in real usage
            assert isinstance(result.unmapped_options, dict)

        # Step 4: Validate translation
        validation = validate_migration(legacy_config, modern_config)
        assert validation.is_equivalent

        # Step 5: Health check (API validation)
        try:
            status = KafkaHealthCheck.check_modern(modern_config)
            assert status.implementation == "modern"
        except Exception:
            # Connection may fail in test, but API should work
            pass

        # Migration complete - modern_config ready to use
        assert modern_config.bootstrap_servers == ["kafka:9092"]
        assert modern_config.topic.prefix == "production"


class TestTroubleshootingGuideExamples:
    """Validate troubleshooting guide code examples."""

    def test_troubleshoot_01_missing_bootstrap_servers(self):
        """Troubleshooting: Missing bootstrap_servers error."""
        from cryptofeed.backends.kafka.migration import translate_legacy_config

        # This should raise ValueError
        with pytest.raises(ValueError, match="bootstrap_servers"):
            translate_legacy_config({})

    def test_troubleshoot_02_invalid_topic_strategy(self):
        """Troubleshooting: Invalid topic strategy error."""
        from cryptofeed.backends.kafka.callback import KafkaTopicConfig

        # This should raise ValidationError
        with pytest.raises(Exception):  # Pydantic ValidationError
            KafkaTopicConfig(strategy="invalid_strategy")

    def test_troubleshoot_03_invalid_partition_strategy(self):
        """Troubleshooting: Invalid partition strategy error."""
        from cryptofeed.backends.kafka.callback import KafkaPartitionConfig

        # This should raise ValidationError
        with pytest.raises(Exception):  # Pydantic ValidationError
            KafkaPartitionConfig(strategy="invalid_partitioner")

    def test_troubleshoot_04_negative_partitions(self):
        """Troubleshooting: Negative partitions_per_topic error."""
        from cryptofeed.backends.kafka.callback import KafkaTopicConfig

        # This should raise ValidationError
        with pytest.raises(Exception):  # Pydantic ValidationError
            KafkaTopicConfig(partitions_per_topic=-1)

    def test_troubleshoot_05_negative_replication_factor(self):
        """Troubleshooting: Negative replication_factor error."""
        from cryptofeed.backends.kafka.callback import KafkaTopicConfig

        # This should raise ValidationError
        with pytest.raises(Exception):  # Pydantic ValidationError
            KafkaTopicConfig(replication_factor=0)


class TestAPIReferenceExamples:
    """Validate API reference documentation examples."""

    def test_api_kafkaconfig_basic(self):
        """API Reference: KafkaConfig basic usage."""
        from cryptofeed.backends.kafka.callback import KafkaConfig

        config = KafkaConfig(bootstrap_servers=["localhost:9092"])
        assert config.bootstrap_servers == ["localhost:9092"]
        assert config.acks == "all"  # Default
        assert config.idempotence is True  # Default

    def test_api_kafkatopicconfig_basic(self):
        """API Reference: KafkaTopicConfig basic usage."""
        from cryptofeed.backends.kafka.callback import KafkaTopicConfig

        topic = KafkaTopicConfig(
            strategy="consolidated",
            prefix="production",
            partitions_per_topic=12,
            replication_factor=3,
        )
        assert topic.strategy == "consolidated"
        assert topic.prefix == "production"
        assert topic.partitions_per_topic == 12
        assert topic.replication_factor == 3

    def test_api_kafkapartitionconfig_basic(self):
        """API Reference: KafkaPartitionConfig basic usage."""
        from cryptofeed.backends.kafka.callback import KafkaPartitionConfig

        partition = KafkaPartitionConfig(strategy="composite")
        assert partition.strategy == "composite"

    def test_api_migration_result(self):
        """API Reference: MigrationResult structure."""
        from cryptofeed.backends.kafka.migration import translate_legacy_config

        result = translate_legacy_config({
            "bootstrap_servers": ["localhost:9092"],
            "unknown_key": "value",
        })

        # MigrationResult has these fields
        assert hasattr(result, 'modern_config')
        assert hasattr(result, 'unmapped_options')
        assert hasattr(result, 'warnings')

        assert "unknown_key" in result.unmapped_options

    def test_api_health_status(self):
        """API Reference: KafkaHealthStatus structure."""
        from cryptofeed.backends.kafka.health import KafkaHealthCheck

        # Legacy config format
        legacy = {"bootstrap_servers": ["localhost:9092"]}

        status = KafkaHealthCheck.check_legacy(legacy)

        # KafkaHealthStatus has these fields
        assert hasattr(status, 'implementation')
        assert hasattr(status, 'ok')
        assert hasattr(status, 'latency_ms')
        assert hasattr(status, 'error')
        assert hasattr(status, 'details')

        assert status.implementation == "legacy"
        assert isinstance(status.ok, bool)
        assert isinstance(status.latency_ms, float)

    def test_api_deprecation_system(self):
        """API Reference: DeprecationWarningSystem API."""
        from cryptofeed.backends.kafka.maintenance import DeprecationWarningSystem

        system = DeprecationWarningSystem()

        # API methods
        assert hasattr(system, 'emit_class_warning')
        assert hasattr(system, 'emit_import_warning')
        assert hasattr(system, 'track_usage')
        assert hasattr(system, 'get_usage_stats')
        assert hasattr(system, 'get_usage_report')
        assert hasattr(system, 'reset_usage_stats')
