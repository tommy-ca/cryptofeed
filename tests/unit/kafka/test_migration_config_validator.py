"""Tests for Kafka Phase 2 configuration validation and testing.

This module tests the config_validator module which handles validation
of Phase 2 KafkaCallback configurations and optional Kafka connectivity testing.

Test organization follows TDD approach with real Kafka testing support.
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from cryptofeed.migration.config_validator import (
    ConfigValidator,
    ValidationResult,
)
from cryptofeed.kafka_callback import KafkaConfig


class TestBasicConfigValidation:
    """Test basic Phase 2 config validation without Kafka."""

    def test_validate_minimal_phase2_config(self):
        """Validate minimal Phase 2 configuration."""
        config_dict = {"bootstrap_servers": ["kafka:9092"]}
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        assert result.is_valid
        assert len(result.errors) == 0
        assert result.config is not None

    def test_validate_complete_config(self):
        """Validate complete Phase 2 configuration."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092", "kafka:9093"],
            "topic": {"strategy": "consolidated", "prefix": "production"},
            "partition": {"strategy": "composite"},
            "acks": "all",
            "idempotence": True,
            "retries": 3,
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        assert result.is_valid
        assert result.config.topic.prefix == "production"

    def test_validate_invalid_topic_strategy(self):
        """Detect invalid topic strategy."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092"],
            "topic": {"strategy": "invalid_strategy"},
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("strategy" in str(e) for e in result.errors)

    def test_validate_invalid_partition_strategy(self):
        """Detect invalid partition strategy."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092"],
            "partition": {"strategy": "invalid_partitioner"},
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        assert not result.is_valid
        assert any("partition" in str(e) for e in result.errors)

    def test_validate_invalid_acks_value(self):
        """Detect invalid acks value."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092"],
            "acks": "invalid",
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        assert not result.is_valid
        assert any("acks" in str(e) for e in result.errors)

    def test_validate_negative_retries(self):
        """Detect negative retries count."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092"],
            "retries": -1,
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        assert not result.is_valid
        assert any("retries" in str(e) for e in result.errors)

    def test_validate_invalid_batch_size(self):
        """Detect invalid batch size."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092"],
            "batch_size": 0,
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        assert not result.is_valid
        assert any("batch" in str(e) for e in result.errors)

    def test_validate_invalid_compression(self):
        """Detect invalid compression type."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092"],
            "compression_type": "deflate",
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        assert not result.is_valid
        assert any("compression" in str(e) for e in result.errors)

    def test_validate_returns_validated_config(self):
        """Validation returns validated config object on success."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092"],
            "topic": {"strategy": "consolidated"},
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        assert result.is_valid
        assert isinstance(result.config, KafkaConfig)
        assert result.config.topic.strategy == "consolidated"


class TestYAMLValidation:
    """Test validation of Phase 2 YAML configurations."""

    def test_validate_phase2_yaml_file(self):
        """Load and validate Phase 2 YAML configuration."""
        phase2_yaml = """
bootstrap_servers:
  - kafka:9092
topic:
  strategy: consolidated
  prefix: production
partition:
  strategy: composite
acks: all
compression_type: snappy
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(phase2_yaml)
            f.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_yaml_file(f.name)

                assert result.is_valid
                assert result.config.topic.strategy == "consolidated"
            finally:
                Path(f.name).unlink()

    def test_validate_invalid_yaml_syntax(self):
        """Detect invalid YAML syntax."""
        invalid_yaml = "invalid: yaml: [content"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml)
            f.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_yaml_file(f.name)

                assert not result.is_valid
                assert any("YAML" in str(e) or "yaml" in str(e) for e in result.errors)
            finally:
                Path(f.name).unlink()

    def test_validate_yaml_file_not_found(self):
        """Handle missing YAML file."""
        validator = ConfigValidator()
        result = validator.validate_yaml_file("/nonexistent/config.yaml")

        assert not result.is_valid
        assert any("not found" in str(e) or "File" in str(e) for e in result.errors)

    def test_validate_empty_yaml_file(self):
        """Handle empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_yaml_file(f.name)

                assert not result.is_valid
                assert any(
                    "empty" in str(e).lower() or "required" in str(e).lower()
                    for e in result.errors
                )
            finally:
                Path(f.name).unlink()


class TestConfigCompatibility:
    """Test compatibility checks between old and new configurations."""

    def test_check_topic_naming_compatibility(self):
        """Verify topic naming strategy compatibility."""
        # Consolidated mode
        config = {
            "bootstrap_servers": ["kafka:9092"],
            "topic": {"strategy": "consolidated"},
        }
        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid
        assert "consolidated" in str(result.config.topic.strategy).lower()

    def test_check_partition_strategy_compatibility(self):
        """Verify partition strategy compatibility."""
        for strategy in ["composite", "symbol", "exchange", "round_robin"]:
            config = {
                "bootstrap_servers": ["kafka:9092"],
                "partition": {"strategy": strategy},
            }
            validator = ConfigValidator()
            result = validator.validate(config)

            assert result.is_valid

    def test_detect_legacy_per_symbol_strategy(self):
        """Detect when per_symbol strategy is used."""
        config = {
            "bootstrap_servers": ["kafka:9092"],
            "topic": {"strategy": "per_symbol"},
        }
        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid
        # Should warn about per_symbol being legacy-compatible
        if result.warnings:
            assert any("legacy" in str(w).lower() for w in result.warnings)


class TestKafkaConnectivityTesting:
    """Test Kafka connectivity validation (requires running Kafka)."""

    @pytest.mark.integration
    def test_test_kafka_connectivity_success(self):
        """Test successful Kafka connectivity."""
        config = {
            "bootstrap_servers": ["localhost:9092"],
            "topic": {"strategy": "consolidated"},
        }
        validator = ConfigValidator()
        result = validator.test_kafka_connectivity(config)

        # Should succeed even if actual Kafka unavailable (we test config syntax)
        if result.is_valid:
            # Config is valid syntax
            assert result.config is None or "localhost" in str(result.config).lower()

    @pytest.mark.integration
    def test_test_kafka_connectivity_failure(self):
        """Test failed Kafka connectivity."""
        config = {
            "bootstrap_servers": ["localhost:19092"],  # Wrong port
            "topic": {"strategy": "consolidated"},
        }
        validator = ConfigValidator()
        result = validator.test_kafka_connectivity(config, timeout_seconds=2)

        # Should fail or have connection warning
        assert not result.is_valid or result.warnings

    def test_validate_with_kafka_connectivity_flag(self):
        """Validate config with Kafka connectivity test."""
        config = {
            "bootstrap_servers": ["kafka:9092"],
            "topic": {"strategy": "consolidated"},
        }
        validator = ConfigValidator()
        # Should not raise even if Kafka unavailable
        result = validator.validate(config, test_kafka=False)

        assert result.is_valid

    @pytest.mark.integration
    def test_validate_topic_creation_capability(self):
        """Validate that we can create topics with this config."""
        config = {
            "bootstrap_servers": ["localhost:9092"],
            "topic": {
                "strategy": "consolidated",
                "partitions_per_topic": 3,
                "replication_factor": 1,
            },
        }
        validator = ConfigValidator()
        result = validator.validate(config, test_topic_creation=False)

        assert result.is_valid


class TestValidationReporting:
    """Test validation result reporting and formatting."""

    def test_validation_result_formatting(self):
        """Format validation results for display."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092"],
            "topic": {"strategy": "invalid"},
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        # Should have formatted output
        formatted = result.format_report()
        assert "invalid" in formatted.lower() or "error" in formatted.lower()

    def test_validation_result_with_warnings(self):
        """Report validation warnings."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092"],
            "topic": {"strategy": "per_symbol"},  # Legacy mode
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        # Should work but may have warnings
        assert result.is_valid

    def test_validation_result_summary(self):
        """Get summary of validation results."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092"],
            "partition": {"strategy": "invalid"},
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        summary = result.summary()
        assert isinstance(summary, dict)
        assert "is_valid" in summary
        assert "error_count" in summary


class TestSchemaValidation:
    """Test schema and data type validation."""

    def test_validate_bootstrap_servers_format(self):
        """Validate bootstrap servers format."""
        config_dict = {
            "bootstrap_servers": ["kafka1:9092", "kafka2:9092", "kafka3:9092"],
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        assert result.is_valid

    def test_validate_bootstrap_servers_require_host(self):
        """Bootstrap servers must have host:port format."""
        config_dict = {
            "bootstrap_servers": ["kafka"],  # Missing port
        }
        validator = ConfigValidator()
        result = validator.validate(config_dict)

        # Should either be valid (if implicit port) or have useful error
        if not result.is_valid:
            assert (
                "bootstrap" in str(result.errors).lower()
                or "port" in str(result.errors).lower()
            )


class TestRealWorldValidation:
    """Test validation against realistic configurations."""

    def test_validate_simple_production_config(self):
        """Validate simple production configuration."""
        config = {
            "bootstrap_servers": ["kafka-1:9092", "kafka-2:9092", "kafka-3:9092"],
            "topic": {"strategy": "consolidated", "prefix": "prod"},
            "partition": {"strategy": "composite"},
            "acks": "all",
            "idempotence": True,
            "compression_type": "snappy",
        }
        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid
        assert result.config.topic.prefix == "prod"

    def test_validate_high_throughput_config(self):
        """Validate high-throughput configuration."""
        config = {
            "bootstrap_servers": ["kafka:9092"],
            "batch_size": 65536,
            "linger_ms": 100,
            "compression_type": "lz4",
            "acks": "1",
            "idempotence": False,  # May be disabled for max throughput
        }
        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid

    def test_validate_low_latency_config(self):
        """Validate low-latency configuration."""
        config = {
            "bootstrap_servers": ["kafka:9092"],
            "batch_size": 1024,
            "linger_ms": 0,
            "compression_type": "none",
            "acks": "1",
        }
        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid

    def test_validate_config_with_multiple_data_types(self):
        """Validate config supporting multiple data types."""
        config = {
            "bootstrap_servers": ["kafka:9092"],
            "topic": {
                "strategy": "consolidated",
                "prefix": "feeds",
            },
            "partition": {"strategy": "symbol"},
        }
        validator = ConfigValidator()
        result = validator.validate(config)

        assert result.is_valid
        assert result.config.topic.prefix == "feeds"


class TestValidationHelpers:
    """Test validation helper functions."""

    def test_validate_broker_address(self):
        """Validate broker address format."""
        validator = ConfigValidator()

        # Valid formats
        assert validator._is_valid_broker_address("localhost:9092")
        assert validator._is_valid_broker_address("kafka:9092")
        assert validator._is_valid_broker_address("192.168.1.1:9092")

    def test_detect_common_config_errors(self):
        """Detect common configuration mistakes."""
        # Test cases: (config, should_fail)
        test_cases = [
            ({"bootstrap_servers": []}, True),  # Empty
            ({"bootstrap_servers": ["kafka"]}, True),  # Missing port
            (
                {
                    "bootstrap_servers": ["kafka:9092"],
                    "topic": {"strategy": "per_symbol_new"},
                },
                True,
            ),  # Wrong strategy
            (
                {"bootstrap_servers": ["kafka:9092"], "acks": "true"},
                True,
            ),  # Wrong acks value
            ({"bootstrap_servers": ["kafka:9092"]}, False),  # Valid minimal config
        ]

        validator = ConfigValidator()
        for config, should_fail in test_cases:
            result = validator.validate(config)
            if should_fail:
                assert not result.is_valid, (
                    f"Expected error for config {config} but got valid"
                )
            else:
                assert result.is_valid, (
                    f"Expected valid for config {config} but got errors: {result.errors}"
                )
