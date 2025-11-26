"""
Unit tests for Task 21.1: Consumer Configuration Validator

Tests the consumer configuration validation script that ensures:
- Topic subscription patterns are valid (wildcard, regex)
- Consumer group configuration is correct
- Offset management strategy is properly configured
- Message header extraction is configured
- Protobuf deserialization is set up correctly

TDD Approach: Write tests first, then implement validator script.
"""

import pytest
import subprocess
import json
from pathlib import Path


class TestConsumerConfigValidator:
    """Test suite for consumer configuration validation."""

    def test_validator_script_exists(self):
        """Test that the validator script exists and is executable."""
        script_path = Path("scripts/validate-consumer-config.py")
        assert script_path.exists(), "Validator script must exist"
        assert script_path.stat().st_mode & 0o111, "Script must be executable"

    def test_validate_flink_consumer_config_valid(self):
        """Test validation of valid Flink consumer configuration."""
        config = {
            "consumer_type": "flink",
            "topics": ["cryptofeed.trades", "cryptofeed.orderbook"],
            "bootstrap_servers": ["kafka1:9092"],
            "consumer_group": "flink-analytics",
            "offset_reset": "earliest",
            "enable_headers": True,
            "protobuf_enabled": True,
        }

        result = self._run_validator(config)
        assert result["valid"] is True
        assert result["errors"] == []
        assert "flink" in result["consumer_type"]

    def test_validate_python_async_consumer_config_valid(self):
        """Test validation of valid Python async consumer configuration."""
        config = {
            "consumer_type": "python-async",
            "topics": ["cryptofeed.trades"],
            "bootstrap_servers": ["kafka1:9092", "kafka2:9092"],
            "consumer_group": "python-processor",
            "offset_reset": "latest",
            "enable_auto_commit": False,
            "batch_size": 100,
            "enable_headers": True,
        }

        result = self._run_validator(config)
        assert result["valid"] is True
        assert result["batch_size"] == 100

    def test_validate_custom_consumer_config_valid(self):
        """Test validation of minimal custom consumer configuration."""
        config = {
            "consumer_type": "custom",
            "topics": ["cryptofeed.*"],  # Wildcard pattern
            "bootstrap_servers": ["localhost:9092"],
            "consumer_group": "my-custom-consumer",
        }

        result = self._run_validator(config)
        assert result["valid"] is True

    def test_validate_wildcard_topic_pattern(self):
        """Test validation of wildcard topic subscription patterns."""
        patterns = [
            "cryptofeed.*",  # All cryptofeed topics
            "cryptofeed.trades.*",  # All trade topics (legacy)
            "cryptofeed.trades",  # Specific consolidated topic
            "cryptofeed.(trades|orderbook)",  # Multiple types
        ]

        for pattern in patterns:
            config = {
                "consumer_type": "custom",
                "topics": [pattern],
                "bootstrap_servers": ["kafka1:9092"],
                "consumer_group": "test",
            }
            result = self._run_validator(config)
            assert result["valid"] is True, f"Pattern {pattern} should be valid"

    def test_validate_rejects_missing_bootstrap_servers(self):
        """Test validator rejects config without bootstrap servers."""
        config = {
            "consumer_type": "python-async",
            "topics": ["cryptofeed.trades"],
            "consumer_group": "test",
            # Missing bootstrap_servers
        }

        result = self._run_validator(config)
        assert result["valid"] is False
        assert any("bootstrap_servers" in error for error in result["errors"])

    def test_validate_rejects_empty_topics(self):
        """Test validator rejects config with empty topic list."""
        config = {
            "consumer_type": "flink",
            "topics": [],  # Empty list
            "bootstrap_servers": ["kafka1:9092"],
            "consumer_group": "test",
        }

        result = self._run_validator(config)
        assert result["valid"] is False
        assert any("topics" in error.lower() for error in result["errors"])

    def test_validate_rejects_invalid_consumer_type(self):
        """Test validator rejects unknown consumer type."""
        config = {
            "consumer_type": "unknown-type",
            "topics": ["cryptofeed.trades"],
            "bootstrap_servers": ["kafka1:9092"],
            "consumer_group": "test",
        }

        result = self._run_validator(config)
        assert result["valid"] is False
        assert any("consumer_type" in error.lower() for error in result["errors"])

    def test_validate_offset_reset_strategy(self):
        """Test validation of offset reset strategies."""
        valid_strategies = ["earliest", "latest", "none"]

        for strategy in valid_strategies:
            config = {
                "consumer_type": "python-async",
                "topics": ["cryptofeed.trades"],
                "bootstrap_servers": ["kafka1:9092"],
                "consumer_group": "test",
                "offset_reset": strategy,
            }
            result = self._run_validator(config)
            assert result["valid"] is True, f"Strategy {strategy} should be valid"

    def test_validate_rejects_invalid_offset_strategy(self):
        """Test validator rejects invalid offset reset strategy."""
        config = {
            "consumer_type": "python-async",
            "topics": ["cryptofeed.trades"],
            "bootstrap_servers": ["kafka1:9092"],
            "consumer_group": "test",
            "offset_reset": "invalid-strategy",
        }

        result = self._run_validator(config)
        assert result["valid"] is False
        assert any("offset" in error.lower() for error in result["errors"])

    def test_validate_header_extraction_config(self):
        """Test validation of header extraction configuration."""
        config = {
            "consumer_type": "python-async",
            "topics": ["cryptofeed.trades"],
            "bootstrap_servers": ["kafka1:9092"],
            "consumer_group": "test",
            "enable_headers": True,
            "header_filters": {
                "exchange": ["coinbase", "binance"],
                "symbol": ["BTC-USD"],
            },
        }

        result = self._run_validator(config)
        assert result["valid"] is True
        assert result["header_filters"]["exchange"] == ["coinbase", "binance"]

    def test_validate_protobuf_deserialization_config(self):
        """Test validation of protobuf deserialization configuration."""
        config = {
            "consumer_type": "flink",
            "topics": ["cryptofeed.trades"],
            "bootstrap_servers": ["kafka1:9092"],
            "consumer_group": "test",
            "protobuf_enabled": True,
            "schema_registry_url": "http://schema-registry:8081",
        }

        result = self._run_validator(config)
        assert result["valid"] is True
        assert result["protobuf_enabled"] is True

    def test_validate_batch_processing_config(self):
        """Test validation of batch processing configuration."""
        config = {
            "consumer_type": "python-async",
            "topics": ["cryptofeed.trades"],
            "bootstrap_servers": ["kafka1:9092"],
            "consumer_group": "test",
            "batch_size": 500,
            "batch_timeout_ms": 5000,
        }

        result = self._run_validator(config)
        assert result["valid"] is True
        assert result["batch_size"] == 500
        assert result["batch_timeout_ms"] == 5000

    def test_validate_consumer_group_naming(self):
        """Test validation of consumer group naming conventions."""
        valid_groups = [
            "flink-analytics",
            "python-processor-v2",
            "custom_consumer_123",
            "my.consumer.group",
        ]

        for group in valid_groups:
            config = {
                "consumer_type": "custom",
                "topics": ["cryptofeed.trades"],
                "bootstrap_servers": ["kafka1:9092"],
                "consumer_group": group,
            }
            result = self._run_validator(config)
            assert result["valid"] is True, f"Group name {group} should be valid"

    # Helper methods

    def _run_validator(self, config):
        """Run validator script with given config and return result."""
        # This will fail initially (RED phase) until we implement the script
        # Validator script should accept JSON on stdin and output JSON
        import tempfile
        import json

        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_file = f.name

        try:
            # Run validator script
            result = subprocess.run(
                ["python", "scripts/validate-consumer-config.py", config_file],
                capture_output=True,
                text=True,
                check=False,
            )

            # Parse JSON output (script outputs JSON even on validation errors)
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {"valid": False, "errors": [result.stderr]}
        finally:
            Path(config_file).unlink(missing_ok=True)


class TestConsumerConfigValidatorCLI:
    """Test CLI interface of validator script."""

    def test_validator_help_output(self):
        """Test validator script shows help message."""
        result = subprocess.run(
            ["python", "scripts/validate-consumer-config.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()

    def test_validator_version_output(self):
        """Test validator script shows version."""
        result = subprocess.run(
            ["python", "scripts/validate-consumer-config.py", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "version" in result.stdout.lower() or len(result.stdout) > 0

    def test_validator_requires_config_file(self):
        """Test validator requires config file argument."""
        result = subprocess.run(
            ["python", "scripts/validate-consumer-config.py"],
            capture_output=True,
            text=True,
        )
        # Should fail without config file
        assert result.returncode != 0
        assert "config" in result.stderr.lower() or "argument" in result.stderr.lower()
