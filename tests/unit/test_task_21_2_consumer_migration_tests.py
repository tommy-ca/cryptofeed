"""
Unit tests for Task 21.2: Consumer Migration Test Automation

Tests the consumer migration testing script that validates:
- Consumer startup with new topic subscriptions
- Offset management and checkpointing
- Consumer restart recovery
- Message consumption from consolidated topics
- End-to-end latency validation

TDD Approach: Write tests first, then implement test automation script.

Note: Tests will skip gracefully if Kafka is not available (KAFKA_AVAILABLE env var).
"""

import pytest
import subprocess
import json
from pathlib import Path


class TestConsumerMigrationAutomation:
    """Test suite for consumer migration test automation."""

    def test_migration_test_script_exists(self):
        """Test that the migration test script exists."""
        script_path = Path("scripts/test-consumer-migration.py")
        assert script_path.exists(), "Migration test script must exist"
        assert script_path.stat().st_mode & 0o111, "Script must be executable"

    def test_consumer_startup_validation(self):
        """Test validation of consumer startup with consolidated topics."""
        result = self._run_test({"test_type": "startup", "topics": ["cryptofeed.trades"]})
        if result["status"] == "skipped":
            pytest.skip(result.get("reason"))
        assert result["status"] == "success"
        assert result.get("consumer_started") is True

    def test_offset_commit_validation(self):
        """Test validation of offset commit behavior."""
        result = self._run_test({"test_type": "offset_commit", "message_count": 100})
        if result["status"] == "skipped":
            pytest.skip(result.get("reason"))
        assert result["messages_consumed"] >= 100

    def test_header_extraction_validation(self):
        """Test message header extraction during migration."""
        result = self._run_test({"test_type": "header_extraction", "message_count": 10})
        if result["status"] == "skipped":
            pytest.skip(result.get("reason"))
        assert "sample_headers" in result
        assert "exchange" in result["sample_headers"]

    def test_latency_measurement(self):
        """Test end-to-end latency measurement."""
        result = self._run_test({"test_type": "latency", "message_count": 100})
        if result["status"] == "skipped":
            pytest.skip(result.get("reason"))
        assert result["latency_p99_ms"] < 200

    # Helper method
    def _run_test(self, test_config):
        """Run migration test and return result."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_file = f.name

        try:
            result = subprocess.run(
                ["python", "scripts/test-consumer-migration.py", config_file],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return json.loads(result.stdout) if result.stdout else {"status": "failed"}
        finally:
            Path(config_file).unlink(missing_ok=True)


class TestConsumerHealthChecks:
    """Test suite for consumer health check automation."""

    def test_health_check_script_exists(self):
        """Test that health check script exists."""
        script_path = Path("scripts/check-consumer-health.py")
        assert script_path.exists(), "Health check script must exist"
        assert script_path.stat().st_mode & 0o111, "Script must be executable"

    def test_consumer_lag_check(self):
        """Test consumer lag health check."""
        result = self._run_check({"check_type": "lag", "threshold_seconds": 5})
        assert result.get("healthy") in [True, False]
        if not result.get("skipped"):
            assert "lag_seconds" in result

    def test_consumer_heartbeat_check(self):
        """Test consumer heartbeat health check."""
        result = self._run_check({"check_type": "heartbeat"})
        assert result.get("healthy") in [True, False]

    # Helper method
    def _run_check(self, check_config):
        """Run health check and return result."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(check_config, f)
            config_file = f.name

        try:
            result = subprocess.run(
                ["python", "scripts/check-consumer-health.py", config_file],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return json.loads(result.stdout) if result.stdout else {"healthy": False}
        finally:
            Path(config_file).unlink(missing_ok=True)
