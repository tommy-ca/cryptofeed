"""
Unit tests for KafkaCallback.get_health_status() method (Task 14.3).

Tests the simplified health check that returns basic producer connectivity status.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch

from cryptofeed.backends.kafka.callback import KafkaCallback


@pytest.fixture
def mock_callback():
    """Create KafkaCallback with mocked producer."""
    with patch('cryptofeed.backends.kafka.producer.Producer') as mock_producer_class:
        # Mock producer factory to avoid actual Kafka connection
        mock_producer_instance = Mock()
        mock_producer_instance.list_topics = Mock(return_value={})
        mock_producer_class.return_value = mock_producer_instance

        callback = KafkaCallback(bootstrap_servers=["localhost:9092"])
        callback._producer._producer = mock_producer_instance  # Access underlying producer

        yield callback


class TestKafkaCallbackHealthStatus:
    """Test KafkaCallback.get_health_status() method."""

    def test_get_health_status_returns_dict_with_required_fields(self, mock_callback):
        """Test health status returns dict with ok, latency_ms, error, bootstrap fields."""
        status = mock_callback.get_health_status()

        assert isinstance(status, dict)
        assert "ok" in status
        assert "latency_ms" in status
        assert "error" in status
        assert "bootstrap" in status

    def test_get_health_status_returns_ok_true_when_producer_connected(self, mock_callback):
        """Test health status returns ok=True when producer can list topics."""
        status = mock_callback.get_health_status()

        assert status["ok"] is True
        assert status["error"] is None
        assert isinstance(status["latency_ms"], (int, float))
        assert status["latency_ms"] >= 0

    def test_get_health_status_returns_ok_false_when_producer_fails(self):
        """Test health status returns ok=False when producer list_topics fails."""
        with patch('cryptofeed.backends.kafka.producer.Producer') as mock_producer_class:
            # Mock producer that succeeds on __init__ but fails on subsequent calls
            mock_producer_instance = Mock()
            call_count = [0]
            def list_topics_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return {}  # Succeed on first call (during connect)
                raise RuntimeError("Connection refused")  # Fail on health check

            mock_producer_instance.list_topics = Mock(side_effect=list_topics_side_effect)
            mock_producer_class.return_value = mock_producer_instance

            callback = KafkaCallback(bootstrap_servers=["localhost:9092"])
            callback._producer._producer = mock_producer_instance

            status = callback.get_health_status()

            assert status["ok"] is False
            assert status["error"] is not None
            assert "Connection refused" in status["error"]
            assert isinstance(status["latency_ms"], (int, float))

    def test_get_health_status_includes_bootstrap_servers(self):
        """Test health status includes bootstrap servers list."""
        bootstrap = ["broker1:9092", "broker2:9092"]

        with patch('cryptofeed.backends.kafka.producer.Producer') as mock_producer_class:
            mock_producer_instance = Mock()
            mock_producer_instance.list_topics = Mock(return_value={})
            mock_producer_class.return_value = mock_producer_instance

            callback = KafkaCallback(bootstrap_servers=bootstrap)
            callback._producer._producer = mock_producer_instance

            status = callback.get_health_status()

            assert status["bootstrap"] == bootstrap

    def test_get_health_status_accepts_optional_timeout(self, mock_callback):
        """Test health status accepts optional timeout_ms parameter."""
        # Should accept timeout parameter without error
        status = mock_callback.get_health_status(timeout_ms=5000)

        assert status["ok"] is True

    def test_get_health_status_measures_latency(self, mock_callback):
        """Test health status measures connection latency in milliseconds."""
        status = mock_callback.get_health_status()

        # Latency should be measured and positive
        assert isinstance(status["latency_ms"], (int, float))
        assert status["latency_ms"] >= 0
        # Should be relatively fast (< 1 second for mock)
        assert status["latency_ms"] < 1000

    def test_get_health_status_returns_ok_false_when_no_producer(self):
        """Test health status returns ok=False when producer not initialized."""
        with patch('cryptofeed.backends.kafka.producer.Producer') as mock_producer_class:
            mock_producer_instance = Mock()
            mock_producer_instance.list_topics = Mock(return_value={})
            mock_producer_class.return_value = mock_producer_instance

            callback = KafkaCallback(bootstrap_servers=["localhost:9092"])

            # Simulate producer not being initialized
            callback._producer._producer = None

            status = callback.get_health_status()

            assert status["ok"] is False
            assert status["error"] is not None

    def test_get_health_status_handles_timeout_gracefully(self):
        """Test health status handles connection timeout gracefully."""
        with patch('cryptofeed.backends.kafka.producer.Producer') as mock_producer_class:
            # Mock timeout that succeeds on connect, fails on health check
            mock_producer_instance = Mock()
            call_count = [0]
            def list_topics_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return {}  # Succeed on connect
                raise TimeoutError("Timed out")  # Fail on health check

            mock_producer_instance.list_topics = Mock(side_effect=list_topics_side_effect)
            mock_producer_class.return_value = mock_producer_instance

            callback = KafkaCallback(bootstrap_servers=["localhost:9092"])
            callback._producer._producer = mock_producer_instance

            status = callback.get_health_status(timeout_ms=100)

            assert status["ok"] is False
            assert "Timed out" in status["error"]

    def test_get_health_status_default_timeout_is_3_seconds(self):
        """Test health status uses 3 second default timeout."""
        with patch('cryptofeed.backends.kafka.producer.Producer') as mock_producer_class:
            # Track the timeout parameter passed to list_topics
            call_args = []
            def track_call(*args, **kwargs):
                call_args.append(kwargs.get('timeout'))
                return {}

            mock_producer_instance = Mock()
            mock_producer_instance.list_topics = Mock(side_effect=track_call)
            mock_producer_class.return_value = mock_producer_instance

            callback = KafkaCallback(bootstrap_servers=["localhost:9092"])
            callback._producer._producer = mock_producer_instance

            callback.get_health_status()

            # Should have been called twice: once during connect, once during get_health_status
            # We only care about the second call (the health check)
            assert len(call_args) == 2
            assert call_args[0] == 5.0  # connect() timeout (5 seconds)
            assert call_args[1] == 3.0  # get_health_status() default timeout (3 seconds)


class TestKafkaCallbackHealthStatusIntegration:
    """Integration tests for health check with real config."""

    def test_get_health_status_preserves_backward_compatibility(self):
        """Test that existing health check infrastructure still works."""
        with patch('cryptofeed.backends.kafka.producer.Producer') as mock_producer_class:
            mock_producer_instance = Mock()
            mock_producer_instance.list_topics = Mock(return_value={})
            mock_producer_class.return_value = mock_producer_instance

            callback = KafkaCallback(bootstrap_servers=["localhost:9092"])
            callback._producer._producer = mock_producer_instance

            # Ensure is_connected() still works independently
            assert callback.is_connected() is True

            # Health status should provide more detailed information
            status = callback.get_health_status()

            # Both should indicate health, but status provides more details
            assert callback.is_connected() is True
            assert status["ok"] is True
            assert len(status) >= 4  # More detailed than just boolean


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
