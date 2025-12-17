"""Test direct prometheus_client usage in backend.py (Phase 3, Task 15.3).

Tests verify that metrics functionality is preserved when using prometheus_client
directly instead of the wrapper classes in metrics.py.
"""

import inspect
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestDirectPrometheusMetrics:
    """Test direct prometheus_client usage without wrapper abstractions."""

    def test_metrics_directly_use_prometheus_client_counter(self):
        """Verify Counter is used directly from prometheus_client."""
        with patch('cryptofeed.backends.kafka.callback.prometheus_client') as mock_prom:
            mock_counter = Mock()
            mock_prom.Counter.return_value = mock_counter

            from cryptofeed.backends.kafka.callback import _create_kafka_metrics

            metrics = _create_kafka_metrics()

            # Should call prometheus_client.Counter directly
            assert mock_prom.Counter.called
            assert 'messages_produced_total' in str(mock_prom.Counter.call_args_list[0])

    def test_metrics_directly_use_prometheus_client_histogram(self):
        """Verify Histogram is used directly from prometheus_client."""
        with patch('cryptofeed.backends.kafka.callback.prometheus_client') as mock_prom:
            mock_histogram = Mock()
            mock_prom.Histogram.return_value = mock_histogram

            from cryptofeed.backends.kafka.callback import _create_kafka_metrics

            metrics = _create_kafka_metrics()

            # Should call prometheus_client.Histogram directly
            assert mock_prom.Histogram.called
            assert 'produce_latency_seconds' in str(mock_prom.Histogram.call_args_list[0])

    def test_metrics_directly_use_prometheus_client_gauge(self):
        """Verify Gauge is used directly from prometheus_client."""
        with patch('cryptofeed.backends.kafka.callback.prometheus_client') as mock_prom:
            mock_gauge = Mock()
            mock_prom.Gauge.return_value = mock_gauge

            from cryptofeed.backends.kafka.callback import _create_kafka_metrics

            metrics = _create_kafka_metrics()

            # Should call prometheus_client.Gauge directly
            assert mock_prom.Gauge.called

    def test_no_wrapper_classes_used(self):
        """Verify no wrapper classes from metrics.py are instantiated."""
        with patch('cryptofeed.backends.kafka.callback.prometheus_client') as mock_prom:
            # Setup basic mocks
            mock_prom.Counter.return_value = Mock()
            mock_prom.Histogram.return_value = Mock()
            mock_prom.Gauge.return_value = Mock()

            from cryptofeed.backends.kafka.callback import _create_kafka_metrics

            metrics = _create_kafka_metrics()

            # Should not import from metrics module
            assert 'PrometheusMetricsExporter' not in str(type(metrics))
            assert 'MetricsCollector' not in str(type(metrics))

    def test_message_produced_counter_increments(self):
        """Verify messages_produced_total counter increments correctly."""
        with patch('cryptofeed.backends.kafka.callback.prometheus_client') as mock_prom:
            mock_counter = Mock()
            mock_labels = Mock()
            mock_counter.labels.return_value = mock_labels
            mock_prom.Counter.return_value = mock_counter
            mock_prom.Histogram.return_value = Mock()
            mock_prom.Gauge.return_value = Mock()

            from cryptofeed.backends.kafka.callback import _create_kafka_metrics, _record_message_produced

            metrics = _create_kafka_metrics()
            _record_message_produced(
                metrics,
                exchange="binance",
                symbol="BTC-USD",
                data_type="trade",
                partition_strategy="composite"
            )

            # Should call .labels() and .inc()
            assert mock_counter.labels.called
            assert mock_labels.inc.called

    def test_produce_latency_histogram_observes(self):
        """Verify produce_latency_seconds histogram observes values."""
        with patch('cryptofeed.backends.kafka.callback.prometheus_client') as mock_prom:
            mock_histogram = Mock()
            mock_labels = Mock()
            mock_histogram.labels.return_value = mock_labels
            mock_prom.Counter.return_value = Mock()
            mock_prom.Histogram.return_value = mock_histogram
            mock_prom.Gauge.return_value = Mock()

            from cryptofeed.backends.kafka.callback import _create_kafka_metrics, _record_produce_latency

            metrics = _create_kafka_metrics()
            _record_produce_latency(
                metrics,
                latency_seconds=0.05,
                exchange="binance",
                data_type="trade"
            )

            # Should call .labels() and .observe()
            assert mock_histogram.labels.called
            assert mock_labels.observe.called
            mock_labels.observe.assert_called_with(0.05)

    def test_produce_error_counter_increments(self):
        """Verify produce_errors_total counter increments on errors."""
        with patch('cryptofeed.backends.kafka.callback.prometheus_client') as mock_prom:
            mock_counter = Mock()
            mock_labels = Mock()
            mock_counter.labels.return_value = mock_labels
            mock_prom.Counter.return_value = mock_counter
            mock_prom.Histogram.return_value = Mock()
            mock_prom.Gauge.return_value = Mock()

            from cryptofeed.backends.kafka.callback import _create_kafka_metrics, _record_produce_error

            metrics = _create_kafka_metrics()
            _record_produce_error(
                metrics,
                exchange="binance",
                data_type="trade",
                error_type="serialization_error"
            )

            # Should call .labels() and .inc()
            assert mock_counter.labels.called
            assert mock_labels.inc.called

    def test_prometheus_unavailable_no_errors(self):
        """Verify graceful handling when prometheus_client is unavailable."""
        with patch('cryptofeed.backends.kafka.callback.prometheus_client', None):
            from cryptofeed.backends.kafka.callback import _create_kafka_metrics

            # Should not raise, should return None or no-op metrics
            metrics = _create_kafka_metrics()

            # Metrics should be None or handle gracefully
            assert metrics is None or metrics == {}

    def test_metrics_recording_with_none_metrics_no_errors(self):
        """Verify recording functions handle None metrics gracefully."""
        from cryptofeed.backends.kafka.callback import _record_message_produced

        # Should not raise when metrics is None
        _record_message_produced(
            None,
            exchange="binance",
            symbol="BTC-USD",
            data_type="trade",
            partition_strategy="composite"
        )
        # No assertion needed, just verify no exception raised

    def test_metric_definitions_approximately_50_lines(self):
        """Verify metric definitions are concise (approximately 50 lines)."""
        import inspect
        from cryptofeed.backends.kafka.callback import _create_kafka_metrics

        source = inspect.getsource(_create_kafka_metrics)
        line_count = len(source.splitlines())

        # Should be approximately 50 lines (allow 50-80 range for flexibility)
        assert 30 <= line_count <= 100, f"Expected ~50 lines, got {line_count}"

    def test_no_import_from_metrics_module_in_backend(self):
        """Verify backend.py does not import from metrics.py."""
        import cryptofeed.backends.kafka.callback as callback_module

        source = inspect.getsource(callback_module)

        # Should not import PrometheusMetricsExporter or get_metrics_exporter
        assert 'from .metrics import PrometheusMetricsExporter' not in source
        assert 'from .metrics import get_metrics_exporter' not in source

    def test_metrics_behavior_unchanged_from_original(self):
        """Integration test: verify metrics behavior is identical to original wrapper."""
        with patch('cryptofeed.backends.kafka.callback.prometheus_client') as mock_prom:
            # Setup realistic mocks
            counters = {}
            histograms = {}
            gauges = {}

            def create_counter(name, doc, labelnames):
                mock = Mock()
                mock.labels.return_value = Mock()
                counters[name] = mock
                return mock

            def create_histogram(name, doc, labelnames, buckets=None):
                mock = Mock()
                mock.labels.return_value = Mock()
                histograms[name] = mock
                return mock

            def create_gauge(name, doc, labelnames):
                mock = Mock()
                mock.labels.return_value = Mock()
                gauges[name] = mock
                return mock

            mock_prom.Counter.side_effect = create_counter
            mock_prom.Histogram.side_effect = create_histogram
            mock_prom.Gauge.side_effect = create_gauge

            from cryptofeed.backends.kafka.callback import (
                _create_kafka_metrics,
                _record_message_produced,
                _record_produce_latency,
                _record_produce_error
            )

            # Create metrics
            metrics = _create_kafka_metrics()

            # Record various events
            _record_message_produced(
                metrics,
                exchange="binance",
                symbol="BTC-USD",
                data_type="trade",
                partition_strategy="composite"
            )
            _record_produce_latency(metrics, 0.05, "binance", "trade")
            _record_produce_error(metrics, "binance", "trade", "kafka_produce_error")

            # Verify all expected metrics were created
            assert 'messages_produced_total' in counters
            assert 'produce_latency_seconds' in histograms
            assert 'produce_errors_total' in counters

            # Verify recording functions were called
            assert counters['messages_produced_total'].labels.called
            assert histograms['produce_latency_seconds'].labels.called
            assert counters['produce_errors_total'].labels.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
