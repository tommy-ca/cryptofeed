"""Prometheus metrics exporter for Kafka producer.

This module provides comprehensive metrics collection for the KafkaCallback producer,
enabling production monitoring and alerting via Prometheus/Grafana.

Metrics collected:
- Producer: messages_produced_total, produce_latency_seconds, produce_errors_total, producer_buffer_usage_bytes
- Kafka: broker_latency_seconds, partition_lag_records, buffer_utilization_percent
- Serialization: message_size_bytes, serialization_latency_seconds

The exporter is designed to be integrated as a decorator/hook with KafkaCallback
to avoid code intrusion and maintain clean separation of concerns.
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Optional

import logging

LOG = logging.getLogger("feedhandler")


# ============================================================================
# Prometheus Metrics Definitions
# ============================================================================


class PrometheusMetricsExporter:
    """Prometheus metrics exporter for Kafka producer.

    This class manages all metrics collection for the KafkaCallback producer,
    providing decorators and hooks for integration with existing code.

    Metrics:
    - messages_produced_total: Counter of successfully produced messages
    - produce_latency_seconds: Histogram of message produce latency
    - produce_errors_total: Counter of produce errors
    - producer_buffer_usage_bytes: Gauge of producer buffer utilization
    - kafka_broker_latency_seconds: Histogram of broker latency
    - kafka_partition_lag_records: Gauge of partition lag
    - kafka_buffer_utilization_percent: Gauge of buffer utilization percentage
    - message_size_bytes: Histogram of serialized message sizes
    - serialization_latency_seconds: Histogram of serialization latency
    """

    def __init__(self, producer_id: str = "default", enabled: bool = True):
        """Initialize metrics exporter.

        Args:
            producer_id: Unique identifier for this producer instance
            enabled: Whether metrics collection is enabled
        """
        self.producer_id = producer_id
        self.enabled = enabled
        self._import_prometheus()

    def _import_prometheus(self) -> None:
        """Lazy import prometheus_client to avoid hard dependency."""
        try:
            from prometheus_client import Counter, Histogram, Gauge, REGISTRY
            self.Counter = Counter
            self.Histogram = Histogram
            self.Gauge = Gauge
            self.REGISTRY = REGISTRY
            self._prometheus_available = True
        except ImportError:
            self._prometheus_available = False
            LOG.warning("prometheus_client not available, metrics collection disabled")

    def _ensure_prometheus(self) -> bool:
        """Check if Prometheus is available."""
        if not self._prometheus_available:
            return False
        return self.enabled

    def _create_counter(self, name: str, documentation: str,
                       labelnames: List[str]) -> Any:
        """Create a Prometheus counter metric."""
        if not self._ensure_prometheus():
            return self._NoOpMetric()

        try:
            return self.Counter(name, documentation, labelnames=labelnames)
        except ValueError:
            # Metric already exists, retrieve it
            return self.REGISTRY._names_to_collectors.get(name)

    def _create_histogram(self, name: str, documentation: str,
                         labelnames: List[str],
                         buckets: Optional[tuple] = None) -> Any:
        """Create a Prometheus histogram metric."""
        if not self._ensure_prometheus():
            return self._NoOpMetric()

        try:
            kwargs = {"labelnames": labelnames}
            if buckets:
                kwargs["buckets"] = buckets
            return self.Histogram(name, documentation, **kwargs)
        except ValueError:
            # Metric already exists, retrieve it
            return self.REGISTRY._names_to_collectors.get(name)

    def _create_gauge(self, name: str, documentation: str,
                     labelnames: List[str]) -> Any:
        """Create a Prometheus gauge metric."""
        if not self._ensure_prometheus():
            return self._NoOpMetric()

        try:
            return self.Gauge(name, documentation, labelnames=labelnames)
        except ValueError:
            # Metric already exists, retrieve it
            return self.REGISTRY._names_to_collectors.get(name)

    class _NoOpMetric:
        """No-op metric for when Prometheus is unavailable."""
        def labels(self, **kwargs) -> "PrometheusMetricsExporter._NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

        def set(self, value: float) -> None:
            pass

    # ========================================================================
    # Producer Metrics
    # ========================================================================

    def create_producer_metrics(self) -> None:
        """Create producer-level metrics."""
        # Counter: messages_produced_total
        self.messages_produced_total = self._create_counter(
            "messages_produced_total",
            "Total number of messages successfully produced to Kafka",
            ["exchange", "symbol", "data_type", "partition_strategy"]
        )

        # Histogram: produce_latency_seconds (buckets: 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s)
        self.produce_latency_seconds = self._create_histogram(
            "produce_latency_seconds",
            "Latency of message production from callback to broker acknowledgment",
            ["exchange", "data_type"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
        )

        # Counter: produce_errors_total
        self.produce_errors_total = self._create_counter(
            "produce_errors_total",
            "Total number of produce errors",
            ["exchange", "data_type", "error_type"]
        )

        # Gauge: producer_buffer_usage_bytes
        self.producer_buffer_usage_bytes = self._create_gauge(
            "producer_buffer_usage_bytes",
            "Current bytes in producer buffer waiting for transmission",
            ["producer_id"]
        )

    # ========================================================================
    # Kafka Infrastructure Metrics
    # ========================================================================

    def create_kafka_metrics(self) -> None:
        """Create Kafka infrastructure metrics."""
        # Histogram: kafka_broker_latency_seconds
        self.kafka_broker_latency_seconds = self._create_histogram(
            "kafka_broker_latency_seconds",
            "Latency to Kafka broker for various operations",
            ["broker_id", "operation"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
        )

        # Gauge: kafka_partition_lag_records
        self.kafka_partition_lag_records = self._create_gauge(
            "kafka_partition_lag_records",
            "Number of records behind in partition (consumer lag)",
            ["partition"]
        )

        # Gauge: kafka_buffer_utilization_percent
        self.kafka_buffer_utilization_percent = self._create_gauge(
            "kafka_buffer_utilization_percent",
            "Percentage of producer buffer pool currently in use",
            ["producer_id"]
        )

    # ========================================================================
    # Serialization Metrics
    # ========================================================================

    def create_serialization_metrics(self) -> None:
        """Create serialization performance metrics."""
        # Histogram: message_size_bytes
        self.message_size_bytes = self._create_histogram(
            "message_size_bytes",
            "Distribution of serialized message sizes in bytes",
            ["data_type", "compression_enabled"],
            buckets=(100, 250, 500, 1000, 2500, 5000, 10000)
        )

        # Histogram: serialization_latency_seconds
        self.serialization_latency_seconds = self._create_histogram(
            "serialization_latency_seconds",
            "Time taken to serialize message to protobuf format",
            ["data_type"],
            buckets=(0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01)
        )

    def initialize(self) -> None:
        """Initialize all metrics."""
        if not self.enabled:
            LOG.debug("Metrics collection disabled")
            return

        self.create_producer_metrics()
        self.create_kafka_metrics()
        self.create_serialization_metrics()
        LOG.info(f"Prometheus metrics initialized for producer: {self.producer_id}")

    # ========================================================================
    # Metric Recording Methods
    # ========================================================================

    def record_message_produced(self, exchange: str, symbol: str,
                               data_type: str, partition_strategy: str) -> None:
        """Record a successfully produced message."""
        if not self.enabled:
            return
        try:
            self.messages_produced_total.labels(
                exchange=exchange,
                symbol=symbol,
                data_type=data_type,
                partition_strategy=partition_strategy
            ).inc()
        except Exception as e:
            LOG.debug(f"Error recording message produced metric: {e}")

    def record_produce_latency(self, latency_seconds: float,
                              exchange: str, data_type: str) -> None:
        """Record message produce latency."""
        if not self.enabled:
            return
        try:
            self.produce_latency_seconds.labels(
                exchange=exchange,
                data_type=data_type
            ).observe(latency_seconds)
        except Exception as e:
            LOG.debug(f"Error recording produce latency metric: {e}")

    def record_produce_error(self, exchange: str, data_type: str,
                            error_type: str) -> None:
        """Record a produce error."""
        if not self.enabled:
            return
        try:
            self.produce_errors_total.labels(
                exchange=exchange,
                data_type=data_type,
                error_type=error_type
            ).inc()
        except Exception as e:
            LOG.debug(f"Error recording produce error metric: {e}")

    def record_buffer_usage(self, bytes_used: float) -> None:
        """Record producer buffer usage in bytes."""
        if not self.enabled:
            return
        try:
            self.producer_buffer_usage_bytes.labels(
                producer_id=self.producer_id
            ).set(bytes_used)
        except Exception as e:
            LOG.debug(f"Error recording buffer usage metric: {e}")

    def record_broker_latency(self, latency_seconds: float,
                             broker_id: str, operation: str) -> None:
        """Record Kafka broker latency."""
        if not self.enabled:
            return
        try:
            self.kafka_broker_latency_seconds.labels(
                broker_id=broker_id,
                operation=operation
            ).observe(latency_seconds)
        except Exception as e:
            LOG.debug(f"Error recording broker latency metric: {e}")

    def record_partition_lag(self, lag_records: int, partition: int) -> None:
        """Record partition lag in records."""
        if not self.enabled:
            return
        try:
            self.kafka_partition_lag_records.labels(
                partition=str(partition)
            ).set(lag_records)
        except Exception as e:
            LOG.debug(f"Error recording partition lag metric: {e}")

    def record_buffer_utilization(self, percent: float) -> None:
        """Record buffer utilization percentage."""
        if not self.enabled:
            return
        try:
            self.kafka_buffer_utilization_percent.labels(
                producer_id=self.producer_id
            ).set(percent)
        except Exception as e:
            LOG.debug(f"Error recording buffer utilization metric: {e}")

    def record_message_size(self, size_bytes: int, data_type: str,
                           compression_enabled: bool) -> None:
        """Record serialized message size."""
        if not self.enabled:
            return
        try:
            self.message_size_bytes.labels(
                data_type=data_type,
                compression_enabled=str(compression_enabled)
            ).observe(size_bytes)
        except Exception as e:
            LOG.debug(f"Error recording message size metric: {e}")

    def record_serialization_latency(self, latency_seconds: float,
                                    data_type: str) -> None:
        """Record message serialization latency."""
        if not self.enabled:
            return
        try:
            self.serialization_latency_seconds.labels(
                data_type=data_type
            ).observe(latency_seconds)
        except Exception as e:
            LOG.debug(f"Error recording serialization latency metric: {e}")

    # ========================================================================
    # Decorators for Integration
    # ========================================================================

    def producer_method(self, func: Callable) -> Callable:
        """Decorator to measure latency of producer methods."""
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                # Log latency but don't record to metrics (to avoid overhead)
                if elapsed > 0.01:  # Log only if > 10ms
                    LOG.debug(f"Producer method {func.__name__} took {elapsed*1000:.2f}ms")
        return wrapper

    def track_produce_latency(self, exchange: str, data_type: str) -> Callable:
        """Decorator factory to track message produce latency."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start_time
                    self.record_produce_latency(elapsed, exchange, data_type)
            return wrapper
        return decorator


# ============================================================================
# Global Metrics Exporter Instance
# ============================================================================


_global_metrics_exporter: Optional[PrometheusMetricsExporter] = None


def get_metrics_exporter(producer_id: str = "default",
                         enabled: bool = True) -> PrometheusMetricsExporter:
    """Get or create the global metrics exporter instance.

    Args:
        producer_id: Unique identifier for this producer
        enabled: Whether metrics collection is enabled

    Returns:
        PrometheusMetricsExporter instance
    """
    global _global_metrics_exporter
    if _global_metrics_exporter is None:
        _global_metrics_exporter = PrometheusMetricsExporter(producer_id, enabled)
        _global_metrics_exporter.initialize()
    return _global_metrics_exporter


def reset_metrics_exporter() -> None:
    """Reset the global metrics exporter (for testing)."""
    global _global_metrics_exporter
    _global_metrics_exporter = None
