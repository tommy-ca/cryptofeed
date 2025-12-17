"""KafkaCallback core implementation for the market data producer."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Optional
import warnings

from cryptofeed.json_utils import dumps_bytes
from cryptofeed.backends.protobuf.bindings import SCHEMA_VERSION as DEFAULT_SCHEMA_VERSION
from .backend import KafkaBackendBase, KafkaQueuedMessage, KafkaProducer, TopicManager
from .config import KafkaConfig, KafkaTopicConfig, KafkaPartitionConfig
from cryptofeed.backends.protobuf.helpers import serialize_to_protobuf
from .normalization import normalize_exchange, normalize_symbol

# Direct prometheus_client usage (Phase 3, Task 15.3)
try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    prometheus_client = None
    Counter = None
    Histogram = None
    Gauge = None
    PROMETHEUS_AVAILABLE = False

DEFAULT_PROTOBUF_CUTOFF = "2026-02-01"

LOG = logging.getLogger("feedhandler")


def _get_cutoff(override=None):
    from datetime import datetime
    import os

    raw = override or os.environ.get("CF_KAFKA_PROTOBUF_CUTOFF", DEFAULT_PROTOBUF_CUTOFF)
    if raw is None:
        raw = DEFAULT_PROTOBUF_CUTOFF
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except Exception:
        return datetime.strptime(DEFAULT_PROTOBUF_CUTOFF, "%Y-%m-%d").date()


def _protobuf_mode_allowed(cutoff_date):
    from datetime import date

    return date.today() < cutoff_date


def _format_disable_date(cutoff_date):
    from datetime import timedelta

    return (cutoff_date - timedelta(days=1)).isoformat()


# Topic strategy helpers moved to cryptofeed.backends.kafka.topic_manager


# ============================================================================
# Inlined Partition Key Generation (Phase 2 - Task 14.2)
# ============================================================================

def _get_partition_key(obj: Any, strategy: str) -> Optional[bytes]:
    """Generate partition key using specified strategy (inlined from partitioner.py).

    Implements 4 partition strategies with simple if/elif logic:
    - symbol: Route by symbol only
    - composite: Route by exchange-symbol combination (default)
    - exchange: Route by exchange only
    - round_robin: No key (let Kafka round-robin)

    Args:
        obj: Message object with exchange and symbol attributes
        strategy: Strategy name ('symbol', 'composite', 'exchange', 'round_robin')

    Returns:
        Partition key as bytes, or None for round-robin strategy
    """
    strategy_lower = strategy.lower() if strategy else "composite"

    if strategy_lower == "symbol":
        symbol = getattr(obj, "symbol", "")
        normalized = normalize_symbol(symbol)
        return normalized.encode("utf-8")
    elif strategy_lower == "composite":
        exchange = getattr(obj, "exchange", "")
        symbol = getattr(obj, "symbol", "")
        normalized_exchange = normalize_exchange(exchange)
        normalized_symbol = normalize_symbol(symbol)
        return f"{normalized_exchange}-{normalized_symbol}".encode("utf-8")
    elif strategy_lower == "exchange":
        exchange = getattr(obj, "exchange", "")
        normalized = normalize_exchange(exchange)
        return normalized.encode("utf-8")
    elif strategy_lower == "round_robin":
        return None
    else:
        # Unknown strategy, default to composite
        exchange = getattr(obj, "exchange", "")
        symbol = getattr(obj, "symbol", "")
        normalized_exchange = normalize_exchange(exchange)
        normalized_symbol = normalize_symbol(symbol)
        return f"{normalized_exchange}-{normalized_symbol}".encode("utf-8")


# ============================================================================
# Backward Compatibility: Partitioner Classes (Phase 2 - Task 14.2)
# ============================================================================


class Partitioner:
    """Base partitioner interface (backward compatibility)."""

    def get_partition_key(self, message: Any) -> bytes | None:  # pragma: no cover - interface
        raise NotImplementedError


class SymbolPartitioner(Partitioner):
    """Partition by normalized symbol (backward compatibility)."""

    def get_partition_key(self, message: Any) -> bytes | None:
        return normalize_symbol(getattr(message, "symbol", None)).encode("utf-8")


class CompositePartitioner(Partitioner):
    """Partition by normalized exchange-symbol combination (backward compatibility)."""

    def get_partition_key(self, message: Any) -> bytes | None:
        exchange = normalize_exchange(getattr(message, "exchange", None))
        symbol = normalize_symbol(getattr(message, "symbol", None))
        return f"{exchange}-{symbol}".encode("utf-8")


class ExchangePartitioner(Partitioner):
    """Partition by normalized exchange (backward compatibility)."""

    def get_partition_key(self, message: Any) -> bytes | None:
        return normalize_exchange(getattr(message, "exchange", None)).encode("utf-8")


class RoundRobinPartitioner(Partitioner):
    """Round robin (no partition key) (backward compatibility)."""

    def get_partition_key(self, message: Any) -> bytes | None:
        return None


class PartitionerFactory:
    """Factory for creating partitioners by strategy name (backward compatibility)."""

    @staticmethod
    def create(strategy: str | None = "composite") -> Partitioner:
        strategy_lower = (strategy or "composite").lower()
        if strategy_lower == "symbol":
            return SymbolPartitioner()
        if strategy_lower == "exchange":
            return ExchangePartitioner()
        if strategy_lower == "round_robin":
            return RoundRobinPartitioner()
        if strategy_lower == "composite":
            return CompositePartitioner()
        raise ValueError(f"Unknown partitioner strategy: {strategy}")


# ============================================================================
# Inlined Header Building (Phase 2 - Task 14.1)
# ============================================================================

def _build_headers(message: Any, data_type: str, content_type: str, schema_version: str = "v1") -> list[tuple[bytes, bytes]]:
    """Build complete set of headers for a message (inlined from headers.py).

    Combines mandatory headers (exchange, symbol, data_type, content-type) with
    optional headers (schema_version, producer_version, timestamp_generated).

    Args:
        message: Message object with exchange and symbol attributes
        data_type: Data type name (e.g., 'trades', 'orderbook')
        content_type: Serialization format (e.g., 'application/x-protobuf')
        schema_version: Protobuf schema version (default: 'v1')

    Returns:
        List of (header_name, header_value) tuples with bytes values
    """
    from datetime import datetime, timezone

    # Extract and normalize metadata
    exchange = normalize_exchange(getattr(message, "exchange", None))
    symbol = normalize_symbol(getattr(message, "symbol", None))

    # Encode helper
    def _enc(val: Any) -> bytes:
        if isinstance(val, bytes):
            return val
        return str(val).encode("utf-8")

    # Build mandatory headers (4)
    headers = [
        (b"content-type", _enc(content_type)),
        (b"exchange", _enc(exchange)),
        (b"symbol", _enc(symbol)),
        (b"data_type", _enc(data_type)),
    ]

    # Build optional headers (3)
    iso_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    headers.extend([
        (b"schema_version", _enc(schema_version)),
        (b"producer_version", b"2.4.1"),
        (b"timestamp_generated", _enc(iso_str)),
        (b"cf.serialization_format", b"json"),
    ])

    return headers


# ============================================================================
# Backward Compatibility: Header Builder Classes (Inlined from headers.py)
# ============================================================================


class MessageHeaders:
    """Build mandatory (and optional) headers for a message."""

    @staticmethod
    def build(message: Any, data_type: str, content_type: str) -> list[tuple[bytes, bytes]]:
        def _enc(val: Any) -> bytes:
            if isinstance(val, bytes):
                return val
            return str(val).encode("utf-8")

        exchange = normalize_exchange(getattr(message, "exchange", None))
        symbol = normalize_symbol(getattr(message, "symbol", None))

        return [
            (b"content-type", _enc(content_type)),
            (b"exchange", _enc(exchange)),
            (b"symbol", _enc(symbol)),
            (b"data_type", _enc(data_type)),
        ]


class OptionalHeaders:
    """Build optional headers only (schema_version, producer_version, timestamp_generated, cf.serialization_format)."""

    @staticmethod
    def build(
        schema_version: str = "v1",
        producer_version: Optional[str] = None,
        timestamp_generated: Optional[str] = None,
        serialization_format: str = "json",
        include_serialization_format: bool = True,
    ) -> list[tuple[bytes, bytes]]:
        from datetime import datetime, timezone

        def _enc(val: Any) -> bytes:
            if isinstance(val, bytes):
                return val
            return str(val).encode("utf-8")

        producer_version = producer_version or "2.4.1"
        if timestamp_generated is None:
            timestamp_generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        headers: list[tuple[bytes, bytes]] = [
            (b"schema_version", _enc(schema_version)),
            (b"producer_version", _enc(producer_version)),
            (b"timestamp_generated", _enc(timestamp_generated)),
        ]

        if include_serialization_format:
            headers.append((b"cf.serialization_format", _enc(serialization_format)))

        return headers


class HeaderEnricher:
    """Legacy enricher facade that forwards to the unified header builder."""

    def __init__(
        self,
        content_type: str = "application/x-protobuf",
        schema_version: str = "v1",
        producer_version: Optional[str] = None,
        timestamp_generated: Optional[str] = None,
        serialization_format: str = "json",
        include_serialization_header: bool = True,
    ) -> None:
        self.content_type = content_type
        self.schema_version = schema_version
        self.producer_version = producer_version
        self.timestamp_generated = timestamp_generated
        self.serialization_format = serialization_format
        self._include_serialization_header = include_serialization_header

    def build(self, message: Any, data_type: str) -> list[tuple[bytes, bytes]]:
        mandatory = MessageHeaders.build(message, data_type, self.content_type)
        optional = OptionalHeaders.build(
            schema_version=self.schema_version,
            producer_version=self.producer_version,
            timestamp_generated=self.timestamp_generated,
            serialization_format=self.serialization_format,
            include_serialization_format=self._include_serialization_header,
        )
        return mandatory + optional


# ============================================================================
# Direct Prometheus Metrics (Phase 3, Task 15.3)
# ============================================================================

def _create_kafka_metrics():
    """Create Kafka metrics using prometheus_client directly (no wrappers).

    Returns:
        Dict of metric objects, or None if prometheus_client unavailable
    """
    if not PROMETHEUS_AVAILABLE:
        LOG.debug("prometheus_client not available, metrics disabled")
        return None

    try:
        from prometheus_client import REGISTRY

        # Helper to get or create metrics (handle duplicate registration)
        def _get_or_create_counter(name, doc, labelnames):
            try:
                return Counter(name, doc, labelnames)
            except ValueError:
                # Metric already exists, retrieve from registry
                return REGISTRY._names_to_collectors.get(name)

        def _get_or_create_histogram(name, doc, labelnames, buckets=None):
            try:
                if buckets:
                    return Histogram(name, doc, labelnames, buckets=buckets)
                return Histogram(name, doc, labelnames)
            except ValueError:
                # Metric already exists, retrieve from registry
                return REGISTRY._names_to_collectors.get(name)

        # Counter: messages_produced_total
        messages_produced_total = _get_or_create_counter(
            'kafka_messages_produced_total',
            'Total number of messages successfully produced to Kafka',
            ['exchange', 'symbol', 'data_type', 'partition_strategy']
        )

        # Histogram: produce_latency_seconds
        produce_latency_seconds = _get_or_create_histogram(
            'kafka_produce_latency_seconds',
            'Latency of message production from callback to broker acknowledgment',
            ['exchange', 'data_type'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
        )

        # Counter: produce_errors_total
        produce_errors_total = _get_or_create_counter(
            'kafka_produce_errors_total',
            'Total number of produce errors',
            ['exchange', 'data_type', 'error_type']
        )

        # Histogram: message_size_bytes
        message_size_bytes = _get_or_create_histogram(
            'kafka_message_size_bytes',
            'Distribution of serialized message sizes in bytes',
            ['data_type', 'compression_enabled'],
            buckets=(100, 250, 500, 1000, 2500, 5000, 10000)
        )

        # Histogram: serialization_latency_seconds
        serialization_latency_seconds = _get_or_create_histogram(
            'kafka_serialization_latency_seconds',
            'Time taken to serialize message to protobuf format',
            ['data_type'],
            buckets=(0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01)
        )

        return {
            'messages_produced_total': messages_produced_total,
            'produce_latency_seconds': produce_latency_seconds,
            'produce_errors_total': produce_errors_total,
            'message_size_bytes': message_size_bytes,
            'serialization_latency_seconds': serialization_latency_seconds,
        }
    except Exception as e:
        LOG.warning(f"Error creating Prometheus metrics: {e}")
        return None


def _record_message_produced(metrics, exchange: str, symbol: str, data_type: str, partition_strategy: str):
    """Record a successfully produced message."""
    if metrics is None:
        return
    try:
        metrics['messages_produced_total'].labels(
            exchange=exchange,
            symbol=symbol,
            data_type=data_type,
            partition_strategy=partition_strategy
        ).inc()
    except Exception as e:
        LOG.debug(f"Error recording message produced metric: {e}")


def _record_produce_latency(metrics, latency_seconds: float, exchange: str, data_type: str):
    """Record message produce latency."""
    if metrics is None:
        return
    try:
        metrics['produce_latency_seconds'].labels(
            exchange=exchange,
            data_type=data_type
        ).observe(latency_seconds)
    except Exception as e:
        LOG.debug(f"Error recording produce latency metric: {e}")


def _record_produce_error(metrics, exchange: str, data_type: str, error_type: str):
    """Record a produce error."""
    if metrics is None:
        return
    try:
        metrics['produce_errors_total'].labels(
            exchange=exchange,
            data_type=data_type,
            error_type=error_type
        ).inc()
    except Exception as e:
        LOG.debug(f"Error recording produce error metric: {e}")


def _record_message_size(metrics, size_bytes: int, data_type: str, compression_enabled: bool):
    """Record serialized message size."""
    if metrics is None:
        return
    try:
        metrics['message_size_bytes'].labels(
            data_type=data_type,
            compression_enabled=str(compression_enabled)
        ).observe(size_bytes)
    except Exception as e:
        LOG.debug(f"Error recording message size metric: {e}")


def _record_serialization_latency(metrics, latency_seconds: float, data_type: str):
    """Record message serialization latency."""
    if metrics is None:
        return
    try:
        metrics['serialization_latency_seconds'].labels(
            data_type=data_type
        ).observe(latency_seconds)
    except Exception as e:
        LOG.debug(f"Error recording serialization latency metric: {e}")


class KafkaCallback(KafkaBackendBase):
    """Backend callback that routes normalized messages to Kafka.

    Supports two initialization modes:
    1. Direct parameters (backward compatible):
       KafkaCallback(bootstrap_servers=['kafka:9092'], acks='all')

    2. KafkaConfig object (recommended):
       config = KafkaConfig(bootstrap_servers=['kafka:9092'])
       KafkaCallback(kafka_config=config)
    """

    def __init__(
        self,
        *,
        bootstrap_servers: Iterable[str] | None = None,
        kafka_config: KafkaConfig | None = None,
        acks: str | None = "all",
        enable_idempotence: bool | None = True,
        connection_timeout_ms: int = 5000,
        producer_factory: Callable[..., Any] | None = None,
        serialization_format: str | None = None,
        numeric_type=float,
        none_to=None,
        queue_maxsize: int = 0,
        enable_batch_drain: bool = True,
        batch_drain_size: int = 50,
        enable_partition_key_cache: bool = True,
        partition_key_cache_size: int = 10000,  # Increased from 1000 for TODO #011
        poll_batch_size: int = 100,  # TODO #010: Batch polling optimization
        enable_header_precomputation: bool = True,
        drain_frequency_ms: int = 10,
        metrics_exporter=None,  # DEPRECATED: kept for backward compatibility
        metrics_enabled: bool = True,
        metrics_producer_id: str | None = None,
        **config: Any,
    ) -> None:
        if not hasattr(self, "_schema_version"):
            self._schema_version = DEFAULT_SCHEMA_VERSION

        # Create direct prometheus metrics (Phase 3, Task 15.3)
        self._metrics = _create_kafka_metrics() if metrics_enabled else None

        # Handle deprecated metrics_exporter parameter (backward compatibility)
        if metrics_exporter is not None:
            warnings.warn(
                "metrics_exporter parameter is deprecated. Metrics now use prometheus_client directly. "
                "Use metrics_enabled=True/False to control metrics collection.",
                DeprecationWarning,
                stacklevel=2
            )

        super().__init__(
            queue_maxsize=queue_maxsize,
            enable_batch_drain=enable_batch_drain,
            batch_drain_size=batch_drain_size,
            drain_frequency_ms=drain_frequency_ms,
            metrics_exporter=None,  # No longer used
        )

        # Provide a ref to module logger so tests can patch callback.LOG
        self._log_ref = lambda: LOG

        # Handle KafkaConfig parameter (Task 4.2 - refactoring)
        if kafka_config is not None:
            # Load settings from KafkaConfig (flattened dataclass)
            self.bootstrap_servers = kafka_config.bootstrap_servers
            self.acks = kafka_config.acks
            self.enable_idempotence = kafka_config.enable_idempotence
            # Extract flattened config fields
            self._topic_strategy = kafka_config.topic_strategy
            self._topic_prefix = kafka_config.topic_prefix
            self._partition_strategy = kafka_config.partition_strategy
            # Extract other producer settings from config
            config.setdefault("batch_size", kafka_config.batch_size)
            config.setdefault("linger_ms", kafka_config.linger_ms)
            config.setdefault("compression_type", kafka_config.compression_type)
            config.setdefault("retries", kafka_config.retries)
            config.setdefault("retry_backoff_ms", kafka_config.retry_backoff_ms)
            cutoff_override = None  # Not in flattened config yet
        elif bootstrap_servers is not None:
            # Backward compatible: direct parameters
            self.bootstrap_servers = list(bootstrap_servers)
            self.acks = acks
            self.enable_idempotence = enable_idempotence if enable_idempotence is not None else True
            # Use default values matching KafkaConfig defaults
            self._topic_strategy = "consolidated"
            self._topic_prefix = "cryptofeed"
            self._partition_strategy = "composite"
            cutoff_override = None
        else:
            raise TypeError(
                "Either 'bootstrap_servers' (direct parameters) or 'kafka_config' "
                "(KafkaConfig object) must be provided"
            )

        self._protobuf_cutoff = _get_cutoff(cutoff_override)

        self.connection_timeout_ms = connection_timeout_ms
        self.numeric_type = numeric_type
        self.none_to = none_to

        # Performance optimization parameters (Task 17.1)
        self._enable_batch_drain = enable_batch_drain
        self._batch_drain_size = batch_drain_size
        self._enable_partition_key_cache = enable_partition_key_cache
        self._partition_key_cache_size = partition_key_cache_size
        self._enable_header_precomputation = enable_header_precomputation

        if serialization_format is not None:
            self.set_serialization_format(serialization_format)
            if (
                serialization_format == "protobuf"
                and self.__class__ is KafkaCallback
            ):
                if _protobuf_mode_allowed(self._protobuf_cutoff):
                    _emit_protobuf_deprecation_warning(self._protobuf_cutoff)
                else:
                    raise RuntimeError(
                        f"KafkaCallback protobuf mode is disabled after {_format_disable_date(self._protobuf_cutoff)}; use KafkaProtobufCallback"
                    )

        # Instantiate topic manager with config strategy (Task 4.3)
        self._topic_manager = TopicManager()

        # Add partition key cache if enabled (Task 17.1 - secondary optimization)
        # TODO #011: Use OrderedDict for proper LRU eviction
        if self._enable_partition_key_cache:
            self._partition_key_cache: OrderedDict[tuple, Optional[bytes]] = OrderedDict()
            self._partition_cache_hits = 0
            self._partition_cache_misses = 0
        else:
            self._partition_key_cache = None

        # TODO #010: Batch polling optimization
        self._poll_counter = 0
        self._poll_batch_size = poll_batch_size

        # Store header configuration (inlined from HeaderEnricher)
        self._header_content_type = (
            "application/x-protobuf"
            if serialization_format == "protobuf"
            else "application/json"
        )
        self._header_schema_version = self._schema_version if hasattr(self, "_schema_version") else "v1"

        self._producer = KafkaProducer(
            self.bootstrap_servers,
            acks=self.acks,
            enable_idempotence=self.enable_idempotence,
            producer_factory=producer_factory,
            connection_timeout_ms=self.connection_timeout_ms,
            **config,
        )
        self._producer.connect()
        # Metrics are now created directly in __init__ (self._metrics)
        compression_value = config.get("compression_type")
        self._compression_enabled = (
            str(compression_value or "").lower() not in ("", "none")
        )

    def is_connected(self) -> bool:
        return self._producer.is_connected

    def queue_size(self) -> int:
        return super().queue_size()

    def get_health_status(self, timeout_ms: int = 3000) -> Dict[str, Any]:
        """
        Check Kafka producer connectivity status (Task 14.3 simplified health check).

        Validates connectivity by calling list_topics on the underlying producer.
        This is a basic health check that returns essential producer status.

        Args:
            timeout_ms: Timeout for connectivity check in milliseconds (default: 3000)

        Returns:
            Dictionary with health check results:
            - ok: bool (True if connected)
            - latency_ms: float (connection latency)
            - error: str | None (error message if failed)
            - bootstrap: list (broker addresses used)
        """
        start = time.time()
        try:
            # Check if producer is initialized
            if self._producer is None or self._producer._producer is None:
                latency_ms = (time.time() - start) * 1000
                return {
                    "ok": False,
                    "latency_ms": latency_ms,
                    "error": "Producer not initialized",
                    "bootstrap": list(self.bootstrap_servers),
                }

            # Attempt to list topics to verify connectivity
            timeout_sec = timeout_ms / 1000 if timeout_ms else None
            self._producer._producer.list_topics(timeout=timeout_sec)

            latency_ms = (time.time() - start) * 1000
            return {
                "ok": True,
                "latency_ms": latency_ms,
                "error": None,
                "bootstrap": list(self.bootstrap_servers),
            }
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            return {
                "ok": False,
                "latency_ms": latency_ms,
                "error": str(exc),
                "bootstrap": list(self.bootstrap_servers),
            }

    @property
    def _header_enricher(self):
        """Backward compatibility property for tests accessing _header_enricher."""
        class _HeaderEnricherCompat:
            def __init__(self, callback):
                self.callback = callback
                self.content_type = callback._header_content_type
                self.schema_version = callback._header_schema_version

            def build(self, message, data_type):
                return _build_headers(message, data_type, self.content_type, self.schema_version)

        return _HeaderEnricherCompat(self)

    @property
    def _partitioner(self):
        """Backward compatibility property for tests accessing _partitioner."""
        return PartitionerFactory.create(self._partition_strategy)

    # ------------------------------------------------------------------
    # Serialization + Kafka writer loop
    # ------------------------------------------------------------------
    def _topic_name(self, data_type: str, obj: Any) -> str:
        """Generate topic name using TopicManager with configured strategy.

        Uses the topic strategy from configuration (consolidated or per_symbol).
        Falls back to old behavior for backward compatibility if TopicManager not available.
        """
        exchange = getattr(obj, "exchange", "unknown")
        symbol = getattr(obj, "symbol", "unknown")

        try:
            # Use TopicManager with configured strategy (Task 4.3)
            # Note: TopicManager already includes 'cryptofeed' in the base topic name,
            # so we only pass custom prefix if it differs from the default
            custom_prefix = (
                self._topic_prefix
                if self._topic_prefix and self._topic_prefix != "cryptofeed"
                else None
            )
            return TopicManager.get_topic(
                data_type=data_type,
                symbol=symbol,
                exchange=exchange,
                strategy=self._topic_strategy,
                prefix=custom_prefix
            )
        except Exception:
            # Fallback to old behavior for backward compatibility
            symbol = symbol.replace("/", "-").replace("_", "-").lower()
            exchange = exchange.lower()
            return f"cryptofeed.{data_type}.{exchange}.{symbol}"

    def _partition_key(self, obj: Any) -> Optional[bytes]:
        """Generate partition key using configured strategy (inlined).

        Uses the partition strategy from configuration (composite, symbol, exchange, round_robin).
        Implements partition key caching optimization (Task 17.1 - secondary).
        """
        # Partition key caching: avoid recomputing keys for same (exchange, symbol) pairs
        if self._enable_partition_key_cache:
            exchange = getattr(obj, "exchange", None)
            symbol = getattr(obj, "symbol", None)
            cache_key = (exchange, symbol)

            # Check cache first
            # TODO #011: Mark as recently used for proper LRU
            if cache_key in self._partition_key_cache:
                self._partition_cache_hits += 1
                self._partition_key_cache.move_to_end(cache_key)  # Mark as recently used
                return self._partition_key_cache[cache_key]

            # Cache miss: compute and store
            self._partition_cache_misses += 1

        try:
            # Use inline partition key function (Task 14.2)
            key = _get_partition_key(obj, self._partition_strategy)

            # Store in cache if enabled
            # TODO #011: Proper LRU eviction with OrderedDict
            if self._enable_partition_key_cache:
                # Add to cache
                self._partition_key_cache[cache_key] = key
                # Evict oldest entry if over capacity (proper LRU)
                if len(self._partition_key_cache) > self._partition_key_cache_size:
                    self._partition_key_cache.popitem(last=False)  # Remove oldest (FIFO)

            return key
        except Exception:
            # Fallback to old behavior for backward compatibility
            symbol = getattr(obj, "symbol", None)
            if symbol is None:
                return None
            return str(symbol).encode()

    def _serialize_payload(self, obj: Any, receipt_timestamp: Optional[float]):
        """Serialize message payload using configured format."""
        timestamp = receipt_timestamp if receipt_timestamp is not None else getattr(obj, "timestamp", None)
        if self.serialization_format == "protobuf":
            # For protobuf, subclasses (KafkaProtobufCallback) override this method.
            # Base path kept for backward compatibility but only sets content-type.

            payload = serialize_to_protobuf(obj)
            headers = [("content-type", b"application/x-protobuf")]
        else:
            payload_dict = self._build_dict_payload(obj, timestamp or 0)
            payload = dumps_bytes(payload_dict)
            headers = [("content-type", b"application/json")]
        return payload, headers

    async def _process_message(self, message: KafkaQueuedMessage) -> None:
        """Process a single queued message (extracted from _drain_once for reuse).

        This method handles the actual message processing pipeline:
        1. Serialize payload
        2. Generate topic name
        3. Generate partition key
        4. Build headers
        5. Produce to Kafka

        Args:
            message: Queued message to process

        Raises:
            Handles all exceptions internally to prevent writer task collapse
        """
        metrics = getattr(self, "_metrics", None)
        try:
            # Extract metadata for error logging
            exchange = getattr(message.obj, "exchange", "unknown")
            symbol = getattr(message.obj, "symbol", "unknown")
            data_type = message.data_type

            # Step 1: Serialize payload
            try:
                serialization_start = time.perf_counter() if metrics else None
                payload, base_headers = self._serialize_payload(message.obj, message.receipt_timestamp)
                if metrics and serialization_start is not None:
                    _record_serialization_latency(
                        metrics,
                        time.perf_counter() - serialization_start,
                        data_type,
                    )
                    _record_message_size(
                        metrics,
                        len(payload),
                        data_type,
                        compression_enabled=self._compression_enabled,
                    )
            except Exception as e:
                LOG.error(
                    "KafkaCallback: Serialization failed for %s message from %s/%s: %s",
                    data_type, exchange, symbol, e,
                    extra={
                        "exchange": exchange,
                        "symbol": symbol,
                        "data_type": data_type,
                        "error_type": "serialization_error",
                        "error": str(e)
                    }
                )
                if metrics:
                    _record_produce_error(metrics, exchange, data_type, "serialization_error")
                return  # Skip this message, continue processing queue

            # Step 2: Generate topic name using TopicManager
            try:
                topic = self._topic_name(data_type, message.obj)
            except Exception as e:
                LOG.error(
                    "KafkaCallback: Topic resolution failed for %s message from %s/%s: %s",
                    data_type, exchange, symbol, e,
                    extra={
                        "exchange": exchange,
                        "symbol": symbol,
                        "data_type": data_type,
                        "error_type": "topic_resolution_error",
                        "error": str(e)
                    }
                )
                if metrics:
                    _record_produce_error(metrics, exchange, data_type, "topic_resolution_error")
                return  # Skip this message, continue processing queue

            # Step 3: Generate partition key using Partitioner (with caching)
            try:
                key = self._partition_key(message.obj)
            except Exception as e:
                LOG.warning(
                    "KafkaCallback: Partition key generation failed for %s/%s, using None: %s",
                    exchange, symbol, e,
                    extra={
                        "exchange": exchange,
                        "symbol": symbol,
                        "data_type": data_type,
                        "error_type": "partition_key_error",
                        "error": str(e)
                    }
                )
                key = None  # Fall back to None (round-robin partition assignment)

            # Step 4: Build enriched headers (inlined from HeaderEnricher)
            try:
                enriched_headers = _build_headers(
                    message=message.obj,
                    data_type=data_type,
                    content_type=self._header_content_type,
                    schema_version=self._header_schema_version,
                )
                # Ensure serialization-format and schema headers from payload/base are preserved
                enriched_headers = self._merge_headers(base_headers, enriched_headers)
                if not enriched_headers:
                    LOG.warning(
                        "KafkaCallback: missing headers after enrichment for %s/%s (%s); base=%s",
                        exchange,
                        symbol,
                        data_type,
                        base_headers,
                        extra={
                            "exchange": exchange,
                            "symbol": symbol,
                            "data_type": data_type,
                            "error_type": "missing_headers",
                        },
                    )
                else:
                    LOG.debug(
                        "KafkaCallback: headers for %s/%s (%s): %s",
                        exchange,
                        symbol,
                        data_type,
                        enriched_headers,
                    )
            except Exception as e:
                LOG.warning(
                    "KafkaCallback: Header enrichment failed for %s/%s, using base headers: %s",
                    exchange, symbol, e,
                    extra={
                        "exchange": exchange,
                        "symbol": symbol,
                        "data_type": data_type,
                        "error_type": "header_enrichment_error",
                        "error": str(e)
                    }
                )
                enriched_headers = self._fallback_headers(base_headers)

            if not self._validate_schema_headers(enriched_headers, exchange, symbol, data_type):
                return

            # Step 5: Produce to Kafka
            try:
                def _to_bytes(val: Any) -> bytes:
                    if isinstance(val, bytes):
                        return val
                    return str(val).encode()

                normalized_headers = [(_to_bytes(k), _to_bytes(v)) for k, v in enriched_headers]

                produce_start = time.perf_counter() if metrics else None
                self._producer.produce(topic, payload, key=key, headers=normalized_headers)

                # TODO #010: Batch polling optimization - only poll every N messages
                self._poll_counter += 1
                if self._poll_counter >= self._poll_batch_size:
                    self._producer.poll(0.0)
                    self._poll_counter = 0

                if metrics and produce_start is not None:
                    _record_produce_latency(
                        metrics,
                        time.perf_counter() - produce_start,
                        exchange,
                        data_type,
                    )
                    _record_message_produced(
                        metrics,
                        exchange,
                        symbol,
                        data_type,
                        self._partition_strategy,
                    )
            except Exception as e:
                LOG.error(
                    "KafkaCallback: Kafka produce failed for %s message from %s/%s on topic %s: %s",
                    data_type, exchange, symbol, topic, e,
                    extra={
                        "exchange": exchange,
                        "symbol": symbol,
                        "data_type": data_type,
                        "topic": topic,
                        "error_type": "kafka_produce_error",
                        "error": str(e)
                    }
                )
                if metrics:
                    _record_produce_error(metrics, exchange, data_type, "kafka_produce_error")
                # Note: Producer retries are configured in KafkaProducer settings
                # We continue processing to avoid blocking the queue on transient errors
        except Exception as e:
            # Catch-all for unexpected errors to prevent writer task collapse
            LOG.error(
                "KafkaCallback: Unexpected error in _process_message: %s",
                e,
                extra={
                    "error_type": "unexpected_error",
                    "error": str(e)
                }
            )

    async def _shutdown_backend(self) -> None:
        self._producer.close()

    def _fallback_headers(self, base_headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
        """Build fallback headers when enrichment fails (inline implementation)."""
        from datetime import datetime, timezone

        def _enc(val: Any) -> bytes:
            if isinstance(val, bytes):
                return val
            return str(val).encode("utf-8")

        iso_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        optional = [
            (b"schema_version", _enc(self._header_schema_version)),
            (b"producer_version", b"2.4.1"),
            (b"timestamp_generated", _enc(iso_str)),
            (b"cf.serialization_format", _enc(self.serialization_format)),
        ]
        return base_headers + optional

    @staticmethod
    def _merge_headers(base: list[tuple[bytes, bytes | str]], enriched: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
        """Merge base and enriched headers, keeping first occurrence per key.

        Ensures schema_version / serialization_format emitted by serializers are
        retained alongside enriched mandatory headers.
        """
        def _to_bytes(val: bytes | str) -> bytes:
            return val if isinstance(val, bytes) else val.encode("utf-8")

        result: list[tuple[bytes, bytes]] = []
        seen = set()
        for name, value in base + enriched:
            name_bytes = _to_bytes(name) if isinstance(name, str) else name
            value_bytes = _to_bytes(value)
            if name_bytes in seen:
                continue
            seen.add(name_bytes)
            result.append((name_bytes, value_bytes))
        return result

    def _validate_schema_headers(
        self,
        headers: list[tuple[bytes, bytes]],
        exchange: str,
        symbol: str,
        data_type: str,
    ) -> bool:
        """Validate that required schema headers are present."""
        header_names = {name for name, _ in headers}
        required = {b"schema_version", b"cf.serialization_format"}
        missing = required - header_names
        if missing:
            LOG.error(
                "KafkaCallback: Missing schema headers %s for %s/%s (%s)",
                ", ".join(name.decode("utf-8") for name in missing),
                exchange,
                symbol,
                data_type,
            )
            return False
        return True

# ------------------------------------------------------------------
# Partition Key Strategies (Task 2)
# ------------------------------------------------------------------


# Partition strategy helpers moved to cryptofeed.backends.kafka.partitioner

# Message header helpers moved to cryptofeed.backends.kafka.headers

class HealthStatus(str, Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(slots=True)
class HealthCheckResponse:
    """Health check response model (Task 17.3).

    Attributes:
        status: Overall health status (healthy/degraded/unhealthy)
        kafka_connected: Whether Kafka broker is connected
        buffer_health: Buffer utilization (0.0-1.0, where 0=empty, 1.0=full)
        queue_size: Current queue size in messages
        messages_produced: Total messages produced
        errors_total: Total errors encountered
        circuit_breaker_state: Circuit breaker state (CLOSED/OPEN/HALF_OPEN)
        last_message_timestamp: Unix timestamp of last message
        memory_bytes: Memory usage in bytes
        uptime_seconds: Producer uptime in seconds
    """
    status: str
    kafka_connected: bool
    buffer_health: float
    queue_size: int
    messages_produced: int
    errors_total: int
    circuit_breaker_state: str
    last_message_timestamp: Optional[float]
    memory_bytes: int
    uptime_seconds: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = self.to_dict()
        # Handle float precision
        data["buffer_health"] = round(data["buffer_health"], 2)
        return dumps_bytes(data).decode("utf-8")


class HealthCheckDeterminer:
    """Determines health status based on metrics (Task 17.3)."""

    @staticmethod
    def determine_status(
        kafka_connected: bool,
        buffer_utilization: float,
        error_rate: float,
        circuit_breaker_state: str,
    ) -> str:
        """Determine health status based on metrics.

        Status Logic:
        - HEALTHY: Kafka connected, buffer < 80%, error rate < 0.1%, circuit CLOSED
        - DEGRADED: Kafka connected, buffer 80-95%, error rate 0.1-1%, circuit HALF_OPEN
        - UNHEALTHY: Kafka disconnected, buffer >= 95%, error rate >= 1%, circuit OPEN

        Args:
            kafka_connected: Whether Kafka broker is accessible
            buffer_utilization: Buffer utilization percentage (0-100)
            error_rate: Error rate as decimal (0-1)
            circuit_breaker_state: Circuit breaker state

        Returns:
            Health status string
        """
        # Check for unhealthy conditions
        if not kafka_connected:
            return HealthStatus.UNHEALTHY.value
        if buffer_utilization >= 95:
            return HealthStatus.UNHEALTHY.value
        if error_rate >= 0.01:  # >= 1%
            return HealthStatus.UNHEALTHY.value
        if circuit_breaker_state == "OPEN":
            return HealthStatus.UNHEALTHY.value

        # Check for degraded conditions
        if buffer_utilization >= 80:
            return HealthStatus.DEGRADED.value
        if error_rate >= 0.001:  # >= 0.1%
            return HealthStatus.DEGRADED.value
        if circuit_breaker_state == "HALF_OPEN":
            return HealthStatus.DEGRADED.value

        # Otherwise healthy
        return HealthStatus.HEALTHY.value

    @staticmethod
    def get_http_status_code(health_status: str) -> int:
        """Get HTTP status code for health status.

        Args:
            health_status: Health status string

        Returns:
            HTTP status code (200 for healthy, 503 for degraded/unhealthy)
        """
        if health_status == HealthStatus.HEALTHY.value:
            return 200
        return 503
def _emit_protobuf_deprecation_warning(cutoff):
    warnings.warn(
        (
            "Use KafkaProtobufCallback for protobuf payloads; "
            f"KafkaCallback protobuf mode will be removed after {_format_disable_date(cutoff)}."
        ),
        DeprecationWarning,
        stacklevel=3,
    )
