"""KafkaCallback core implementation for the market data producer."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict

from cryptofeed.json_utils import dumps_bytes

from .base import KafkaBackendBase, KafkaQueuedMessage
from .producer import KafkaProducer
from .topic_manager import TopicManager, TopicStrategy
from .partitioner import (
    Partitioner,
    SymbolPartitioner,
    CompositePartitioner,
    ExchangePartitioner,
    RoundRobinPartitioner,
    PartitionerFactory,
)
from .headers import HeaderEnricher, OptionalHeaders
from .metrics import PrometheusMetricsExporter


LOG = logging.getLogger("feedhandler")


class KafkaTopicConfig(BaseModel):
    """Configuration for Kafka topic management.

    Attributes:
        strategy: Topic naming strategy ('consolidated' or 'per_symbol').
                 Default: 'consolidated'
        prefix: Topic name prefix (e.g., 'cryptofeed', 'production').
               Whitespace-only prefixes default to 'cryptofeed'.
               Default: 'cryptofeed'
        partitions_per_topic: Number of partitions per topic.
                             Must be > 0. Default: 3
        replication_factor: Replication factor for topic.
                           Must be > 0. Default: 3

    Example:
        >>> config = KafkaTopicConfig(
        ...     strategy='consolidated',
        ...     prefix='production',
        ...     partitions_per_topic=12,
        ...     replication_factor=3
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(default="consolidated", description="Topic naming strategy")
    prefix: str = Field(default="cryptofeed", description="Topic name prefix")
    partitions_per_topic: int = Field(default=3, description="Partitions per topic")
    replication_factor: int = Field(default=3, description="Replication factor")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate topic strategy is supported."""
        if v not in {"consolidated", "per_symbol"}:
            raise ValueError(
                f"Invalid topic strategy: {v}. "
                f"Must be 'consolidated' or 'per_symbol'"
            )
        return v

    @field_validator("prefix", mode="before")
    @classmethod
    def normalize_prefix(cls, v: Optional[str]) -> str:
        """Normalize prefix: whitespace-only becomes 'cryptofeed'."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return "cryptofeed"
        return v

    @field_validator("partitions_per_topic")
    @classmethod
    def validate_partitions(cls, v: int) -> int:
        """Validate partitions_per_topic is positive."""
        if v <= 0:
            raise ValueError("partitions_per_topic must be > 0")
        return v

    @field_validator("replication_factor")
    @classmethod
    def validate_replication(cls, v: int) -> int:
        """Validate replication_factor is positive."""
        if v <= 0:
            raise ValueError("replication_factor must be > 0")
        return v


class KafkaPartitionConfig(BaseModel):
    """Configuration for partition key strategies.

    Attributes:
        strategy: Partitioner strategy to use.
                 Options: 'composite' (default), 'symbol', 'exchange', 'round_robin'
                 - composite: Route by exchange-symbol (recommended)
                 - symbol: Route by symbol only (cross-exchange analysis)
                 - exchange: Route by exchange only (exchange-specific processing)
                 - round_robin: No ordering (maximum parallelism)

    Example:
        >>> config = KafkaPartitionConfig(strategy='composite')
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(default="composite", description="Partition key strategy")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate partition strategy is supported."""
        valid_strategies = {"composite", "symbol", "exchange", "round_robin"}
        if v.lower() not in valid_strategies:
            raise ValueError(
                f"Invalid partition strategy: {v}. "
                f"Must be one of: {', '.join(sorted(valid_strategies))}"
            )
        return v.lower()


class KafkaProducerConfig(BaseModel):
    """Configuration for Kafka producer client.

    Attributes:
        bootstrap_servers: List of Kafka broker addresses (required).
                          Example: ['kafka1:9092', 'kafka2:9092']
        acks: Delivery guarantee ('0', '1', 'all'). Default: 'all'
        idempotence: Enable idempotent producer (prevent duplicates). Default: True
        retries: Number of retries on failure. Default: 3
        retry_backoff_ms: Backoff time between retries in milliseconds. Default: 100
        batch_size: Maximum batch size in bytes. Default: 16384
        linger_ms: Time to wait before sending batch (ms). Default: 10
        compression_type: Compression algorithm. Default: 'snappy'
                         Options: 'none', 'gzip', 'snappy', 'lz4', 'zstd'

    Example:
        >>> config = KafkaProducerConfig(
        ...     bootstrap_servers=['kafka:9092'],
        ...     acks='all',
        ...     compression_type='snappy'
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    bootstrap_servers: list[str] = Field(
        description="Kafka broker addresses (required)"
    )
    acks: str = Field(default="all", description="Delivery guarantee")
    idempotence: bool = Field(default=True, description="Enable idempotence")
    retries: int = Field(default=3, description="Number of retries")
    retry_backoff_ms: int = Field(default=100, description="Retry backoff (ms)")
    batch_size: int = Field(default=16384, description="Batch size (bytes)")
    linger_ms: int = Field(default=10, description="Linger time (ms)")
    compression_type: str = Field(default="snappy", description="Compression type")

    @field_validator("bootstrap_servers")
    @classmethod
    def validate_bootstrap_servers(cls, v: list[str]) -> list[str]:
        """Validate bootstrap_servers is not empty."""
        if not v:
            raise ValueError("bootstrap_servers cannot be empty")
        return v

    @field_validator("acks")
    @classmethod
    def validate_acks(cls, v: str) -> str:
        """Validate acks value."""
        if v not in {"0", "1", "all"}:
            raise ValueError(f"acks must be '0', '1', or 'all', got {v}")
        return v

    @field_validator("retries")
    @classmethod
    def validate_retries(cls, v: int) -> int:
        """Validate retries is non-negative."""
        if v < 0:
            raise ValueError("retries must be >= 0")
        return v

    @field_validator("retry_backoff_ms")
    @classmethod
    def validate_retry_backoff(cls, v: int) -> int:
        """Validate retry_backoff_ms is non-negative."""
        if v < 0:
            raise ValueError("retry_backoff_ms must be >= 0")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch_size is positive."""
        if v <= 0:
            raise ValueError("batch_size must be > 0")
        return v

    @field_validator("linger_ms")
    @classmethod
    def validate_linger(cls, v: int) -> int:
        """Validate linger_ms is non-negative."""
        if v < 0:
            raise ValueError("linger_ms must be >= 0")
        return v

    @field_validator("compression_type")
    @classmethod
    def validate_compression(cls, v: str) -> str:
        """Validate compression_type is supported."""
        valid = {"none", "gzip", "snappy", "lz4", "zstd"}
        if v not in valid:
            raise ValueError(
                f"compression_type must be one of {valid}, got {v}"
            )
        return v


class KafkaConfig(BaseModel):
    """Top-level Kafka configuration combining all settings.

    Combines producer, topic, and partition configurations into a single,
    loadable configuration object. Supports loading from YAML files and
    Python dictionaries.

    Attributes:
        bootstrap_servers: Kafka broker addresses (required).
        topic: Topic configuration (nested KafkaTopicConfig).
        partition: Partition configuration (nested KafkaPartitionConfig).
        acks: Producer acks setting. Default: 'all'
        idempotence: Enable idempotence. Default: True
        retries: Retry count. Default: 3
        retry_backoff_ms: Retry backoff. Default: 100
        batch_size: Batch size. Default: 16384
        linger_ms: Linger time. Default: 10
        compression_type: Compression type. Default: 'snappy'

    Example:
        >>> # From dictionary
        >>> config = KafkaConfig.from_dict({
        ...     'bootstrap_servers': ['kafka:9092'],
        ...     'acks': 'all',
        ...     'topic': {'strategy': 'consolidated'},
        ...     'partition': {'strategy': 'composite'}
        ... })
        >>>
        >>> # From YAML file
        >>> config = KafkaConfig.from_yaml('config/kafka.yaml')
    """

    model_config = ConfigDict(extra="forbid")

    bootstrap_servers: list[str] = Field(description="Kafka broker addresses")
    topic: KafkaTopicConfig = Field(
        default_factory=KafkaTopicConfig,
        description="Topic configuration"
    )
    partition: KafkaPartitionConfig = Field(
        default_factory=KafkaPartitionConfig,
        description="Partition configuration"
    )
    acks: str = Field(default="all", description="Delivery guarantee")
    idempotence: bool = Field(default=True, description="Enable idempotence")
    retries: int = Field(default=3, description="Number of retries")
    retry_backoff_ms: int = Field(default=100, description="Retry backoff (ms)")
    batch_size: int = Field(default=16384, description="Batch size (bytes)")
    linger_ms: int = Field(default=10, description="Linger time (ms)")
    compression_type: str = Field(default="snappy", description="Compression type")

    @field_validator("bootstrap_servers")
    @classmethod
    def validate_bootstrap_servers(cls, v: list[str]) -> list[str]:
        """Validate bootstrap_servers is not empty."""
        if not v:
            raise ValueError("bootstrap_servers cannot be empty")
        return v

    @field_validator("acks")
    @classmethod
    def validate_acks(cls, v: str) -> str:
        """Validate acks value."""
        if v not in {"0", "1", "all"}:
            raise ValueError(f"acks must be '0', '1', or 'all', got {v}")
        return v

    @field_validator("compression_type")
    @classmethod
    def validate_compression(cls, v: str) -> str:
        """Validate compression_type."""
        valid = {"none", "gzip", "snappy", "lz4", "zstd"}
        if v not in valid:
            raise ValueError(f"compression_type must be one of {valid}")
        return v

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> KafkaConfig:
        """Load configuration from dictionary.

        Args:
            config_dict: Configuration dictionary with bootstrap_servers,
                        optional topic, partition, and producer settings.

        Returns:
            KafkaConfig instance

        Raises:
            ValueError: If configuration is invalid
            KeyError: If required fields are missing
        """
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> KafkaConfig:
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            KafkaConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid or configuration is incomplete
        """
        import yaml

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(f"Configuration file is empty: {yaml_path}")

        return cls.from_dict(config_dict)


# Topic strategy helpers moved to cryptofeed.backends.kafka.topic_manager


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
        partition_key_cache_size: int = 1000,
        enable_header_precomputation: bool = True,
        drain_frequency_ms: int = 10,
        metrics_exporter: PrometheusMetricsExporter | None = None,
        metrics_enabled: bool = True,
        metrics_producer_id: str | None = None,
        **config: Any,
    ) -> None:
        if not hasattr(self, "_schema_version"):
            self._schema_version = "v1"
        if metrics_exporter is None:
            producer_id = metrics_producer_id or self.__class__.__name__
            metrics_exporter = PrometheusMetricsExporter(
                producer_id=producer_id,
                enabled=metrics_enabled,
            )
            metrics_exporter.initialize()

        super().__init__(
            queue_maxsize=queue_maxsize,
            enable_batch_drain=enable_batch_drain,
            batch_drain_size=batch_drain_size,
            drain_frequency_ms=drain_frequency_ms,
            metrics_exporter=metrics_exporter,
        )

        # Handle KafkaConfig parameter (Task 4.2 - refactoring)
        if kafka_config is not None:
            # Load settings from KafkaConfig
            self.bootstrap_servers = kafka_config.bootstrap_servers
            self.acks = kafka_config.acks
            self.enable_idempotence = kafka_config.idempotence
            self.topic_config = kafka_config.topic
            self.partition_config = kafka_config.partition
            # Extract other producer settings from config
            config.setdefault("batch_size", kafka_config.batch_size)
            config.setdefault("linger_ms", kafka_config.linger_ms)
            config.setdefault("compression_type", kafka_config.compression_type)
            config.setdefault("retries", kafka_config.retries)
            config.setdefault("retry_backoff_ms", kafka_config.retry_backoff_ms)
        elif bootstrap_servers is not None:
            # Backward compatible: direct parameters
            self.bootstrap_servers = list(bootstrap_servers)
            self.acks = acks
            self.enable_idempotence = enable_idempotence if enable_idempotence is not None else True
            # Create default configs for backward compatibility
            self.topic_config = KafkaTopicConfig()
            self.partition_config = KafkaPartitionConfig()
        else:
            raise TypeError(
                "Either 'bootstrap_servers' (direct parameters) or 'kafka_config' "
                "(KafkaConfig object) must be provided"
            )

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

        # Instantiate topic manager with config strategy (Task 4.3)
        self._topic_manager = TopicManager()
        self._topic_strategy = self.topic_config.strategy
        self._topic_prefix = self.topic_config.prefix

        # Instantiate partitioner based on config (Task 4.3)
        self._partitioner = PartitionerFactory.create(self.partition_config.strategy)

        # Add partition key cache if enabled (Task 17.1 - secondary optimization)
        if self._enable_partition_key_cache:
            self._partition_key_cache: Dict[tuple, Optional[bytes]] = {}
            self._partitioner.cache_hits = 0
            self._partitioner.cache_misses = 0
        else:
            self._partition_key_cache = None

        # Instantiate header enricher (Task 4.3)
        self._header_enricher = HeaderEnricher(
            content_type="application/x-protobuf"
            if serialization_format == "protobuf"
            else "application/json"
            ,
            schema_version=self._schema_version if hasattr(self, "_schema_version") else "v1",
            serialization_format=self.serialization_format,
        )

        self._producer = KafkaProducer(
            self.bootstrap_servers,
            acks=self.acks,
            enable_idempotence=self.enable_idempotence,
            producer_factory=producer_factory,
            connection_timeout_ms=self.connection_timeout_ms,
            **config,
        )
        self._producer.connect()
        self._metrics = metrics_exporter
        compression_value = config.get("compression_type")
        self._compression_enabled = (
            str(compression_value or "").lower() not in ("", "none")
        )

    def is_connected(self) -> bool:
        return self._producer.is_connected

    def queue_size(self) -> int:
        return super().queue_size()

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
        """Generate partition key using configured partitioner strategy.

        Uses the partitioner from configuration (composite, symbol, exchange, round_robin).
        Implements partition key caching optimization (Task 17.1 - secondary).
        """
        # Partition key caching: avoid recomputing keys for same (exchange, symbol) pairs
        if self._enable_partition_key_cache:
            exchange = getattr(obj, "exchange", None)
            symbol = getattr(obj, "symbol", None)
            cache_key = (exchange, symbol)

            # Check cache first
            if cache_key in self._partition_key_cache:
                self._partitioner.cache_hits += 1
                return self._partition_key_cache[cache_key]

            # Cache miss: compute and store
            self._partitioner.cache_misses += 1

        try:
            # Use partitioner from configuration (Task 4.3)
            key = self._partitioner.get_partition_key(obj)

            # Store in cache if enabled
            if self._enable_partition_key_cache:
                # Simple LRU: clear cache if it gets too large
                if len(self._partition_key_cache) >= self._partition_key_cache_size:
                    self._partition_key_cache.clear()
                self._partition_key_cache[cache_key] = key

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
            from cryptofeed.backends.protobuf_helpers import serialize_to_protobuf

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
                    metrics.record_serialization_latency(
                        time.perf_counter() - serialization_start,
                        data_type,
                    )
                    metrics.record_message_size(
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
                    metrics.record_produce_error(exchange, data_type, "serialization_error")
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
                    metrics.record_produce_error(exchange, data_type, "topic_resolution_error")
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

            # Step 4: Build enriched headers using HeaderEnricher
            try:
                enriched_headers = self._header_enricher.build(
                    message=message.obj,
                    data_type=data_type
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
                produce_start = time.perf_counter() if metrics else None
                self._producer.produce(topic, payload, key=key, headers=enriched_headers)
                self._producer.poll(0.0)
                if metrics and produce_start is not None:
                    metrics.record_produce_latency(
                        time.perf_counter() - produce_start,
                        exchange,
                        data_type,
                    )
                    metrics.record_message_produced(
                        exchange,
                        symbol,
                        data_type,
                        self.partition_config.strategy,
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
                    metrics.record_produce_error(exchange, data_type, "kafka_produce_error")
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
        optional = OptionalHeaders.build(
            schema_version=self._schema_version,
            producer_version=self._header_enricher.producer_version,
            timestamp_generated=None,
            serialization_format=self.serialization_format,
        )
        return base_headers + optional

    def _validate_schema_headers(
        self,
        headers: list[tuple[bytes, bytes]],
        exchange: str,
        symbol: str,
        data_type: str,
    ) -> bool:
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
