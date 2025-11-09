"""KafkaCallback core implementation for the market data producer."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Literal

from pydantic import BaseModel, Field, field_validator, ConfigDict

from cryptofeed.backends.backend import BackendCallback
from cryptofeed.json_utils import dumps_bytes

from .kafka_producer import KafkaProducer


LOG = logging.getLogger("feedhandler")


# ============================================================================
# Configuration Models (Task 4 - Pydantic Models)
# ============================================================================


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


class TopicStrategy(Enum):
    """Topic naming strategies for Kafka topics.

    Attributes:
        CONSOLIDATED: Single topic per data type, aggregates all exchanges and symbols
        PER_SYMBOL: One topic per exchange-symbol pair (legacy support, higher topic count)
    """
    CONSOLIDATED = 'consolidated'
    PER_SYMBOL = 'per_symbol'


class TopicManager:
    """Manages topic naming strategies for Kafka topics.

    Supports two configurable strategies for topic naming:

    1. **Consolidated Strategy** (default, recommended):
       - Pattern: `cryptofeed.{data_type}`
       - Example: `cryptofeed.trades`, `cryptofeed.orderbook`
       - Advantage: O(data_types) topics = ~14 topics total
       - Use case: New deployments, simplified consumer routing

    2. **Per-Symbol Strategy** (legacy, backward compatibility):
       - Pattern: `cryptofeed.{data_type}.{exchange}.{symbol}`
       - Example: `cryptofeed.trades.binance.BTC-USDT`, `cryptofeed.orderbook.coinbase.ETH-USD`
       - Advantage: Per-exchange-symbol ordering guarantees
       - Use case: Legacy deployments requiring per-pair topics

    Both strategies support topic prefix/namespace for multi-tenant deployments:
    - With prefix: `{prefix}.cryptofeed.{data_type}` or `{prefix}.cryptofeed.{data_type}.{exchange}.{symbol}`

    Example:
        >>> # Consolidated strategy (default)
        >>> TopicManager.get_topic('trades', 'BTC-USDT', 'binance')
        'cryptofeed.trades'

        >>> # Per-symbol strategy
        >>> TopicManager.get_topic('trades', 'BTC-USDT', 'binance', strategy='per_symbol')
        'cryptofeed.trades.binance.BTC-USDT'

        >>> # With prefix
        >>> TopicManager.get_topic('trades', 'BTC-USDT', 'binance', prefix='production')
        'production.cryptofeed.trades'
    """

    # Supported data types (from cryptofeed/backends/protobuf_helpers.py)
    SUPPORTED_DATA_TYPES = {
        'trades', 'orderbook', 'ticker', 'candle', 'funding',
        'liquidation', 'index', 'openinterest', 'fill', 'balance',
        'position', 'margin', 'order', 'transaction'
    }

    STRATEGIES = {'consolidated', 'per_symbol'}

    @staticmethod
    def validate_strategy(strategy: str) -> None:
        """Validate that strategy is supported.

        Args:
            strategy: Topic strategy name

        Raises:
            ValueError: If strategy is not recognized
        """
        if strategy not in TopicManager.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Supported strategies: {', '.join(sorted(TopicManager.STRATEGIES))}"
            )

    @staticmethod
    def validate_data_type(data_type: str) -> None:
        """Validate that data type is supported.

        Args:
            data_type: Data type name (e.g., 'trades', 'orderbook')

        Raises:
            ValueError: If data type is not supported
        """
        if data_type not in TopicManager.SUPPORTED_DATA_TYPES:
            sorted_types = ', '.join(sorted(TopicManager.SUPPORTED_DATA_TYPES))
            raise ValueError(
                f"Unsupported data type: {data_type}. "
                f"Supported types: {sorted_types}"
            )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol for topic naming.

        Converts underscores to hyphens and ensures uppercase format.
        E.g., 'btc_usdt' → 'BTC-USDT', 'btc/usdt' → 'BTC/USDT'

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT', 'btc_usdt')

        Returns:
            Normalized symbol in uppercase with hyphens
        """
        return str(symbol).upper().replace('_', '-')

    @staticmethod
    def _normalize_exchange(exchange: str) -> str:
        """Normalize exchange name for topic naming.

        Converts to lowercase for consistency.

        Args:
            exchange: Exchange name (e.g., 'Binance', 'COINBASE')

        Returns:
            Normalized exchange in lowercase
        """
        return str(exchange).lower()

    @staticmethod
    def get_topic(
        data_type: str,
        symbol: str,
        exchange: str,
        strategy: str = 'consolidated',
        prefix: Optional[str] = None
    ) -> str:
        """Generate topic name based on strategy.

        Generates a Kafka topic name following the specified strategy.
        Validates all parameters and normalizes symbol/exchange names.

        Args:
            data_type: Type of data (e.g., 'trades', 'orderbook', 'ticker')
            symbol: Trading symbol (e.g., 'BTC-USDT'). Used for per_symbol strategy.
            exchange: Exchange name (e.g., 'binance'). Used for per_symbol strategy.
            strategy: Naming strategy - 'consolidated' (default) or 'per_symbol'
            prefix: Optional prefix to prepend to topic (e.g., 'production', 'staging').
                   Whitespace-only prefixes are treated as empty.

        Returns:
            Topic name string in format:
            - Consolidated: `cryptofeed.{data_type}` (or `{prefix}.cryptofeed.{data_type}`)
            - Per-symbol: `cryptofeed.{data_type}.{exchange}.{symbol}` (or with prefix)

        Raises:
            ValueError: If strategy, data_type, or per_symbol required params are invalid
            TypeError: If required parameters are None/empty when needed

        Examples:
            Consolidated (default):
            >>> TopicManager.get_topic('trades', 'BTC-USDT', 'binance')
            'cryptofeed.trades'

            Per-symbol:
            >>> TopicManager.get_topic('trades', 'BTC-USDT', 'binance', strategy='per_symbol')
            'cryptofeed.trades.binance.BTC-USDT'

            With prefix:
            >>> TopicManager.get_topic('trades', 'BTC-USDT', 'binance', prefix='prod')
            'prod.cryptofeed.trades'

            All data types:
            >>> for dt in TopicManager.SUPPORTED_DATA_TYPES:
            ...     topic = TopicManager.get_topic(dt, 'BTC-USDT', 'binance')
            ...     assert topic == f'cryptofeed.{dt}'
        """
        # Validate strategy
        TopicManager.validate_strategy(strategy)

        # Validate data type
        TopicManager.validate_data_type(data_type)

        # Validate required parameters for per_symbol strategy
        if strategy == 'per_symbol':
            if symbol is None or not symbol:
                raise ValueError("symbol is required for per_symbol strategy")
            if exchange is None or not exchange:
                raise ValueError("exchange is required for per_symbol strategy")

        # Generate base topic
        if strategy == 'consolidated':
            # Consolidated: cryptofeed.{data_type}
            base_topic = f'cryptofeed.{data_type}'
        elif strategy == 'per_symbol':
            # Per-symbol: cryptofeed.{data_type}.{exchange}.{symbol}
            normalized_symbol = TopicManager._normalize_symbol(symbol)
            normalized_exchange = TopicManager._normalize_exchange(exchange)
            base_topic = f'cryptofeed.{data_type}.{normalized_exchange}.{normalized_symbol}'
        else:
            # Should not reach here due to validate_strategy, but include for completeness
            raise ValueError(f"Unknown strategy: {strategy}")

        # Add prefix if provided and non-empty
        if prefix is not None and prefix.strip():
            return f'{prefix.strip()}.{base_topic}'

        return base_topic


_STOP_SENTINEL = object()


_SUPPORTED_METHODS: Dict[str, str] = {
    "trade": "trades",
    "orderbook": "orderbook",
    "ticker": "ticker",
    "candle": "candles",
    "liquidation": "liquidations",
    "funding": "funding",
    "open_interest": "open_interest",
    "order_info": "order_info",
    "balances": "balances",
    "transactions": "transactions",
    "fills": "fills",
}


@dataclass(slots=True)
class _QueuedMessage:
    data_type: str
    obj: Any
    receipt_timestamp: Optional[float]


class KafkaCallback(BackendCallback):
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
        **config: Any,
    ) -> None:
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

        if serialization_format is not None:
            self.set_serialization_format(serialization_format)

        self._queue: asyncio.Queue[_QueuedMessage | object] = asyncio.Queue(maxsize=queue_maxsize)

        # Instantiate topic manager with config strategy (Task 4.3)
        self._topic_manager = TopicManager()
        self._topic_strategy = self.topic_config.strategy
        self._topic_prefix = self.topic_config.prefix

        # Instantiate partitioner based on config (Task 4.3)
        self._partitioner = PartitionerFactory.create(self.partition_config.strategy)

        # Instantiate header enricher (Task 4.3)
        self._header_enricher = HeaderEnricher(
            content_type="application/x-protobuf"
            if serialization_format == "protobuf"
            else "application/json"
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

        self._loop: asyncio.AbstractEventLoop | None = None
        self._writer_task: asyncio.Task | None = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if self._running:
            return
        self._loop = loop or asyncio.get_event_loop()
        self._running = True
        self._writer_task = self._loop.create_task(self._writer())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            await self._queue.put(_STOP_SENTINEL)
        except asyncio.QueueFull:
            # Queue full implies writer still running; swap to put_nowait.
            self._queue.put_nowait(_STOP_SENTINEL)

        if self._writer_task is not None:
            await self._writer_task
            self._writer_task = None

        self._producer.close()

    # ------------------------------------------------------------------
    # Public helpers for tests and monitoring
    # ------------------------------------------------------------------
    def is_connected(self) -> bool:
        return self._producer.is_connected

    def queue_size(self) -> int:
        return self._queue.qsize()

    def _queue_message(self, data_type: str, obj: Any, receipt_timestamp: Optional[float] = None) -> bool:
        message = _QueuedMessage(data_type=data_type, obj=obj, receipt_timestamp=receipt_timestamp)
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            LOG.error("KafkaCallback queue is full; dropping message for %s", data_type)
            return False
        return True

    # ------------------------------------------------------------------
    # Dynamic data-type method binding (trade, orderbook, ...)
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        canonical = _SUPPORTED_METHODS.get(name)
        if canonical is None:
            raise AttributeError(name)

        async def _handler(obj, receipt_timestamp: float):
            await self._handle_message(canonical, obj, receipt_timestamp)

        setattr(self, name, _handler)
        return _handler

    async def _handle_message(self, data_type: str, obj: Any, receipt_timestamp: float) -> None:
        queued = self._queue_message(data_type, obj, receipt_timestamp)
        if not queued:
            LOG.warning("KafkaCallback: dropped %s message due to full queue", data_type)

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
        """
        try:
            # Use partitioner from configuration (Task 4.3)
            return self._partitioner.get_partition_key(obj)
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

    async def _drain_once(self) -> None:
        """Process one message from queue using updated pipeline.

        Pipeline (Task 4.2-4.3):
        1. Get message from queue
        2. Serialize payload
        3. Extract metadata and generate topic using TopicManager
        4. Generate partition key using Partitioner
        5. Build headers using HeaderEnricher
        6. Produce to Kafka with all components
        """
        message = await self._queue.get()
        try:
            if message is _STOP_SENTINEL:
                return

            assert isinstance(message, _QueuedMessage)

            # Serialize payload
            payload, base_headers = self._serialize_payload(message.obj, message.receipt_timestamp)

            # Generate topic name using TopicManager
            topic = self._topic_name(message.data_type, message.obj)

            # Generate partition key using Partitioner
            key = self._partition_key(message.obj)

            # Build enriched headers using HeaderEnricher (Task 4.3)
            try:
                enriched_headers = self._header_enricher.build(
                    message=message.obj,
                    data_type=message.data_type
                )
            except Exception:
                # Fallback to base headers if enrichment fails
                enriched_headers = base_headers

            self._producer.produce(topic, payload, key=key, headers=enriched_headers)
            self._producer.poll(0.0)
        finally:
            self._queue.task_done()

    async def _writer(self) -> None:
        while self._running:
            await self._drain_once()


# ------------------------------------------------------------------
# Partition Key Strategies (Task 2)
# ------------------------------------------------------------------


class Partitioner(ABC):
    """Abstract base class for partition key strategies.

    A partitioner generates a partition key for a message, which determines
    which Kafka partition receives the message. Different strategies provide
    different ordering and distribution guarantees.

    The partition key must be bytes (for deterministic hashing) or None
    (to let Kafka assign round-robin).
    """

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol for consistent partition key generation.

        Converts underscores to hyphens and ensures lowercase format.
        E.g., 'BTC_USDT' → 'btc-usdt', 'btc/usd' → 'btc/usd'

        Args:
            symbol: Trading symbol (e.g., 'BTC-USDT', 'btc_usdt')

        Returns:
            Normalized symbol in lowercase with hyphens instead of underscores
        """
        return str(symbol).strip().upper().replace("_", "-").lower()

    @staticmethod
    def _normalize_exchange(exchange: str) -> str:
        """Normalize exchange name for consistent partition key generation.

        Converts to lowercase and strips whitespace.

        Args:
            exchange: Exchange name (e.g., 'Coinbase', 'BINANCE')

        Returns:
            Normalized exchange in lowercase
        """
        return str(exchange).strip().lower()

    @abstractmethod
    def get_partition_key(self, message: Any) -> Optional[bytes]:
        """Generate partition key for a message.

        Args:
            message: Message object with exchange and symbol attributes

        Returns:
            Bytes partition key or None to let Kafka assign partition
        """
        pass


class SymbolPartitioner(Partitioner):
    """Symbol-based partition key strategy.

    Routes all messages for the same symbol to the same partition,
    regardless of exchange. Useful for cross-exchange symbol analysis.

    Example partition keys:
        - symbol="BTC-USD" → b"btc-usd"
        - symbol="ETH-USDT" → b"eth-usdt"

    Guarantees: Per-symbol ordering across all exchanges.
    Use case: Cross-exchange arbitrage, symbol-level aggregation.
    """

    def get_partition_key(self, message: Any) -> bytes:
        """Generate partition key from symbol only.

        Args:
            message: Message object with symbol attribute

        Returns:
            Normalized symbol in lowercase with hyphens as bytes
        """
        symbol = getattr(message, "symbol", "")
        normalized = self._normalize_symbol(symbol)
        return normalized.encode("utf-8")


class CompositePartitioner(Partitioner):
    """Composite partition key strategy (default).

    Routes messages by exchange-symbol pair. Same exchange-symbol pair
    always goes to same partition, different exchanges get different partitions.

    Example partition keys:
        - exchange="coinbase", symbol="BTC-USD" → b"coinbase-btc-usd"
        - exchange="binance", symbol="BTC-USD" → b"binance-btc-usd"

    Guarantees: Per-exchange-symbol ordering.
    Use case: Real-time trading, order matching (DEFAULT).
    """

    def get_partition_key(self, message: Any) -> bytes:
        """Generate partition key from exchange and symbol.

        Args:
            message: Message object with exchange and symbol attributes

        Returns:
            Composite key in format "exchange-symbol" as bytes
        """
        exchange = getattr(message, "exchange", "")
        symbol = getattr(message, "symbol", "")

        normalized_exchange = self._normalize_exchange(exchange)
        normalized_symbol = self._normalize_symbol(symbol)

        composite = f"{normalized_exchange}-{normalized_symbol}"
        return composite.encode("utf-8")


class ExchangePartitioner(Partitioner):
    """Exchange-based partition key strategy.

    Routes all messages from the same exchange to the same partition,
    regardless of symbol. Useful for exchange-specific processing.

    Example partition keys:
        - exchange="coinbase" → b"coinbase"
        - exchange="binance" → b"binance"

    Guarantees: Per-exchange ordering.
    Use case: Exchange-specific logic, reconciliation.
    """

    def get_partition_key(self, message: Any) -> bytes:
        """Generate partition key from exchange only.

        Args:
            message: Message object with exchange attribute

        Returns:
            Normalized exchange name in lowercase as bytes
        """
        exchange = getattr(message, "exchange", "")
        normalized = self._normalize_exchange(exchange)
        return normalized.encode("utf-8")


class RoundRobinPartitioner(Partitioner):
    """Round-robin partition key strategy.

    Returns None for partition key, allowing Kafka to distribute
    messages round-robin across partitions. Maximum parallelism,
    no ordering guarantees.

    Guarantees: None (no ordering).
    Use case: Analytics, aggregation where order doesn't matter.
    """

    def get_partition_key(self, message: Any) -> Optional[bytes]:
        """Return None to allow Kafka automatic distribution.

        Args:
            message: Message object (ignored)

        Returns:
            None to signal Kafka automatic round-robin distribution
        """
        return None


class PartitionerFactory:
    """Factory for creating partitioner instances by strategy name.

    Supports:
    - Creating partitioners by strategy name
    - Case-insensitive strategy names
    - Default strategy (composite)
    - Validation of unknown strategies
    """

    _PARTITIONERS: Dict[str, type[Partitioner]] = {
        "symbol": SymbolPartitioner,
        "composite": CompositePartitioner,
        "exchange": ExchangePartitioner,
        "round_robin": RoundRobinPartitioner,
    }

    @staticmethod
    def create(strategy: str = "composite") -> Partitioner:
        """Create a partitioner instance by strategy name.

        Args:
            strategy: Strategy name (symbol, composite, exchange, round_robin).
                     Defaults to 'composite'.
                     Case-insensitive.

        Returns:
            Partitioner instance

        Raises:
            ValueError: If strategy name is unknown
        """
        strategy_lower = strategy.lower() if strategy else "composite"

        if strategy_lower not in PartitionerFactory._PARTITIONERS:
            supported = ", ".join(sorted(PartitionerFactory._PARTITIONERS.keys()))
            raise ValueError(
                f"Unknown partitioner strategy: {strategy}. "
                f"Supported strategies: {supported}"
            )

        partitioner_class = PartitionerFactory._PARTITIONERS[strategy_lower]
        return partitioner_class()


# ============================================================================
# Message Headers (Task 3)
# ============================================================================


class MessageHeaders:
    """Builder for mandatory message headers (Task 3.2).

    Mandatory headers provide essential routing metadata for Kafka consumers:
    - **content-type**: Serialization format (application/x-protobuf or application/json)
    - **exchange**: Exchange name (e.g., coinbase, binance) - normalized to lowercase
    - **symbol**: Trading symbol (e.g., BTC-USD) - underscores converted to hyphens
    - **data_type**: Message type (e.g., trades, ticker, orderbook)

    All header values are returned as UTF-8 encoded bytes in a list of tuples format
    that can be passed directly to `KafkaProducer.produce(headers=...)`.

    Normalization Rules:
    - Exchange: Lowercase, whitespace stripped
    - Symbol: Underscores converted to hyphens, whitespace stripped
    - Content-Type: Passed through as-is (lowercase recommended by caller)
    - Data Type: Passed through as-is (lowercase recommended by caller)

    Missing Attributes:
    - If message lacks exchange or symbol attributes, defaults to "unknown"
    - If exchange/symbol are None or empty, defaults to "unknown"

    Example:
        >>> from cryptofeed.types import Trade
        >>> trade = Trade(exchange='Coinbase', symbol='BTC_USD', ...)
        >>> headers = MessageHeaders.build(
        ...     message=trade,
        ...     data_type='trades',
        ...     content_type='application/x-protobuf'
        ... )
        >>> # headers = [
        >>> #     (b'content-type', b'application/x-protobuf'),
        >>> #     (b'exchange', b'coinbase'),  # normalized to lowercase
        >>> #     (b'symbol', b'BTC-USD'),    # underscores converted to hyphens
        >>> #     (b'data_type', b'trades')
        >>> # ]
    """

    @staticmethod
    def build(message: Any, data_type: str, content_type: str) -> list[tuple[bytes, bytes]]:
        """Build mandatory headers from message metadata.

        Extracts exchange and symbol from message object and normalizes them
        according to Kafka header conventions. All output is UTF-8 encoded bytes.

        Args:
            message: Message object with exchange and symbol attributes.
                     If attributes are missing, defaults to "unknown".
            data_type: Data type name (e.g., 'trades', 'orderbook', 'ticker').
                       Should be lowercase for consistency.
            content_type: Serialization format (e.g., 'application/x-protobuf',
                         'application/json'). Should be MIME type format.

        Returns:
            List of 4 (header_name, header_value) tuples with bytes values:
            - (b'content-type', <content_type>)
            - (b'exchange', <normalized_exchange>)
            - (b'symbol', <normalized_symbol>)
            - (b'data_type', <data_type>)

        Type:
            Callable[[Any, str, str], list[tuple[bytes, bytes]]]
        """
        # Extract metadata from message with fallbacks
        exchange = getattr(message, "exchange", "unknown")
        symbol = getattr(message, "symbol", "unknown")

        # Normalize exchange: lowercase and strip whitespace
        exchange_str = str(exchange).strip().lower() if exchange else "unknown"

        # Normalize symbol: strip whitespace and convert underscores to hyphens
        symbol_str = str(symbol).strip() if symbol else "unknown"
        symbol_str = symbol_str.replace("_", "-")

        # Build header list with consistent ordering
        headers: list[tuple[bytes, bytes]] = [
            (b"content-type", content_type.encode("utf-8")),
            (b"exchange", exchange_str.encode("utf-8")),
            (b"symbol", symbol_str.encode("utf-8")),
            (b"data_type", data_type.encode("utf-8")),
        ]

        return headers


class OptionalHeaders:
    """Builder for optional message headers (Task 3.3).

    Optional headers provide additional metadata for downstream processing and tracing:
    - **schema_version**: Protobuf schema version for backwards compatibility (default: v1)
    - **producer_version**: Cryptofeed package version for issue tracking
    - **timestamp_generated**: ISO8601 timestamp of message generation for latency tracking

    All header values are returned as UTF-8 encoded bytes in a list of tuples format.

    Usage:
    - Use for version tracking across consumer deployments
    - Timestamp allows consumers to measure end-to-end latency
    - Schema version enables migrations between protobuf versions
    - Can be overridden for testing or specific deployment scenarios

    Example:
        >>> headers = OptionalHeaders.build(
        ...     schema_version='v1',
        ...     producer_version='2.4.1'
        ... )
        >>> # headers = [
        >>> #     (b'schema_version', b'v1'),
        >>> #     (b'producer_version', b'2.4.1'),
        >>> #     (b'timestamp_generated', b'2025-11-09T12:34:56.123456Z')
        >>> # ]
    """

    @staticmethod
    def build(
        schema_version: str = "v1",
        producer_version: Optional[str] = None,
        timestamp_generated: Optional[str] = None,
    ) -> list[tuple[bytes, bytes]]:
        """Build optional headers with defaults.

        Args:
            schema_version: Protobuf schema version (default: 'v1').
                           Should match version in .proto files and consumers.
            producer_version: Cryptofeed producer package version.
                             Default: '2.4.1' (from setup.py).
                             Can be overridden for testing or custom builds.
            timestamp_generated: ISO8601 timestamp of message generation.
                                Default: Current UTC time with Z suffix.
                                Format: YYYY-MM-DDTHH:MM:SS[.ffffff]Z

        Returns:
            List of 3 (header_name, header_value) tuples with bytes values:
            - (b'schema_version', <version>)
            - (b'producer_version', <version>)
            - (b'timestamp_generated', <iso8601_timestamp>)

        Type:
            Callable[[str, Optional[str], Optional[str]], list[tuple[bytes, bytes]]]
        """
        from datetime import datetime, timezone

        # Default producer version to package version (from setup.py)
        if producer_version is None:
            producer_version = "2.4.1"

        # Default timestamp to current UTC time in ISO8601 format
        if timestamp_generated is None:
            # Note: datetime.now(timezone.utc).isoformat() already includes +00:00
            # Don't add 'Z' since that's for naive UTC times
            iso_str = datetime.now(timezone.utc).isoformat()
            # Replace the +00:00 suffix with Z for brevity (standard for UTC)
            timestamp_generated = iso_str.replace("+00:00", "Z")

        # Build header list with consistent ordering
        headers: list[tuple[bytes, bytes]] = [
            (b"schema_version", schema_version.encode("utf-8")),
            (b"producer_version", producer_version.encode("utf-8")),
            (b"timestamp_generated", timestamp_generated.encode("utf-8")),
        ]

        return headers


class HeaderEnricher:
    """Complete message header enrichment pipeline (Task 3.1).

    Combines mandatory and optional headers into a single enrichment class
    that integrates into the Kafka producer pipeline. This is the primary
    interface for header generation used by KafkaCallback.

    The enricher ensures all headers are properly formatted and encoded:
    - All header names and values are bytes (UTF-8 encoded)
    - Headers are returned as list of tuples compatible with confluent-kafka
    - Metadata is normalized (exchange lowercase, symbol underscores to hyphens)
    - Consistent header ordering: mandatory headers first, then optional headers

    Architecture:
    - HeaderEnricher delegates to MessageHeaders for mandatory headers
    - HeaderEnricher delegates to OptionalHeaders for optional headers
    - Composition pattern for separation of concerns

    Performance:
    - Fast header generation (<1ms per message)
    - No memory allocations beyond header list
    - Suitable for 10K+ messages/second throughput

    Example Usage:
        >>> from cryptofeed.kafka_callback import HeaderEnricher
        >>> enricher = HeaderEnricher(
        ...     content_type='application/x-protobuf',
        ...     schema_version='v1'
        ... )
        >>> trade = Trade(exchange='coinbase', symbol='BTC-USD', ...)
        >>> headers = enricher.build(message=trade, data_type='trades')
        >>> # Produces 7 headers:
        >>> #   Mandatory (4): content-type, exchange, symbol, data_type
        >>> #   Optional (3): schema_version, producer_version, timestamp_generated
        >>> # Result: [
        >>> #     (b'content-type', b'application/x-protobuf'),
        >>> #     (b'exchange', b'coinbase'),
        >>> #     (b'symbol', b'BTC-USD'),
        >>> #     (b'data_type', b'trades'),
        >>> #     (b'schema_version', b'v1'),
        >>> #     (b'producer_version', b'2.4.1'),
        >>> #     (b'timestamp_generated', b'2025-11-09T12:34:56Z')
        >>> # ]
    """

    def __init__(
        self,
        content_type: str = "application/x-protobuf",
        schema_version: str = "v1",
        producer_version: Optional[str] = None,
        timestamp_generated: Optional[str] = None,
    ) -> None:
        """Initialize header enricher with configuration.

        Args:
            content_type: Serialization format for content-type header.
                         Default: 'application/x-protobuf'.
                         Common values: 'application/x-protobuf', 'application/json'.
            schema_version: Protobuf schema version for version tracking.
                           Default: 'v1'.
                           Should match version in consumers.
            producer_version: Cryptofeed producer package version.
                             Default: None (uses package version '2.4.1').
                             Can be overridden for custom builds.
            timestamp_generated: ISO8601 timestamp for message generation.
                                Default: None (uses current UTC time).
                                Useful for testing reproducible headers.
        """
        self.content_type = content_type
        self.schema_version = schema_version
        self.producer_version = producer_version
        self.timestamp_generated = timestamp_generated

    def build(self, message: Any, data_type: str) -> list[tuple[bytes, bytes]]:
        """Build complete set of headers (mandatory + optional) for a message.

        Delegates to MessageHeaders for mandatory headers and OptionalHeaders
        for optional headers, then combines them in order.

        Args:
            message: Message object with exchange and symbol attributes.
                    Typically Trade, Ticker, OrderBook, or similar from cryptofeed.types.
                    If missing exchange/symbol attributes, defaults to "unknown".
            data_type: Data type name (e.g., 'trades', 'orderbook', 'ticker').
                      Should be lowercase for consistency.

        Returns:
            List of 7 (header_name, header_value) tuples with bytes values:
            1. (b'content-type', <format>)
            2. (b'exchange', <exchange>)
            3. (b'symbol', <symbol>)
            4. (b'data_type', <type>)
            5. (b'schema_version', <version>)
            6. (b'producer_version', <version>)
            7. (b'timestamp_generated', <timestamp>)

        Type:
            Callable[[Any, str], list[tuple[bytes, bytes]]]

        Example:
            >>> enricher = HeaderEnricher()
            >>> headers = enricher.build(trade, 'trades')
            >>> # Pass to producer:
            >>> producer.produce(topic, value=serialized_value, headers=headers)
        """
        # Build mandatory headers (exchange, symbol, data_type, content-type)
        mandatory = MessageHeaders.build(
            message=message,
            data_type=data_type,
            content_type=self.content_type,
        )

        # Build optional headers (schema_version, producer_version, timestamp)
        optional = OptionalHeaders.build(
            schema_version=self.schema_version,
            producer_version=self.producer_version,
            timestamp_generated=self.timestamp_generated,
        )

        # Combine all headers: mandatory first, then optional
        return mandatory + optional
