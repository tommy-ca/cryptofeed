"""KafkaCallback core implementation for the market data producer."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Optional, Literal

from cryptofeed.backends.backend import BackendCallback
from cryptofeed.json_utils import dumps_bytes

from .kafka_producer import KafkaProducer


LOG = logging.getLogger("feedhandler")


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
    """Backend callback that routes normalized messages to Kafka."""

    def __init__(
        self,
        *,
        bootstrap_servers: Iterable[str],
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
        self.bootstrap_servers = list(bootstrap_servers)
        self.acks = acks
        self.enable_idempotence = enable_idempotence if enable_idempotence is not None else True
        self.connection_timeout_ms = connection_timeout_ms
        self.numeric_type = numeric_type
        self.none_to = none_to

        if serialization_format is not None:
            self.set_serialization_format(serialization_format)

        self._queue: asyncio.Queue[_QueuedMessage | object] = asyncio.Queue(maxsize=queue_maxsize)

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
        exchange = getattr(obj, "exchange", "unknown").lower()
        symbol = getattr(obj, "symbol", "unknown").replace("/", "-").replace("_", "-").lower()
        return f"cryptofeed.{data_type}.{exchange}.{symbol}"

    def _partition_key(self, obj: Any) -> Optional[bytes]:
        symbol = getattr(obj, "symbol", None)
        if symbol is None:
            return None
        return str(symbol).encode()

    def _serialize_payload(self, obj: Any, receipt_timestamp: Optional[float]):
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
        message = await self._queue.get()
        try:
            if message is _STOP_SENTINEL:
                return

            assert isinstance(message, _QueuedMessage)

            payload, headers = self._serialize_payload(message.obj, message.receipt_timestamp)
            topic = self._topic_name(message.data_type, message.obj)
            key = self._partition_key(message.obj)

            self._producer.produce(topic, payload, key=key, headers=headers)
            self._producer.poll(0.0)
        finally:
            self._queue.task_done()

    async def _writer(self) -> None:
        while self._running:
            await self._drain_once()
