"""Consolidated Kafka backend module (Phase 3, Task 15.1).

Combines KafkaBackendBase, KafkaProducer, and TopicManager into a single module
to reduce file count and simplify the Kafka backend architecture.

This module contains:
1. Message types (KafkaQueuedMessage)
2. Producer wrapper (KafkaProducer)
3. Topic naming (TopicManager, TopicStrategy)
4. Backend base class (KafkaBackendBase)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from confluent_kafka import KafkaException, Producer

from cryptofeed.backends.backend import BackendCallback
from .normalization import normalize_exchange, normalize_symbol


LOG = logging.getLogger("feedhandler")


# ============================================================================
# Section 1: Message Types
# ============================================================================

_STOP_SENTINEL = object()


@dataclass(slots=True)
class KafkaQueuedMessage:
    """Internal message container shared across Kafka callbacks."""

    data_type: str
    obj: Any
    receipt_timestamp: Optional[float]


_SUPPORTED_METHODS: Dict[str, str] = {
    "trade": "trade",
    "orderbook": "orderbook",
    "ticker": "ticker",
    "candle": "candle",
    "liquidation": "liquidation",
    "funding": "funding",
    "open_interest": "openinterest",
    "order_info": "order",
    "balances": "balance",
    "transactions": "transaction",
    "fills": "fill",
    "index": "index",
    "indices": "index",
    "position": "position",
    "positions": "position",
    "top_of_book": "top_of_book",
    "nbbo": "top_of_book",
    "l2_delta": "l2_delta",
    "level2_delta": "l2_delta",
    "candles": "candle",
}


# ============================================================================
# Section 2: Kafka Producer (merged from producer.py)
# ============================================================================

def _normalize_bootstrap_servers(servers: Iterable[str]) -> str:
    parts = [str(server).strip() for server in servers if str(server).strip()]
    if not parts:
        raise ValueError("bootstrap_servers must not be empty")
    return ",".join(parts)


@dataclass(slots=True)
class DeliveryReport:
    topic: str
    partition: int
    offset: int


class KafkaProducer:
    """Kafka producer helper with connection checks and delivery semantics."""

    def __init__(
        self,
        bootstrap_servers: Sequence[str],
        *,
        acks: str | None = "all",
        enable_idempotence: bool | None = True,
        producer_factory: Callable[[Mapping[str, Any]], Producer] | None = None,
        connection_timeout_ms: int = 5000,
        logger: logging.Logger | None = None,
        **config: Any,
    ) -> None:
        self._bootstrap_servers = list(bootstrap_servers)
        self._acks = acks
        self._enable_idempotence = enable_idempotence
        self._config = dict(config)
        self._producer_factory = producer_factory or Producer
        self._connection_timeout_ms = connection_timeout_ms
        self._producer: Producer | None = None
        self._logger = logger or LOG
        self._connected: bool = False

    @property
    def bootstrap_servers(self) -> Sequence[str]:
        return list(self._bootstrap_servers)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _build_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = dict(self._config)
        config["bootstrap.servers"] = _normalize_bootstrap_servers(self._bootstrap_servers)
        if self._acks is not None:
            config["acks"] = self._acks
        if self._enable_idempotence is not None:
            config["enable.idempotence"] = "true" if self._enable_idempotence else "false"
        return config

    def connect(self) -> None:
        if self._producer is None:
            config = self._build_config()
            try:
                self._producer = self._producer_factory(config)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ConnectionError("failed to initialize Kafka producer") from exc

        try:
            timeout_sec = self._connection_timeout_ms / 1000 if self._connection_timeout_ms else None
            self._producer.list_topics(timeout=timeout_sec)
        except KafkaException as exc:
            self._connected = False
            message = str(exc.args[0] if exc.args else exc)
            raise ConnectionError(f"kafka brokers unreachable: {message}") from exc
        else:
            self._connected = True
            self._logger.info(
                "Kafka producer connected [bootstrap=%s, idempotent=%s]",
                ",".join(self._bootstrap_servers),
                bool(self._enable_idempotence),
            )

    def produce(
        self,
        topic: str,
        value: bytes,
        *,
        key: Optional[bytes] = None,
        headers: Optional[list[tuple[str, bytes]]] = None,
        on_delivery: Optional[Callable[[DeliveryReport], None]] = None,
    ) -> None:
        if self._producer is None:
            raise RuntimeError("producer not connected")

        def _delivery_callback(err, msg):
            if err is not None:
                self._logger.error("Kafka delivery error: %s", err)
                return
            if on_delivery is not None:
                on_delivery(
                    DeliveryReport(
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                    )
                )

        self._producer.produce(
            topic,
            value=value,
            key=key,
            headers=headers,
            on_delivery=_delivery_callback if on_delivery else None,
        )

    def poll(self, timeout: float = 0.0) -> int:
        if self._producer is None:
            return 0
        return self._producer.poll(timeout)

    def flush(self, timeout: Optional[float] = None) -> int:
        if self._producer is None:
            return 0
        return self._producer.flush(timeout)

    def close(self, timeout: Optional[float] = None) -> None:
        if self._producer is not None:
            try:
                flush_timeout = 5.0 if timeout is None else timeout
                self._producer.flush(flush_timeout)
            finally:
                self._producer = None
                self._connected = False


# ============================================================================
# Section 3: Topic Naming (merged from topic_manager.py)
# ============================================================================

class TopicStrategy(Enum):
    """Topic naming strategies for Kafka topics."""

    CONSOLIDATED = "consolidated"
    PER_SYMBOL = "per_symbol"


class TopicManager:
    """Manages topic naming strategies for Kafka topics."""

    # Supported data types (normalized to singular form for topic naming)
    SUPPORTED_DATA_TYPES = {
        "trade",
        "trades",  # plural form supported for backward compatibility
        "orderbook",
        "l2_book",
        "ticker",
        "candle",
        "funding",
        "liquidation",
        "index",
        "open_interest",
        "openinterest",
        "fill",
        "balance",
        "position",
        "margin",
        "order",
        "transaction",
    }

    STRATEGIES = {"consolidated", "per_symbol"}

    @staticmethod
    def validate_strategy(strategy: str) -> None:
        if strategy not in TopicManager.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Supported strategies: {', '.join(sorted(TopicManager.STRATEGIES))}"
            )

    @staticmethod
    def validate_data_type(data_type: str) -> None:
        if data_type not in TopicManager.SUPPORTED_DATA_TYPES:
            sorted_types = ", ".join(sorted(TopicManager.SUPPORTED_DATA_TYPES))
            raise ValueError(
                f"Unsupported data type: {data_type}. Supported types: {sorted_types}"
            )

    @staticmethod
    def get_topic(
        data_type: str,
        symbol: str,
        exchange: str,
        strategy: str = "consolidated",
        prefix: Optional[str] = None,
    ) -> str:
        TopicManager.validate_strategy(strategy)
        TopicManager.validate_data_type(data_type)

        strategy = strategy.lower()
        prefix_clean = prefix.strip() if prefix else ""

        if strategy == TopicStrategy.CONSOLIDATED.value:
            topic_body = f"cryptofeed.{data_type}"
        else:
            # Validate required fields for per_symbol strategy
            if not symbol:
                raise ValueError("Symbol is required for per_symbol topic strategy")
            if not exchange:
                raise ValueError("Exchange is required for per_symbol topic strategy")

            normalized_exchange = normalize_exchange(exchange)
            normalized_symbol = normalize_symbol(symbol)
            topic_body = (
                f"cryptofeed.{data_type}.{normalized_exchange}.{normalized_symbol}"
            )

        if prefix_clean:
            return f"{prefix_clean}.{topic_body}"
        return topic_body


# ============================================================================
# Section 4: Backend Base Class (merged from base.py)
# ============================================================================

class KafkaBackendBase(BackendCallback, ABC):
    """
    Base class that owns queue lifecycles, batching, and writer orchestration.
    """

    SUPPORTED_METHODS: Dict[str, str] = _SUPPORTED_METHODS

    def __init__(
        self,
        *,
        queue_maxsize: int = 0,
        enable_batch_drain: bool = True,
        batch_drain_size: int = 50,
        drain_frequency_ms: int = 10,
        metrics_exporter: Any | None = None,
    ) -> None:
        super().__init__()
        self._queue: asyncio.Queue[KafkaQueuedMessage | object] = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._enable_batch_drain = enable_batch_drain
        self._batch_drain_size = batch_drain_size
        self._drain_frequency_ms = drain_frequency_ms
        self._loop: asyncio.AbstractEventLoop | None = None
        self._writer_task: asyncio.Task | None = None
        self._running: bool = False
        self._log_name = self.__class__.__name__
        self._metrics = metrics_exporter

    # ------------------------------------------------------------------ #
    # Lifecycle helpers
    # ------------------------------------------------------------------ #
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
            self._queue.put_nowait(_STOP_SENTINEL)

        if self._writer_task is not None:
            await self._writer_task
            self._writer_task = None

        await self._shutdown_backend()

    def queue_size(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------ #
    # Message handling helpers
    # ------------------------------------------------------------------ #
    def __getattr__(self, name: str):
        canonical = self.SUPPORTED_METHODS.get(name)
        if canonical is None:
            raise AttributeError(name)

        async def _handler(obj, receipt_timestamp: float):
            await self._handle_message(canonical, obj, receipt_timestamp)

        setattr(self, name, _handler)
        return _handler

    async def _handle_message(
        self,
        data_type: str,
        obj: Any,
        receipt_timestamp: Optional[float],
    ) -> None:
        queued = self._queue_message(data_type, obj, receipt_timestamp)
        if not queued:
            LOG.warning("%s: dropped %s message due to full queue", self._log_name, data_type)

    def _queue_message(
        self,
        data_type: str,
        obj: Any,
        receipt_timestamp: Optional[float] = None,
    ) -> bool:
        if receipt_timestamp is None:
            receipt_timestamp = time.time()
        message = KafkaQueuedMessage(
            data_type=data_type,
            obj=obj,
            receipt_timestamp=receipt_timestamp,
        )
        exchange = getattr(obj, "exchange", "unknown")
        symbol = getattr(obj, "symbol", "unknown")

        logger = LOG
        if hasattr(self, "_log_ref"):
            try:
                logger = self._log_ref()
            except Exception:
                logger = LOG
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.error(
                "%s queue is full; dropping %s message from %s/%s (queue size: %d)",
                self._log_name,
                data_type,
                exchange,
                symbol,
                self._queue.maxsize,
                extra={
                    "exchange": exchange,
                    "symbol": symbol,
                    "data_type": data_type,
                    "queue_size": self._queue.maxsize,
                    "error_type": "queue_full",
                },
            )
            return False
        return True

    # ------------------------------------------------------------------ #
    # Writer orchestration
    # ------------------------------------------------------------------ #
    async def _writer(self) -> None:
        while self._running:
            if self._enable_batch_drain:
                await self._drain_batch()
            else:
                await self._drain_once()

    async def _drain_once(self) -> None:
        """Process a single queued message if available.

        Uses a non-blocking poll so callers can safely invoke this in tests
        (e.g., empty queue sanity checks) without hanging indefinitely.
        """
        try:
            message = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            if message is _STOP_SENTINEL:
                return
            await self._process_message(message)
        finally:
            try:
                self._queue.task_done()
            except Exception as e:
                logger = LOG
                if hasattr(self, "_log_ref"):
                    try:
                        logger = self._log_ref()
                    except Exception:
                        logger = LOG
                logger.error(
                    "%s: Failed to mark task as done: %s",
                    self._log_name,
                    e,
                    extra={
                        "error_type": "task_done_error",
                        "error": str(e),
                    },
                )

    async def _drain_batch(self) -> None:
        batch_count = 0
        max_batch = self._batch_drain_size

        while batch_count < max_batch:
            try:
                message = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                if message is _STOP_SENTINEL:
                    self._running = False
                    return
                await self._process_message(message)
                batch_count += 1
            finally:
                try:
                    self._queue.task_done()
                except Exception as e:
                    LOG.error(
                        "%s: Failed to mark task as done: %s",
                        self._log_name,
                        e,
                        extra={
                            "error_type": "task_done_error",
                            "error": str(e),
                        },
                    )

        await asyncio.sleep(0)

    # ------------------------------------------------------------------ #
    # Extension points
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def _process_message(self, message: KafkaQueuedMessage) -> None:
        ...

    @abstractmethod
    async def _shutdown_backend(self) -> None:
        ...


# ============================================================================
# Section 5: Public Exports
# ============================================================================

__all__ = [
    "KafkaBackendBase",
    "KafkaQueuedMessage",
    "KafkaProducer",
    "DeliveryReport",
    "TopicStrategy",
    "TopicManager",
]
