"""KafkaCallback core implementation for the market data producer."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

from cryptofeed.backends.backend import BackendCallback
from cryptofeed.json_utils import dumps_bytes

from .kafka_producer import KafkaProducer


LOG = logging.getLogger("feedhandler")


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
