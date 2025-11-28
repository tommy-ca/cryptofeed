"""
Shared infrastructure for Kafka backends (queueing, writers, method binding).
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from cryptofeed.backends.backend import BackendCallback


LOG = logging.getLogger("feedhandler")


_STOP_SENTINEL = object()


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
}


@dataclass(slots=True)
class KafkaQueuedMessage:
    """Internal message container shared across Kafka callbacks."""

    data_type: str
    obj: Any
    receipt_timestamp: Optional[float]


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


__all__ = [
    "KafkaBackendBase",
    "KafkaQueuedMessage",
]
