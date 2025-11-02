"""Unit tests for the KafkaCallback foundation layer (Spec 3.1).

These tests cover configuration, connection handling, queue semantics,
and BackendCallback integration guarantees required by the Market Data
Kafka Producer specification.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional

import pytest
from confluent_kafka import KafkaException

from cryptofeed.backends.backend import BackendCallback
from cryptofeed.types import Trade

kafka_module = pytest.importorskip("cryptofeed.kafka_callback")
KafkaCallback = kafka_module.KafkaCallback


@dataclass
class _RecordedMessage:
    topic: str
    key: Optional[bytes]
    value: bytes
    headers: Dict[str, Any]


class _StubProducer:
    """In-memory producer used for unit tests without Kafka brokers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self.messages: List[_RecordedMessage] = []

    def list_topics(self, timeout: Optional[float] = None):
        self.connected = True
        return {"topics": []}

    def produce(self, topic: str, value: bytes, key: Optional[bytes] = None, headers=None, on_delivery=None):
        headers = headers or {}
        self.messages.append(
            _RecordedMessage(topic=topic, key=key, value=value, headers=headers)
        )
        if on_delivery:
            on_delivery(None, None)

    def poll(self, timeout: float):
        return 0

    def flush(self, timeout: Optional[float] = None):
        return 0


class _FailingProducer:
    """Producer factory that simulates broker connection failure."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def list_topics(self, timeout: Optional[float] = None):
        raise KafkaException("broker unreachable")


def _producer_factory(cls):
    def _factory(config):
        return cls(config)

    return _factory


def _sample_trade() -> Trade:
    return Trade(
        exchange="coinbase",
        symbol="BTC-USD",
        side="buy",
        amount=Decimal("0.25"),
        price=Decimal("68000.10"),
        timestamp=1700000000.0,
        id="trade-1",
        type="spot",
        raw=None,
    )


def test_kafka_callback_initialization():
    callback = KafkaCallback(
        bootstrap_servers=["kafka:9092"],
        acks="all",
        enable_idempotence=True,
        connection_timeout_ms=50,
        producer_factory=_producer_factory(_StubProducer),
    )

    assert callback.bootstrap_servers == ["kafka:9092"]
    assert callback.acks == "all"
    assert callback.enable_idempotence is True
    assert callback.is_connected() is True


def test_kafka_callback_invalid_bootstrap_servers():
    with pytest.raises(ConnectionError, match="broker"):
        KafkaCallback(
            bootstrap_servers=["invalid:9092"],
            connection_timeout_ms=10,
            producer_factory=_producer_factory(_FailingProducer),
        )


def test_kafka_callback_is_connected_with_stub():
    callback = KafkaCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=_producer_factory(_StubProducer),
    )

    assert callback.is_connected() is True


def test_kafka_callback_message_queueing():
    callback = KafkaCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=_producer_factory(_StubProducer),
    )

    trade = _sample_trade()
    queued = callback._queue_message("trade", trade)

    assert queued is True
    assert callback.queue_size() == 1


def test_kafka_callback_inherits_backend_callback():
    callback = KafkaCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=_producer_factory(_StubProducer),
    )

    assert isinstance(callback, BackendCallback)


def test_kafka_callback_supports_all_data_types():
    callback = KafkaCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=_producer_factory(_StubProducer),
    )

    supported = {
        "trade",
        "orderbook",
        "ticker",
        "candle",
        "liquidation",
        "funding",
        "open_interest",
        "order_info",
        "balances",
        "transactions",
        "fills",
    }

    missing: Iterable[str] = [name for name in supported if not hasattr(callback, name)]
    assert not missing


@pytest.mark.asyncio
async def test_kafka_callback_writer_drains_queue():
    stub_producer = _StubProducer({})

    callback = KafkaCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=lambda config: stub_producer,
    )

    trade = _sample_trade()
    callback._queue_message("trade", trade)

    async def _run_writer():
        await asyncio.wait_for(callback._drain_once(), timeout=1)

    await _run_writer()

    assert len(stub_producer.messages) == 1
