"""Integration tests for the KafkaCallback backend."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, List, Optional

import pytest

from cryptofeed.kafka_callback import KafkaCallback
from cryptofeed.types import Trade


@dataclass(slots=True)
class RecordedMessage:
    topic: str
    value: bytes
    key: Optional[bytes]
    headers: Optional[list[tuple[str, bytes]]]


class InMemoryProducer:
    """Minimal in-memory producer that mimics confluent-kafka Producer."""

    def __init__(self, config):
        self.config = config
        self.messages: List[RecordedMessage] = []

    def list_topics(self, timeout: Optional[float] = None):  # pragma: no cover - trivial
        return {}

    def produce(self, topic, value, key=None, headers=None, on_delivery=None):
        self.messages.append(RecordedMessage(topic=topic, value=value, key=key, headers=headers))
        if on_delivery is not None:
            on_delivery(None, _DeliveryMessage(topic, len(self.messages) - 1))

    def poll(self, timeout):  # pragma: no cover - no-op
        return 0

    def flush(self, timeout=None):  # pragma: no cover - no-op
        return 0


class _DeliveryMessage:
    def __init__(self, topic: str, partition: int):
        self._topic = topic
        self._partition = partition

    def topic(self):  # pragma: no cover - defensive guard
        return self._topic

    def partition(self):  # pragma: no cover - defensive guard
        return self._partition

    def offset(self):  # pragma: no cover - defensive guard
        return 0


def _trade_sample(symbol: str = "BTC-USD") -> Trade:
    return Trade(
        exchange="COINBASE",
        symbol=symbol,
        side="buy",
        amount=Decimal("0.5"),
        price=Decimal("68000.00"),
        timestamp=1700000000.0,
        id="kafka-test",
        type="spot",
        raw=None,
    )


def _callback(factory: Callable) -> KafkaCallback:
    return KafkaCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=factory,
        queue_maxsize=16,
    )


async def _drain(callback: KafkaCallback, expected: int, producer: InMemoryProducer):
    for _ in range(100):
        if len(producer.messages) >= expected:
            return
        await asyncio.sleep(0)
    raise AssertionError("KafkaCallback did not flush messages as expected")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("serialization_format", ["protobuf", "json"], ids=["protobuf", "json"])
@pytest.mark.parametrize("compression_type", [None, "lz4", "zstd"], ids=["no-compression", "lz4", "zstd"])
async def test_kafka_callback_serialization_with_compression(serialization_format, compression_type):
    producer_holder: dict[str, InMemoryProducer] = {}

    def factory(config):
        producer = InMemoryProducer(config)
        producer_holder["instance"] = producer
        producer_holder["config"] = config
        return producer

    kwargs = {
        "bootstrap_servers": ["kafka:9092"],
        "serialization_format": serialization_format,
        "producer_factory": factory,
        "queue_maxsize": 16,
    }

    if compression_type:
        kwargs["compression_type"] = compression_type

    callback = KafkaCallback(**kwargs)

    loop = asyncio.get_event_loop()
    callback.start(loop)

    try:
        symbol = "BTC-USD" if serialization_format == "protobuf" else "ETH-USD"
        trade = _trade_sample(symbol)
        await callback.trade(trade, receipt_timestamp=trade.timestamp + 0.5)
        producer = producer_holder["instance"]
        await _drain(callback, expected=1, producer=producer)

        message = producer.messages[0]

        if serialization_format == "protobuf":
            assert message.topic == f"cryptofeed.trades.coinbase.{symbol.lower()}"
            assert message.headers and ("content-type", b"application/x-protobuf") in message.headers
        else:
            assert message.topic == "cryptofeed.trades.coinbase.eth-usd"
            assert message.headers and ("content-type", b"application/json") in message.headers

        assert message.key == symbol.encode()
        assert message.value

        config = producer_holder.get("config", {})
        if compression_type:
            assert config.get("compression_type") == compression_type
        else:
            assert "compression_type" not in config
    finally:
        await callback.stop()
