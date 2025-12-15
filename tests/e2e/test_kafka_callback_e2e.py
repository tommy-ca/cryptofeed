"""End-to-end test for KafkaCallback with concurrent message flow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

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
    """In-memory stand-in for confluent-kafka Producer."""

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

    def topic(self):
        return self._topic

    def partition(self):  # pragma: no cover - unused
        return self._partition

    def offset(self):  # pragma: no cover - unused
        return 0


def _trade(exchange: str, symbol: str, uid: int) -> Trade:
    return Trade(
        exchange=exchange,
        symbol=symbol,
        side="buy",
        amount=Decimal("1.00"),
        price=Decimal("48000.00") + Decimal(uid),
        timestamp=1700000100.0 + uid,
        id=f"trade-{uid}",
        type="spot",
        raw=None,
    )


async def _drain(producer: InMemoryProducer, expected: int) -> None:
    for _ in range(200):
        if len(producer.messages) >= expected:
            return
        await asyncio.sleep(0)
    raise AssertionError("KafkaCallback did not emit expected message count")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_kafka_callback_concurrent_flow_e2e():
    producer: InMemoryProducer | None = None

    def factory(config):
        nonlocal producer
        producer = InMemoryProducer(config)
        return producer

    callback = KafkaCallback(
        bootstrap_servers=["localhost:9092"],
        serialization_format="protobuf",
        queue_maxsize=32,
        producer_factory=factory,
    )

    loop = asyncio.get_event_loop()
    callback.start(loop)

    try:
        assert callback.is_connected()

        trades = [
            _trade("COINBASE", "BTC-USD", idx)
            for idx in range(5)
        ] + [
            _trade("BINANCE", "ETH-USDT", idx)
            for idx in range(5, 10)
        ]

        await asyncio.gather(
            *(callback.trade(trade, receipt_timestamp=trade.timestamp + 0.25) for trade in trades)
        )

        assert producer is not None
        await _drain(producer, expected=len(trades))

        topics = {message.topic for message in producer.messages}
        # Consolidated topic strategy (default): all trades go to single topic
        assert "cryptofeed.trade" in topics
        assert len(topics) == 1  # All messages use consolidated topic

        for message in producer.messages:
            assert message.value
            assert message.headers and (b"content-type", b"application/x-protobuf") in message.headers

        assert callback.queue_size() == 0
    finally:
        await callback.stop()
