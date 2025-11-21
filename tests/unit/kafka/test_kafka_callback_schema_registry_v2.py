"""Schema Registry integration path for KafkaCallback (v2 protobuf).

This test exercises the end-to-end path inside KafkaCallback when the
Schema Registry mode is enabled, without requiring a live registry or
Kafka broker. It verifies:
 - subject naming ({topic}.v2-value)
 - schema registration & caching via SchemaRegistry.create()
 - Confluent wire framing (magic byte + schema id + payload)
 - dual-production (v2 + legacy v1) when enabled
 - header enrichment carries schema_version=v2
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pytest

import cryptofeed.kafka_callback as kafka_module
from cryptofeed.kafka_callback import KafkaCallback
from cryptofeed.types import Trade


class _RecordedMessage:
    def __init__(self, topic: str, key: Optional[bytes], value: bytes, headers):
        self.topic = topic
        self.key = key
        self.value = value
        self.headers = headers


class _StubProducer:
    """Simple in-memory producer used to avoid a real Kafka broker."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.messages: List[_RecordedMessage] = []
        self.poll_count = 0

    def list_topics(self, timeout: Optional[float] = None):
        return {"topics": {}}

    def produce(self, topic: str, value: bytes, key=None, headers=None, on_delivery=None):
        self.messages.append(_RecordedMessage(topic, key, value, headers or []))
        if on_delivery:
            on_delivery(None, None)

    def poll(self, timeout: float):
        self.poll_count += 1
        return 0

    def flush(self, timeout: Optional[float] = None):
        return 0


def _producer_factory(cls):
    def _factory(config):
        return cls(config)

    return _factory


class _FakeRegistry:
    """Minimal Schema Registry stub to capture interactions."""

    def __init__(self):
        self.register_calls: List[tuple[str, str]] = []

    def register_schema(self, subject: str, schema: str, schema_type: str):
        self.register_calls.append((subject, schema_type))
        return 42

    def embed_schema_id_in_message(self, payload: bytes, schema_id: int) -> bytes:
        # Confluent wire format: magic byte 0 + 4-byte schema id + payload
        return b"\x00" + schema_id.to_bytes(4, "big") + payload

    def get_schema_id_header(self, schema_id: int) -> bytes:
        return str(schema_id).encode()


@pytest.mark.asyncio
async def test_kafka_callback_schema_registry_dual_production(monkeypatch):
    """Ensure v2 + v1 production works with Schema Registry enabled."""

    fake_registry = _FakeRegistry()
    # Ensure KafkaCallback uses our fake registry instead of making HTTP calls
    monkeypatch.setattr(
        kafka_module.SchemaRegistry, "create", lambda config: fake_registry
    )

    callback = KafkaCallback(
        bootstrap_servers=["kafka:9092"],
        producer_factory=_producer_factory(_StubProducer),
        serialization_format="protobuf",
        schema_registry_config={
            "registry_type": "confluent",
            "url": "https://schema-registry:8081",
        },
        dual_production=True,
    )

    trade = Trade(
        exchange="coinbase",
        symbol="BTC-USD",
        side="buy",
        amount=Decimal("0.25"),
        price=Decimal("68000.10"),
        timestamp=1700000000.123,
        id="t-1",
    )

    assert callback._queue_message("trade", trade) is True

    # Drain one message to trigger production
    await callback._drain_once()

    # Expect two messages when dual_production is enabled: v2 first, then v1
    produced = callback._producer._producer.messages
    assert len(produced) == 2

    v2_msg = produced[0]
    v1_msg = produced[1]

    # Topic suffix .v2 is applied for registry path
    assert v2_msg.topic.endswith(".v2")
    assert v1_msg.topic.endswith(".trade") or v1_msg.topic.endswith(".trades")

    # Registry was invoked with {topic}-value subject
    assert fake_registry.register_calls
    subject, schema_type = fake_registry.register_calls[0]
    assert subject.endswith(".v2-value")
    assert schema_type == "PROTOBUF"

    # Confluent wire format framing present (magic byte + schema id + payload)
    assert v2_msg.value[:1] == b"\x00"
    assert v2_msg.value[1:5] == (42).to_bytes(4, "big")
    assert len(v2_msg.value) > 5  # payload not empty

    # Headers include schema_version v2 for registry path
    header_dict = {k: v for k, v in v2_msg.headers}
    assert header_dict.get(b"schema_version") == b"v2"
    assert header_dict.get(b"schema_id") == b"42"

    # Producer poll invoked to flush delivery callbacks
    assert callback._producer._producer.poll_count >= 1
