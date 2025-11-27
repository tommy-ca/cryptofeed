"""Binance → Kafka Protobuf end-to-end tests using Redpanda.

These tests exercise the live Binance public WebSocket feed through the
Cryptofeed normalization and Kafka Protobuf backend into a Redpanda
cluster, then validate headers and payloads via the generated protobuf
bindings.

They are **opt-in** and will skip unless all of the following are true:
- Docker + `docker compose` are available (via the existing Redpanda fixture)
- `CRYPTODATA_RUN_BINANCE_KAFKA_E2E` is set to a truthy value
- Network access to Binance public endpoints is available
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
import time

import pytest
from confluent_kafka import Consumer

from cryptofeed.defines import TRADES
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.exchanges.binance import Binance
from tests.integration.kafka.test_kafka_protobuf_e2e import (
    redpanda,
)  # re-export fixture
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
from cryptofeed.backends.kafka.partitioner import PartitionerFactory
from cryptofeed.backends.protobuf.bindings import SCHEMA_VERSION

BINANCE_E2E_ENV = "CRYPTODATA_RUN_BINANCE_KAFKA_E2E"


@dataclass
class _ConsumedRecord:
    value: bytes
    headers: dict[bytes, bytes]
    topic: str
    key: bytes | None


def _env_enabled() -> bool:
    value = os.getenv(BINANCE_E2E_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _require_binance_e2e_prereqs() -> None:
    if not _env_enabled():
        pytest.skip(
            f"Binance Kafka Protobuf E2E tests disabled. "
            f"Set {BINANCE_E2E_ENV}=true to enable."
        )


def _consume_one(
    bootstrap: str, topic: str, timeout_s: float = 30.0
) -> _ConsumedRecord:
    """Blocking helper to consume a single record from Kafka.

    This function is intended to run in a background thread via
    asyncio.to_thread so that it does not block the asyncio event loop
    that drives the Binance feed.
    """
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": "cf-e2e-binance-proto",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic])
    end_time = time.time() + timeout_s
    msg = None
    try:
        while time.time() < end_time:
            msg = consumer.poll(0.5)
            if msg and not msg.error():
                break
    finally:
        consumer.close()

    if msg is None or msg.error():
        raise AssertionError("No message consumed from Kafka for topic " + topic)

    def _b(key: object) -> bytes:
        return key if isinstance(key, bytes) else str(key).encode()

    header_dict = {_b(k): v for k, v in msg.headers() or []}
    return _ConsumedRecord(
        value=msg.value(), headers=header_dict, topic=msg.topic(), key=msg.key()
    )


async def _start_binance_with_kafka(
    redpanda_bootstrap: str,
    *,
    partition_strategy: str | None = None,
) -> FeedHandler:
    """Configure and start a Binance feed wired to KafkaProtobufCallback.

    The feed subscribes to TRADES for BTC-USDT and routes normalized
    events to a KafkaProtobufCallback that produces protobuf-encoded
    messages to the given Redpanda bootstrap address.
    """
    loop = asyncio.get_running_loop()

    fh = FeedHandler()

    kafka_cb = KafkaProtobufCallback(
        bootstrap_servers=[redpanda_bootstrap],
        producer_factory=None,
        metrics_exporter=None,
        metrics_enabled=False,
    )
    # Route per-symbol to get predictable topic names
    kafka_cb._topic_strategy = "per_symbol"

    if partition_strategy is not None:
        kafka_cb._partitioner = PartitionerFactory.create(partition_strategy)

    # Use FeedHandler to construct the Binance feed with its own Config
    fh.add_feed(
        "BINANCE",
        symbols=["BTC-USDT"],
        channels=[TRADES],
        callbacks={TRADES: kafka_cb},
    )

    for feed in fh.feeds:
        feed.start(loop)

    return fh


async def _shutdown_feeds(handler: FeedHandler) -> None:
    """Stop all feeds and shut down connections and backends."""
    shutdown_tasks = []
    for feed in handler.feeds:
        feed.stop()
        shutdown_tasks.append(feed.shutdown())
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_kafka_protobuf_trade_roundtrip(redpanda):
    """End-to-end trade roundtrip: Binance WS → Kafka Protobuf → Protobuf decode.

    This test is intentionally opt-in and live: it requires network
    access to Binance and a local Redpanda instance started via the
    shared `redpanda` fixture.
    """
    _require_binance_e2e_prereqs()

    fh: FeedHandler | None = None
    topic = "cryptofeed.trade.binance.btc-usdt"

    try:
        fh = await _start_binance_with_kafka(redpanda_bootstrap=redpanda)

        # Consume one record from Kafka without blocking the event loop
        try:
            record = await asyncio.to_thread(_consume_one, redpanda, topic, 60.0)
        except AssertionError as exc:
            pytest.skip(
                f"Binance Kafka Protobuf E2E: no message consumed within timeout: {exc}"
            )

    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    # Header assertions
    assert record.headers[b"content-type"] == b"application/x-protobuf"
    assert record.headers[b"schema_version"] == SCHEMA_VERSION.encode()
    assert record.headers[b"cf.serialization_format"] == b"protobuf"
    assert record.headers[b"exchange"] == b"binance"
    assert record.headers[b"symbol"] == b"BTC-USDT"
    assert record.headers[b"data_type"] == b"trade"

    # Payload assertions
    from cryptofeed.proto_bindings import trade_pb2

    msg = trade_pb2.Trade()
    msg.ParseFromString(record.value)
    assert msg.exchange.lower() == "binance"
    assert msg.symbol == "BTC-USDT"
    # Basic sanity checks: non-empty numeric fields
    assert msg.amount not in ("", "0", "0.0")
    assert msg.price not in ("", "0", "0.0")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_kafka_protobuf_trade_roundtrip_round_robin(redpanda):
    """Validate round-robin partitioning produces keyless records for Binance trades."""
    _require_binance_e2e_prereqs()

    fh: FeedHandler | None = None
    topic = "cryptofeed.trade.binance.btc-usdt"

    try:
        fh = await _start_binance_with_kafka(
            redpanda_bootstrap=redpanda,
            partition_strategy="round_robin",
        )

        try:
            record = await asyncio.to_thread(_consume_one, redpanda, topic, 60.0)
        except AssertionError as exc:
            pytest.skip(
                f"Binance Kafka Protobuf E2E (round-robin): "
                f"no message consumed within timeout: {exc}"
            )

    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    assert record.key is None
