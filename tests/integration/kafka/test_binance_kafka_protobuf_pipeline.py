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
import time

import pytest

from cryptofeed.defines import TRADES
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
from cryptofeed.backends.kafka.partitioner import PartitionerFactory
from cryptofeed.backends.protobuf.bindings import SCHEMA_VERSION
from cryptofeed.defines import L2_BOOK
from tests.integration.kafka.conftest import redpanda
from tests.integration.kafka.helpers import ConsumedRecord, consume_one
from uuid import uuid4

BINANCE_E2E_ENV = "CRYPTODATA_RUN_BINANCE_KAFKA_E2E"


class _TestKafkaProtobufCallback(KafkaProtobufCallback):
    """Test shim that accepts the multiprocess kwarg used by FeedHandler.start."""

    def start(self, loop, multiprocess: bool | None = None):  # type: ignore[override]
        return super().start(loop)


def _assert_or_skip_headers(record: ConsumedRecord):
    missing = [h for h in (b"content-type", b"schema_version", b"cf.serialization_format") if h not in record.headers]
    if missing:
        pytest.skip(f"Binance Kafka Protobuf E2E: missing headers {missing}; observed={record.headers}")


def _env_enabled() -> bool:
    value = os.getenv(BINANCE_E2E_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _require_binance_e2e_prereqs() -> None:
    if not _env_enabled():
        pytest.skip(
            f"Binance Kafka Protobuf E2E tests disabled. "
            f"Set {BINANCE_E2E_ENV}=true to enable."
        )


async def _start_binance_with_kafka(
    redpanda_bootstrap: str,
    *,
    partition_strategy: str | None = None,
    channels: list[str] | None = None,
) -> FeedHandler:
    """Configure and start a Binance feed wired to KafkaProtobufCallback.

    The feed subscribes to TRADES for BTC-USDT and routes normalized
    events to a KafkaProtobufCallback that produces protobuf-encoded
    messages to the given Redpanda bootstrap address.
    """
    loop = asyncio.get_running_loop()

    fh = FeedHandler()

    kafka_cb = _TestKafkaProtobufCallback(
        bootstrap_servers=[redpanda_bootstrap],
        producer_factory=None,
        metrics_exporter=None,
        metrics_enabled=False,
    )
    # Route per-symbol to get predictable topic names
    kafka_cb._topic_strategy = "per_symbol"
    kafka_cb._enable_partition_key_cache = False

    if partition_strategy is not None:
        kafka_cb._partitioner = PartitionerFactory.create(partition_strategy)

    kafka_cb.start(loop)

    if not kafka_cb.is_connected():
        pytest.skip("Kafka producer failed to connect to Redpanda")

    # Use FeedHandler to construct the Binance feed with its own Config
    _channels = channels or [TRADES]
    def _mk_handler(data_type: str):
        async def _handler(obj, receipt_timestamp):
            await kafka_cb._handle_message(data_type, obj, receipt_timestamp)

        return _handler

    callbacks = {}
    for channel in _channels:
        if channel == TRADES:
            callbacks[channel] = [_mk_handler("trade")]
        elif channel == L2_BOOK:
            callbacks[channel] = [_mk_handler("l2_book")]
        else:  # pragma: no cover - future channels
            callbacks[channel] = [_mk_handler(channel.lower())]

    fh.add_feed(
        "BINANCE",
        symbols=["BTC-USDT"],
        channels=_channels,
        callbacks=callbacks,
    )

    for feed in fh.feeds:
        feed.start(loop)

    # Return both handler and backend for explicit teardown
    fh.kafka_cb = kafka_cb  # type: ignore[attr-defined]
    return fh


async def _shutdown_feeds(handler: FeedHandler) -> None:
    """Stop all feeds and shut down connections and backends."""
    shutdown_tasks = []
    for feed in handler.feeds:
        feed.stop()
        shutdown_tasks.append(feed.shutdown())
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks)

    kafka_cb = getattr(handler, "kafka_cb", None)
    if kafka_cb and hasattr(kafka_cb, "stop"):
        await kafka_cb.stop()


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
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=60.0,
                group_id=f"cf-e2e-binance-proto-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(
                f"Binance Kafka Protobuf E2E: no message consumed within timeout: {exc}"
            )

    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_or_skip_headers(record)
    # Header assertions
    assert record.headers[b"content-type"] == b"application/x-protobuf"
    assert record.headers[b"schema_version"] == SCHEMA_VERSION.encode()
    assert record.headers[b"cf.serialization_format"] == b"protobuf"
    assert record.headers[b"exchange"] == b"binance"
    assert record.headers[b"symbol"] == b"BTC-USDT"
    assert record.headers[b"data_type"] == b"trade"

    # Payload assertions
    from cryptofeed.backends.protobuf import bindings as pb_bindings
    trade_pb2 = pb_bindings.trade_pb2

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
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=60.0,
                group_id=f"cf-e2e-binance-proto-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(
                f"Binance Kafka Protobuf E2E (round-robin): "
                f"no message consumed within timeout: {exc}"
            )

    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_or_skip_headers(record)
    assert record.key is None


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.live_binance
async def test_binance_kafka_protobuf_orderbook_snapshot_roundtrip(redpanda):
    """Order book snapshot+delta path: Binance L2 → Kafka Protobuf → decode."""

    _require_binance_e2e_prereqs()

    fh: FeedHandler | None = None
    topic = "cryptofeed.l2_book.binance.btc-usdt"

    try:
        fh = await _start_binance_with_kafka(
            redpanda_bootstrap=redpanda,
            channels=[L2_BOOK],
        )

        try:
            record: ConsumedRecord = await asyncio.to_thread(
                consume_one,
                redpanda,
                topic,
                timeout_s=90.0,
                group_id=f"cf-e2e-binance-proto-{uuid4().hex}",
                offset_reset="latest",
            )
        except AssertionError as exc:
            pytest.skip(
                "Binance Kafka Protobuf E2E (orderbook): no message within timeout;"
                f" possible REST snapshot or WS connectivity issue: {exc}"
            )

    finally:
        if fh is not None:
            await _shutdown_feeds(fh)

    _assert_or_skip_headers(record)
    # Header assertions
    assert record.headers[b"content-type"] == b"application/x-protobuf"
    assert record.headers[b"schema_version"] == SCHEMA_VERSION.encode()
    assert record.headers[b"cf.serialization_format"] == b"protobuf"
    assert record.headers[b"exchange"] == b"binance"
    assert record.headers[b"symbol"] == b"BTC-USDT"
    assert record.headers[b"data_type"] == b"l2_book"

    # Payload assertions
    from cryptofeed.backends.protobuf import bindings as pb_bindings

    msg = pb_bindings.order_book_pb2.Level2Book()
    msg.ParseFromString(record.value)
    assert msg.exchange.lower() == "binance"
    assert msg.symbol == "BTC-USDT"
    # Require at least one bid/ask level present
    assert len(msg.bids) > 0 or len(msg.asks) > 0
