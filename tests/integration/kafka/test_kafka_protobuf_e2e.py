"""End-to-end Kafka protobuf roundtrip tests using Redpanda."""

from __future__ import annotations

import time

import pytest

from cryptofeed.backends.kafka.base import KafkaQueuedMessage
from cryptofeed.backends.kafka.partitioner import PartitionerFactory
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
from cryptofeed.backends.protobuf.bindings import SCHEMA_VERSION
from cryptofeed.types import Trade
from tests.integration.kafka.helpers import ConsumedRecord, consume_one


# ---------------------------------------------------------------------------


async def _produce_trade(bootstrap: str, trade: Trade) -> None:
    cb = KafkaProtobufCallback(
        bootstrap_servers=[bootstrap],
        producer_factory=None,
        metrics_exporter=None,
    )
    # route per-symbol for easier verification
    cb._topic_strategy = "per_symbol"
    message = KafkaQueuedMessage(
        data_type="trade", obj=trade, receipt_timestamp=time.time()
    )
    await cb._process_message(message)
    # ensure message leaves client buffer
    cb._producer.flush(2)


@pytest.mark.asyncio
async def test_kafka_protobuf_trade_roundtrip(redpanda):
    trade = Trade(
        exchange="coinbase",
        symbol="BTC-USD",
        side="buy",
        amount=1.0,
        price=68000.1,
        timestamp=1700000000.0,
        id="t-1",
        type="spot",
        raw=None,
    )

    await _produce_trade(redpanda, trade)

    topic = "cryptofeed.trade.coinbase.btc-usd"
    record: ConsumedRecord = consume_one(redpanda, topic)

    # Headers
    assert record.headers[b"content-type"] == b"application/x-protobuf"
    assert record.headers[b"schema_version"] == SCHEMA_VERSION.encode()
    assert record.headers[b"cf.serialization_format"] == b"protobuf"
    assert record.headers[b"exchange"] == b"coinbase"
    assert record.headers[b"symbol"] == b"BTC-USD"
    assert record.headers[b"data_type"] == b"trade"

    # Payload
    from cryptofeed.backends.protobuf.bindings import trade_pb2

    msg = trade_pb2.Trade()
    msg.ParseFromString(record.value)
    assert msg.exchange == "coinbase"
    assert msg.symbol == "BTC-USD"
    assert msg.amount == "1.0"
    assert msg.price == "68000.1"


@pytest.mark.asyncio
async def test_kafka_protobuf_partition_key_round_robin(redpanda):
    # round_robin partitioner should emit None key
    trade = Trade(
        exchange="binance",
        symbol="ETH-USDT",
        side="sell",
        amount=0.5,
        price=3500.0,
        timestamp=1700000001.0,
        id="t-2",
        type="spot",
        raw=None,
    )

    cb = KafkaProtobufCallback(
        bootstrap_servers=[redpanda],
        producer_factory=None,
        metrics_exporter=None,
    )
    cb._partitioner = PartitionerFactory.create("round_robin")
    cb._topic_strategy = "per_symbol"

    message = KafkaQueuedMessage(
        data_type="trade", obj=trade, receipt_timestamp=time.time()
    )
    await cb._process_message(message)

    topic = "cryptofeed.trade.binance.eth-usdt"
    record: ConsumedRecord = consume_one(redpanda, topic)
    assert record.key is None
