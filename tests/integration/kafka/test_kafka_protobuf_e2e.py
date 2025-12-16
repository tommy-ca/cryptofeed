"""End-to-end Kafka protobuf roundtrip tests using Redpanda."""

from __future__ import annotations

import time

import pytest

from cryptofeed.backends.kafka.backend import KafkaQueuedMessage
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback
from cryptofeed.backends.protobuf.bindings import SCHEMA_VERSION
from cryptofeed.types import Trade
from tests.integration.kafka.helpers import ConsumedRecord, consume_one


def _make_obj(type_name: str):
    from decimal import Decimal

    def obj(**fields):
        return type(type_name, (), fields)()

    if type_name == "Funding":
        return obj(
            exchange="binance",
            symbol="BTC-PERP",
            rate=Decimal("0.0001"),
            rate_daily=Decimal("0.001"),
            next_funding_rate=Decimal("0.0002"),
            timestamp=1700000000.0,
        )
    if type_name == "OpenInterest":
        return obj(
            exchange="binance",
            symbol="BTC-PERP",
            open_interest=Decimal("12345"),
            timestamp=1700000000.0,
        )
    if type_name == "Index":
        return obj(
            exchange="binance",
            symbol="BTC-USDT",
            price=Decimal("68000.12"),
            timestamp=1700000000.0,
        )
    if type_name == "Liquidation":
        return obj(
            exchange="binance",
            symbol="BTC-USDT",
            side="sell",
            price=Decimal("67900"),
            quantity=Decimal("0.25"),
            order_type="market",
            liquidation_type="partial",
            timestamp=1700000000.0,
        )
    if type_name == "TopOfBook":
        return obj(
            exchange="binance",
            symbol="BTC-USDT",
            bid_price=Decimal("67950"),
            bid_size=Decimal("1.1"),
            ask_price=Decimal("67955"),
            ask_size=Decimal("0.8"),
            timestamp=1700000000.0,
        )
    if type_name == "Level2Delta":
        return obj(
            exchange="binance",
            symbol="BTC-USDT",
            bids=[(Decimal("67940"), Decimal("0.5"))],
            asks=[(Decimal("67960"), Decimal("0.4"))],
            timestamp=1700000000.0,
            sequence=101,
            checksum="abc",
        )
    if type_name == "Ticker":
        return obj(
            exchange="binance",
            symbol="BTC-USDT",
            bid=Decimal("67950"),
            ask=Decimal("67955"),
            timestamp=1700000000.0,
        )
    if type_name == "Candle":
        return obj(
            exchange="binance",
            symbol="BTC-USDT",
            start=1700000000.0,
            stop=1700000600.0,
            interval="1m",
            open=Decimal("67900"),
            high=Decimal("68010"),
            low=Decimal("67890"),
            close=Decimal("67980"),
            volume=Decimal("10.5"),
            trades=1200,
            closed=True,
            timestamp=1700000600.0,
        )
    raise ValueError(f"Unsupported type {type_name}")



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


async def _produce_generic(bootstrap: str, data_type: str, obj: object) -> str:
    cb = KafkaProtobufCallback(
        bootstrap_servers=[bootstrap],
        producer_factory=None,
        metrics_exporter=None,
    )
    cb._topic_strategy = "per_symbol"
    message = KafkaQueuedMessage(
        data_type=data_type, obj=obj, receipt_timestamp=time.time()
    )
    await cb._process_message(message)
    cb._producer.flush(2)
    topic_symbol = obj.symbol.lower().replace('/', '-').replace(':', '-')  # type: ignore[attr-defined]
    return f"cryptofeed.{data_type}.{obj.exchange.lower()}.{topic_symbol}"


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
    assert record.headers[b"symbol"] == b"btc-usd"  # Normalized to lowercase
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
@pytest.mark.integration
@pytest.mark.parametrize(
    "type_name,data_type,pb_attr,field_check",
    [
        ("Funding", "funding", "funding_pb2.Funding", lambda m: m.rate),
        ("OpenInterest", "open_interest", "open_interest_pb2.OpenInterest", lambda m: m.open_interest),
        ("Index", "index", "index_price_pb2.IndexPrice", lambda m: m.price),
        ("Liquidation", "liquidation", "liquidation_pb2.Liquidation", lambda m: m.quantity),
        ("TopOfBook", "top_of_book", "top_of_book_pb2.TopOfBook", lambda m: m.bid_price or m.ask_price),
        ("Level2Delta", "l2_delta", "level2_delta_pb2.Level2Delta", lambda m: m.bids or m.asks),
        ("Ticker", "ticker", "ticker_pb2.Ticker", lambda m: m.bid or m.ask),
        ("Candle", "candle", "candle_pb2.Candle", lambda m: m.open or m.close),
    ],
)
async def test_kafka_protobuf_misc_roundtrip(redpanda, type_name, data_type, pb_attr, field_check):
    obj = _make_obj(type_name)
    topic = await _produce_generic(redpanda, data_type, obj)

    record: ConsumedRecord = consume_one(redpanda, topic)

    # Headers
    assert record.headers[b"content-type"] == b"application/x-protobuf"
    assert record.headers[b"schema_version"] == SCHEMA_VERSION.encode()
    assert record.headers[b"cf.serialization_format"] == b"protobuf"
    assert record.headers[b"exchange"] == obj.exchange.encode()
    # Symbol is normalized to lowercase in headers (REQ-4 normalization)
    assert record.headers[b"symbol"] == obj.symbol.lower().replace('/', '-').replace(':', '-').encode()
    assert record.headers[b"data_type"] == data_type.encode()

    # Payload
    from cryptofeed.backends.protobuf import bindings as pb_bindings

    module_name, cls_name = pb_attr.split(".")
    module = getattr(pb_bindings, module_name)
    pb_cls = getattr(module, cls_name)
    msg = pb_cls()
    msg.ParseFromString(record.value)
    assert field_check(msg)


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

    # Use KafkaConfig to set partition and topic strategies
    from cryptofeed.backends.kafka.config import KafkaConfig

    config = KafkaConfig(
        bootstrap_servers=[redpanda],
        partition_strategy="round_robin",
        topic_strategy="per_symbol",
    )
    cb = KafkaProtobufCallback(
        kafka_config=config,
        producer_factory=None,
        metrics_exporter=None,
    )

    message = KafkaQueuedMessage(
        data_type="trade", obj=trade, receipt_timestamp=time.time()
    )
    await cb._process_message(message)

    topic = "cryptofeed.trade.binance.eth-usdt"
    record: ConsumedRecord = consume_one(redpanda, topic)
    assert record.key is None
