import asyncio

import pytest

from cryptofeed.backends.kafka import TradeKafka
from cryptofeed.serializers.formats import CALLBACK_FORMAT_ENV_VAR
from cryptofeed.proto_bindings import trade_pb2, trade_side_pb2


def create_kafka_backend(**kwargs):
    return TradeKafka(bootstrap_servers="localhost:9092", **kwargs)


def test_trade_kafka_defaults_to_json(monkeypatch):
    monkeypatch.delenv(CALLBACK_FORMAT_ENV_VAR, raising=False)
    backend = create_kafka_backend()
    assert backend.serialization_format == "json"


def test_trade_kafka_explicit_protobuf():
    backend = create_kafka_backend(serialization_format="protobuf")
    assert backend.serialization_format == "protobuf"


def test_trade_kafka_invalid_format():
    with pytest.raises(ValueError, match="Invalid serialization format"):
        create_kafka_backend(serialization_format="avro")


def test_trade_kafka_env_override(monkeypatch):
    monkeypatch.setenv(CALLBACK_FORMAT_ENV_VAR, "PROTOBUF")
    backend = create_kafka_backend(serialization_format="json")
    assert backend.serialization_format == "protobuf"


def test_trade_kafka_serialization_lock():
    backend = create_kafka_backend(serialization_format="protobuf")
    assert backend.serialization_format == "protobuf"

    with pytest.raises(RuntimeError, match="serialization format already locked"):
        backend.set_serialization_format("json")


def test_trade_kafka_mixed_instances(monkeypatch):
    monkeypatch.delenv(CALLBACK_FORMAT_ENV_VAR, raising=False)
    json_backend = create_kafka_backend()
    proto_backend = create_kafka_backend(serialization_format="protobuf")

    assert json_backend.serialization_format == "json"
    assert proto_backend.serialization_format == "protobuf"


class _DummyTrade:
    def __init__(self, exchange: str, symbol: str):
        self.exchange = exchange
        self.symbol = symbol
        self.timestamp = None

    def to_dict(self, numeric_type=float, none_to=None):
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
        }

    def to_proto(self):
        msg = trade_pb2.Trade()
        msg.exchange = self.exchange
        msg.symbol = self.symbol
        msg.price = "100.0"
        msg.amount = "1"
        msg.side = trade_side_pb2.TRADE_SIDE_BUY
        msg.timestamp = 0
        return msg


@pytest.mark.asyncio
async def test_kafka_protobuf_message_enqueued():
    backend = create_kafka_backend(serialization_format="protobuf")
    backend.multiprocess = False
    backend.queue = asyncio.Queue()

    trade = _DummyTrade("coinbase", "BTC-USD")
    await backend.__call__(trade, receipt_timestamp=123.0)

    message = await backend.queue.get()
    assert message["format"] == "protobuf"
    assert isinstance(message["payload"], bytes)
    assert message["metadata"]["exchange"] == "coinbase"
    assert message["metadata"]["symbol"] == "BTC-USD"

    topic = backend.topic(message)
    assert topic == "cryptofeed.market.trades.coinbase"

    partition_key = backend.partition_key(message)
    assert partition_key == b"BTC-USD"


@pytest.mark.asyncio
async def test_kafka_json_message_enqueued(monkeypatch):
    monkeypatch.delenv(CALLBACK_FORMAT_ENV_VAR, raising=False)
    backend = create_kafka_backend()
    backend.multiprocess = False
    backend.queue = asyncio.Queue()

    trade = _DummyTrade("binance", "ETH-USDT")
    await backend.__call__(trade, receipt_timestamp=456.0)

    message = await backend.queue.get()
    assert "format" not in message
    assert message["exchange"] == "binance"
    assert message["symbol"] == "ETH-USDT"

    topic = backend.topic(message)
    assert topic == "trades-binance-ETH-USDT"

    partition_key = backend.partition_key(message)
    assert partition_key is None
