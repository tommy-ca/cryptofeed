import asyncio
import base64

import pytest

from cryptofeed.backends.redis import TradeRedis
from cryptofeed.proto_bindings import trade_pb2, trade_side_pb2


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
async def test_redis_json_message_default():
    backend = TradeRedis()
    backend.multiprocess = False
    backend.queue = asyncio.Queue()

    trade = _DummyTrade("binance", "ETH-USDT")
    await backend.__call__(trade, receipt_timestamp=10.0)

    message = await backend.queue.get()
    assert "format" not in message
    assert message["exchange"] == "binance"
    assert message["symbol"] == "ETH-USDT"


@pytest.mark.asyncio
async def test_redis_protobuf_message_packaging():
    backend = TradeRedis(serialization_format="protobuf")
    backend.multiprocess = False
    backend.queue = asyncio.Queue()

    trade = _DummyTrade("coinbase", "BTC-USD")
    await backend.__call__(trade, receipt_timestamp=20.0)

    message = await backend.queue.get()
    assert message["format"] == "protobuf"
    assert isinstance(message["payload"], bytes)
    assert message["metadata"]["exchange"] == "coinbase"
    assert message["metadata"]["symbol"] == "BTC-USD"

    record = backend._prepare_json_record(message)
    assert record["payload_b64"] == base64.b64encode(message["payload"]).decode()
