import asyncio

import pytest

pytest.importorskip("zmq")

from cryptofeed.backends.zmq import TradeZMQ
from cryptofeed.backends.protobuf.bindings import trade_pb2, trade_side_pb2


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
        msg.price = "42"
        msg.amount = "0.5"
        msg.side = trade_side_pb2.TRADE_SIDE_BUY
        msg.timestamp = 0
        return msg


@pytest.mark.asyncio
async def test_zmq_json_message_default():
    backend = TradeZMQ()
    backend.multiprocess = False
    backend.queue = asyncio.Queue()

    trade = _DummyTrade("binance", "ETH-USDT")
    await backend.__call__(trade, receipt_timestamp=5.0)

    message = await backend.queue.get()
    assert "format" not in message
    assert message["exchange"] == "binance"
    assert message["symbol"] == "ETH-USDT"


@pytest.mark.asyncio
async def test_zmq_protobuf_message_packaging():
    backend = TradeZMQ(serialization_format="protobuf")
    backend.multiprocess = False
    backend.queue = asyncio.Queue()

    trade = _DummyTrade("coinbase", "BTC-USD")
    await backend.__call__(trade, receipt_timestamp=15.0)

    message = await backend.queue.get()
    assert message["format"] == "protobuf"
    assert isinstance(message["payload"], bytes)
    assert message["metadata"]["exchange"] == "coinbase"
    assert message["metadata"]["symbol"] == "BTC-USD"
