import asyncio
import json

import pytest

from cryptofeed.defines import TRADES, COINBASE
from cryptofeed.symbols import Symbols
from cryptofeed.exchanges.coinbase import Coinbase


class _Collector:
    def __init__(self):
        self.items = []

    async def __call__(self, obj, receipt_timestamp):
        self.items.append((obj, receipt_timestamp))


@pytest.mark.asyncio
async def test_coinbase_trade_snapshot_calls_trade_callback():
    collector = _Collector()
    # Avoid network symbol discovery by pre-populating symbol mapping
    Symbols.set(COINBASE, {"BTC-USD": "BTC-USD"}, {"tick_size": {}, "instrument_type": {}})
    cb = Coinbase(callbacks={TRADES: collector})

    # Craft a snapshot message with two trades, matching the structure expected by the handler
    snapshot = {
        "channel": "market_trades",
        "events": [
            {
                "type": "snapshot",
                "trades": [
                    {
                        "trade_id": 1,
                        "side": "BUY",
                        "size": "0.10",
                        "price": "50000.00",
                        "product_id": "BTC-USD",
                        "time": "2024-01-01T00:00:00Z",
                    },
                    {
                        "trade_id": 2,
                        "side": "SELL",
                        "size": "0.20",
                        "price": "50010.00",
                        "product_id": "BTC-USD",
                        "time": "2024-01-01T00:00:01Z",
                    },
                ],
            }
        ],
    }

    await cb.message_handler(json.dumps(snapshot), conn=None, timestamp=0.0)

    # Expect two callback invocations with proper values
    assert len(collector.items) == 2
    trade0, _ = collector.items[0]
    trade1, _ = collector.items[1]
    assert trade0.id == "1" and trade0.price == trade0.price.__class__("50000.00")
    assert trade1.id == "2" and trade1.price == trade1.price.__class__("50010.00")
