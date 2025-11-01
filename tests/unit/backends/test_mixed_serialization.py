import asyncio

import pytest

from cryptofeed.backends.kafka import TradeKafka
from cryptofeed.backends.redis import TradeRedis
from cryptofeed.serializers.formats import CALLBACK_FORMAT_ENV_VAR
from cryptofeed.types import Trade
from decimal import Decimal


def _make_trade(symbol: str) -> Trade:
    return Trade(
        exchange="coinbase",
        symbol=symbol,
        side="buy",
        amount=Decimal("1.0"),
        price=Decimal("50000"),
        timestamp=1_700_000_000.0,
    )


@pytest.mark.asyncio
async def test_mixed_serialization_callbacks(monkeypatch):
    monkeypatch.delenv(CALLBACK_FORMAT_ENV_VAR, raising=False)

    kafka = TradeKafka(serialization_format="protobuf")
    redis = TradeRedis()

    kafka.multiprocess = False
    redis.multiprocess = False

    kafka.queue = asyncio.Queue()
    redis.queue = asyncio.Queue()

    trade = _make_trade("BTC-USD")

    await asyncio.gather(
        kafka.__call__(trade, receipt_timestamp=123.0),
        redis.__call__(trade, receipt_timestamp=123.0),
    )

    kafka_message = await kafka.queue.get()
    redis_message = await redis.queue.get()

    assert kafka_message["format"] == "protobuf"
    assert redis_message["exchange"] == "coinbase"
    assert redis_message["symbol"] == "BTC-USD"
