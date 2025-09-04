import os
import sys
from decimal import Decimal

import pytest


# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _ensure_gen_on_path():
    gen_root = os.path.join(ROOT, "gen", "python")
    if gen_root not in sys.path:
        sys.path.insert(0, gen_root)
    # Shadow runtime 'cryptofeed' with a shim pointing to generated package
    import types
    cf_pkg = types.ModuleType('cryptofeed')
    cf_pkg.__path__ = [os.path.join(gen_root, 'cryptofeed')]
    sys.modules['cryptofeed'] = cf_pkg


def test_trade_serializer_roundtrip():
    _ensure_gen_on_path()

    # Import generated modules and serializer example
    from cryptofeed.v1 import common_pb2 as common
    from cryptofeed.v1 import events_pb2 as ev
    from examples.kafka_protobuf_serializer import make_value_serializer

    serializer = make_value_serializer(common.DATA_CHANNEL_TRADES)

    # Simulate backend dict for a trade
    payload = {
        'exchange': 'BINANCE',
        'symbol': 'BTC-USDT',
        'side': 'buy',
        'amount': Decimal('0.0105'),
        'price': Decimal('60250.12'),
        'id': 't-12345',
        'type': 'match',
        'timestamp': 1725456789.123456,
    }

    data = serializer(payload)

    # Decode and assert fields
    out = ev.DataFeedEvent()
    out.ParseFromString(data)

    assert out.channel == common.DATA_CHANNEL_TRADES
    assert out.exchange == common.EXCHANGE_BINANCE

    # Symbol mapping
    assert out.symbol.symbol == 'BTC-USDT'
    assert out.symbol.base == 'BTC'
    assert out.symbol.quote == 'USDT'

    # Timestamp mapping (seconds + nanos)
    assert out.event_timestamp.seconds == int(payload['timestamp'])
    assert out.event_timestamp.nanos == int((payload['timestamp'] - int(payload['timestamp'])) * 1e9)

    # Payload set in oneof
    assert out.WhichOneof('data') == 'trade'
    trade = out.trade
    assert trade.exchange == common.EXCHANGE_BINANCE
    assert trade.symbol.symbol == 'BTC-USDT'
    assert trade.side == common.SIDE_BUY
    assert trade.amount.value == str(payload['amount'])
    assert trade.price.value == str(payload['price'])
    assert trade.id == payload['id']
    assert trade.type == payload['type']
    assert trade.timestamp.seconds == int(payload['timestamp'])


def test_trade_serializer_symbol_without_delimiter():
    _ensure_gen_on_path()

    from cryptofeed.v1 import common_pb2 as common
    from cryptofeed.v1 import events_pb2 as ev
    from examples.kafka_protobuf_serializer import make_value_serializer

    serializer = make_value_serializer(common.DATA_CHANNEL_TRADES)

    payload = {
        'exchange': 'COINBASE',
        'symbol': 'BTCUSD',  # no hyphen; base/quote not inferred
        'side': 'sell',
        'amount': 1,
        'price': 100,
        'timestamp': 1600000000.0,
    }

    data = serializer(payload)
    out = ev.DataFeedEvent()
    out.ParseFromString(data)

    assert out.exchange == common.EXCHANGE_COINBASE
    assert out.symbol.symbol == 'BTCUSD'
    assert out.symbol.base == ''
    assert out.symbol.quote == ''
    assert out.trade.side == common.SIDE_SELL

