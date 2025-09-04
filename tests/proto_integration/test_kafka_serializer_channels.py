import os
import sys
from decimal import Decimal


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _ensure_gen_on_path():
    gen_root = os.path.join(ROOT, "gen", "python")
    if gen_root not in sys.path:
        sys.path.insert(0, gen_root)
    import types
    cf_pkg = types.ModuleType('cryptofeed')
    cf_pkg.__path__ = [os.path.join(gen_root, 'cryptofeed')]
    sys.modules['cryptofeed'] = cf_pkg


def test_ticker_serializer_roundtrip():
    _ensure_gen_on_path()
    from cryptofeed.v1 import common_pb2 as common
    from cryptofeed.v1 import events_pb2 as ev
    from examples.kafka_protobuf_serializer import make_value_serializer

    serializer = make_value_serializer(common.DATA_CHANNEL_TICKER)

    payload = {
        'exchange': 'OKX',
        'symbol': 'BTC-USDT',
        'bid': Decimal('60200.1'),
        'ask': Decimal('60210.2'),
        'timestamp': 1725456000.5,
    }

    data = serializer(payload)
    out = ev.DataFeedEvent()
    out.ParseFromString(data)

    assert out.channel == common.DATA_CHANNEL_TICKER
    assert out.exchange == common.EXCHANGE_OKX
    assert out.WhichOneof('data') == 'ticker'
    assert out.ticker.bid.value == str(payload['bid'])
    assert out.ticker.ask.value == str(payload['ask'])


def test_l1_book_serializer_roundtrip():
    _ensure_gen_on_path()
    from cryptofeed.v1 import common_pb2 as common
    from cryptofeed.v1 import events_pb2 as ev
    from examples.kafka_protobuf_serializer import make_value_serializer

    serializer = make_value_serializer(common.DATA_CHANNEL_L1_BOOK)
    payload = {
        'exchange': 'BYBIT',
        'symbol': 'ETH-USDT',
        'bid_price': Decimal('2500.01'),
        'bid_size': Decimal('3.5'),
        'ask_price': Decimal('2500.50'),
        'ask_size': Decimal('2.1'),
        'timestamp': 1700000000.0,
    }
    data = serializer(payload)
    out = ev.DataFeedEvent()
    out.ParseFromString(data)

    assert out.channel == common.DATA_CHANNEL_L1_BOOK
    assert out.exchange == common.EXCHANGE_BYBIT
    assert out.WhichOneof('data') == 'l1_book'
    b = out.l1_book
    assert b.bid_price.value == str(payload['bid_price'])
    assert b.bid_size.value == str(payload['bid_size'])
    assert b.ask_price.value == str(payload['ask_price'])
    assert b.ask_size.value == str(payload['ask_size'])


def test_l2_book_serializer_roundtrip():
    _ensure_gen_on_path()
    from cryptofeed.v1 import common_pb2 as common
    from cryptofeed.v1 import events_pb2 as ev
    from examples.kafka_protobuf_serializer import make_value_serializer

    serializer = make_value_serializer(common.DATA_CHANNEL_L2_BOOK)
    payload = {
        'exchange': 'BINANCE',
        'symbol': 'BTC-USDT',
        'bids': [(Decimal('60000'), Decimal('1.2')), (Decimal('59900'), Decimal('2.3'))],
        'asks': [(Decimal('60100'), Decimal('1.1'))],
        'timestamp': 1725456123.123,
        'sequence_number': 123456789,
    }
    data = serializer(payload)
    out = ev.DataFeedEvent()
    out.ParseFromString(data)

    assert out.channel == common.DATA_CHANNEL_L2_BOOK
    assert out.exchange == common.EXCHANGE_BINANCE
    assert out.WhichOneof('data') == 'l2_book'
    book = out.l2_book
    assert len(book.bids) == 2
    assert len(book.asks) == 1
    assert book.bids[0].price.value == '60000'
    assert book.bids[0].size.value == '1.2'
    assert book.asks[0].price.value == '60100'
    assert book.asks[0].size.value == '1.1'


def test_funding_serializer_roundtrip():
    _ensure_gen_on_path()
    from cryptofeed.v1 import common_pb2 as common
    from cryptofeed.v1 import events_pb2 as ev
    from examples.kafka_protobuf_serializer import make_value_serializer

    serializer = make_value_serializer(common.DATA_CHANNEL_FUNDING)
    payload = {
        'exchange': 'BYBIT',
        'symbol': 'BTC-USDT-PERP',
        'mark_price': Decimal('60222.2'),
        'rate': Decimal('0.0001'),
        'predicted_rate': Decimal('0.0002'),
        'next_funding_time': 1725457200.0,
        'timestamp': 1725456000.0,
    }
    data = serializer(payload)
    out = ev.DataFeedEvent()
    out.ParseFromString(data)

    assert out.channel == common.DATA_CHANNEL_FUNDING
    assert out.exchange == common.EXCHANGE_BYBIT
    assert out.WhichOneof('data') == 'funding'
    f = out.funding
    assert f.mark_price.value == str(payload['mark_price'])
    assert f.rate.value == str(payload['rate'])
    assert f.predicted_rate.value == str(payload['predicted_rate'])
    assert f.timestamp.seconds == int(payload['timestamp'])

