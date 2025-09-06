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


def test_data_feed_event_sizes_reasonable():
    _ensure_gen_on_path()
    from cryptofeed.v1 import common_pb2 as common
    from cryptofeed.v1 import events_pb2 as ev
    from examples.kafka_protobuf_serializer import to_event

    # Trade
    trade_payload = {
        'exchange': 'BINANCE',
        'symbol': 'BTC-USDT',
        'side': 'buy',
        'amount': Decimal('0.1'),
        'price': Decimal('60000'),
        'timestamp': 1700000000.0,
    }
    trade_evt = to_event(common.DATA_CHANNEL_TRADES, trade_payload)
    trade_bytes = trade_evt.SerializeToString()
    assert len(trade_bytes) < 512

    # Ticker
    tick_payload = {
        'exchange': 'OKX',
        'symbol': 'ETH-USDT',
        'bid': Decimal('2500.01'),
        'ask': Decimal('2500.50'),
        'timestamp': 1700000001.0,
    }
    tick_evt = to_event(common.DATA_CHANNEL_TICKER, tick_payload)
    tick_bytes = tick_evt.SerializeToString()
    assert len(tick_bytes) < 512

    # L1
    l1_payload = {
        'exchange': 'BYBIT',
        'symbol': 'BTC-USDT',
        'bid_price': Decimal('59999.9'),
        'bid_size': Decimal('1.5'),
        'ask_price': Decimal('60000.1'),
        'ask_size': Decimal('2.0'),
        'timestamp': 1700000002.0,
    }
    l1_evt = to_event(common.DATA_CHANNEL_L1_BOOK, l1_payload)
    l1_bytes = l1_evt.SerializeToString()
    assert len(l1_bytes) < 512

    # L2 (small)
    l2_payload = {
        'exchange': 'BINANCE',
        'symbol': 'BTC-USDT',
        'bids': [(Decimal('60000'), Decimal('1.2')), (Decimal('59900'), Decimal('2.3'))],
        'asks': [(Decimal('60100'), Decimal('1.1'))],
        'sequence_number': 123,
        'timestamp': 1700000003.0,
    }
    l2_evt = to_event(common.DATA_CHANNEL_L2_BOOK, l2_payload)
    l2_bytes = l2_evt.SerializeToString()
    assert len(l2_bytes) < 1024

    # Funding
    funding_payload = {
        'exchange': 'BITGET',
        'symbol': 'BTC-USDT-PERP',
        'mark_price': Decimal('60001.5'),
        'rate': Decimal('0.0001'),
        'predicted_rate': Decimal('0.0002'),
        'timestamp': 1700000004.0,
    }
    funding_evt = to_event(common.DATA_CHANNEL_FUNDING, funding_payload)
    funding_bytes = funding_evt.SerializeToString()
    assert len(funding_bytes) < 768

