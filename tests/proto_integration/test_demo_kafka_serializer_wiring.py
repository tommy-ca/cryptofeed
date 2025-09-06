import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_demo_kafka_build_callbacks_includes_serializer_for_trades():
    from cryptofeed.defines import TRADES
    from examples.demo_kafka import build_callbacks

    cbs = build_callbacks(use_protobuf_serializer=True)
    trade_backend = cbs[TRADES]

    # Ensure serializer is configured
    assert hasattr(trade_backend, 'producer_config')
    serializer = trade_backend.producer_config.get('value_serializer')
    assert callable(serializer)

    # Minimal payload roundtrip
    payload = {
        'exchange': 'BINANCE',
        'symbol': 'BTC-USDT',
        'side': 'buy',
        'amount': 1,
        'price': 100,
        'timestamp': 1700000000.0,
    }
    data = serializer(payload)
    assert isinstance(data, (bytes, bytearray)) and len(data) > 0
