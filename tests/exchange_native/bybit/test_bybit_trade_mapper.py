
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

def test_bybit_trade_to_common():
    from cryptofeed.proto_mappers import bybit as mapper
    y = load_pb2(ROOT, "gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py")
    md = load_pb2(ROOT, "gen/python/cryptofeed/v1/market_data_pb2.py")
    cmn = load_pb2(ROOT, "gen/python/cryptofeed/v1/common_pb2.py")

    native = y.Trade(symbol='BTCUSDT', trade_id='x1', side='Sell')
    native.segment = y.MARKET_SEGMENT_LINEAR
    native.price.value = '10'
    native.quantity.value = '2'

    out = mapper.to_common_trade(native, md)
    assert out.exchange == cmn.EXCHANGE_BYBIT
    assert out.symbol.base == 'BTC' and out.symbol.quote == 'USDT'
    assert out.side == cmn.SIDE_SELL
    assert out.amount.value == '2'
