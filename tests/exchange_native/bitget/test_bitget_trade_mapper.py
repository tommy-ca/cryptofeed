
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

def test_bitget_trade_to_common():
    from cryptofeed.proto_mappers import bitget as mapper
    g = load_pb2(ROOT, "gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py")
    md = load_pb2(ROOT, "gen/python/cryptofeed/v1/market_data_pb2.py")
    cmn = load_pb2(ROOT, "gen/python/cryptofeed/v1/common_pb2.py")

    native = g.Trade(inst_id='BTCUSDT', trade_id='z1', side='buy')
    native.segment = g.MARKET_SEGMENT_USDT_PERP
    native.price.value = '1.1'
    native.size.value = '3.3'

    out = mapper.to_common_trade(native, md)
    assert out.exchange == cmn.EXCHANGE_BITGET
    assert out.symbol.base == 'BTC' and out.symbol.quote == 'USDT'
    assert out.side == cmn.SIDE_BUY
    assert out.amount.value == '3.3'
