
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

def test_binance_depth_to_bookdelta():
    from cryptofeed.proto_mappers import binance as mapper
    b = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')

    native = b.DepthUpdate(symbol='BTCUSDT')
    native.segment = b.MARKET_SEGMENT_SPOT
    native.final_update_id = 42
    bl = md.PriceLevel(); bl.price.value='1'; bl.size.value='2'
    al = md.PriceLevel(); al.price.value='3'; al.size.value='0'  # deletion
    native.bids.append(bl)
    native.asks.append(al)

    out = mapper.to_common_book_delta_from_depth(native, md)
    assert out.exchange == cmn.EXCHANGE_BINANCE
    assert out.sequence_number == 42
    assert out.bid_changes[0].size.value == '2'
    assert out.ask_changes[0].size.value == '0'
