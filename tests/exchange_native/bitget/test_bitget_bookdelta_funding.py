
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

def test_bitget_bookdelta():
    from cryptofeed.proto_mappers import bitget as mapper
    g = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    native = g.OrderBook(inst_id='BTCUSDT'); native.segment = g.MARKET_SEGMENT_USDT_PERP
    out = mapper.to_common_book_delta(native, md)
    assert hasattr(out, 'bid_changes')


def test_bitget_funding():
    from cryptofeed.proto_mappers import bitget as mapper
    g = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    native = g.Funding(inst_id='BTCUSDT'); native.segment = g.MARKET_SEGMENT_USDT_PERP
    native.rate.value = '0.0003'
    native.raw_data = b'raw-bitget-funding'
    out = mapper.to_common_funding(native, md)
    assert out.rate.value == '0.0003'
    assert out.raw_data == native.raw_data
