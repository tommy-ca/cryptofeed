
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

def test_bybit_bookdelta():
    from cryptofeed.proto_mappers import bybit as mapper
    y = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    native = y.OrderBook(symbol='BTCUSDT'); native.segment = y.MARKET_SEGMENT_LINEAR
    b = md.PriceLevel(); b.price.value='1'; b.size.value='0.1'
    a = md.PriceLevel(); a.price.value='2'; a.size.value='0'
    native.bids.append(b); native.asks.append(a)
    out = mapper.to_common_book_delta(native, md)
    assert out.bid_changes[0].size.value == '0.1'
    assert out.ask_changes[0].size.value == '0'


def test_bybit_funding():
    from cryptofeed.proto_mappers import bybit as mapper
    y = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    native = y.Funding(symbol='BTCUSDT'); native.segment = y.MARKET_SEGMENT_LINEAR
    native.rate.value = '0.0002'
    native.raw_data = b'raw-bybit-funding'
    out = mapper.to_common_funding(native, md)
    assert out.rate.value == '0.0002'
    assert out.raw_data == native.raw_data
