import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def test_bybit_funding_inverse_segment():
    from cryptofeed.proto_mappers import bybit as mapper
    y = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')

    native = y.Funding(symbol='BTCUSD'); native.segment = y.MARKET_SEGMENT_INVERSE
    native.rate.value = '0.0004'
    native.raw_data = b'raw-bybit-inverse-funding'

    out = mapper.to_common_funding(native, md)
    assert out.rate.value == '0.0004'
    assert out.symbol.base == 'BTC' and out.symbol.quote == 'USD'
    assert out.symbol.type == cmn.INSTRUMENT_TYPE_PERPETUAL
    assert out.raw_data == native.raw_data
