import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def test_bitget_funding_coin_perp_segment():
    from cryptofeed.proto_mappers import bitget as mapper
    g = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')

    native = g.Funding(inst_id='BTCUSD'); native.segment = g.MARKET_SEGMENT_COIN_PERP
    native.rate.value = '0.0005'
    native.raw_data = b'raw-bitget-coin-funding'

    out = mapper.to_common_funding(native, md)
    assert out.rate.value == '0.0005'
    assert out.symbol.base == 'BTC' and out.symbol.quote == 'USD'
    assert out.symbol.type == cmn.INSTRUMENT_TYPE_PERPETUAL
    assert out.raw_data == native.raw_data
