import os, sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

@pytest.mark.parametrize(
    'mapper_mod, pb2_path, build_native',
    [
        ('cryptofeed.proto_mappers.okx', 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py', lambda m: m.Funding(inst_id='BTC-USD-SWAP')),
        ('cryptofeed.proto_mappers.bybit', 'gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py', lambda m: m.Funding(symbol='BTCUSDT')),
        ('cryptofeed.proto_mappers.bitget', 'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py', lambda m: m.Funding(inst_id='BTCUSDT')),
        ('cryptofeed.proto_mappers.binance', 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py', lambda m: m.Funding(symbol='BTCUSDT')),
    ],
)
def test_param_funding_map(mapper_mod, pb2_path, build_native):
    mapper = __import__(mapper_mod, fromlist=['dummy'])
    pb2 = load_pb2(ROOT, pb2_path)
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    native = build_native(pb2)
    if hasattr(native, 'rate'):
        native.rate.value = '0.0001'
    if hasattr(native, 'raw_data'):
        native.raw_data = b'raw-funding'
    out = mapper.to_common_funding(native, md)
    assert out.rate.value == '0.0001'
    if hasattr(native, 'raw_data'):
        assert out.raw_data == b'raw-funding'

