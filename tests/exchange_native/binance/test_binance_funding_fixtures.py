
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

def test_binance_funding_um_fixture():
    from cryptofeed.proto_mappers import binance as mapper
    b = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/binance/funding_um.json')))
    native = b.Funding(symbol=data['s']); native.segment = b.MARKET_SEGMENT_FUTURES_UM
    native.rate.value = data['r']
    out = mapper.to_common_funding(native, md)
    assert out.rate.value == data['r']


def test_binance_funding_cm_fixture():
    from cryptofeed.proto_mappers import binance as mapper
    b = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/binance/funding_cm.json')))
    native = b.Funding(symbol=data['s']); native.segment = b.MARKET_SEGMENT_FUTURES_CM
    native.rate.value = data['r']
    out = mapper.to_common_funding(native, md)
    assert out.rate.value == data['r']
