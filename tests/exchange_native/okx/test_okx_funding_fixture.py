import os, json, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

def test_okx_funding_fixture():
    from cryptofeed.proto_mappers import okx as mapper
    o = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/okx/funding_swap.json')))
    native = o.Funding(inst_id=data['instId']); native.segment = o.MARKET_SEGMENT_SWAP
    native.rate.value = data['fundingRate']
    native.raw_data = json.dumps(data).encode()
    out = mapper.to_common_funding(native, md)
    assert out.rate.value == data['fundingRate']
    assert out.raw_data == native.raw_data
