import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def test_okx_option_ticker_maps_to_option_type():
    from cryptofeed.proto_mappers import okx as mapper
    o = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/okx/ticker_option.json')))
    native = o.Ticker(inst_id=data['instId'])
    native.segment = o.MARKET_SEGMENT_OPTIONS
    native.best_bid.value = data['bestBid']
    native.best_ask.value = data['bestAsk']

    t = mapper.to_common_ticker(native, md)
    assert t.symbol.type == cmn.INSTRUMENT_TYPE_OPTION
    assert t.symbol.base == 'BTC' and t.symbol.quote == 'USD'
    assert t.bid.value == data['bestBid'] and t.ask.value == data['bestAsk']
