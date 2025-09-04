import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def test_okx_option_trade_maps_to_option_type_and_side():
    from cryptofeed.proto_mappers import okx as mapper
    o = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/okx/trade_option.json')))
    native = o.Trade(inst_id=data['instId'], trade_id=data['tradeId'], side=data['side'])
    native.segment = o.MARKET_SEGMENT_OPTIONS
    native.price.value = data['px']
    native.size.value = data['sz']

    t = mapper.to_common_trade(native, md)
    assert t.symbol.type == cmn.INSTRUMENT_TYPE_OPTION
    assert t.symbol.base == 'BTC' and t.symbol.quote == 'USD'
    assert t.side == cmn.SIDE_BUY
    assert t.price.value == data['px'] and t.amount.value == data['sz']
