import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def _build_price_level(md, p, q):
    pl = md.PriceLevel()
    pl.price.value = str(p)
    pl.size.value = str(q)
    return pl


def _map_fixture(fname, seg_const):
    from cryptofeed.proto_mappers import binance as mapper
    b = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    data = json.load(open(os.path.join(ROOT, 'tests/fixtures/exchange-native/binance', fname)))
    native = b.DepthUpdate(symbol=data['s']); native.segment = getattr(b, seg_const)
    # Convert price levels
    if data.get('b'):
        p, q = data['b'][0]
        native.bids.append(_build_price_level(md, p, q))
    if data.get('a'):
        p, q = data['a'][0]
        native.asks.append(_build_price_level(md, p, q))
    native.final_update_id = data['u']
    native.raw_data = json.dumps(data).encode()

    d = mapper.to_common_book_delta_from_depth(native, md)
    assert d.raw_data == native.raw_data
    return d


def test_binance_um_depth_delta_fixture():
    delta = _map_fixture('depth_delta_um.json', 'MARKET_SEGMENT_FUTURES_UM')
    assert delta.bid_changes[0].price.value == '100.5'
    assert delta.ask_changes[0].size.value == '0'
    assert delta.sequence_number == 123456


def test_binance_cm_depth_delta_fixture():
    delta = _map_fixture('depth_delta_cm.json', 'MARKET_SEGMENT_FUTURES_CM')
    assert delta.bid_changes[0].price.value == '200.5'
    assert delta.ask_changes[0].size.value == '0'
    assert delta.sequence_number == 654321
