import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def test_binance_depth_update_first_last_ids_to_sequence():
    from cryptofeed.proto_mappers import binance as mapper
    b = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/binance/depth_delta_um_full.json')))
    native = b.DepthUpdate(symbol=data['s']); native.segment = b.MARKET_SEGMENT_FUTURES_UM
    native.first_update_id = data['U']
    native.final_update_id = data['u']
    bl = md.PriceLevel(); bl.price.value=data['b'][0][0]; bl.size.value=data['b'][0][1]
    al = md.PriceLevel(); al.price.value=data['a'][0][0]; al.size.value=data['a'][0][1]
    native.bids.append(bl); native.asks.append(al)

    delta = mapper.to_common_book_delta_from_depth(native, md)
    assert delta.sequence_number == data['u']
    assert delta.bid_changes[0].price.value == data['b'][0][0]
    assert delta.ask_changes[0].size.value == data['a'][0][1]
