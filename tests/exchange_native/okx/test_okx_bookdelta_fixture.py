import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def test_okx_bookdelta_fixture_maps_changes_and_seq():
    from cryptofeed.proto_mappers import okx as mapper
    o = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/okx/book_delta.json')))
    native = o.OrderBook(inst_id=data['instId']); native.segment = o.MARKET_SEGMENT_SPOT
    # Convert JSON price/size into PriceLevel items
    b = md.PriceLevel(); b.price.value = data['bids'][0][0]; b.size.value = data['bids'][0][1]
    a = md.PriceLevel(); a.price.value = data['asks'][0][0]; a.size.value = data['asks'][0][1]
    native.bids.append(b); native.asks.append(a)
    native.seq_id = data['seqId']
    native.raw_data = json.dumps(data).encode()

    delta = mapper.to_common_book_delta(native, md)
    assert delta.raw_data == native.raw_data
    assert delta.bid_changes[0].price.value == '100'
    assert delta.ask_changes[0].size.value == '0'
    assert delta.sequence_number == 555
