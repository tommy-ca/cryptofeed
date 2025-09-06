import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def test_bitget_bookdelta_fixture_seq_mapping():
    from cryptofeed.proto_mappers import bitget as mapper
    g = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/bitget/book_delta_usdt.json')))
    native = g.OrderBook(inst_id=data['instId']); native.segment = g.MARKET_SEGMENT_USDT_PERP
    b = md.PriceLevel(); b.price.value = data['bids'][0][0]; b.size.value = data['bids'][0][1]
    a = md.PriceLevel(); a.price.value = data['asks'][0][0]; a.size.value = data['asks'][0][1]
    native.bids.append(b); native.asks.append(a)
    native.seq = data['seq']
    native.raw_data = json.dumps(data).encode()

    delta = mapper.to_common_book_delta(native, md)
    assert delta.raw_data == native.raw_data
    assert delta.bid_changes[0].price.value == '30000'
    assert delta.ask_changes[0].size.value == '0.5'
    assert delta.sequence_number == 777
