
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

def test_bitget_ticker_to_common_raw():
    from cryptofeed.proto_mappers import bitget as mapper
    g = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/bitget/ticker_spot.json')))
    native = g.Ticker(inst_id=data['instId'])
    native.segment = g.MARKET_SEGMENT_SPOT
    native.best_bid.value = data['bestBid']
    native.best_ask.value = data['bestAsk']
    native.raw_data = json.dumps(data).encode()

    out = mapper.to_common_ticker(native, md)
    assert out.bid.value == data['bestBid']
    assert out.raw_data == native.raw_data


def test_bitget_orderbook_to_common_l2():
    from cryptofeed.proto_mappers import bitget as mapper
    g = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    native = g.OrderBook(inst_id='BTCUSDT')
    native.segment = g.MARKET_SEGMENT_USDT_PERP
    b = md.PriceLevel(); b.price.value='1'; b.size.value='2'
    a = md.PriceLevel(); a.price.value='3'; a.size.value='4'
    native.bids.append(b); native.asks.append(a)

    out = mapper.to_common_l2_from_orderbook(native, md)
    assert out.bids[0].size.value == '2'
