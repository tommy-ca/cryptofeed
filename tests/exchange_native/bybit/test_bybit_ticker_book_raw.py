
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

def test_bybit_ticker_to_common_raw():
    from cryptofeed.proto_mappers import bybit as mapper
    y = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/bybit/ticker_linear.json')))
    native = y.Ticker(symbol=data['symbol'])
    native.segment = y.MARKET_SEGMENT_LINEAR
    native.best_bid.value = data['bestBidPrice']
    native.best_ask.value = data['bestAskPrice']
    native.raw_data = json.dumps(data).encode()

    out = mapper.to_common_ticker(native, md)
    assert out.bid.value == data['bestBidPrice']
    assert out.raw_data == native.raw_data


def test_bybit_orderbook_to_common_l2():
    from cryptofeed.proto_mappers import bybit as mapper
    y = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    native = y.OrderBook(symbol='BTCUSDT')
    native.segment = y.MARKET_SEGMENT_LINEAR
    b = md.PriceLevel(); b.price.value='1'; b.size.value='2'
    a = md.PriceLevel(); a.price.value='3'; a.size.value='4'
    native.bids.append(b); native.asks.append(a)

    out = mapper.to_common_l2_from_orderbook(native, md)
    assert out.bids[0].size.value == '2'
