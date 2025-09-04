
import os, json, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

def test_okx_ticker_to_common_raw():
    from cryptofeed.proto_mappers import okx as mapper
    o = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/okx/ticker_spot.json')))
    native = o.Ticker(inst_id=data['instId'])
    native.segment = o.MARKET_SEGMENT_SPOT
    native.best_bid.value = data['bestBid']
    native.best_ask.value = data['bestAsk']
    native.raw_data = json.dumps(data).encode()

    out = mapper.to_common_ticker(native, md)
    assert out.exchange == cmn.EXCHANGE_OKX
    assert out.symbol.base=='BTC' and out.symbol.quote=='USDT'
    assert out.raw_data == native.raw_data


def test_okx_orderbook_to_common_l2():
    from cryptofeed.proto_mappers import okx as mapper
    o = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')

    native = o.OrderBook(inst_id='BTC-USDT')
    native.segment = o.MARKET_SEGMENT_SWAP
    native.seq_id = 999
    b = md.PriceLevel(); b.price.value='1'; b.size.value='2'
    a = md.PriceLevel(); a.price.value='3'; a.size.value='4'
    native.bids.append(b); native.asks.append(a)

    out = mapper.to_common_l2_from_orderbook(native, md)
    assert out.exchange == cmn.EXCHANGE_OKX
    assert out.sequence_number == 999
    assert out.bids[0].size.value == '2'
