import os
import importlib.util
import json


def _project_root(start):
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, 'gen', 'python')) and os.path.isfile(os.path.join(cur, 'buf.yaml')):
            return cur
        nxt = os.path.abspath(os.path.join(cur, os.pardir))
        if nxt == cur:
            return cur
        cur = nxt


ROOT = _project_root(os.path.dirname(__file__))


def _ensure_gen_on_path():
    import sys, types
    gen_root = os.path.join(ROOT, "gen", "python")
    if gen_root not in sys.path:
        sys.path.insert(0, gen_root)
    cf_pkg = types.ModuleType('cryptofeed')
    cf_pkg.__path__ = [os.path.join(gen_root, 'cryptofeed')]
    sys.modules['cryptofeed'] = cf_pkg


def _load_pb2(rel_path):
    _ensure_gen_on_path()
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location("_pb2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def test_binance_bookticker_to_common_ticker_and_l1():
    from cryptofeed.proto_mappers import binance as mapper
    b = _load_pb2("gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py")
    md = _load_pb2("gen/python/cryptofeed/v1/market_data_pb2.py")
    cmn = _load_pb2("gen/python/cryptofeed/v1/common_pb2.py")

    # Load fixture
    fx_path = os.path.join(ROOT, 'tests', 'fixtures', 'exchange-native', 'binance', 'bookTicker_spot.json')
    data = json.load(open(fx_path))

    native = b.BookTicker(symbol=data['s'])
    native.segment = b.MARKET_SEGMENT_SPOT
    native.bid_price.value = data['b']
    native.bid_qty.value = data['B']
    native.ask_price.value = data['a']
    native.ask_qty.value = data['A']
    native.raw_data = json.dumps(data).encode()

    t = mapper.to_common_ticker(native, md)
    assert t.exchange == cmn.EXCHANGE_BINANCE
    assert t.symbol.base == 'BTC' and t.symbol.quote == 'USDT'
    assert t.bid.value == data['b']
    assert t.ask.value == data['a']
    assert t.raw_data == native.raw_data

    l1 = mapper.to_common_l1_from_bookticker(native, md)
    assert l1.bid_price.value == data['b']
    assert l1.bid_size.value == data['B']
    assert l1.ask_price.value == data['a']
    assert l1.ask_size.value == data['A']


def test_binance_depth_to_common_l2():
    from cryptofeed.proto_mappers import binance as mapper
    b = _load_pb2("gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py")
    md = _load_pb2("gen/python/cryptofeed/v1/market_data_pb2.py")
    cmn = _load_pb2("gen/python/cryptofeed/v1/common_pb2.py")

    native = b.DepthUpdate(symbol='BTCUSDT')
    native.segment = b.MARKET_SEGMENT_FUTURES_UM
    native.final_update_id = 12345
    bl = md.PriceLevel(); bl.price.value='62000.0'; bl.size.value='1.0'
    al = md.PriceLevel(); al.price.value='62001.0'; al.size.value='2.0'
    native.bids.append(bl)
    native.asks.append(al)

    out = mapper.to_common_l2_from_depth(native, md)
    assert out.exchange == cmn.EXCHANGE_BINANCE_FUTURES
    assert out.sequence_number == 12345
    assert out.bids[0].price.value == '62000.0'
    assert out.asks[0].size.value == '2.0'


def test_binance_funding_to_common():
    from cryptofeed.proto_mappers import binance as mapper
    b = _load_pb2("gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py")
    md = _load_pb2("gen/python/cryptofeed/v1/market_data_pb2.py")

    native = b.Funding(symbol='BTCUSDT')
    native.segment = b.MARKET_SEGMENT_FUTURES_UM
    native.mark_price.value = '62010.0'
    native.rate.value = '0.0001'

    out = mapper.to_common_funding(native, md)
    assert out.mark_price.value == '62010.0'
    assert out.rate.value == '0.0001'
