import os, json, importlib.util


def _project_root(start):
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, 'gen', 'python')) and os.path.isfile(os.path.join(cur, 'buf.yaml')):
            return cur
        nxt = os.path.abspath(os.path.join(cur, os.pardir))
        if nxt == cur: return cur
        cur = nxt
ROOT = _project_root(os.path.dirname(__file__))


def _ensure_gen_on_path():
    import sys, types
    gen_root = os.path.join(ROOT, 'gen', 'python')
    if gen_root not in sys.path: sys.path.insert(0, gen_root)
    cf_pkg = types.ModuleType('cryptofeed'); cf_pkg.__path__=[os.path.join(gen_root,'cryptofeed')]
    sys.modules['cryptofeed']=cf_pkg


def _load_pb2(rel):
    _ensure_gen_on_path()
    path = os.path.join(ROOT, rel)
    spec = importlib.util.spec_from_file_location('_pb2', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_binance_cm_aggtrade_to_common_trade():
    from cryptofeed.proto_mappers import binance as mapper
    b = _load_pb2('gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')
    md = _load_pb2('gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = _load_pb2('gen/python/cryptofeed/v1/common_pb2.py')

    data = json.load(open(os.path.join(ROOT,'tests/fixtures/exchange-native/binance/aggTrade_cm.json')))
    native = b.Trade(symbol=data['s']); native.segment = b.MARKET_SEGMENT_FUTURES_CM
    native.trade_id = str(data['a']); native.price.value = data['p']; native.quantity.value = data['q']; native.is_buyer_maker = data['m']
    native.raw_data = json.dumps(data).encode()

    out = mapper.to_common_trade(native, md)
    assert out.exchange == cmn.EXCHANGE_BINANCE_DELIVERY
    assert out.price.value == data['p'] and out.amount.value == data['q']
    # m=true => maker is seller, taker buys -> is_buyer_maker True => side SELL per mapper
    assert out.side == cmn.SIDE_SELL
    assert out.raw_data == native.raw_data
