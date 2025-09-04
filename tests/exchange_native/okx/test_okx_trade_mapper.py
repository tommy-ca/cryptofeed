import os
import importlib.util

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


def test_okx_trade_to_common():
    from cryptofeed.proto_mappers import okx as mapper
    o = _load_pb2("gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py")
    md = _load_pb2("gen/python/cryptofeed/v1/market_data_pb2.py")
    cmn = _load_pb2("gen/python/cryptofeed/v1/common_pb2.py")

    native = o.Trade(inst_id='BTC-USDT', trade_id='t1', side='buy')
    native.segment = o.MARKET_SEGMENT_SPOT
    native.price.value = '100.0'
    native.size.value = '0.5'

    out = mapper.to_common_trade(native, md)
    assert out.exchange == cmn.EXCHANGE_OKX
    assert out.symbol.base == 'BTC' and out.symbol.quote == 'USDT'
    assert out.price.value == '100.0'
    assert out.amount.value == '0.5'
    assert out.side == cmn.SIDE_BUY
