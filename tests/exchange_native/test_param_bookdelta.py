import os, sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
try:
    from tests.util.pb2_loader import project_root, load_pb2
except Exception:
    import importlib.util, types
    def project_root(start: str) -> str:
        cur = os.path.abspath(start)
        while True:
            if os.path.isdir(os.path.join(cur, 'gen', 'python')) and os.path.isfile(os.path.join(cur, 'buf.yaml')):
                return cur
            nxt = os.path.abspath(os.path.join(cur, os.pardir))
            if nxt == cur:
                return cur
            cur = nxt
    def _ensure_gen_on_path(root: str) -> None:
        gen_root = os.path.join(root, 'gen', 'python')
        if gen_root not in sys.path:
            sys.path.insert(0, gen_root)
        cf_pkg = types.ModuleType('cryptofeed')
        cf_pkg.__path__ = [os.path.join(gen_root, 'cryptofeed')]
        sys.modules['cryptofeed'] = cf_pkg
    def load_pb2(root: str, rel_path: str):
        _ensure_gen_on_path(root)
        path = os.path.join(root, rel_path)
        spec = importlib.util.spec_from_file_location('_pb2', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod

ROOT = project_root(os.path.dirname(__file__))

def _build_pl(md, p, q):
    pl = md.PriceLevel(); pl.price.value=str(p); pl.size.value=str(q); return pl

def build_okx(pb2, md):
    ob = pb2.OrderBook(inst_id='BTC-USDT')
    ob.bids.append(_build_pl(md,'100','1.1'))
    ob.asks.append(_build_pl(md,'101','0'))
    return ob

def build_bybit(pb2, md):
    ob = pb2.OrderBook(symbol='BTCUSDT')
    ob.bids.append(_build_pl(md,'100','1.1'))
    ob.asks.append(_build_pl(md,'101','0'))
    return ob

def build_bitget(pb2, md):
    ob = pb2.OrderBook(inst_id='BTCUSDT')
    ob.bids.append(_build_pl(md,'100','1.1'))
    ob.asks.append(_build_pl(md,'101','0'))
    return ob

@pytest.mark.parametrize(
    'mapper_mod, pb2_path, builder, mapper_fn_name',
    [
        ('cryptofeed.proto_mappers.okx', 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py', build_okx, 'to_common_book_delta'),
        ('cryptofeed.proto_mappers.bybit', 'gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py', build_bybit, 'to_common_book_delta'),
        ('cryptofeed.proto_mappers.bitget', 'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py', build_bitget, 'to_common_book_delta'),
    ],
)
def test_param_bookdelta(mapper_mod, pb2_path, builder, mapper_fn_name):
    mapper = __import__(mapper_mod, fromlist=['dummy'])
    pb2 = load_pb2(ROOT, pb2_path)
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    native = builder(pb2, md)
    # add raw_data for propagation assertion
    if hasattr(native, 'raw_data'):
        native.raw_data = b'raw-delta'
    delta = getattr(mapper, mapper_fn_name)(native, md)
    assert delta.bid_changes and delta.ask_changes
    assert delta.bid_changes[0].price.value == '100'
    assert delta.ask_changes[0].size.value == '0'
    if hasattr(native, 'raw_data'):
        assert delta.raw_data == b'raw-delta'
