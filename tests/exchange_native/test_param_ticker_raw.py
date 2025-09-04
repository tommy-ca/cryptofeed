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

@pytest.mark.parametrize(
    'mapper_mod, pb2_path, build_native',
    [
        ('cryptofeed.proto_mappers.binance', 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py', lambda m: m.BookTicker(symbol='BTCUSDT')),
        ('cryptofeed.proto_mappers.okx', 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py', lambda m: m.Ticker(inst_id='BTC-USDT')),
        ('cryptofeed.proto_mappers.bybit', 'gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py', lambda m: m.Ticker(symbol='BTCUSDT')),
        ('cryptofeed.proto_mappers.bitget', 'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py', lambda m: m.Ticker(inst_id='BTCUSDT')),
    ]
)
def test_param_ticker_raw(mapper_mod, pb2_path, build_native):
    mapper = __import__(mapper_mod, fromlist=['dummy'])
    pb2 = load_pb2(ROOT, pb2_path)
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    native = build_native(pb2)
    # set minimal prices for mapping
    if hasattr(native, 'bid_price'):
        native.bid_price.value = '1'
        native.ask_price.value = '2'
    else:
        native.best_bid.value = '1'
        native.best_ask.value = '2'
    # raw attachment
    native.raw_data = b'xyz'

    t = mapper.to_common_ticker(native, md)
    assert t.raw_data == b'xyz'
    assert t.symbol.base == 'BTC'
    assert t.symbol.quote in ('USDT', 'USD')
