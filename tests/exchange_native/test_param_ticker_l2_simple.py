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


def build_price_level(md, price, size):
    pl = md.PriceLevel()
    pl.price.value = price
    pl.size.value = size
    return pl


@pytest.mark.parametrize(
    'ex, pb2_path, mapper_mod, ticker_ctor, book_ctor',
    [
        ('binance', 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py', 'cryptofeed.proto_mappers.binance', lambda m: m.BookTicker(symbol='BTCUSDT'), lambda m: m.DepthUpdate(symbol='BTCUSDT')),
        ('okx', 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py', 'cryptofeed.proto_mappers.okx', lambda m: m.Ticker(inst_id='BTC-USDT'), lambda m: m.OrderBook(inst_id='BTC-USDT')),
        ('bybit', 'gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py', 'cryptofeed.proto_mappers.bybit', lambda m: m.Ticker(symbol='BTCUSDT'), lambda m: m.OrderBook(symbol='BTCUSDT')),
        ('bitget', 'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py', 'cryptofeed.proto_mappers.bitget', lambda m: m.Ticker(inst_id='BTCUSDT'), lambda m: m.OrderBook(inst_id='BTCUSDT')),
    ],
)
def test_ticker_l2_param(ex, pb2_path, mapper_mod, ticker_ctor, book_ctor):
    mapper = __import__(mapper_mod, fromlist=['dummy'])
    pb2 = load_pb2(ROOT, pb2_path)
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    t_native = ticker_ctor(pb2)
    b_native = book_ctor(pb2)
    # add levels
    b_native.bids.append(build_price_level(md, '1', '2'))
    b_native.asks.append(build_price_level(md, '3', '4'))
    # raw
    if hasattr(t_native, 'raw_data'):
        t_native.raw_data = b'raw'
    if hasattr(b_native, 'raw_data'):
        b_native.raw_data = b'raw2'

    t = mapper.to_common_ticker(t_native, md)
    assert t.symbol.base == 'BTC'
    assert t.symbol.quote in ('USDT', 'USD')

    l2 = getattr(mapper, 'to_common_l2_from_depth', None)
    l2 = l2(b_native, md) if l2 and b_native.__class__.__name__ == 'DepthUpdate' else mapper.to_common_l2_from_orderbook(b_native, md)
    assert l2.bids and l2.asks and l2.bids[0].price.value == '1' and l2.asks[0].size.value == '4'
