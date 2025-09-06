import os, sys
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
    def load_pb2(root: str, rel_path: str):
        gen_root = os.path.join(root, 'gen', 'python')
        if gen_root not in sys.path:
            sys.path.insert(0, gen_root)
        cf_pkg = types.ModuleType('cryptofeed'); cf_pkg.__path__=[os.path.join(gen_root,'cryptofeed')]
        sys.modules['cryptofeed']=cf_pkg
        path = os.path.join(root, rel_path)
        spec = importlib.util.spec_from_file_location('_pb2', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod


def test_registry_maps_binance_trade_via_fullname():
    # Import registry first to avoid pb2 loader shadowing source packages
    from cryptofeed.proto_mappers.registry import default_registry
    root = project_root(os.path.dirname(__file__))
    md = load_pb2(root, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = load_pb2(root, 'gen/python/cryptofeed/v1/common_pb2.py')
    b = load_pb2(root, 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')

    # Create native trade (module name likely '_pb2')
    native = b.Trade(symbol='BTCUSDT', trade_id='1', is_buyer_maker=False)
    native.segment = b.MARKET_SEGMENT_SPOT
    native.price.value = '60000'
    native.quantity.value = '0.5'

    reg = default_registry()
    t = reg.map(native, md)

    assert t.exchange == cmn.EXCHANGE_BINANCE
    assert t.symbol.base == 'BTC' and t.symbol.quote == 'USDT'
    assert t.side == cmn.SIDE_BUY
    assert t.price.value == '60000' and t.amount.value == '0.5'


def test_registry_maps_okx_orderbook_via_fullname():
    from cryptofeed.proto_mappers.registry import default_registry
    root = project_root(os.path.dirname(__file__))
    md = load_pb2(root, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = load_pb2(root, 'gen/python/cryptofeed/v1/common_pb2.py')
    o = load_pb2(root, 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py')

    ob = o.OrderBook(inst_id='BTC-USDT')
    # add one level to each side
    plb = md.PriceLevel(); plb.price.value='100'; plb.size.value='1'
    pla = md.PriceLevel(); pla.price.value='101'; pla.size.value='2'
    ob.bids.append(plb); ob.asks.append(pla)
    ob.seq_id = 123

    from cryptofeed.proto_mappers.registry import default_registry
    reg = default_registry()
    l2 = reg.map(ob, md)
    assert l2.exchange == cmn.EXCHANGE_OKX
    assert l2.symbol.base == 'BTC' and l2.symbol.quote == 'USDT'
    assert l2.bids and l2.asks and l2.sequence_number == 123
