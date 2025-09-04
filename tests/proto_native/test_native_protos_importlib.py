import os
import importlib.util
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ensure_gen_on_path():
    gen_root = os.path.join(ROOT, "gen", "python")
    if gen_root not in sys.path:
        sys.path.insert(0, gen_root)
    # Shadow runtime 'cryptofeed' with a shim pointing to generated package
    import types
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


def test_binance_trade_roundtrip():
    b = _load_pb2("gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py")
    msg = b.Trade(symbol="BTCUSDT", trade_id="123", is_buyer_maker=True)
    data = msg.SerializeToString()
    out = b.Trade()
    out.ParseFromString(data)
    assert out.symbol == "BTCUSDT"
    assert out.trade_id == "123"
    assert out.is_buyer_maker is True


def test_okx_orderbook_has_levels():
    o = _load_pb2("gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py")
    md = _load_pb2("gen/python/cryptofeed/v1/market_data_pb2.py")
    ob = o.OrderBook(inst_id="BTC-USDT")
    ob.bids.append(md.PriceLevel())
    ob.asks.append(md.PriceLevel())
    assert len(ob.bids) == 1
    assert len(ob.asks) == 1
