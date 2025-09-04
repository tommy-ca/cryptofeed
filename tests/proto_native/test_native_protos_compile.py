import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def test_binance_trade_roundtrip():
    b = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')
    msg = b.Trade(
        symbol="BTCUSDT",
        trade_id="123",
        is_buyer_maker=True,
    )
    data = msg.SerializeToString()
    out = b.Trade()
    out.ParseFromString(data)
    assert out.symbol == "BTCUSDT"
    assert out.trade_id == "123"
    assert out.is_buyer_maker is True


def test_okx_orderbook_has_levels():
    o = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/okx/v1/okx_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    ob = o.OrderBook(inst_id="BTC-USDT")
    ob.bids.append(md.PriceLevel())
    ob.asks.append(md.PriceLevel())
    assert len(ob.bids) == 1
    assert len(ob.asks) == 1


def test_bybit_ticker_fields_present():
    y = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bybit/v1/bybit_pb2.py')
    tick = y.Ticker(symbol="BTCUSDT")
    assert tick.symbol == "BTCUSDT"


def test_bitget_trade_fields_present():
    g = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/bitget/v1/bitget_pb2.py')
    t = g.Trade(inst_id="BTCUSDT", trade_id="abc")
    assert t.inst_id == "BTCUSDT"
    assert t.trade_id == "abc"

