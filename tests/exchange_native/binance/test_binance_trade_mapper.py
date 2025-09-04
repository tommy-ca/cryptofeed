
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))

def test_binance_trade_to_common():
    from cryptofeed.proto_mappers import binance as mapper
    b = load_pb2(ROOT, "gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py")
    md = load_pb2(ROOT, "gen/python/cryptofeed/v1/market_data_pb2.py")
    cmn = load_pb2(ROOT, "gen/python/cryptofeed/v1/common_pb2.py")

    # Build native trade
    native = b.Trade(symbol="BTCUSDT", trade_id="1001", is_buyer_maker=False)
    native.segment = b.MARKET_SEGMENT_FUTURES_UM
    native.price.value = "62000"
    native.quantity.value = "0.01"

    # Map to common
    from cryptofeed.proto_mappers import binance as mapper
    out = mapper.to_common_trade(native, md)

    assert out.exchange == cmn.EXCHANGE_BINANCE_FUTURES
    assert out.symbol.base == "BTC"
    assert out.symbol.quote == "USDT"
    assert out.symbol.type == cmn.INSTRUMENT_TYPE_PERPETUAL
    assert out.price.value == "62000"
    assert out.amount.value == "0.01"
    assert out.id == "1001"
    assert out.side == cmn.SIDE_BUY  # taker buy when is_buyer_maker=False
