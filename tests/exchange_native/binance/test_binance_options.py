import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def test_binance_options_trade_maps_to_option_type_and_side():
    from cryptofeed.proto_mappers import binance as mapper
    b = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')

    native = b.Trade(symbol='BTCUSDT')
    native.segment = b.MARKET_SEGMENT_OPTIONS
    native.price.value = '100.0'
    native.quantity.value = '1.5'
    native.is_buyer_maker = False  # taker buys -> side BUY

    out = mapper.to_common_trade(native, md)
    assert out.symbol.type == cmn.INSTRUMENT_TYPE_OPTION
    assert out.symbol.base == 'BTC' and out.symbol.quote == 'USDT'
    assert out.side == cmn.SIDE_BUY
    assert out.price.value == '100.0' and out.amount.value == '1.5'


def test_binance_options_ticker_maps_to_option_type():
    from cryptofeed.proto_mappers import binance as mapper
    b = load_pb2(ROOT, 'gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')

    native = b.BookTicker(symbol='BTCUSDT')
    native.segment = b.MARKET_SEGMENT_OPTIONS
    native.bid_price.value = '99.5'
    native.ask_price.value = '100.5'

    t = mapper.to_common_ticker(native, md)
    assert t.symbol.type == cmn.INSTRUMENT_TYPE_OPTION
    assert t.symbol.base == 'BTC' and t.symbol.quote == 'USDT'
    assert t.bid.value == '99.5' and t.ask.value == '100.5'
