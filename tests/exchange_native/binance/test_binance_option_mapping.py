import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def _load_pb2(path: str):
    return load_pb2(ROOT, path)


def test_binance_option_trade_maps_to_option_type():
    from cryptofeed.proto_mappers import binance as mapper
    b = _load_pb2('gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')
    md = _load_pb2('gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = _load_pb2('gen/python/cryptofeed/v1/common_pb2.py')

    # Binance options symbols are not hyphen split; keep native symbol string
    native = b.Trade(symbol='BTC-240927-60000-C', trade_id='1', is_buyer_maker=False)
    native.segment = b.MARKET_SEGMENT_OPTIONS
    native.price.value = '100.5'
    native.quantity.value = '2'

    out = mapper.to_common_trade(native, md, cmn)
    assert out.symbol.type == cmn.INSTRUMENT_TYPE_OPTION
    assert out.symbol.symbol == 'BTC-240927-60000-C'
    # We do not over-normalize options for Binance; base/quote may be empty
    assert out.symbol.base == '' and out.symbol.quote == ''
    # Buyer-not-maker implies BUY per mapper logic
    assert out.side == cmn.SIDE_BUY
    assert out.price.value == '100.5' and out.amount.value == '2'


def test_binance_option_ticker_maps_to_option_type():
    from cryptofeed.proto_mappers import binance as mapper
    b = _load_pb2('gen/python/cryptofeed/exchanges/binance/v1/binance_pb2.py')
    md = _load_pb2('gen/python/cryptofeed/v1/market_data_pb2.py')
    cmn = _load_pb2('gen/python/cryptofeed/v1/common_pb2.py')

    bt = b.BookTicker(symbol='BTC-240927-60000-C')
    bt.segment = b.MARKET_SEGMENT_OPTIONS
    bt.bid_price.value = '99.9'
    bt.ask_price.value = '100.1'

    out = mapper.to_common_ticker(bt, md, cmn)
    assert out.symbol.type == cmn.INSTRUMENT_TYPE_OPTION
    assert out.symbol.symbol == 'BTC-240927-60000-C'
    assert out.symbol.base == '' and out.symbol.quote == ''
    assert out.bid.value == '99.9' and out.ask.value == '100.1'

