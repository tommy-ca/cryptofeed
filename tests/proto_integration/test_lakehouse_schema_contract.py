import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tests.util.pb2_loader import project_root, load_pb2

ROOT = project_root(os.path.dirname(__file__))


def test_trade_message_supports_lakehouse_columns():
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    t = md.Trade()
    t.exchange = cmn.EXCHANGE_OKX
    t.symbol.base = 'BTC'
    t.symbol.quote = 'USDT'
    t.symbol.symbol = 'BTC-USDT'
    t.symbol.type = cmn.INSTRUMENT_TYPE_SPOT
    t.side = cmn.SIDE_BUY
    t.amount.value = '0.5'
    t.price.value = '60000.00'
    t.id = 'abc-1'
    t.type = 'market'

    data = t.SerializeToString()
    out = md.Trade(); out.ParseFromString(data)
    assert out.exchange == cmn.EXCHANGE_OKX
    assert out.symbol.base == 'BTC' and out.symbol.quote == 'USDT' and out.symbol.type == cmn.INSTRUMENT_TYPE_SPOT
    assert out.side == cmn.SIDE_BUY
    assert out.amount.value == '0.5' and out.price.value == '60000.00'
    assert out.id == 'abc-1' and out.type == 'market'


def test_ticker_message_supports_lakehouse_columns():
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    t = md.Ticker()
    t.exchange = cmn.EXCHANGE_BINANCE
    t.symbol.base = 'ETH'
    t.symbol.quote = 'USDT'
    t.symbol.symbol = 'ETH-USDT'
    t.symbol.type = cmn.INSTRUMENT_TYPE_SPOT
    t.bid.value = '3000.1'
    t.ask.value = '3001.2'

    data = t.SerializeToString()
    out = md.Ticker(); out.ParseFromString(data)
    assert out.symbol.base == 'ETH' and out.symbol.quote == 'USDT'
    assert out.bid.value == '3000.1' and out.ask.value == '3001.2'


def test_bookdelta_message_supports_array_of_structs():
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    d = md.BookDelta()
    d.exchange = cmn.EXCHANGE_BYBIT
    d.symbol.base = 'BTC'
    d.symbol.quote = 'USDT'
    d.symbol.symbol = 'BTC-USDT'
    d.symbol.type = cmn.INSTRUMENT_TYPE_PERPETUAL
    d.sequence_number = 123
    b = md.PriceLevel(); b.price.value='1'; b.size.value='2'
    a = md.PriceLevel(); a.price.value='3'; a.size.value='0'
    d.bid_changes.append(b)
    d.ask_changes.append(a)

    data = d.SerializeToString()
    out = md.BookDelta(); out.ParseFromString(data)
    assert out.sequence_number == 123
    assert out.bid_changes[0].price.value == '1' and out.bid_changes[0].size.value == '2'
    assert out.ask_changes[0].price.value == '3' and out.ask_changes[0].size.value == '0'


def test_decimal_and_symbol_shapes_are_stable_for_etl():
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')
    dec = cmn.Decimal(value='1.2345')
    sy = cmn.Symbol(base='BTC', quote='USD', symbol='BTC-USD', type=cmn.INSTRUMENT_TYPE_SPOT)
    assert hasattr(dec, 'value') and isinstance(dec.value, str)
    assert sy.base == 'BTC' and sy.quote == 'USD' and sy.symbol == 'BTC-USD'


def test_funding_message_supports_lakehouse_columns():
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    f = md.Funding()
    f.exchange = cmn.EXCHANGE_OKX
    f.symbol.base = 'BTC'; f.symbol.quote = 'USDT'; f.symbol.symbol = 'BTC-USDT'
    f.symbol.type = cmn.INSTRUMENT_TYPE_PERPETUAL
    f.mark_price.value = '65000.00'
    f.rate.value = '0.0001'

    data = f.SerializeToString(); out = md.Funding(); out.ParseFromString(data)
    assert out.mark_price.value == '65000.00'
    assert out.rate.value == '0.0001'


def test_l1book_message_supports_lakehouse_columns():
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    b = md.L1Book()
    b.exchange = cmn.EXCHANGE_BINANCE
    b.symbol.base='BTC'; b.symbol.quote='USDT'; b.symbol.symbol='BTC-USDT'; b.symbol.type=cmn.INSTRUMENT_TYPE_SPOT
    b.bid_price.value='1'; b.bid_size.value='2'; b.ask_price.value='3'; b.ask_size.value='4'
    data=b.SerializeToString(); out=md.L1Book(); out.ParseFromString(data)
    assert out.bid_price.value=='1' and out.bid_size.value=='2'
    assert out.ask_price.value=='3' and out.ask_size.value=='4'


def test_candle_message_supports_lakehouse_columns():
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    c = md.Candle()
    c.exchange = cmn.EXCHANGE_BYBIT
    c.symbol.base='ETH'; c.symbol.quote='USDT'; c.symbol.symbol='ETH-USDT'; c.symbol.type=cmn.INSTRUMENT_TYPE_SPOT
    c.interval='1m'
    c.open.value='1'; c.close.value='2'; c.high.value='3'; c.low.value='0.5'; c.volume.value='10'
    c.closed=True
    data=c.SerializeToString(); out=md.Candle(); out.ParseFromString(data)
    assert out.interval=='1m' and out.open.value=='1' and out.closed is True


def test_open_interest_message_supports_lakehouse_columns():
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    oi = md.OpenInterest()
    oi.exchange = cmn.EXCHANGE_BITGET
    oi.symbol.base='BTC'; oi.symbol.quote='USDT'; oi.symbol.symbol='BTC-USDT'; oi.symbol.type=cmn.INSTRUMENT_TYPE_PERPETUAL
    oi.open_interest.value='12345'
    data=oi.SerializeToString(); out=md.OpenInterest(); out.ParseFromString(data)
    assert out.open_interest.value=='12345'


def test_liquidation_message_supports_lakehouse_columns():
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    liq = md.Liquidation()
    liq.exchange = cmn.EXCHANGE_OKX
    liq.symbol.base='BTC'; liq.symbol.quote='USDT'; liq.symbol.symbol='BTC-USDT'; liq.symbol.type=cmn.INSTRUMENT_TYPE_PERPETUAL
    liq.side = cmn.SIDE_SELL
    liq.quantity.value='0.01'
    liq.price.value='58000'
    liq.id='liq-1'
    liq.status='filled'
    data=liq.SerializeToString(); out=md.Liquidation(); out.ParseFromString(data)
    assert out.side==cmn.SIDE_SELL and out.quantity.value=='0.01' and out.price.value=='58000'


def test_index_message_supports_lakehouse_columns():
    cmn = load_pb2(ROOT, 'gen/python/cryptofeed/v1/common_pb2.py')
    md = load_pb2(ROOT, 'gen/python/cryptofeed/v1/market_data_pb2.py')

    idx = md.Index()
    idx.exchange = cmn.EXCHANGE_OKX
    idx.symbol.base='BTC'; idx.symbol.quote='USDT'; idx.symbol.symbol='BTC-USDT'; idx.symbol.type=cmn.INSTRUMENT_TYPE_SPOT
    idx.price.value='60000'
    data=idx.SerializeToString(); out=md.Index(); out.ParseFromString(data)
    assert out.price.value=='60000'
