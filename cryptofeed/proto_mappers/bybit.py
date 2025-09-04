from . import util

def _exchange(common):
    return common.EXCHANGE_BYBIT

def _instrument_type_from_segment(common, segment: int):
    s = int(segment)
    if s == 1: return common.INSTRUMENT_TYPE_SPOT
    if s == 2: return common.INSTRUMENT_TYPE_PERPETUAL
    if s == 3: return common.INSTRUMENT_TYPE_PERPETUAL
    if s == 4: return common.INSTRUMENT_TYPE_OPTION
    return common.INSTRUMENT_TYPE_UNSPECIFIED

def _symbol_from_native(common, symbol: str, segment: int):
    base, quote = util.split_base_quote_concat(symbol)
    sym = common.Symbol()
    sym.base = base
    sym.quote = quote
    sym.symbol = f"{base}-{quote}" if quote else symbol
    sym.type = _instrument_type_from_segment(common, segment)
    return sym

def _side_from_text(common, side: str):
    s = (side or '').lower()
    if s == 'buy': return common.SIDE_BUY
    if s == 'sell': return common.SIDE_SELL
    return common.SIDE_UNSPECIFIED

def to_common_trade(native, md, cmn=None):
    if cmn is None:
        import importlib
        cmn = importlib.import_module('cryptofeed.v1.common_pb2')
    common = cmn
    out = md.Trade()
    out.exchange = _exchange(common)
    out.symbol.CopyFrom(_symbol_from_native(common, native.symbol, native.segment))
    out.side = _side_from_text(common, getattr(native, 'side', ''))
    out.price.CopyFrom(util.make_decimal(cmn, getattr(getattr(native, 'price', None), 'value', getattr(native, 'price', None))))
    out.amount.CopyFrom(util.make_decimal(cmn, getattr(getattr(native, 'quantity', None), 'value', getattr(native, 'quantity', None))))
    out.id = getattr(native, 'trade_id', '')
    if getattr(native, 'event_timestamp', None):
        out.timestamp.CopyFrom(native.event_timestamp)
    if getattr(native, 'raw_data', None):
        out.raw_data = native.raw_data
    return out


def to_common_ticker(native, md, cmn=None):
    if cmn is None:
        import importlib
        cmn = importlib.import_module('cryptofeed.v1.common_pb2')
    common = cmn
    out = md.Ticker()
    out.exchange = _exchange(common)
    out.symbol.CopyFrom(_symbol_from_native(common, native.symbol, native.segment))
    out.bid.CopyFrom(util.make_decimal(cmn, getattr(getattr(native, 'best_bid', None), 'value', getattr(native, 'best_bid', None))))
    out.ask.CopyFrom(util.make_decimal(cmn, getattr(getattr(native, 'best_ask', None), 'value', getattr(native, 'best_ask', None))))
    if getattr(native, 'event_timestamp', None):
        out.timestamp.CopyFrom(native.event_timestamp)
    if getattr(native, 'raw_data', None):
        out.raw_data = native.raw_data
    return out


def to_common_l2_from_orderbook(native, md, cmn=None):
    if cmn is None:
        import importlib
        cmn = importlib.import_module('cryptofeed.v1.common_pb2')
    common = cmn
    out = md.L2Book()
    out.exchange = _exchange(common)
    out.symbol.CopyFrom(_symbol_from_native(common, native.symbol, native.segment))
    for lvl in getattr(native, 'bids', []):
        pl = md.PriceLevel()
        pl.price.CopyFrom(util.make_decimal(cmn, getattr(getattr(lvl, 'price', None), 'value', getattr(lvl, 'price', None))))
        pl.size.CopyFrom(util.make_decimal(cmn, getattr(getattr(lvl, 'size', None), 'value', getattr(lvl, 'size', None))))
        out.bids.append(pl)
    for lvl in getattr(native, 'asks', []):
        pl = md.PriceLevel()
        pl.price.CopyFrom(util.make_decimal(cmn, getattr(getattr(lvl, 'price', None), 'value', getattr(lvl, 'price', None))))
        pl.size.CopyFrom(util.make_decimal(cmn, getattr(getattr(lvl, 'size', None), 'value', getattr(lvl, 'size', None))))
        out.asks.append(pl)
    if getattr(native, 'seq', None):
        out.sequence_number = native.seq
    if getattr(native, 'event_timestamp', None):
        out.timestamp.CopyFrom(native.event_timestamp)
    if getattr(native, 'raw_data', None):
        out.raw_data = native.raw_data
    return out


def to_common_funding(native, md, cmn=None):
    if cmn is None:
        import importlib
        cmn = importlib.import_module('cryptofeed.v1.common_pb2')
    common = cmn
    out = md.Funding()
    out.exchange = _exchange(common)
    out.symbol.CopyFrom(_symbol_from_native(common, native.symbol, native.segment))
    if getattr(native, 'mark_price', None):
        out.mark_price.CopyFrom(util.make_decimal(cmn, getattr(getattr(native, 'mark_price', None), 'value', getattr(native, 'mark_price', None))))
    if getattr(native, 'rate', None):
        out.rate.CopyFrom(util.make_decimal(cmn, getattr(getattr(native, 'rate', None), 'value', getattr(native, 'rate', None))))
    if getattr(native, 'next_funding_time', None):
        out.next_funding_time.CopyFrom(native.next_funding_time)
    if getattr(native, 'event_timestamp', None):
        out.timestamp.CopyFrom(native.event_timestamp)
    if getattr(native, 'raw_data', None):
        out.raw_data = native.raw_data
    return out


def to_common_book_delta(native, md, cmn=None):
    if cmn is None:
        import importlib
        cmn = importlib.import_module('cryptofeed.v1.common_pb2')
    common = cmn
    out = md.BookDelta()
    out.exchange = _exchange(common)
    out.symbol.CopyFrom(_symbol_from_native(common, native.symbol, native.segment))
    for lvl in getattr(native, 'bids', []):
        pl = md.PriceLevel(); pl.price.CopyFrom(util.make_decimal(cmn, getattr(getattr(lvl, 'price', None), 'value', getattr(lvl, 'price', None)))); pl.size.CopyFrom(util.make_decimal(cmn, getattr(getattr(lvl, 'size', None), 'value', getattr(lvl, 'size', None))))
        out.bid_changes.append(pl)
    for lvl in getattr(native, 'asks', []):
        pl = md.PriceLevel(); pl.price.CopyFrom(util.make_decimal(cmn, getattr(getattr(lvl, 'price', None), 'value', getattr(lvl, 'price', None)))); pl.size.CopyFrom(util.make_decimal(cmn, getattr(getattr(lvl, 'size', None), 'value', getattr(lvl, 'size', None))))
        out.ask_changes.append(pl)
    if getattr(native, 'seq_id', None): out.sequence_number = native.seq_id
    if getattr(native, 'seq', None): out.sequence_number = native.seq
    if getattr(native, 'event_timestamp', None): out.timestamp.CopyFrom(native.event_timestamp)
    if getattr(native, 'raw_data', None): out.raw_data = native.raw_data
    return out
