from typing import Any
from . import util


def _decimal(cmn, val: Any):
    return util.make_decimal(cmn, val)


def _exchange_from_segment(common, segment):
    seg = int(segment)
    if seg == 1:  # SPOT
        return common.EXCHANGE_BINANCE
    if seg == 2:  # FUTURES_UM
        return common.EXCHANGE_BINANCE_FUTURES
    if seg == 3:  # FUTURES_CM
        return common.EXCHANGE_BINANCE_DELIVERY
    return common.EXCHANGE_BINANCE


def _instrument_type_from_segment(common, segment):
    seg = int(segment)
    if seg == 1:
        return common.INSTRUMENT_TYPE_SPOT
    if seg == 2:
        return common.INSTRUMENT_TYPE_PERPETUAL
    if seg == 3:
        return common.INSTRUMENT_TYPE_FUTURES
    if seg == 4:
        return common.INSTRUMENT_TYPE_OPTION
    return common.INSTRUMENT_TYPE_UNSPECIFIED


def _symbol_from_native(common, cmn, native_symbol: str, segment):
    base, quote = util.split_base_quote_concat(native_symbol)
    sym = common.Symbol()
    sym.base = base
    sym.quote = quote
    sym.symbol = f"{base}-{quote}" if quote else native_symbol
    sym.type = _instrument_type_from_segment(common, segment)
    return sym


def _side_from_buyer_maker(common, is_buyer_maker: bool):
    return common.SIDE_SELL if is_buyer_maker else common.SIDE_BUY


def to_common_trade(native, md, cmn=None):
    if cmn is None:
        import importlib
        cmn = importlib.import_module('cryptofeed.v1.common_pb2')
    common = cmn
    out = md.Trade()
    ex = getattr(native, 'exchange', 0)
    if ex == getattr(common, 'EXCHANGE_UNSPECIFIED', 0):
        ex = _exchange_from_segment(common, native.segment)
    out.exchange = ex
    out.symbol.CopyFrom(_symbol_from_native(common, cmn, native.symbol, native.segment))
    out.side = _side_from_buyer_maker(common, native.is_buyer_maker)
    out.amount.CopyFrom(_decimal(cmn, getattr(getattr(native, 'quantity', None), 'value', getattr(native, 'quantity', None))))
    out.price.CopyFrom(_decimal(cmn, getattr(getattr(native, 'price', None), 'value', getattr(native, 'price', None))))
    if getattr(native, 'event_timestamp', None):
        out.timestamp.CopyFrom(native.event_timestamp)
    out.id = native.trade_id
    if getattr(native, 'raw_data', None):
        out.raw_data = native.raw_data
    return out


def to_common_ticker(native, md, cmn=None):
    if cmn is None:
        import importlib
        cmn = importlib.import_module('cryptofeed.v1.common_pb2')
    common = cmn
    out = md.Ticker()
    ex = getattr(native, 'exchange', 0)
    if ex == getattr(common, 'EXCHANGE_UNSPECIFIED', 0):
        ex = _exchange_from_segment(common, native.segment)
    out.exchange = ex
    out.symbol.CopyFrom(_symbol_from_native(common, cmn, native.symbol, native.segment))
    # Support either BookTicker( bid_price/bid_qty/ask_price/ask_qty ) or Ticker(best_bid/best_ask)
    bid = getattr(native, 'bid_price', None) or getattr(native, 'best_bid', None)
    ask = getattr(native, 'ask_price', None) or getattr(native, 'best_ask', None)
    out.bid.CopyFrom(_decimal(cmn, getattr(bid, 'value', bid)))
    out.ask.CopyFrom(_decimal(cmn, getattr(ask, 'value', ask)))
    if getattr(native, 'event_timestamp', None):
        out.timestamp.CopyFrom(native.event_timestamp)
    if getattr(native, 'raw_data', None):
        out.raw_data = native.raw_data
    return out


def to_common_l1_from_bookticker(native, md, cmn=None):
    if cmn is None:
        import importlib
        cmn = importlib.import_module('cryptofeed.v1.common_pb2')
    common = cmn
    out = md.L1Book()
    ex = getattr(native, 'exchange', 0)
    if ex == getattr(common, 'EXCHANGE_UNSPECIFIED', 0):
        ex = _exchange_from_segment(common, native.segment)
    out.exchange = ex
    out.symbol.CopyFrom(_symbol_from_native(common, cmn, native.symbol, native.segment))
    out.bid_price.CopyFrom(_decimal(cmn, getattr(getattr(native, 'bid_price', None), 'value', getattr(native, 'bid_price', None))))
    out.bid_size.CopyFrom(_decimal(cmn, getattr(getattr(native, 'bid_qty', None), 'value', getattr(native, 'bid_qty', None))))
    out.ask_price.CopyFrom(_decimal(cmn, getattr(getattr(native, 'ask_price', None), 'value', getattr(native, 'ask_price', None))))
    out.ask_size.CopyFrom(_decimal(cmn, getattr(getattr(native, 'ask_qty', None), 'value', getattr(native, 'ask_qty', None))))
    if getattr(native, 'event_timestamp', None):
        out.timestamp.CopyFrom(native.event_timestamp)
    if getattr(native, 'raw_data', None):
        out.raw_data = native.raw_data
    return out


def to_common_l2_from_depth(native, md, cmn=None):
    if cmn is None:
        import importlib
        cmn = importlib.import_module('cryptofeed.v1.common_pb2')
    common = cmn
    out = md.L2Book()
    ex = getattr(native, 'exchange', 0)
    if ex == getattr(common, 'EXCHANGE_UNSPECIFIED', 0):
        ex = _exchange_from_segment(common, native.segment)
    out.exchange = ex
    out.symbol.CopyFrom(_symbol_from_native(common, cmn, native.symbol, native.segment))
    for lvl in getattr(native, 'bids', []):
        pl = md.PriceLevel()
        pl.price.CopyFrom(_decimal(cmn, getattr(getattr(lvl, 'price', None), 'value', getattr(lvl, 'price', None))))
        pl.size.CopyFrom(_decimal(cmn, getattr(getattr(lvl, 'size', None), 'value', getattr(lvl, 'size', None))))
        out.bids.append(pl)
    for lvl in getattr(native, 'asks', []):
        pl = md.PriceLevel()
        pl.price.CopyFrom(_decimal(cmn, getattr(getattr(lvl, 'price', None), 'value', getattr(lvl, 'price', None))))
        pl.size.CopyFrom(_decimal(cmn, getattr(getattr(lvl, 'size', None), 'value', getattr(lvl, 'size', None))))
        out.asks.append(pl)
    if getattr(native, 'final_update_id', None):
        out.sequence_number = native.final_update_id
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
    ex = getattr(native, 'exchange', 0)
    if ex == getattr(common, 'EXCHANGE_UNSPECIFIED', 0):
        ex = _exchange_from_segment(common, native.segment)
    out.exchange = ex
    out.symbol.CopyFrom(_symbol_from_native(common, cmn, native.symbol, native.segment))
    if getattr(native, 'mark_price', None):
        out.mark_price.CopyFrom(_decimal(cmn, getattr(getattr(native, 'mark_price', None), 'value', getattr(native, 'mark_price', None))))
    if getattr(native, 'rate', None):
        out.rate.CopyFrom(_decimal(cmn, getattr(getattr(native, 'rate', None), 'value', getattr(native, 'rate', None))))
    if getattr(native, 'next_funding_time', None):
        out.next_funding_time.CopyFrom(native.next_funding_time)
    if getattr(native, 'event_timestamp', None):
        out.timestamp.CopyFrom(native.event_timestamp)
    if getattr(native, 'raw_data', None):
        out.raw_data = native.raw_data
    return out


def to_common_book_delta_from_depth(native, md, cmn=None):
    if cmn is None:
        import importlib
        cmn = importlib.import_module('cryptofeed.v1.common_pb2')
    common = cmn
    out = md.BookDelta()
    ex = getattr(native, 'exchange', 0)
    if ex == getattr(common, 'EXCHANGE_UNSPECIFIED', 0):
        ex = _exchange_from_segment(common, native.segment)
    out.exchange = ex
    out.symbol.CopyFrom(_symbol_from_native(common, cmn, native.symbol, native.segment))
    for lvl in getattr(native, 'bids', []):
        pl = md.PriceLevel()
        pl.price.CopyFrom(_decimal(cmn, getattr(getattr(lvl, 'price', None), 'value', getattr(lvl, 'price', None))))
        pl.size.CopyFrom(_decimal(cmn, getattr(getattr(lvl, 'size', None), 'value', getattr(lvl, 'size', None))))
        out.bid_changes.append(pl)
    for lvl in getattr(native, 'asks', []):
        pl = md.PriceLevel()
        pl.price.CopyFrom(_decimal(cmn, getattr(getattr(lvl, 'price', None), 'value', getattr(lvl, 'price', None))))
        pl.size.CopyFrom(_decimal(cmn, getattr(getattr(lvl, 'size', None), 'value', getattr(lvl, 'size', None))))
        out.ask_changes.append(pl)
    if getattr(native, 'final_update_id', None):
        out.sequence_number = native.final_update_id
    if getattr(native, 'event_timestamp', None):
        out.timestamp.CopyFrom(native.event_timestamp)
    if getattr(native, 'raw_data', None):
        out.raw_data = native.raw_data
    return out
