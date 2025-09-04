# Example serializer for publishing cryptofeed updates as protobuf DataFeedEvent to Kafka

import os

# Ensure generated Python modules are importable if running this script directly
GEN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gen', 'python'))
if GEN_PATH not in os.sys.path:
    os.sys.path.insert(0, GEN_PATH)

from google.protobuf.timestamp_pb2 import Timestamp
from cryptofeed.v1 import common_pb2 as common
from cryptofeed.v1 import market_data_pb2 as md
from cryptofeed.v1 import events_pb2 as ev

EXCHANGE_ENUM = {name.replace('EXCHANGE_', ''): getattr(common.Exchange, name) for name in common.Exchange.keys()}
SIDE_ENUM = {name.replace('SIDE_', '').lower(): getattr(common.Side, name) for name in common.Side.keys()}


def _to_ts(ts_float: float) -> Timestamp:
    ts = Timestamp()
    if ts_float is not None:
        seconds = int(ts_float)
        nanos = int((ts_float - seconds) * 1e9)
        ts.seconds = seconds
        ts.nanos = nanos
    return ts


def _build_symbol(symbol_str: str) -> common.Symbol:
    s = common.Symbol(symbol=symbol_str)
    if '-' in symbol_str:
        base, quote = symbol_str.split('-', 1)
        s.base = base
        s.quote = quote
    return s


def build_trade(d: dict) -> md.Trade:
    msg = md.Trade()
    msg.exchange = EXCHANGE_ENUM.get(d['exchange'], common.EXCHANGE_UNSPECIFIED)
    msg.symbol.CopyFrom(_build_symbol(d['symbol']))
    msg.side = SIDE_ENUM.get(d.get('side', ''), common.SIDE_UNSPECIFIED)
    msg.amount.value = str(d['amount'])
    msg.price.value = str(d['price'])
    if d.get('id'):
        msg.id = str(d['id'])
    if d.get('type'):
        msg.type = str(d['type'])
    msg.timestamp.CopyFrom(_to_ts(d.get('timestamp')))
    return msg

def build_ticker(d: dict) -> md.Ticker:
    msg = md.Ticker()
    msg.exchange = EXCHANGE_ENUM.get(d['exchange'], common.EXCHANGE_UNSPECIFIED)
    msg.symbol.CopyFrom(_build_symbol(d['symbol']))
    if d.get('bid') is not None:
        msg.bid.value = str(d['bid'])
    if d.get('ask') is not None:
        msg.ask.value = str(d['ask'])
    msg.timestamp.CopyFrom(_to_ts(d.get('timestamp')))
    return msg

def build_l1_book(d: dict) -> md.L1Book:
    msg = md.L1Book()
    msg.exchange = EXCHANGE_ENUM.get(d['exchange'], common.EXCHANGE_UNSPECIFIED)
    msg.symbol.CopyFrom(_build_symbol(d['symbol']))
    if d.get('bid_price') is not None:
        msg.bid_price.value = str(d['bid_price'])
    if d.get('bid_size') is not None:
        msg.bid_size.value = str(d['bid_size'])
    if d.get('ask_price') is not None:
        msg.ask_price.value = str(d['ask_price'])
    if d.get('ask_size') is not None:
        msg.ask_size.value = str(d['ask_size'])
    msg.timestamp.CopyFrom(_to_ts(d.get('timestamp')))
    return msg

def build_l2_book(d: dict) -> md.L2Book:
    msg = md.L2Book()
    msg.exchange = EXCHANGE_ENUM.get(d['exchange'], common.EXCHANGE_UNSPECIFIED)
    msg.symbol.CopyFrom(_build_symbol(d['symbol']))
    for price, size in d.get('bids', []):
        level = md.PriceLevel()
        level.price.value = str(price)
        level.size.value = str(size)
        msg.bids.append(level)
    for price, size in d.get('asks', []):
        level = md.PriceLevel()
        level.price.value = str(price)
        level.size.value = str(size)
        msg.asks.append(level)
    if d.get('sequence_number') is not None:
        msg.sequence_number = int(d['sequence_number'])
    if d.get('checksum'):
        msg.checksum = str(d['checksum'])
    msg.timestamp.CopyFrom(_to_ts(d.get('timestamp')))
    return msg

def build_funding(d: dict) -> md.Funding:
    msg = md.Funding()
    msg.exchange = EXCHANGE_ENUM.get(d['exchange'], common.EXCHANGE_UNSPECIFIED)
    msg.symbol.CopyFrom(_build_symbol(d['symbol']))
    if d.get('mark_price') is not None:
        msg.mark_price.value = str(d['mark_price'])
    if d.get('rate') is not None:
        msg.rate.value = str(d['rate'])
    if d.get('predicted_rate') is not None:
        msg.predicted_rate.value = str(d['predicted_rate'])
    if d.get('next_funding_time') is not None:
        msg.next_funding_time.CopyFrom(_to_ts(d['next_funding_time']))
    msg.timestamp.CopyFrom(_to_ts(d.get('timestamp')))
    return msg


def to_event(channel: int, d: dict) -> ev.DataFeedEvent:
    e = ev.DataFeedEvent()
    e.channel = channel
    e.exchange = EXCHANGE_ENUM.get(d['exchange'], common.EXCHANGE_UNSPECIFIED)
    e.symbol.CopyFrom(_build_symbol(d['symbol']))
    e.event_timestamp.CopyFrom(_to_ts(d.get('timestamp')))
    if channel == common.DATA_CHANNEL_TRADES:
        e.trade.CopyFrom(build_trade(d))
    elif channel == common.DATA_CHANNEL_TICKER:
        e.ticker.CopyFrom(build_ticker(d))
    elif channel == common.DATA_CHANNEL_L1_BOOK:
        e.l1_book.CopyFrom(build_l1_book(d))
    elif channel == common.DATA_CHANNEL_L2_BOOK:
        e.l2_book.CopyFrom(build_l2_book(d))
    elif channel == common.DATA_CHANNEL_FUNDING:
        e.funding.CopyFrom(build_funding(d))
    return e


def make_value_serializer(channel: int):
    def serializer(d: dict) -> bytes:
        evt = to_event(channel, d)
        return evt.SerializeToString()
    return serializer
