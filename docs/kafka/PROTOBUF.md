# Kafka + Protobuf for Cryptofeed

This guide shows how to publish protobuf-encoded messages to Kafka using the existing Kafka backend. We will set a value_serializer on AIOKafkaProducer to encode dict payloads produced by cryptofeed into protobuf messages.

Two approaches
- Channel messages: Serialize specific messages (Trade, Ticker, L1Book, BookDelta, etc.). Use per-channel topics.
- Event envelope: Serialize DataFeedEvent to a single topic or channel-specific topics.

Recommended
- Use DataFeedEvent for maximum flexibility and consistency across channels.

Topic conventions
- Channel-specific: cryptofeed.<channel>
- Multi-channel: cryptofeed.events

Keys
- Suggested: symbol or event_id depending on ordering needs. For per-symbol partitioning, use symbol.

Setup
- Generate Python code from proto via `buf generate` (already configured to output to gen/python).
- Create a value_serializer callable that takes the dict produced by the backend and returns bytes (protobuf-encoded).

Example serializer (Python)

```python path=null start=null
# examples/kafka_protobuf_serializer.py
from decimal import Decimal
from google.protobuf.timestamp_pb2 import Timestamp

# Generated modules (ensure gen/python on PYTHONPATH)
from cryptofeed.v1 import common_pb2 as common
from cryptofeed.v1 import market_data_pb2 as md
from cryptofeed.v1 import events_pb2 as ev

EXCHANGE_ENUM = {name.replace('EXCHANGE_', ''): getattr(common.Exchange, name) for name in common.Exchange.keys()}
SIDE_ENUM = {name.replace('SIDE_', '').lower(): getattr(common.Side, name) for name in common.Side.keys()}


def to_ts(ts_float: float) -> Timestamp:
    ts = Timestamp()
    if ts_float is not None:
        seconds = int(ts_float)
        nanos = int((ts_float - seconds) * 1e9)
        ts.seconds = seconds
        ts.nanos = nanos
    return ts


def build_symbol(symbol_str: str, exchange: str) -> common.Symbol:
    s = common.Symbol(symbol=symbol_str)
    if '-' in symbol_str:
        base, quote = symbol_str.split('-', 1)
        s.base = base
        s.quote = quote
    return s


def build_trade(d: dict) -> md.Trade:
    msg = md.Trade()
    msg.exchange = EXCHANGE_ENUM.get(d['exchange'], common.EXCHANGE_UNSPECIFIED)
    msg.symbol.CopyFrom(build_symbol(d['symbol'], d['exchange']))
    msg.side = SIDE_ENUM.get(d.get('side', ''), common.SIDE_UNSPECIFIED)
    msg.amount.value = str(d['amount'])
    msg.price.value = str(d['price'])
    if d.get('id'):
        msg.id = str(d['id'])
    if d.get('type'):
        msg.type = str(d['type'])
    msg.timestamp.CopyFrom(to_ts(d.get('timestamp')))
    return msg


def to_event(channel: common.DataChannel, d: dict) -> ev.DataFeedEvent:
    e = ev.DataFeedEvent()
    e.channel = channel
    e.exchange = EXCHANGE_ENUM.get(d['exchange'], common.EXCHANGE_UNSPECIFIED)
    e.symbol.CopyFrom(build_symbol(d['symbol'], d['exchange']))
    e.event_timestamp.CopyFrom(to_ts(d.get('timestamp')))
    # You can optionally set receipt_timestamp here
    # set payload
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


def make_value_serializer(channel: common.DataChannel):
    def serializer(d: dict) -> bytes:
        # d is the dict produced by the backend for this channel
        evt = to_event(channel, d)
        return evt.SerializeToString()
    return serializer
```

Usage with Kafka backend

```python path=null start=null
from cryptofeed import FeedHandler
from cryptofeed.backends.kafka import TradeKafka
from cryptofeed.defines import TRADES
from cryptofeed.exchanges import Coinbase

# Ensure PYTHONPATH includes gen/python so generated modules can be imported
import sys, os
sys.path.insert(0, os.path.abspath('gen/python'))

from cryptofeed.v1 import common_pb2 as common
from examples.kafka_protobuf_serializer import make_value_serializer

fh = FeedHandler()
producer_config = {
    'bootstrap_servers': 'localhost:9092',
    'value_serializer': make_value_serializer(common.DATA_CHANNEL_TRADES),
}
fh.add_feed(Coinbase(symbols=['BTC-USD'], channels=[TRADES],
                     callbacks={TRADES: TradeKafka(**producer_config)}))
fh.run()
```

Notes
- The Kafka backend already supports user-provided value_serializer. The example above converts backend dicts into protobuf DataFeedEvent bytes.
- Extended channels: Trade, Ticker, L1Book, L2Book, Funding are supported by builders and oneof assignment.
- Headers: when wrapping in KafkaDataFeedEvent/KafkaRecord, include headers such as `schema.version` and `content.type=application/x-protobuf`.
- For Buf Schema Registry, publish your proto module to BSR and pin versions. Consumers only need the generated code or the .proto with pinned version to decode.
- For TypeScript/Go/Rust producers/consumers, use the generated code in gen/* from buf generate.
