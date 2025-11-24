"""Protobuf v2 serialization helpers (native numeric types).

These helpers mirror ``cryptofeed.backends.protobuf_helpers`` but target the
normalized v2 schemas that use native numeric types (`double`, `uint64`) and
`google.protobuf.Timestamp`. They are schema-registry friendly and avoid the
string-based decimal encoding used in v1.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Callable, Dict

from google.protobuf import timestamp_pb2

from cryptofeed.exceptions import ProtobufEncodeError, SerializationError

try:
    from gen.python.cryptofeed.normalized.v2 import trade_pb2 as trade_v2_pb2
    from gen.python.cryptofeed.normalized.v2 import ticker_pb2 as ticker_v2_pb2
    from gen.python.cryptofeed.normalized.v2 import order_book_pb2 as order_book_v2_pb2
    from gen.python.cryptofeed.normalized.v2 import candle_pb2 as candle_v2_pb2
except ImportError as exc:  # pragma: no cover - import guarded by tests
    raise ImportError(
        "Missing generated v2 protobuf bindings. Run 'buf generate proto/' first."
    ) from exc


_DEFAULT_SCHEMA_VERSION = "v2"


def _to_timestamp_proto(value: Any) -> timestamp_pb2.Timestamp:
    """Convert float/int/Decimal seconds to Timestamp."""

    proto_ts = timestamp_pb2.Timestamp()
    if value is None:
        return proto_ts

    seconds = int(math.floor(float(value)))
    nanos = int(round((float(value) - seconds) * 1_000_000_000))

    proto_ts.seconds = seconds
    proto_ts.nanos = nanos
    return proto_ts


def _to_double(value: Any) -> float:
    """Lossy conversion that accepts Decimal/str/float/int."""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)  # intentional lossy conversion per v2 spec
    return float(value)


def trade_to_proto_v2(trade_obj) -> trade_v2_pb2.Trade:
    proto = trade_v2_pb2.Trade()

    proto.exchange = getattr(trade_obj, "exchange", "") or ""
    proto.symbol = getattr(trade_obj, "symbol", "") or ""

    side = getattr(trade_obj, "side", None)
    if side:
        if str(side).lower() == "buy":
            proto.side = trade_v2_pb2.Trade.SIDE_BUY
        elif str(side).lower() == "sell":
            proto.side = trade_v2_pb2.Trade.SIDE_SELL
        else:
            proto.side = trade_v2_pb2.Trade.SIDE_UNSPECIFIED

    trade_id = getattr(trade_obj, "id", None) or getattr(trade_obj, "trade_id", None)
    if trade_id is not None:
        proto.trade_id = str(trade_id)

    price = getattr(trade_obj, "price", None)
    amount = getattr(trade_obj, "amount", None)
    proto.price = _to_double(price)
    proto.amount = _to_double(amount)

    timestamp_val = getattr(trade_obj, "timestamp", None)
    proto.timestamp.CopyFrom(_to_timestamp_proto(timestamp_val))

    seq = getattr(trade_obj, "sequence_number", None)
    if seq is not None:
        proto.sequence_number = int(seq)

    # scale is kept at default 0 unless explicitly provided by callers who
    # opt into bytes+scale semantics in the future.
    if hasattr(trade_obj, "scale") and getattr(trade_obj, "scale") is not None:
        proto.scale = int(getattr(trade_obj, "scale"))

    return proto


def ticker_to_proto_v2(ticker_obj) -> ticker_v2_pb2.Ticker:
    proto = ticker_v2_pb2.Ticker()
    proto.exchange = getattr(ticker_obj, "exchange", "") or ""
    proto.symbol = getattr(ticker_obj, "symbol", "") or ""

    proto.best_bid_price = _to_double(getattr(ticker_obj, "bid", None) or getattr(ticker_obj, "best_bid", None))
    proto.best_ask_price = _to_double(getattr(ticker_obj, "ask", None) or getattr(ticker_obj, "best_ask", None))
    proto.best_bid_size = _to_double(getattr(ticker_obj, "bid_size", None) or getattr(ticker_obj, "best_bid_size", None))
    proto.best_ask_size = _to_double(getattr(ticker_obj, "ask_size", None) or getattr(ticker_obj, "best_ask_size", None))

    proto.timestamp.CopyFrom(_to_timestamp_proto(getattr(ticker_obj, "timestamp", None)))

    seq = getattr(ticker_obj, "sequence_number", None)
    if seq is not None:
        proto.sequence_number = int(seq)

    if hasattr(ticker_obj, "scale") and getattr(ticker_obj, "scale") is not None:
        proto.scale = int(getattr(ticker_obj, "scale"))

    return proto


def orderbook_to_proto_v2(book_obj) -> order_book_v2_pb2.OrderBook:
    proto = order_book_v2_pb2.OrderBook()
    proto.exchange = getattr(book_obj, "exchange", "") or ""
    proto.symbol = getattr(book_obj, "symbol", "") or ""

    bids = getattr(book_obj, "bids", None) or {}
    asks = getattr(book_obj, "asks", None) or {}

    for price, qty in getattr(bids, "items", bids.items)():
        level = proto.bids.add()
        level.price = _to_double(price)
        level.quantity = _to_double(qty)

    for price, qty in getattr(asks, "items", asks.items)():
        level = proto.asks.add()
        level.price = _to_double(price)
        level.quantity = _to_double(qty)

    proto.timestamp.CopyFrom(_to_timestamp_proto(getattr(book_obj, "timestamp", None)))

    seq = getattr(book_obj, "sequence_number", None)
    if seq is not None:
        proto.sequence_number = int(seq)

    checksum = getattr(book_obj, "checksum", None)
    if checksum is not None:
        proto.checksum = str(checksum)

    if hasattr(book_obj, "scale") and getattr(book_obj, "scale") is not None:
        proto.scale = int(getattr(book_obj, "scale"))

    return proto


def candle_to_proto_v2(candle_obj) -> candle_v2_pb2.Candle:
    proto = candle_v2_pb2.Candle()
    proto.exchange = getattr(candle_obj, "exchange", "") or ""
    proto.symbol = getattr(candle_obj, "symbol", "") or ""

    proto.start.CopyFrom(_to_timestamp_proto(getattr(candle_obj, "start", getattr(candle_obj, "open_time", None))))
    proto.end.CopyFrom(_to_timestamp_proto(getattr(candle_obj, "end", getattr(candle_obj, "stop", None))))

    interval = getattr(candle_obj, "interval", None)
    if interval is not None:
        proto.interval = str(interval)

    trades = getattr(candle_obj, "trades", None)
    if trades is not None:
        proto.trades = int(trades)

    proto.open = _to_double(getattr(candle_obj, "open", None))
    proto.close = _to_double(getattr(candle_obj, "close", None))
    proto.high = _to_double(getattr(candle_obj, "high", None))
    proto.low = _to_double(getattr(candle_obj, "low", None))
    proto.volume = _to_double(getattr(candle_obj, "volume", None))

    closed = getattr(candle_obj, "closed", None)
    if closed is not None:
        proto.closed = bool(closed)

    proto.timestamp.CopyFrom(_to_timestamp_proto(getattr(candle_obj, "timestamp", None)))

    seq = getattr(candle_obj, "sequence_number", None)
    if seq is not None:
        proto.sequence_number = int(seq)

    if hasattr(candle_obj, "scale") and getattr(candle_obj, "scale") is not None:
        proto.scale = int(getattr(candle_obj, "scale"))

    return proto


_CONVERTER_MAP: Dict[str, Callable[[Any], Any]] = {
    "Trade": trade_to_proto_v2,
    "Ticker": ticker_to_proto_v2,
    "OrderBook": orderbook_to_proto_v2,
    "Candle": candle_to_proto_v2,
    # common alternates in the codebase
    "L2Book": orderbook_to_proto_v2,
}


def get_converter_v2(type_name: str) -> Callable[[Any], Any] | None:
    normalized = type_name.lstrip("_")
    return _CONVERTER_MAP.get(normalized)


def _ensure_message(proto_obj, type_name: str, source: str):
    from google.protobuf.message import Message

    if not isinstance(proto_obj, Message):  # pragma: no cover - defensive guard
        raise ProtobufEncodeError(
            "Converter did not return a protobuf Message",
            data_type=type_name,
            schema_version=_DEFAULT_SCHEMA_VERSION,
            schema_name=source,
        )
    return proto_obj


def serialize_to_protobuf_v2(obj: Any) -> bytes:
    type_name = type(obj).__name__

    converter = get_converter_v2(type_name)
    if not converter:
        raise SerializationError(
            "No protobuf v2 converter registered for data type.",
            data_type=type_name,
        )

    try:
        proto_msg = converter(obj)
    except Exception as exc:
        raise ProtobufEncodeError(
            "Converter raised an exception",
            data_type=type_name,
            schema_name=type_name,
            schema_version=_DEFAULT_SCHEMA_VERSION,
        ) from exc

    proto_msg = _ensure_message(proto_msg, type_name, "converter")

    try:
        return proto_msg.SerializeToString()
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ProtobufEncodeError(
            "SerializeToString() failed",
            data_type=type_name,
            schema_name=type(proto_msg).__name__,
            schema_version=_DEFAULT_SCHEMA_VERSION,
        ) from exc


__all__ = [
    "trade_to_proto_v2",
    "ticker_to_proto_v2",
    "orderbook_to_proto_v2",
    "candle_to_proto_v2",
    "get_converter_v2",
    "serialize_to_protobuf_v2",
]
