'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Consolidated protobuf serialization helpers for all data types.

This module contains all converter functions that were previously
distributed across cryptofeed/proto_wrappers/.

Functions:
    - All 14 converter functions (trade_to_proto, ticker_to_proto, etc.)
    - get_converter(data_type) - Lookup function for converters
'''

from google.protobuf.message import Message

from cryptofeed.exceptions import ProtobufEncodeError, SerializationError
from cryptofeed.proto_bindings import (
    trade_pb2, trade_side_pb2,
    ticker_pb2, candle_pb2, funding_pb2, order_book_pb2,
    liquidation_pb2, open_interest_pb2, index_price_pb2,
    balance_pb2, position_pb2, fill_pb2,
    order_info_pb2, order_pb2, transaction_pb2
)


# =============================================================================
# Market Data Type Converters
# =============================================================================

def trade_to_proto(trade_obj) -> trade_pb2.Trade:
    """
    Convert Trade to protobuf representation.

    Conversions:
    - Decimal (price, amount) → string (preserves full precision)
    - float seconds (timestamp) → int64 microseconds
    - string (side: 'buy'/'sell') → enum (TRADE_SIDE_BUY/SELL)
    """
    proto = trade_pb2.Trade()
    proto.exchange = trade_obj.exchange or ''
    proto.symbol = trade_obj.symbol or ''

    if trade_obj.side:
        if trade_obj.side.lower() == 'buy':
            proto.side = trade_side_pb2.TRADE_SIDE_BUY
        elif trade_obj.side.lower() == 'sell':
            proto.side = trade_side_pb2.TRADE_SIDE_SELL
        else:
            proto.side = trade_side_pb2.TRADE_SIDE_UNSPECIFIED
    else:
        proto.side = trade_side_pb2.TRADE_SIDE_UNSPECIFIED

    if trade_obj.id:
        proto.trade_id = str(trade_obj.id)
    if trade_obj.price is not None:
        proto.price = str(trade_obj.price)
    if trade_obj.amount is not None:
        proto.amount = str(trade_obj.amount)
    if trade_obj.timestamp is not None:
        proto.timestamp = int(trade_obj.timestamp * 1_000_000)
    if hasattr(trade_obj, 'type') and trade_obj.type:
        proto.trade_type = str(trade_obj.type)

    return proto


def ticker_to_proto(ticker_obj) -> ticker_pb2.Ticker:
    """
    Convert Ticker to protobuf representation.

    Conversions:
    - Decimal (bid, ask) → string (preserves full precision)
    - float seconds (timestamp) → int64 microseconds
    """
    proto = ticker_pb2.Ticker()
    proto.exchange = ticker_obj.exchange or ''
    proto.symbol = ticker_obj.symbol or ''

    if ticker_obj.bid is not None:
        proto.bid = str(ticker_obj.bid)
    if ticker_obj.ask is not None:
        proto.ask = str(ticker_obj.ask)
    if ticker_obj.timestamp is not None:
        proto.timestamp = int(ticker_obj.timestamp * 1_000_000)

    return proto


def candle_to_proto(candle_obj) -> candle_pb2.Candle:
    """
    Convert Candle to protobuf representation.

    Conversions:
    - Decimal (open, high, low, close, volume) → string (preserves precision)
    - float seconds (start, stop, timestamp) → int64 microseconds
    """
    proto = candle_pb2.Candle()
    proto.exchange = candle_obj.exchange or ''
    proto.symbol = candle_obj.symbol or ''

    if candle_obj.start is not None:
        proto.start = int(candle_obj.start * 1_000_000)
    if candle_obj.stop is not None:
        proto.end = int(candle_obj.stop * 1_000_000)

    if hasattr(candle_obj, 'interval') and candle_obj.interval:
        proto.interval = str(candle_obj.interval)
    if hasattr(candle_obj, 'trades') and candle_obj.trades is not None:
        proto.trades = int(candle_obj.trades)

    if candle_obj.open is not None:
        proto.open = str(candle_obj.open)
    if candle_obj.close is not None:
        proto.close = str(candle_obj.close)
    if candle_obj.high is not None:
        proto.high = str(candle_obj.high)
    if candle_obj.low is not None:
        proto.low = str(candle_obj.low)
    if candle_obj.volume is not None:
        proto.volume = str(candle_obj.volume)

    if hasattr(candle_obj, 'closed') and candle_obj.closed is not None:
        proto.closed = bool(candle_obj.closed)
    if candle_obj.timestamp is not None:
        proto.timestamp = int(candle_obj.timestamp * 1_000_000)

    return proto


def funding_to_proto(funding_obj) -> funding_pb2.Funding:
    """
    Convert Funding to protobuf representation.

    Conversions:
    - Decimal (mark_price, rate, predicted_rate) → string (preserves precision)
    - float seconds (timestamp, next_funding_time) → int64 microseconds
    """
    proto = funding_pb2.Funding()
    proto.exchange = funding_obj.exchange or ''
    proto.symbol = funding_obj.symbol or ''

    if hasattr(funding_obj, 'mark_price') and funding_obj.mark_price is not None:
        proto.mark_price = str(funding_obj.mark_price)
    if hasattr(funding_obj, 'rate') and funding_obj.rate is not None:
        proto.rate = str(funding_obj.rate)
    if (hasattr(funding_obj, 'predicted_rate') and
            funding_obj.predicted_rate is not None):
        proto.predicted_rate = str(funding_obj.predicted_rate)
    if (hasattr(funding_obj, 'next_funding_time') and
            funding_obj.next_funding_time is not None):
        proto.next_funding_time = int(funding_obj.next_funding_time * 1_000_000)

    if funding_obj.timestamp is not None:
        proto.timestamp = int(funding_obj.timestamp * 1_000_000)

    return proto


def orderbook_to_proto(orderbook_obj) -> order_book_pb2.Level2Book:
    """
    Convert OrderBook to protobuf representation.

    Conversions:
    - Decimal (price, quantity in bids/asks) → string (preserves precision)
    - float seconds (timestamp) → int64 microseconds
    - SortedDict (bids/asks) → repeated PriceLevel
    """
    proto = order_book_pb2.Level2Book()
    proto.exchange = orderbook_obj.exchange or ''
    proto.symbol = orderbook_obj.symbol or ''

    if orderbook_obj.bids:
        for price in orderbook_obj.bids:
            bid_level = proto.bids.add()
            bid_level.price = str(price)
            bid_level.quantity = str(orderbook_obj.bids[price])

    if orderbook_obj.asks:
        for price in orderbook_obj.asks:
            ask_level = proto.asks.add()
            ask_level.price = str(price)
            ask_level.quantity = str(orderbook_obj.asks[price])

    if orderbook_obj.timestamp is not None:
        proto.timestamp = int(orderbook_obj.timestamp * 1_000_000)
    if (hasattr(orderbook_obj, 'sequence_number') and
            orderbook_obj.sequence_number is not None):
        proto.sequence = int(orderbook_obj.sequence_number)
    if hasattr(orderbook_obj, 'checksum') and orderbook_obj.checksum is not None:
        proto.checksum = str(orderbook_obj.checksum)

    return proto


def liquidation_to_proto(liquidation_obj) -> liquidation_pb2.Liquidation:
    """Convert Liquidation to protobuf representation."""
    proto = liquidation_pb2.Liquidation()
    proto.exchange = liquidation_obj.exchange or ''
    proto.symbol = liquidation_obj.symbol or ''

    if hasattr(liquidation_obj, 'side') and liquidation_obj.side:
        if liquidation_obj.side.lower() == 'buy':
            proto.side = trade_side_pb2.TRADE_SIDE_BUY
        elif liquidation_obj.side.lower() == 'sell':
            proto.side = trade_side_pb2.TRADE_SIDE_SELL
        else:
            proto.side = trade_side_pb2.TRADE_SIDE_UNSPECIFIED

    if hasattr(liquidation_obj, 'quantity') and liquidation_obj.quantity is not None:
        proto.quantity = str(liquidation_obj.quantity)
    if hasattr(liquidation_obj, 'price') and liquidation_obj.price is not None:
        proto.price = str(liquidation_obj.price)
    if hasattr(liquidation_obj, 'id') and liquidation_obj.id:
        proto.liquidation_id = str(liquidation_obj.id)
    if hasattr(liquidation_obj, 'status') and liquidation_obj.status:
        proto.status = str(liquidation_obj.status)
    if liquidation_obj.timestamp is not None:
        proto.timestamp = int(liquidation_obj.timestamp * 1_000_000)

    return proto


def open_interest_to_proto(oi_obj) -> open_interest_pb2.OpenInterest:
    """Convert OpenInterest to protobuf representation."""
    proto = open_interest_pb2.OpenInterest()
    proto.exchange = oi_obj.exchange or ''
    proto.symbol = oi_obj.symbol or ''

    if hasattr(oi_obj, 'open_interest') and oi_obj.open_interest is not None:
        proto.open_interest = str(oi_obj.open_interest)
    if oi_obj.timestamp is not None:
        proto.timestamp = int(oi_obj.timestamp * 1_000_000)

    return proto


def index_to_proto(index_obj) -> index_price_pb2.IndexPrice:
    """Convert Index to protobuf representation."""
    proto = index_price_pb2.IndexPrice()
    proto.exchange = index_obj.exchange or ''
    proto.symbol = index_obj.symbol or ''

    if hasattr(index_obj, 'price') and index_obj.price is not None:
        proto.price = str(index_obj.price)
    if index_obj.timestamp is not None:
        proto.timestamp = int(index_obj.timestamp * 1_000_000)

    return proto


# =============================================================================
# Account/Order Data Type Converters
# =============================================================================

def balance_to_proto(balance_obj) -> balance_pb2.Balance:
    """Convert Balance to protobuf representation."""
    proto = balance_pb2.Balance()
    proto.exchange = balance_obj.exchange or ''

    if hasattr(balance_obj, 'currency') and balance_obj.currency:
        proto.currency = str(balance_obj.currency)
    if hasattr(balance_obj, 'balance') and balance_obj.balance is not None:
        proto.balance = str(balance_obj.balance)
    if hasattr(balance_obj, 'reserved') and balance_obj.reserved is not None:
        proto.reserved = str(balance_obj.reserved)

    return proto


def position_to_proto(position_obj) -> position_pb2.Position:
    """Convert Position to protobuf representation."""
    proto = position_pb2.Position()
    proto.exchange = position_obj.exchange or ''
    proto.symbol = position_obj.symbol or ''

    if hasattr(position_obj, 'position') and position_obj.position is not None:
        proto.position = str(position_obj.position)
    if hasattr(position_obj, 'entry_price') and position_obj.entry_price is not None:
        proto.entry_price = str(position_obj.entry_price)
    if hasattr(position_obj, 'side') and position_obj.side:
        proto.side = str(position_obj.side)
    if (hasattr(position_obj, 'unrealised_pnl') and
            position_obj.unrealised_pnl is not None):
        proto.unrealised_pnl = str(position_obj.unrealised_pnl)
    if hasattr(position_obj, 'timestamp') and position_obj.timestamp is not None:
        proto.timestamp = int(position_obj.timestamp * 1_000_000)

    return proto


def fill_to_proto(fill_obj) -> fill_pb2.Fill:
    """Convert Fill to protobuf representation."""
    proto = fill_pb2.Fill()
    proto.exchange = fill_obj.exchange or ''
    proto.symbol = fill_obj.symbol or ''

    if hasattr(fill_obj, 'side') and fill_obj.side:
        if fill_obj.side.lower() == 'buy':
            proto.side = trade_side_pb2.TRADE_SIDE_BUY
        elif fill_obj.side.lower() == 'sell':
            proto.side = trade_side_pb2.TRADE_SIDE_SELL
        else:
            proto.side = trade_side_pb2.TRADE_SIDE_UNSPECIFIED

    if hasattr(fill_obj, 'amount') and fill_obj.amount is not None:
        proto.amount = str(fill_obj.amount)
    if hasattr(fill_obj, 'price') and fill_obj.price is not None:
        proto.price = str(fill_obj.price)
    if hasattr(fill_obj, 'fee') and fill_obj.fee is not None:
        proto.fee = str(fill_obj.fee)
    if hasattr(fill_obj, 'liquidity') and fill_obj.liquidity:
        proto.liquidity = str(fill_obj.liquidity)
    if hasattr(fill_obj, 'id') and fill_obj.id:
        proto.fill_id = str(fill_obj.id)
    if hasattr(fill_obj, 'order_id') and fill_obj.order_id:
        proto.order_id = str(fill_obj.order_id)
    if hasattr(fill_obj, 'type') and fill_obj.type:
        proto.type = str(fill_obj.type)
    if hasattr(fill_obj, 'account') and fill_obj.account:
        proto.account = str(fill_obj.account)
    if fill_obj.timestamp is not None:
        proto.timestamp = int(fill_obj.timestamp * 1_000_000)

    return proto


def order_info_to_proto(order_info_obj) -> order_info_pb2.OrderInfo:
    """Convert OrderInfo to protobuf representation."""
    proto = order_info_pb2.OrderInfo()
    proto.exchange = order_info_obj.exchange or ''
    proto.symbol = order_info_obj.symbol or ''

    if hasattr(order_info_obj, 'id') and order_info_obj.id:
        proto.order_id = str(order_info_obj.id)
    if hasattr(order_info_obj, 'client_order_id') and order_info_obj.client_order_id:
        proto.client_order_id = str(order_info_obj.client_order_id)
    if hasattr(order_info_obj, 'side') and order_info_obj.side:
        proto.side = str(order_info_obj.side)
    if hasattr(order_info_obj, 'status') and order_info_obj.status:
        proto.status = str(order_info_obj.status)
    if hasattr(order_info_obj, 'type') and order_info_obj.type:
        proto.type = str(order_info_obj.type)
    if hasattr(order_info_obj, 'price') and order_info_obj.price is not None:
        proto.price = str(order_info_obj.price)
    if hasattr(order_info_obj, 'amount') and order_info_obj.amount is not None:
        proto.amount = str(order_info_obj.amount)
    if hasattr(order_info_obj, 'remaining') and order_info_obj.remaining is not None:
        proto.remaining = str(order_info_obj.remaining)
    if hasattr(order_info_obj, 'account') and order_info_obj.account:
        proto.account = str(order_info_obj.account)
    if hasattr(order_info_obj, 'timestamp') and order_info_obj.timestamp is not None:
        proto.timestamp = int(order_info_obj.timestamp * 1_000_000)

    return proto


def order_to_proto(order_obj) -> order_pb2.Order:
    """Convert Order to protobuf representation."""
    proto = order_pb2.Order()
    proto.exchange = order_obj.exchange or ''
    proto.symbol = order_obj.symbol or ''

    if hasattr(order_obj, 'client_order_id') and order_obj.client_order_id:
        proto.client_order_id = str(order_obj.client_order_id)
    if hasattr(order_obj, 'side') and order_obj.side:
        proto.side = str(order_obj.side)
    if hasattr(order_obj, 'type') and order_obj.type:
        proto.type = str(order_obj.type)
    if hasattr(order_obj, 'price') and order_obj.price is not None:
        proto.price = str(order_obj.price)
    if hasattr(order_obj, 'amount') and order_obj.amount is not None:
        proto.amount = str(order_obj.amount)
    if hasattr(order_obj, 'account') and order_obj.account:
        proto.account = str(order_obj.account)
    if hasattr(order_obj, 'timestamp') and order_obj.timestamp is not None:
        proto.timestamp = int(order_obj.timestamp * 1_000_000)

    return proto


def transaction_to_proto(transaction_obj) -> transaction_pb2.Transaction:
    """Convert Transaction to protobuf representation."""
    proto = transaction_pb2.Transaction()
    proto.exchange = transaction_obj.exchange or ''

    if hasattr(transaction_obj, 'currency') and transaction_obj.currency:
        proto.currency = str(transaction_obj.currency)
    if hasattr(transaction_obj, 'type') and transaction_obj.type:
        proto.type = str(transaction_obj.type)
    if hasattr(transaction_obj, 'status') and transaction_obj.status:
        proto.status = str(transaction_obj.status)
    if hasattr(transaction_obj, 'amount') and transaction_obj.amount is not None:
        proto.amount = str(transaction_obj.amount)
    if transaction_obj.timestamp is not None:
        proto.timestamp = int(transaction_obj.timestamp * 1_000_000)

    return proto


# =============================================================================
# Converter Registry and Lookup
# =============================================================================

_CONVERTER_MAP = {
    'Trade': trade_to_proto,
    'Ticker': ticker_to_proto,
    'Candle': candle_to_proto,
    'Funding': funding_to_proto,
    'OrderBook': orderbook_to_proto,
    'Liquidation': liquidation_to_proto,
    'OpenInterest': open_interest_to_proto,
    'Index': index_to_proto,
    'Balance': balance_to_proto,
    'Position': position_to_proto,
    'Fill': fill_to_proto,
    'OrderInfo': order_info_to_proto,
    'Order': order_to_proto,
    'Transaction': transaction_to_proto,
}


_SCHEMA_CLASS_MAP = {
    'Trade': trade_pb2.Trade,
    'Ticker': ticker_pb2.Ticker,
    'Candle': candle_pb2.Candle,
    'Funding': funding_pb2.Funding,
    'OrderBook': order_book_pb2.Level2Book,
    'Liquidation': liquidation_pb2.Liquidation,
    'OpenInterest': open_interest_pb2.OpenInterest,
    'Index': index_price_pb2.IndexPrice,
    'Balance': balance_pb2.Balance,
    'Position': position_pb2.Position,
    'Fill': fill_pb2.Fill,
    'OrderInfo': order_info_pb2.OrderInfo,
    'Order': order_pb2.Order,
    'Transaction': transaction_pb2.Transaction,
}


_DEFAULT_SCHEMA_VERSION = "v0.1.0"


def _resolve_schema_name(schema_message: Message | None, type_name: str) -> str | None:
    """Return protobuf schema identifier for diagnostics."""

    if schema_message is not None and hasattr(schema_message, 'DESCRIPTOR'):
        descriptor = schema_message.DESCRIPTOR
        if descriptor is not None:
            return descriptor.full_name

    schema_class = _SCHEMA_CLASS_MAP.get(type_name)
    if schema_class is not None and hasattr(schema_class, 'DESCRIPTOR'):
        descriptor = schema_class.DESCRIPTOR
        if descriptor is not None:
            return descriptor.full_name

    return f"{type_name.lower()}_pb2.{type_name}"


def _ensure_message(instance, type_name: str, context: str) -> Message:
    """Validate converter/to_proto output is a protobuf Message instance."""

    if isinstance(instance, Message):
        return instance

    if hasattr(instance, 'SerializeToString') and callable(getattr(instance, 'SerializeToString')):
        return instance

    raise ProtobufEncodeError(
        f"{context} returned non-protobuf instance; expected protobuf Message",
        data_type=type_name,
        schema_name=_resolve_schema_name(None, type_name),
        schema_version=_DEFAULT_SCHEMA_VERSION,
    )


def get_converter(type_name: str):
    """
    Get the protobuf converter function for a data type.

    Args:
        type_name: The data type class name as string (e.g., 'Trade', 'Ticker')

    Returns:
        Converter function or None if not found

    Example:
        >>> converter = get_converter('Trade')
        >>> proto_msg = converter(trade_obj)
    """
    return _CONVERTER_MAP.get(type_name)


def serialize_to_protobuf(obj):
    """
    Serialize any cryptofeed data object to protobuf.

    Args:
        obj: Any cryptofeed data object (Trade, Ticker, etc.)

    Returns:
        Serialized protobuf message (bytes)

    Raises:
        SerializationError: If no converter exists for the object's type
        ProtobufEncodeError: If conversion or serialization fails
    """
    type_name = type(obj).__name__

    # First, check if the object exposes a to_proto() method (test doubles)
    if hasattr(obj, 'to_proto') and callable(getattr(obj, 'to_proto')):
        try:
            proto_msg = obj.to_proto()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ProtobufEncodeError(
                "to_proto() raised an exception",
                data_type=type_name,
                schema_version=_DEFAULT_SCHEMA_VERSION,
            ) from exc

        proto_msg = _ensure_message(proto_msg, type_name, "to_proto()")

        try:
            return proto_msg.SerializeToString()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ProtobufEncodeError(
                "SerializeToString() failed",
                data_type=type_name,
                schema_name=_resolve_schema_name(proto_msg, type_name),
                schema_version=_DEFAULT_SCHEMA_VERSION,
            ) from exc

    # Otherwise, use the converter lookup
    converter = get_converter(type_name)

    if not converter:
        raise SerializationError(
            "No protobuf converter registered for data type.",
            data_type=type_name,
        )

    try:
        proto_msg = converter(obj)
    except Exception as exc:
        raise ProtobufEncodeError(
            "Converter raised an exception",
            data_type=type_name,
            schema_name=_resolve_schema_name(None, type_name),
            schema_version=_DEFAULT_SCHEMA_VERSION,
        ) from exc

    proto_msg = _ensure_message(proto_msg, type_name, "converter")

    try:
        return proto_msg.SerializeToString()
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ProtobufEncodeError(
            "SerializeToString() failed",
            data_type=type_name,
            schema_name=_resolve_schema_name(proto_msg, type_name),
            schema_version=_DEFAULT_SCHEMA_VERSION,
        ) from exc


__all__ = [
    # Market data converters
    'trade_to_proto',
    'ticker_to_proto',
    'candle_to_proto',
    'funding_to_proto',
    'orderbook_to_proto',
    'liquidation_to_proto',
    'open_interest_to_proto',
    'index_to_proto',
    # Account/Order converters
    'balance_to_proto',
    'position_to_proto',
    'fill_to_proto',
    'order_info_to_proto',
    'order_to_proto',
    'transaction_to_proto',
    # Registry functions
    'get_converter',
    'serialize_to_protobuf',
]
