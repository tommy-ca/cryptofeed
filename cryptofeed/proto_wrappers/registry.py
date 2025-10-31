'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Registry for C extension type to_proto() converters.

Since C extension types cannot be monkey-patched, we maintain a registry
of conversion functions that ProtobufSerializer uses.
'''
from typing import Callable, Dict, Any
from cryptofeed.types import Trade


# Registry mapping type to conversion function
_PROTO_CONVERTERS: Dict[type, Callable] = {}


def register_proto_converter(data_type: type, converter_func: Callable):
    """
    Register a protobuf converter function for a data type.

    Args:
        data_type: The data type class (e.g., Trade, OrderBook)
        converter_func: Function that converts instance to protobuf
    """
    _PROTO_CONVERTERS[data_type] = converter_func


def get_proto_converter(data_type: type) -> Callable | None:
    """
    Get the protobuf converter function for a data type.

    Args:
        data_type: The data type class

    Returns:
        Converter function or None if not registered
    """
    return _PROTO_CONVERTERS.get(data_type)


def convert_to_proto(obj: Any):
    """
    Convert an object to protobuf using registered converters.

    First tries obj.to_proto() if available (for pure Python types).
    Falls back to registered converter for C extension types.

    Args:
        obj: Data object to convert

    Returns:
        Protobuf message

    Raises:
        AttributeError: If no converter found
    """
    # Try direct to_proto() method first
    if hasattr(obj, 'to_proto') and callable(getattr(obj, 'to_proto')):
        return obj.to_proto()

    # Fall back to registered converter
    converter = get_proto_converter(type(obj))
    if converter:
        return converter(obj)

    raise AttributeError(
        f"{type(obj).__name__} has no to_proto() method and no registered converter"
    )


# Register all converters on module import
def _register_converters():
    """Register all data type converters."""
    from cryptofeed.types import (
        Ticker, Candle, Funding, OrderBook,
        Liquidation, OpenInterest, Index,
        Balance, Position, Fill, OrderInfo, Order, Transaction
    )
    from cryptofeed.proto_wrappers.trade import trade_to_proto
    from cryptofeed.proto_wrappers.ticker import ticker_to_proto
    from cryptofeed.proto_wrappers.candle import candle_to_proto
    from cryptofeed.proto_wrappers.funding import funding_to_proto
    from cryptofeed.proto_wrappers.orderbook import orderbook_to_proto
    from cryptofeed.proto_wrappers.liquidation import liquidation_to_proto
    from cryptofeed.proto_wrappers.open_interest import open_interest_to_proto
    from cryptofeed.proto_wrappers.index import index_to_proto
    from cryptofeed.proto_wrappers.balance import balance_to_proto
    from cryptofeed.proto_wrappers.position import position_to_proto
    from cryptofeed.proto_wrappers.fill import fill_to_proto
    from cryptofeed.proto_wrappers.order_info import order_info_to_proto
    from cryptofeed.proto_wrappers.order import order_to_proto
    from cryptofeed.proto_wrappers.transaction import transaction_to_proto

    # Market data types
    register_proto_converter(Trade, trade_to_proto)
    register_proto_converter(Ticker, ticker_to_proto)
    register_proto_converter(Candle, candle_to_proto)
    register_proto_converter(Funding, funding_to_proto)
    register_proto_converter(OrderBook, orderbook_to_proto)
    register_proto_converter(Liquidation, liquidation_to_proto)
    register_proto_converter(OpenInterest, open_interest_to_proto)
    register_proto_converter(Index, index_to_proto)

    # Account/Order data types
    register_proto_converter(Balance, balance_to_proto)
    register_proto_converter(Position, position_to_proto)
    register_proto_converter(Fill, fill_to_proto)
    register_proto_converter(OrderInfo, order_info_to_proto)
    register_proto_converter(Order, order_to_proto)
    register_proto_converter(Transaction, transaction_to_proto)


_register_converters()
