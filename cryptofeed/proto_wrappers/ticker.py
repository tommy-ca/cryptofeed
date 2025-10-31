'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for Ticker data type.
'''
from cryptofeed.proto_bindings import ticker_pb2


def ticker_to_proto(ticker_obj) -> ticker_pb2.Ticker:
    """
    Convert Ticker to protobuf representation.

    Conversions:
    - Decimal (bid, ask) → string (preserves full precision)
    - float seconds (timestamp) → int64 microseconds

    Args:
        ticker_obj: Ticker instance

    Returns:
        ticker_pb2.Ticker: Protobuf message ready for serialization
    """
    proto = ticker_pb2.Ticker()

    # Required fields
    proto.exchange = ticker_obj.exchange or ''
    proto.symbol = ticker_obj.symbol or ''

    # Decimal fields as strings (preserves precision)
    if ticker_obj.bid is not None:
        proto.bid = str(ticker_obj.bid)

    if ticker_obj.ask is not None:
        proto.ask = str(ticker_obj.ask)

    # Optional timestamp: float seconds → int64 microseconds
    if ticker_obj.timestamp is not None:
        proto.timestamp = int(ticker_obj.timestamp * 1_000_000)

    return proto


__all__ = ['ticker_to_proto']
