'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for Candle data type.
'''
from cryptofeed.proto_bindings import candle_pb2


def candle_to_proto(candle_obj) -> candle_pb2.Candle:
    """
    Convert Candle to protobuf representation.

    Conversions:
    - Decimal (open, high, low, close, volume) → string (preserves precision)
    - float seconds (start, stop, timestamp) → int64 microseconds

    Args:
        candle_obj: Candle instance

    Returns:
        candle_pb2.Candle: Protobuf message ready for serialization
    """
    proto = candle_pb2.Candle()

    # Required fields
    proto.exchange = candle_obj.exchange or ''
    proto.symbol = candle_obj.symbol or ''

    # Timestamps: float seconds → int64 microseconds
    if candle_obj.start is not None:
        proto.start = int(candle_obj.start * 1_000_000)

    if candle_obj.stop is not None:
        proto.end = int(candle_obj.stop * 1_000_000)

    # Interval string
    if hasattr(candle_obj, 'interval') and candle_obj.interval:
        proto.interval = str(candle_obj.interval)

    # Optional trades count
    if hasattr(candle_obj, 'trades') and candle_obj.trades is not None:
        proto.trades = int(candle_obj.trades)

    # OHLCV as strings (preserves Decimal precision)
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

    # Closed flag
    if hasattr(candle_obj, 'closed') and candle_obj.closed is not None:
        proto.closed = bool(candle_obj.closed)

    # Optional timestamp
    if candle_obj.timestamp is not None:
        proto.timestamp = int(candle_obj.timestamp * 1_000_000)

    return proto


__all__ = ['candle_to_proto']
