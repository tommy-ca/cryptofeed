'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for Trade data type.
'''
from cryptofeed.proto_bindings import trade_pb2, trade_side_pb2


def trade_to_proto(trade_obj) -> trade_pb2.Trade:
    """
    Convert Trade to protobuf representation.

    Since Trade is a C extension type and immutable, we cannot monkey-patch.
    Instead, we add this method to each Trade instance dynamically.

    Conversions:
    - Decimal (price, amount) → string (preserves full precision)
    - float seconds (timestamp) → int64 microseconds
    - string (side: 'buy'/'sell') → enum (TRADE_SIDE_BUY/SELL)

    Args:
        trade_obj: Trade instance

    Returns:
        trade_pb2.Trade: Protobuf message ready for serialization
    """
    proto = trade_pb2.Trade()

    # Required fields
    proto.exchange = trade_obj.exchange or ''
    proto.symbol = trade_obj.symbol or ''

    # Side enum conversion
    if trade_obj.side:
        if trade_obj.side.lower() == 'buy':
            proto.side = trade_side_pb2.TRADE_SIDE_BUY
        elif trade_obj.side.lower() == 'sell':
            proto.side = trade_side_pb2.TRADE_SIDE_SELL
        else:
            proto.side = trade_side_pb2.TRADE_SIDE_UNSPECIFIED
    else:
        proto.side = trade_side_pb2.TRADE_SIDE_UNSPECIFIED

    # Optional trade_id (maps to id)
    if trade_obj.id:
        proto.trade_id = str(trade_obj.id)

    # Decimal fields as strings (preserves precision)
    if trade_obj.price is not None:
        proto.price = str(trade_obj.price)

    if trade_obj.amount is not None:
        proto.amount = str(trade_obj.amount)

    # Timestamp: float seconds → int64 microseconds
    if trade_obj.timestamp is not None:
        proto.timestamp = int(trade_obj.timestamp * 1_000_000)

    # Optional raw_id (not commonly used)
    # proto.raw_id can be left empty

    # Optional trade_type (maps to type)
    if hasattr(trade_obj, 'type') and trade_obj.type:
        proto.trade_type = str(trade_obj.type)

    return proto


# Note: Trade is a C extension type and cannot be monkey-patched.
# Users should either:
# 1. Call trade_to_proto(trade_obj) directly
# 2. Use ProtobufSerializer which will call trade_to_proto() internally


# Export for convenience
__all__ = ['trade_to_proto']
