'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for Position data type.
'''
from cryptofeed.proto_bindings import position_pb2


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


__all__ = ['position_to_proto']
