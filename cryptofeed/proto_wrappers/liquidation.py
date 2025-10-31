'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for Liquidation data type.
'''
from cryptofeed.proto_bindings import liquidation_pb2, trade_side_pb2


def liquidation_to_proto(liquidation_obj) -> liquidation_pb2.Liquidation:
    """Convert Liquidation to protobuf representation."""
    proto = liquidation_pb2.Liquidation()

    proto.exchange = liquidation_obj.exchange or ''
    proto.symbol = liquidation_obj.symbol or ''

    # Side enum
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


__all__ = ['liquidation_to_proto']
