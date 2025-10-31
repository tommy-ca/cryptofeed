'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for Fill data type.
'''
from cryptofeed.proto_bindings import fill_pb2, trade_side_pb2


def fill_to_proto(fill_obj) -> fill_pb2.Fill:
    """Convert Fill to protobuf representation."""
    proto = fill_pb2.Fill()

    proto.exchange = fill_obj.exchange or ''
    proto.symbol = fill_obj.symbol or ''

    # Side enum
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


__all__ = ['fill_to_proto']
