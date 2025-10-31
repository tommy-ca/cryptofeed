'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for Order data type.
'''
from cryptofeed.proto_bindings import order_pb2


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


__all__ = ['order_to_proto']
