'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for OrderInfo data type.
'''
from cryptofeed.proto_bindings import order_info_pb2


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


__all__ = ['order_info_to_proto']
