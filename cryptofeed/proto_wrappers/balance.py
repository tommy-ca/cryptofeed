'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for Balance data type.
'''
from cryptofeed.proto_bindings import balance_pb2


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


__all__ = ['balance_to_proto']
