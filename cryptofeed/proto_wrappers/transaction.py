'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for Transaction data type.
'''
from cryptofeed.proto_bindings import transaction_pb2


def transaction_to_proto(transaction_obj) -> transaction_pb2.Transaction:
    """Convert Transaction to protobuf representation."""
    proto = transaction_pb2.Transaction()

    proto.exchange = transaction_obj.exchange or ''

    if hasattr(transaction_obj, 'currency') and transaction_obj.currency:
        proto.currency = str(transaction_obj.currency)

    if hasattr(transaction_obj, 'type') and transaction_obj.type:
        proto.type = str(transaction_obj.type)

    if hasattr(transaction_obj, 'status') and transaction_obj.status:
        proto.status = str(transaction_obj.status)

    if hasattr(transaction_obj, 'amount') and transaction_obj.amount is not None:
        proto.amount = str(transaction_obj.amount)

    if transaction_obj.timestamp is not None:
        proto.timestamp = int(transaction_obj.timestamp * 1_000_000)

    return proto


__all__ = ['transaction_to_proto']
