'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for OpenInterest data type.
'''
from cryptofeed.proto_bindings import open_interest_pb2


def open_interest_to_proto(oi_obj) -> open_interest_pb2.OpenInterest:
    """Convert OpenInterest to protobuf representation."""
    proto = open_interest_pb2.OpenInterest()

    proto.exchange = oi_obj.exchange or ''
    proto.symbol = oi_obj.symbol or ''

    if hasattr(oi_obj, 'open_interest') and oi_obj.open_interest is not None:
        proto.open_interest = str(oi_obj.open_interest)

    if oi_obj.timestamp is not None:
        proto.timestamp = int(oi_obj.timestamp * 1_000_000)

    return proto


__all__ = ['open_interest_to_proto']
