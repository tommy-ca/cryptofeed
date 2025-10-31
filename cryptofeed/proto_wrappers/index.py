'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for Index data type.
'''
from cryptofeed.proto_bindings import index_price_pb2


def index_to_proto(index_obj) -> index_price_pb2.IndexPrice:
    """Convert Index to protobuf representation."""
    proto = index_price_pb2.IndexPrice()

    proto.exchange = index_obj.exchange or ''
    proto.symbol = index_obj.symbol or ''

    if hasattr(index_obj, 'price') and index_obj.price is not None:
        proto.price = str(index_obj.price)

    if index_obj.timestamp is not None:
        proto.timestamp = int(index_obj.timestamp * 1_000_000)

    return proto


__all__ = ['index_to_proto']
