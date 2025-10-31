'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for Funding data type.
'''
from cryptofeed.proto_bindings import funding_pb2


def funding_to_proto(funding_obj) -> funding_pb2.Funding:
    """
    Convert Funding to protobuf representation.

    Conversions:
    - Decimal (mark_price, rate, predicted_rate) → string (preserves precision)
    - float seconds (timestamp, next_funding_time) → int64 microseconds

    Args:
        funding_obj: Funding instance

    Returns:
        funding_pb2.Funding: Protobuf message ready for serialization
    """
    proto = funding_pb2.Funding()

    # Required fields
    proto.exchange = funding_obj.exchange or ''
    proto.symbol = funding_obj.symbol or ''

    # Optional mark_price
    if hasattr(funding_obj, 'mark_price') and funding_obj.mark_price is not None:
        proto.mark_price = str(funding_obj.mark_price)

    # Optional rate
    if hasattr(funding_obj, 'rate') and funding_obj.rate is not None:
        proto.rate = str(funding_obj.rate)

    # Optional predicted_rate
    if (hasattr(funding_obj, 'predicted_rate') and
            funding_obj.predicted_rate is not None):
        proto.predicted_rate = str(funding_obj.predicted_rate)

    # Optional next_funding_time: float seconds → int64 microseconds
    if (hasattr(funding_obj, 'next_funding_time') and
            funding_obj.next_funding_time is not None):
        proto.next_funding_time = int(funding_obj.next_funding_time * 1_000_000)

    # Required timestamp: float seconds → int64 microseconds
    if funding_obj.timestamp is not None:
        proto.timestamp = int(funding_obj.timestamp * 1_000_000)

    return proto


__all__ = ['funding_to_proto']
