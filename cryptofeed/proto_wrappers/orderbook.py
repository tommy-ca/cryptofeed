'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf wrapper for OrderBook data type.
'''
from cryptofeed.proto_bindings import order_book_pb2


def orderbook_to_proto(orderbook_obj) -> order_book_pb2.Level2Book:
    """
    Convert OrderBook to protobuf representation.

    Conversions:
    - Decimal (price, quantity in bids/asks) → string (preserves precision)
    - float seconds (timestamp) → int64 microseconds
    - SortedDict (bids/asks) → repeated PriceLevel

    Args:
        orderbook_obj: OrderBook instance

    Returns:
        order_book_pb2.Level2Book: Protobuf message ready for serialization
    """
    proto = order_book_pb2.Level2Book()

    # Required fields
    proto.exchange = orderbook_obj.exchange or ''
    proto.symbol = orderbook_obj.symbol or ''

    # Convert bids (SortedDict) to repeated PriceLevel
    # SortedDict can be iterated directly for keys, access values via indexing
    if orderbook_obj.bids:
        for price in orderbook_obj.bids:
            bid_level = proto.bids.add()
            bid_level.price = str(price)
            bid_level.quantity = str(orderbook_obj.bids[price])

    # Convert asks (SortedDict) to repeated PriceLevel
    if orderbook_obj.asks:
        for price in orderbook_obj.asks:
            ask_level = proto.asks.add()
            ask_level.price = str(price)
            ask_level.quantity = str(orderbook_obj.asks[price])

    # Optional timestamp: float seconds → int64 microseconds
    if orderbook_obj.timestamp is not None:
        proto.timestamp = int(orderbook_obj.timestamp * 1_000_000)

    # Optional sequence number
    if (hasattr(orderbook_obj, 'sequence_number') and
            orderbook_obj.sequence_number is not None):
        proto.sequence = int(orderbook_obj.sequence_number)

    # Optional checksum
    if hasattr(orderbook_obj, 'checksum') and orderbook_obj.checksum is not None:
        proto.checksum = str(orderbook_obj.checksum)

    return proto


__all__ = ['orderbook_to_proto']
