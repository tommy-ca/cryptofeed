"""Mappings between Cryptofeed order book types and normalized protobuf messages."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from typing import Sequence, Tuple

from cryptofeed.defines import ASK, BID

from cryptofeed.normalized.v1 import level2_delta_pb2, order_book_pb2, price_level_pb2

# OrderBook is a cython-backed class; type ignored to avoid mypy dependency on C extension
from cryptofeed.types import OrderBook  # type: ignore

def level2_book_from_order_book(book: OrderBook) -> order_book_pb2.Level2Book:
    """Convert a snapshot OrderBook into a Level2Book protobuf message."""

    message = order_book_pb2.Level2Book()
    message.exchange = book.exchange
    message.symbol = book.symbol
    _extend_price_levels(message.bids, book.book.bids, ascending=False)
    _extend_price_levels(message.asks, book.book.asks, ascending=True)

    timestamp = _seconds_to_micros(book.timestamp)
    if timestamp is not None:
        message.timestamp = timestamp

    if book.sequence_number is not None:
        message.sequence = int(book.sequence_number)

    if book.checksum is not None:
        message.checksum = str(book.checksum)

    return message


def level2_delta_from_order_book(book: OrderBook) -> level2_delta_pb2.Level2Delta:
    """Convert an OrderBook delta into a Level2Delta protobuf message.

    Raises:
        ValueError: if the order book does not have delta information.
    """

    if not book.delta:
        raise ValueError("OrderBook.delta is empty; cannot build Level2Delta message")

    message = level2_delta_pb2.Level2Delta()
    message.exchange = book.exchange
    message.symbol = book.symbol

    bids = book.delta.get(BID, [])
    asks = book.delta.get(ASK, [])
    _extend_price_levels(message.bids, bids, ascending=False)
    _extend_price_levels(message.asks, asks, ascending=True)

    timestamp = _seconds_to_micros(book.timestamp)
    if timestamp is not None:
        message.timestamp = timestamp

    if book.sequence_number is not None:
        message.sequence = int(book.sequence_number)

    if book.checksum is not None:
        message.checksum = str(book.checksum)

    return message


def _extend_price_levels(
    repeated_field,
    levels: Sequence[Tuple],
    *,
    ascending: bool | None = None,
) -> None:
    """Populate repeated PriceLevel fields with normalized decimal strings."""

    if hasattr(levels, "items"):
        iterable = list(levels.items())
    else:
        iterable = list(levels)
        if iterable and not isinstance(iterable[0], (tuple, list)) and hasattr(levels, "__getitem__"):
            iterable = [(price, levels[price]) for price in iterable]

    if ascending is True:
        sorted_levels = sorted(iterable, key=lambda entry: entry[0])
    elif ascending is False:
        sorted_levels = sorted(iterable, key=lambda entry: entry[0], reverse=True)
    else:
        sorted_levels = iterable

    for entry in sorted_levels:
        if len(entry) < 2:
            raise ValueError("Delta entry must contain at least price and size")
        price, size = entry[0], entry[1]
        repeated_field.append(
            price_level_pb2.PriceLevel(
                price=_decimal_to_str(price),
                quantity=_decimal_to_str(size),
            )
        )


def _decimal_to_str(value: Decimal | float | int | str) -> str:
    decimal_value = _coerce_decimal(value)
    scaled = decimal_value.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_EVEN)
    return format(scaled, "f")


def _coerce_decimal(value: Decimal | float | int | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    return Decimal(value)


def _seconds_to_micros(timestamp: float | None) -> int | None:
    if timestamp is None:
        return None
    return int(round(timestamp * 1_000_000))
