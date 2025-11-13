from decimal import Decimal

import pytest

from cryptofeed.defines import ASK, BID
from cryptofeed.proto_mappers import (
    level2_book_from_order_book,
    level2_delta_from_order_book,
)
from cryptofeed.types import OrderBook  # type: ignore


def test_level2_book_from_order_book_populates_snapshot_fields():
    book = OrderBook("BINANCE", "BTC-USDT")
    book.bids = {Decimal("50000"): Decimal("1.50")}
    book.asks = {Decimal("50010"): Decimal("0.75")}
    book.timestamp = 1696160000.123456
    book.sequence_number = 42
    book.checksum = 123456

    message = level2_book_from_order_book(book)

    assert message.exchange == "BINANCE"
    assert message.symbol == "BTC-USDT"
    assert message.bids[0].price == "50000.00000000"
    assert message.bids[0].quantity == "1.50000000"
    assert message.asks[0].price == "50010.00000000"
    assert message.asks[0].quantity == "0.75000000"
    assert message.HasField("timestamp")
    assert message.timestamp == 1_696_160_000_123_456
    assert message.sequence == 42
    assert message.checksum == "123456"


def test_level2_delta_from_order_book_sorts_and_formats_levels():
    book = OrderBook("BITSTAMP", "ETH-USD")
    book.timestamp = 1696161000.000001
    book.sequence_number = 7
    book.checksum = "abcd"
    book.delta = {
        BID: [
            (Decimal("1850.1"), Decimal("0.25")),
            (Decimal("1850.3"), Decimal("1.10")),
        ],
        ASK: [
            (Decimal("1850.5"), Decimal("0.75")),
            (Decimal("1850.4"), Decimal("0.10")),
        ],
    }

    message = level2_delta_from_order_book(book)

    assert message.exchange == "BITSTAMP"
    assert message.symbol == "ETH-USD"
    # bids should be sorted descending by price
    assert [level.price for level in message.bids] == [
        "1850.30000000",
        "1850.10000000",
    ]
    assert [level.quantity for level in message.bids] == [
        "1.10000000",
        "0.25000000",
    ]
    # asks should be sorted ascending by price
    assert [level.price for level in message.asks] == [
        "1850.40000000",
        "1850.50000000",
    ]
    assert message.HasField("timestamp")
    assert message.timestamp == 1_696_161_000_000_001
    assert message.sequence == 7
    assert message.checksum == "abcd"


def test_level2_delta_requires_delta_entries():
    book = OrderBook("COINBASE", "SOL-USD")
    with pytest.raises(ValueError):
        level2_delta_from_order_book(book)


def test_level2_delta_rejects_malformed_entries():
    book = OrderBook("COINBASE", "SOL-USD")
    book.delta = {BID: [(Decimal("100"),)]}

    with pytest.raises(ValueError):
        level2_delta_from_order_book(book)


def test_level2_delta_leaves_optional_fields_unset_when_missing():
    book = OrderBook("DERIBIT", "BTC-PERP")
    book.delta = {
        BID: [(Decimal("40000"), Decimal("2"))],
        ASK: [(Decimal("40010"), Decimal("1.5"))],
    }

    message = level2_delta_from_order_book(book)

    assert not message.HasField("timestamp")
    assert not message.HasField("sequence")
    assert not message.HasField("checksum")
