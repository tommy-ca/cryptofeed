from decimal import Decimal

from cryptofeed.backends.protobuf.converters import orderbook_to_proto


class StubOrderBook:
    def __init__(
        self,
        exchange: str,
        symbol: str,
        bids,
        asks,
        timestamp=None,
        sequence_number=None,
        checksum=None,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.bids = bids
        self.asks = asks
        self.timestamp = timestamp
        self.sequence_number = sequence_number
        self.checksum = checksum


def test_orderbook_to_proto_formats_levels_and_metadata():
    book = StubOrderBook(
        exchange="BINANCE",
        symbol="BTC-USDT",
        bids={Decimal("50000"): Decimal("1.50")},
        asks={Decimal("50010"): Decimal("0.75")},
        timestamp=1696160000.123456,
        sequence_number=42,
        checksum="123456",
    )

    message = orderbook_to_proto(book)

    assert message.exchange == "BINANCE"
    assert message.symbol == "BTC-USDT"
    assert message.bids[0].price == "50000"
    assert message.bids[0].quantity == "1.50"
    assert message.asks[0].price == "50010"
    assert message.asks[0].quantity == "0.75"
    assert message.timestamp == 1_696_160_000_123_456
    assert message.sequence == 42
    assert message.checksum == "123456"


def test_orderbook_to_proto_handles_missing_optional_fields():
    book = StubOrderBook(
        exchange="COINBASE",
        symbol="ETH-USD",
        bids={Decimal("1850.1"): Decimal("0.25")},
        asks={Decimal("1850.5"): Decimal("0.75")},
    )

    message = orderbook_to_proto(book)

    assert message.exchange == "COINBASE"
    assert message.symbol == "ETH-USD"
    assert message.bids[0].price == "1850.1"
    assert message.bids[0].quantity == "0.25"
    assert message.asks[0].price == "1850.5"
    assert message.asks[0].quantity == "0.75"
    # Optional fields should remain unset/default when not provided
    assert message.timestamp == 0
    assert message.sequence == 0
    assert message.checksum == ""
