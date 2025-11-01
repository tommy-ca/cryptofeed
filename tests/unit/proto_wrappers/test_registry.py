import pytest

from cryptofeed.proto_wrappers import registry
from cryptofeed.types import (
    Balance,
    Candle,
    Fill,
    Funding,
    Index,
    Liquidation,
    OpenInterest,
    Order,
    OrderBook,
    OrderInfo,
    Position,
    Ticker,
    Trade,
    Transaction,
)


EXPECTED_CONVERTERS = {
    Balance,
    Candle,
    Fill,
    Funding,
    Index,
    Liquidation,
    OpenInterest,
    Order,
    OrderBook,
    OrderInfo,
    Position,
    Ticker,
    Trade,
    Transaction,
}


def test_all_expected_converters_present():
    registered = set(registry._PROTO_CONVERTERS.keys())
    assert EXPECTED_CONVERTERS.issubset(registered)


def test_missing_converter_raises_serialization_error():
    class CustomType:
        pass

    with pytest.raises(AttributeError):
        registry.convert_to_proto(CustomType())
