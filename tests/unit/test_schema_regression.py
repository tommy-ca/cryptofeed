from __future__ import annotations

from decimal import Decimal

import pytest

from tools import schema_regression as sr


def test_decimal_loader_preserves_decimal(tmp_path):
    fixture = tmp_path / "events.jsonl"
    fixture.write_text(
        """
{"type": "trade", "exchange": "TEST", "symbol": "BTC-USD", "side": "buy", "amount": 1.2345, "price": 50000.5, "timestamp": 1234567890.0}
""".strip()
    )

    loader = sr.DecimalLoader()
    events = loader.load(fixture)

    assert isinstance(events[0]["amount"], Decimal)
    assert isinstance(events[0]["price"], Decimal)


def test_factory_adapter_builds_trade():
    adapter = sr.DataclassFactoryAdapter()
    event = {
        "type": "trade",
        "exchange": "TEST",
        "symbol": "BTC-USD",
        "side": "buy",
        "amount": "1.0",
        "price": "100.0",
        "timestamp": 1.0,
        "id": "trade-1",
    }

    trade = adapter.build(event)
    data = trade.to_dict()

    assert data["exchange"] == "TEST"
    assert data["symbol"] == "BTC-USD"
    assert data["amount"] == Decimal("1.0")


def test_factory_adapter_missing_type():
    adapter = sr.DataclassFactoryAdapter()

    with pytest.raises(sr.FactoryNotFound):
        adapter.build({"exchange": "TEST"})
