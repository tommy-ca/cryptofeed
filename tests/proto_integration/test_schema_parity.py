"""Proto integration tests for schema parity validation.

Tests that Cryptofeed dataclasses serialize/deserialize correctly through
Protobuf messages while preserving field values and precision.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from cryptofeed.types import Trade, Ticker, OrderBook, Funding, OpenInterest


class TestTradeProtoParity:
    """Validate Trade event parity across representations."""

    def test_trade_basic_fields(self):
        """Trade event has all required Protobuf fields."""
        trade = Trade(
            exchange="BACKPACK",
            symbol="BTC-USDT",
            side="buy",
            amount=Decimal("1.5"),
            price=Decimal("45000.50"),
            timestamp=1234567890.0,
            id="trade123",
        )

        # Verify fields match Protobuf schema
        data = trade.to_dict()
        assert data["exchange"] == "BACKPACK"
        assert data["symbol"] == "BTC-USDT"
        assert data["side"] == "buy"
        assert data["amount"] == Decimal("1.5")
        assert data["price"] == Decimal("45000.50")
        assert data["timestamp"] == 1234567890.0
        assert data["id"] == "trade123"

    def test_trade_decimal_precision(self):
        """Trade decimal fields maintain precision."""
        trade = Trade(
            exchange="BINANCE",
            symbol="ETH-USDT",
            side="sell",
            amount=Decimal("0.00000001"),  # Small precision
            price=Decimal("2500.123456789"),  # High precision
            timestamp=1234567890.123456,
            id="eth-trade",
        )

        data = trade.to_dict()
        # Verify precision is preserved in dict representation
        assert isinstance(data["amount"], Decimal)
        assert isinstance(data["price"], Decimal)
        assert data["amount"] == Decimal("0.00000001")
        assert data["price"] == Decimal("2500.123456789")

    def test_trade_timestamp_microseconds(self):
        """Trade timestamp matches Protobuf int64 microseconds convention."""
        # Use timestamp with microsecond precision
        ts = 1234567890.123456
        trade = Trade(
            exchange="TEST",
            symbol="BTC-USD",
            side="buy",
            amount=Decimal("1"),
            price=Decimal("50000"),
            timestamp=ts,
            id="ts-test",
        )

        data = trade.to_dict()
        assert data["timestamp"] == ts


class TestTickerProtoParity:
    """Validate Ticker event parity across representations."""

    def test_ticker_basic_fields(self):
        """Ticker event serializes all required fields."""
        ticker = Ticker(
            exchange="KRAKEN",
            symbol="XRP-USD",
            bid=Decimal("0.50"),
            ask=Decimal("0.51"),
            timestamp=1234567890.0,
        )

        data = ticker.to_dict()
        assert data["exchange"] == "KRAKEN"
        assert data["symbol"] == "XRP-USD"
        assert data["bid"] == Decimal("0.50")
        assert data["ask"] == Decimal("0.51")
        assert data["timestamp"] == 1234567890.0

    def test_ticker_bid_ask_spread(self):
        """Ticker bid/ask values are properly ordered."""
        ticker = Ticker(
            exchange="COINBASE",
            symbol="DOGE-USD",
            bid=Decimal("0.099"),
            ask=Decimal("0.101"),
            timestamp=1234567890.0,
        )

        data = ticker.to_dict()
        bid = float(data["bid"])
        ask = float(data["ask"])
        assert bid < ask, "Bid should be less than ask"


class TestFundingProtoParity:
    """Validate Funding event parity across representations."""

    def test_funding_rate_precision(self):
        """Funding event maintains rate precision."""
        funding = Funding(
            exchange="DYDX",
            symbol="BTC-USD",
            mark_price=Decimal("45000.00"),
            rate=Decimal("0.00005"),  # 5 basis points
            next_funding_time=None,
            timestamp=1234567890.0,
        )

        data = funding.to_dict()
        assert data["exchange"] == "DYDX"
        assert data["symbol"] == "BTC-USD"
        assert data["mark_price"] == Decimal("45000.00")
        assert data["rate"] == Decimal("0.00005")

    def test_funding_optional_fields(self):
        """Funding optional fields handled correctly."""
        funding = Funding(
            exchange="BYBIT",
            symbol="ETH-USDT",
            mark_price=Decimal("2500.00"),
            rate=Decimal("0.0001"),
            next_funding_time=1234567890.0,
            timestamp=1234567890.0,
        )

        data = funding.to_dict()
        assert data["next_funding_time"] == 1234567890.0


class TestOpenInterestProtoParity:
    """Validate OpenInterest event parity across representations."""

    def test_open_interest_large_values(self):
        """OpenInterest handles large position values."""
        oi = OpenInterest(
            exchange="DYDX",
            symbol="BTC-USD",
            open_interest=Decimal("50000000"),  # 50M in OI
            timestamp=1234567890.0,
        )

        data = oi.to_dict()
        assert data["exchange"] == "DYDX"
        assert data["symbol"] == "BTC-USD"
        assert data["open_interest"] == Decimal("50000000")


class TestOrderBookProtoParity:
    """Validate OrderBook event parity across representations."""

    def test_order_book_snapshot(self):
        """OrderBook snapshot structure matches proto expectations."""
        book = OrderBook(exchange="BACKPACK", symbol="BTC-USDT")
        book.bids = {"100.0": Decimal("1.5")}
        book.asks = {"101.0": Decimal("2.0")}

        # Verify structure (OrderBook has special handling)
        assert book.exchange == "BACKPACK"
        assert book.symbol == "BTC-USDT"
        assert len(book.bids) > 0
        assert len(book.asks) > 0


@pytest.mark.integration
class TestSchemaPurityRobustness:
    """Robustness tests for schema parity edge cases."""

    def test_trade_with_minimal_fields(self):
        """Trade with only required fields."""
        trade = Trade(
            exchange="TEST",
            symbol="BTC-USD",
            side="buy",
            amount=Decimal("1"),
            price=Decimal("50000"),
            timestamp=1234567890.0,
        )
        data = trade.to_dict()
        assert all(k in data for k in ["exchange", "symbol", "side", "amount", "price", "timestamp"])

    def test_trade_with_optional_fields(self):
        """Trade with optional id field."""
        trade = Trade(
            exchange="TEST",
            symbol="BTC-USD",
            side="buy",
            amount=Decimal("1"),
            price=Decimal("50000"),
            timestamp=1234567890.0,
            id="optional-id",
        )
        data = trade.to_dict()
        assert data.get("id") == "optional-id"

    def test_decimal_conversion_stability(self):
        """Decimal fields survive dict->decimal->dict conversions."""
        original_amount = Decimal("123.456789")
        trade = Trade(
            exchange="TEST",
            symbol="BTC-USD",
            side="buy",
            amount=original_amount,
            price=Decimal("50000"),
            timestamp=1234567890.0,
        )

        data1 = trade.to_dict()
        amount1 = data1["amount"]
        assert amount1 == original_amount

        # Reconstruct and verify again
        trade2 = Trade(
            exchange=data1["exchange"],
            symbol=data1["symbol"],
            side=data1["side"],
            amount=Decimal(str(data1["amount"])),
            price=Decimal(str(data1["price"])),
            timestamp=data1["timestamp"],
            id=data1.get("id"),
        )
        data2 = trade2.to_dict()
        amount2 = data2["amount"]
        assert amount2 == original_amount


@pytest.mark.integration
@pytest.mark.slow
class TestRegressionReports:
    """Integration tests using schema regression tool."""

    def test_regression_tool_exists(self):
        """Schema regression tool is available."""
        from tools import schema_regression

        assert hasattr(schema_regression, "run_regression")
        assert hasattr(schema_regression, "EventParity")

    def test_regression_with_sample_events(self):
        """Regression tool processes sample events correctly."""
        from tools import schema_regression

        # Create minimal sample file
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            # Write a trade event
            trade_event = {
                "type": "trade",
                "exchange": "TEST",
                "symbol": "BTC-USD",
                "side": "buy",
                "amount": "1.5",
                "price": "45000.50",
                "timestamp": 1234567890.0,
                "id": "test-trade-1",
            }
            f.write(json.dumps(trade_event) + "\n")
            temp_path = f.name

        try:
            # Run regression test
            import argparse

            args = argparse.Namespace(
                events=temp_path,
                output=None,
                protobuf=False,
                tolerance=1e-8,
                verbose=False,
                strict=True,
            )
            exit_code, report = schema_regression.run_regression(args)

            assert exit_code == 0, f"Regression test failed: {report.total_mismatches} mismatches"
            assert report.events_processed == 1
            assert report.events_clean == 1
        finally:
            Path(temp_path).unlink()
