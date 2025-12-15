"""
Integration tests verifying normalization consistency across all Kafka backend modules.

Tests ensure that topic_manager, partitioner, and headers all use the same shared
normalization functions and produce consistent normalized values.
"""

import pytest
from decimal import Decimal

from cryptofeed.backends.kafka.normalization import normalize_exchange, normalize_symbol
from cryptofeed.backends.kafka.topic_manager import TopicManager
from cryptofeed.backends.kafka.callback import (
    SymbolPartitioner,
    CompositePartitioner,
    ExchangePartitioner,
    MessageHeaders,
)
from cryptofeed.types import Trade


class TestNormalizationConsistency:
    """Verify normalization consistency across all usage sites."""

    @pytest.fixture
    def sample_trade(self):
        """Create sample trade for testing."""
        return Trade(
            exchange="Binance",
            symbol="BTC/USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=123.456,
            id="12345",
        )

    def test_symbol_normalization_consistency(self, sample_trade):
        """Verify symbol normalization produces identical output across all modules."""
        # Extract raw symbol
        raw_symbol = sample_trade.symbol

        # Normalize using shared function
        normalized = normalize_symbol(raw_symbol)

        # Verify topic_manager uses same normalization
        topic = TopicManager.get_topic(
            data_type="trade",
            symbol=raw_symbol,
            exchange="binance",
            strategy="per_symbol",
        )
        assert normalized in topic
        assert "btc-usd" in topic

        # Verify partitioner uses same normalization
        partitioner = SymbolPartitioner()
        partition_key = partitioner.get_partition_key(sample_trade)
        assert partition_key == normalized.encode("utf-8")

        # Verify headers use same normalization
        headers = MessageHeaders.build(
            message=sample_trade,
            data_type="trades",
            content_type="application/x-protobuf",
        )
        symbol_header = next(h for h in headers if h[0] == b"symbol")
        assert symbol_header[1] == normalized.encode("utf-8")

    def test_exchange_normalization_consistency(self, sample_trade):
        """Verify exchange normalization produces identical output across all modules."""
        # Extract raw exchange
        raw_exchange = sample_trade.exchange

        # Normalize using shared function
        normalized = normalize_exchange(raw_exchange)

        # Verify topic_manager uses same normalization
        topic = TopicManager.get_topic(
            data_type="trade",
            symbol="btc-usd",
            exchange=raw_exchange,
            strategy="per_symbol",
        )
        assert normalized in topic
        assert "binance" in topic

        # Verify partitioner uses same normalization
        partitioner = ExchangePartitioner()
        partition_key = partitioner.get_partition_key(sample_trade)
        assert partition_key == normalized.encode("utf-8")

        # Verify composite partitioner uses same normalization
        composite = CompositePartitioner()
        composite_key = composite.get_partition_key(sample_trade)
        assert normalized.encode("utf-8") in composite_key

        # Verify headers use same normalization
        headers = MessageHeaders.build(
            message=sample_trade,
            data_type="trades",
            content_type="application/x-protobuf",
        )
        exchange_header = next(h for h in headers if h[0] == b"exchange")
        assert exchange_header[1] == normalized.encode("utf-8")

    def test_full_consistency_across_modules(self):
        """Verify same input produces same normalized output in all contexts."""
        test_symbols = ["BTC/USD", "ETH_USDT", " SOL-PERP ", None, ""]
        test_exchanges = ["Binance", " OKX ", "COINBASE", None, ""]

        for symbol in test_symbols:
            # Skip None/empty for modules that require values
            if not symbol or not symbol.strip():
                continue

            # Create trade with this symbol
            trade = Trade(
                exchange="binance",
                symbol=symbol,
                side="buy",
                amount=Decimal("1.0"),
                price=Decimal("50000"),
                timestamp=123.456,
            )

            # Get normalized values from all sources
            norm_symbol = normalize_symbol(symbol)

            # Topic normalization
            topic = TopicManager.get_topic(
                "trade", symbol, "binance", "per_symbol"
            )
            assert norm_symbol in topic

            # Partition normalization
            partitioner = SymbolPartitioner()
            partition_key = partitioner.get_partition_key(trade)
            assert partition_key == norm_symbol.encode("utf-8")

            # Header normalization
            headers = MessageHeaders.build(trade, "trades", "application/x-protobuf")
            symbol_header = next(h for h in headers if h[0] == b"symbol")
            assert symbol_header[1] == norm_symbol.encode("utf-8")

        for exchange in test_exchanges:
            # Skip None/empty for modules that require values
            if not exchange or not exchange.strip():
                continue

            # Create trade with this exchange
            trade = Trade(
                exchange=exchange,
                symbol="btc-usd",
                side="buy",
                amount=Decimal("1.0"),
                price=Decimal("50000"),
                timestamp=123.456,
            )

            # Get normalized values from all sources
            norm_exchange = normalize_exchange(exchange)

            # Topic normalization
            topic = TopicManager.get_topic(
                "trade", "btc-usd", exchange, "per_symbol"
            )
            assert norm_exchange in topic

            # Partition normalization
            partitioner = ExchangePartitioner()
            partition_key = partitioner.get_partition_key(trade)
            assert partition_key == norm_exchange.encode("utf-8")

            # Header normalization
            headers = MessageHeaders.build(trade, "trades", "application/x-protobuf")
            exchange_header = next(h for h in headers if h[0] == b"exchange")
            assert exchange_header[1] == norm_exchange.encode("utf-8")

    def test_composite_partitioner_consistency(self, sample_trade):
        """Verify composite partitioner uses both normalize functions consistently."""
        norm_exchange = normalize_exchange(sample_trade.exchange)
        norm_symbol = normalize_symbol(sample_trade.symbol)

        composite = CompositePartitioner()
        partition_key = composite.get_partition_key(sample_trade)

        expected = f"{norm_exchange}-{norm_symbol}".encode("utf-8")
        assert partition_key == expected
        assert partition_key == b"binance-btc-usd"
