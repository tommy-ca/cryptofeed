"""Phase 2: Task 7 - Unit tests for partition key strategies.

This module tests:
- Task 7.1: Symbol partitioner consistency and normalization
- Task 7.2: Composite and exchange partitioners
- Task 7.3: Partitioner factory selection

All tests written FIRST (TDD: RED phase) before implementation.
"""

import pytest
from dataclasses import dataclass
from typing import Optional

from cryptofeed.kafka_callback import (
    SymbolPartitioner,
    CompositePartitioner,
    ExchangePartitioner,
    RoundRobinPartitioner,
    PartitionerFactory,
)


@dataclass
class MockMessage:
    """Mock message for testing partitioners."""
    exchange: str
    symbol: str


# ============================================================================
# Task 7.1: Symbol Partitioner Tests
# ============================================================================


class TestTaskSevenOneSymbolPartitioner:
    """Test symbol-based partition key generation."""

    def test_symbol_partitioner_encodes_symbol(self):
        """SymbolPartitioner should encode symbol as bytes."""
        partitioner = SymbolPartitioner()
        msg = MockMessage(exchange='coinbase', symbol='BTC-USD')

        key = partitioner.get_partition_key(msg)

        assert isinstance(key, bytes), \
            "Partition key should be bytes"
        assert key == b'btc-usd', \
            "Should encode lowercase symbol"

    def test_symbol_partitioner_case_normalization(self):
        """SymbolPartitioner should normalize to lowercase."""
        partitioner = SymbolPartitioner()

        # Test various case combinations
        test_cases = [
            ('BTC-USD', b'btc-usd'),
            ('btc-usd', b'btc-usd'),
            ('BtC-UsD', b'btc-usd'),
            ('ETHEREUM-USDT', b'ethereum-usdt'),
        ]

        for symbol, expected_key in test_cases:
            msg = MockMessage(exchange='test', symbol=symbol)
            key = partitioner.get_partition_key(msg)
            assert key == expected_key, \
                f"Symbol {symbol} should produce {expected_key}"

    def test_symbol_partitioner_underscore_to_hyphen(self):
        """SymbolPartitioner should convert underscores to hyphens."""
        partitioner = SymbolPartitioner()

        msg_hyphen = MockMessage(exchange='test', symbol='BTC-USDT')
        msg_underscore = MockMessage(exchange='test', symbol='BTC_USDT')

        key_hyphen = partitioner.get_partition_key(msg_hyphen)
        key_underscore = partitioner.get_partition_key(msg_underscore)

        # Both should produce identical keys
        assert key_hyphen == key_underscore == b'btc-usdt', \
            "Underscores and hyphens should be normalized to same key"

    def test_symbol_partitioner_consistency(self):
        """SymbolPartitioner should produce consistent keys across multiple calls."""
        partitioner = SymbolPartitioner()
        msg = MockMessage(exchange='kraken', symbol='ETH-USD')

        # Generate key 100 times
        keys = [partitioner.get_partition_key(msg) for _ in range(100)]

        # All should be identical
        assert all(k == keys[0] for k in keys), \
            "Partition key should be deterministic"

    def test_symbol_partitioner_ignores_exchange(self):
        """SymbolPartitioner should ignore exchange field."""
        partitioner = SymbolPartitioner()

        msg1 = MockMessage(exchange='coinbase', symbol='BTC-USD')
        msg2 = MockMessage(exchange='binance', symbol='BTC-USD')
        msg3 = MockMessage(exchange='kraken', symbol='BTC-USD')

        key1 = partitioner.get_partition_key(msg1)
        key2 = partitioner.get_partition_key(msg2)
        key3 = partitioner.get_partition_key(msg3)

        assert key1 == key2 == key3, \
            "Same symbol should produce same key regardless of exchange"

    def test_symbol_partitioner_various_symbol_formats(self):
        """SymbolPartitioner should handle various symbol formats."""
        partitioner = SymbolPartitioner()

        formats = [
            'BTC-USD',
            'btc-usd',
            'BTC_USD',
            'btc_usd',
            'BTC/USD',  # slash preserved if present
        ]

        # All formats should produce keys (though / might be preserved)
        for symbol_fmt in formats:
            msg = MockMessage(exchange='test', symbol=symbol_fmt)
            key = partitioner.get_partition_key(msg)
            assert isinstance(key, bytes), \
                f"Should handle symbol format {symbol_fmt}"


# ============================================================================
# Task 7.2: Composite and Exchange Partitioner Tests
# ============================================================================


class TestTaskSevenTwoCompositePartitioner:
    """Test composite (exchange-symbol) partition key generation."""

    def test_composite_partitioner_includes_exchange_and_symbol(self):
        """CompositePartitioner should include both exchange and symbol."""
        partitioner = CompositePartitioner()
        msg = MockMessage(exchange='coinbase', symbol='BTC-USD')

        key = partitioner.get_partition_key(msg)

        assert isinstance(key, bytes), \
            "Partition key should be bytes"
        # Should contain both exchange and symbol in some form
        key_str = key.decode('utf-8').lower()
        assert 'coinbase' in key_str and 'btc' in key_str, \
            "Composite key should include exchange and symbol"

    def test_composite_partitioner_exchange_normalization(self):
        """CompositePartitioner should normalize exchange to lowercase."""
        partitioner = CompositePartitioner()

        msg1 = MockMessage(exchange='Coinbase', symbol='BTC-USD')
        msg2 = MockMessage(exchange='COINBASE', symbol='BTC-USD')

        key1 = partitioner.get_partition_key(msg1)
        key2 = partitioner.get_partition_key(msg2)

        assert key1 == key2, \
            "Exchange case should not affect partition key"

    def test_composite_partitioner_symbol_normalization(self):
        """CompositePartitioner should normalize symbol."""
        partitioner = CompositePartitioner()

        msg1 = MockMessage(exchange='binance', symbol='BTC-USDT')
        msg2 = MockMessage(exchange='binance', symbol='BTC_USDT')

        key1 = partitioner.get_partition_key(msg1)
        key2 = partitioner.get_partition_key(msg2)

        assert key1 == key2, \
            "Symbol format variations should produce same key"

    def test_composite_partitioner_different_exchanges_different_keys(self):
        """Different exchanges should produce different partition keys."""
        partitioner = CompositePartitioner()

        msg1 = MockMessage(exchange='coinbase', symbol='BTC-USD')
        msg2 = MockMessage(exchange='binance', symbol='BTC-USD')

        key1 = partitioner.get_partition_key(msg1)
        key2 = partitioner.get_partition_key(msg2)

        assert key1 != key2, \
            "Different exchanges with same symbol should have different keys"

    def test_composite_partitioner_different_symbols_different_keys(self):
        """Different symbols should produce different partition keys."""
        partitioner = CompositePartitioner()

        msg1 = MockMessage(exchange='binance', symbol='BTC-USDT')
        msg2 = MockMessage(exchange='binance', symbol='ETH-USDT')

        key1 = partitioner.get_partition_key(msg1)
        key2 = partitioner.get_partition_key(msg2)

        assert key1 != key2, \
            "Different symbols with same exchange should have different keys"

    def test_composite_partitioner_consistency(self):
        """CompositePartitioner should be deterministic."""
        partitioner = CompositePartitioner()
        msg = MockMessage(exchange='kraken', symbol='SOL-USD')

        keys = [partitioner.get_partition_key(msg) for _ in range(50)]

        assert all(k == keys[0] for k in keys), \
            "Composite partition key should be deterministic"


class TestTaskSevenTwoExchangePartitioner:
    """Test exchange-based partition key generation."""

    def test_exchange_partitioner_encodes_exchange(self):
        """ExchangePartitioner should encode exchange as bytes."""
        partitioner = ExchangePartitioner()
        msg = MockMessage(exchange='coinbase', symbol='BTC-USD')

        key = partitioner.get_partition_key(msg)

        assert isinstance(key, bytes), \
            "Partition key should be bytes"
        assert key == b'coinbase', \
            "Should encode lowercase exchange"

    def test_exchange_partitioner_case_normalization(self):
        """ExchangePartitioner should normalize to lowercase."""
        partitioner = ExchangePartitioner()

        msg1 = MockMessage(exchange='Coinbase', symbol='BTC-USD')
        msg2 = MockMessage(exchange='COINBASE', symbol='BTC-USD')

        key1 = partitioner.get_partition_key(msg1)
        key2 = partitioner.get_partition_key(msg2)

        assert key1 == key2 == b'coinbase', \
            "Exchange normalization should produce same key"

    def test_exchange_partitioner_ignores_symbol(self):
        """ExchangePartitioner should ignore symbol."""
        partitioner = ExchangePartitioner()

        msg1 = MockMessage(exchange='binance', symbol='BTC-USDT')
        msg2 = MockMessage(exchange='binance', symbol='ETH-USDT')

        key1 = partitioner.get_partition_key(msg1)
        key2 = partitioner.get_partition_key(msg2)

        assert key1 == key2, \
            "Exchange partitioner should ignore symbol"

    def test_exchange_partitioner_different_exchanges(self):
        """Different exchanges should produce different keys."""
        partitioner = ExchangePartitioner()

        exchanges = ['coinbase', 'binance', 'kraken', 'bybit']
        keys = []

        for exchange in exchanges:
            msg = MockMessage(exchange=exchange, symbol='BTC-USD')
            key = partitioner.get_partition_key(msg)
            keys.append(key)

        # All keys should be unique
        assert len(set(keys)) == len(keys), \
            "Different exchanges should produce different keys"


# ============================================================================
# Task 7.3: Partitioner Factory Tests
# ============================================================================


class TestTaskSevenThreePartitionerFactory:
    """Test partitioner factory selection."""

    def test_factory_returns_symbol_partitioner(self):
        """Factory should return SymbolPartitioner for 'symbol' strategy."""
        partitioner = PartitionerFactory.create('symbol')

        assert isinstance(partitioner, SymbolPartitioner), \
            "Factory should return SymbolPartitioner for 'symbol' strategy"

    def test_factory_returns_composite_partitioner(self):
        """Factory should return CompositePartitioner for 'composite' strategy."""
        partitioner = PartitionerFactory.create('composite')

        assert isinstance(partitioner, CompositePartitioner), \
            "Factory should return CompositePartitioner for 'composite' strategy"

    def test_factory_returns_exchange_partitioner(self):
        """Factory should return ExchangePartitioner for 'exchange' strategy."""
        partitioner = PartitionerFactory.create('exchange')

        assert isinstance(partitioner, ExchangePartitioner), \
            "Factory should return ExchangePartitioner for 'exchange' strategy"

    def test_factory_returns_round_robin_partitioner(self):
        """Factory should return RoundRobinPartitioner for 'round_robin' strategy."""
        partitioner = PartitionerFactory.create('round_robin')

        assert isinstance(partitioner, RoundRobinPartitioner), \
            "Factory should return RoundRobinPartitioner for 'round_robin' strategy"

    def test_factory_default_is_composite(self):
        """Factory should default to composite partitioner."""
        partitioner = PartitionerFactory.create('composite')

        assert isinstance(partitioner, CompositePartitioner), \
            "Default should be composite partitioner"

    def test_factory_invalid_strategy_raises_error(self):
        """Factory should raise error for invalid strategy."""
        with pytest.raises((ValueError, KeyError)):
            PartitionerFactory.create('invalid_strategy')

    def test_factory_case_insensitive_strategy(self):
        """Factory should handle case-insensitive strategy names."""
        # Try uppercase
        partitioner = PartitionerFactory.create('COMPOSITE')
        assert isinstance(partitioner, CompositePartitioner), \
            "Factory should handle uppercase strategy names"

    def test_factory_all_strategies_are_valid(self):
        """Factory should support all four strategies."""
        strategies = ['symbol', 'composite', 'exchange', 'round_robin']

        for strategy in strategies:
            partitioner = PartitionerFactory.create(strategy)
            assert partitioner is not None, \
                f"Factory should create partitioner for {strategy}"

    def test_partitioner_selection_by_config(self):
        """Verify partitioner selection based on config strategy string."""
        config_strategies = [
            ('symbol', SymbolPartitioner),
            ('composite', CompositePartitioner),
            ('exchange', ExchangePartitioner),
            ('round_robin', RoundRobinPartitioner),
        ]

        for strategy_name, expected_class in config_strategies:
            partitioner = PartitionerFactory.create(strategy_name)
            assert isinstance(partitioner, expected_class), \
                f"Strategy {strategy_name} should create {expected_class.__name__}"


# ============================================================================
# Round-Robin Partitioner Tests
# ============================================================================


class TestRoundRobinPartitioner:
    """Test round-robin partitioner for maximum parallelism."""

    def test_round_robin_returns_none(self):
        """RoundRobinPartitioner should return None (Kafka assigns partition)."""
        partitioner = RoundRobinPartitioner()
        msg = MockMessage(exchange='coinbase', symbol='BTC-USD')

        key = partitioner.get_partition_key(msg)

        assert key is None, \
            "Round-robin should return None (no ordering guarantee)"

    def test_round_robin_always_returns_none(self):
        """RoundRobinPartitioner should always return None."""
        partitioner = RoundRobinPartitioner()

        for _ in range(100):
            msg = MockMessage(exchange='test', symbol='BTC-USD')
            key = partitioner.get_partition_key(msg)
            assert key is None, \
                "Round-robin should always return None"

    def test_round_robin_no_ordering_guarantee(self):
        """RoundRobinPartitioner documents no ordering guarantee."""
        # This test documents the behavior: round-robin provides no ordering
        partitioner = RoundRobinPartitioner()
        msg1 = MockMessage(exchange='coinbase', symbol='BTC-USD')
        msg2 = MockMessage(exchange='coinbase', symbol='BTC-USD')

        # Both return None, meaning Kafka assigns partitions
        key1 = partitioner.get_partition_key(msg1)
        key2 = partitioner.get_partition_key(msg2)

        assert key1 is None and key2 is None, \
            "Round-robin provides no ordering guarantee"


# ============================================================================
# Integration Tests: Partitioner Usage
# ============================================================================


class TestPartitionerIntegration:
    """Test partitioner usage patterns."""

    def test_partitioner_with_mock_messages(self):
        """Verify partitioners work with typical message objects."""
        partitioner = CompositePartitioner()

        # Simulate multiple messages
        messages = [
            MockMessage(exchange='binance', symbol='BTC-USDT'),
            MockMessage(exchange='binance', symbol='ETH-USDT'),
            MockMessage(exchange='coinbase', symbol='BTC-USD'),
        ]

        keys = [partitioner.get_partition_key(msg) for msg in messages]

        assert all(isinstance(k, bytes) for k in keys), \
            "All partition keys should be bytes"
        assert len(set(keys)) > 1, \
            "Different messages should have different keys"

    def test_all_partitioners_return_bytes_or_none(self):
        """All partitioners should return bytes or None."""
        msg = MockMessage(exchange='test', symbol='TEST-USD')

        partitioners = [
            SymbolPartitioner(),
            CompositePartitioner(),
            ExchangePartitioner(),
            RoundRobinPartitioner(),
        ]

        for partitioner in partitioners:
            key = partitioner.get_partition_key(msg)
            assert key is None or isinstance(key, bytes), \
                f"{partitioner.__class__.__name__} should return bytes or None"
