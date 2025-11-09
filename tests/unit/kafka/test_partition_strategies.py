"""Unit tests for Kafka partition key strategies (Spec 3, Task 2).

Tests cover four partition key strategies:
1. Composite: {exchange}-{symbol} (default)
2. Symbol: {symbol} (per-symbol ordering)
3. Exchange: {exchange} (per-exchange ordering)
4. Round-robin: None (maximum parallelism)

Each strategy is tested for:
- Correct partition key generation
- Consistency (deterministic mapping)
- Normalization (uppercase, special characters)
- Edge cases (empty values, special characters)
- Factory pattern selection
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any, Optional


# Import or skip partitioner implementations
pytest.importorskip("cryptofeed.kafka_callback")
from cryptofeed.kafka_callback import (
    SymbolPartitioner,
    CompositePartitioner,
    ExchangePartitioner,
    RoundRobinPartitioner,
    PartitionerFactory,
)


@dataclass
class MockMessage:
    """Mock message object for testing partitioners."""
    exchange: str
    symbol: str

    def __repr__(self) -> str:
        return f"MockMessage(exchange={self.exchange!r}, symbol={self.symbol!r})"


class TestSymbolPartitioner:
    """Test symbol-based partition key strategy.

    Symbol partitioner generates keys based on trading symbol only.
    All messages for same symbol → same partition across all exchanges.

    Examples:
        - BTC-USD (all exchanges) → partition key b'btc-usd'
        - ETH-USDT (all exchanges) → partition key b'eth-usdt'
    """

    def test_symbol_partitioner_basic_generation(self):
        """SymbolPartitioner should encode symbol as partition key."""
        partitioner = SymbolPartitioner()
        msg = MockMessage(exchange="coinbase", symbol="BTC-USD")

        key = partitioner.get_partition_key(msg)

        assert isinstance(key, bytes)
        assert key == b"btc-usd"

    def test_symbol_partitioner_case_insensitive(self):
        """SymbolPartitioner should normalize to lowercase."""
        partitioner = SymbolPartitioner()

        # All cases should produce same key
        msg_upper = MockMessage(exchange="coinbase", symbol="BTC-USD")
        msg_lower = MockMessage(exchange="coinbase", symbol="btc-usd")
        msg_mixed = MockMessage(exchange="coinbase", symbol="BtC-UsD")

        key_upper = partitioner.get_partition_key(msg_upper)
        key_lower = partitioner.get_partition_key(msg_lower)
        key_mixed = partitioner.get_partition_key(msg_mixed)

        assert key_upper == key_lower == key_mixed == b"btc-usd"

    def test_symbol_partitioner_underscore_to_hyphen(self):
        """SymbolPartitioner should convert underscores to hyphens."""
        partitioner = SymbolPartitioner()

        msg_hyphen = MockMessage(exchange="binance", symbol="BTC-USDT")
        msg_underscore = MockMessage(exchange="binance", symbol="BTC_USDT")

        key_hyphen = partitioner.get_partition_key(msg_hyphen)
        key_underscore = partitioner.get_partition_key(msg_underscore)

        # Both should map to same key
        assert key_hyphen == key_underscore == b"btc-usdt"

    def test_symbol_partitioner_consistency(self):
        """SymbolPartitioner should produce consistent keys for same symbol."""
        partitioner = SymbolPartitioner()
        msg = MockMessage(exchange="kraken", symbol="ETH-USD")

        # Generate key multiple times
        keys = [partitioner.get_partition_key(msg) for _ in range(10)]

        # All keys should be identical
        assert all(k == keys[0] for k in keys)
        assert keys[0] == b"eth-usd"

    def test_symbol_partitioner_different_symbols(self):
        """SymbolPartitioner should generate different keys for different symbols."""
        partitioner = SymbolPartitioner()

        msg_btc = MockMessage(exchange="coinbase", symbol="BTC-USD")
        msg_eth = MockMessage(exchange="coinbase", symbol="ETH-USD")
        msg_sol = MockMessage(exchange="coinbase", symbol="SOL-USD")

        key_btc = partitioner.get_partition_key(msg_btc)
        key_eth = partitioner.get_partition_key(msg_eth)
        key_sol = partitioner.get_partition_key(msg_sol)

        assert key_btc == b"btc-usd"
        assert key_eth == b"eth-usd"
        assert key_sol == b"sol-usd"
        assert len({key_btc, key_eth, key_sol}) == 3

    def test_symbol_partitioner_exchange_independent(self):
        """SymbolPartitioner should ignore exchange name."""
        partitioner = SymbolPartitioner()

        msg_coinbase = MockMessage(exchange="coinbase", symbol="BTC-USD")
        msg_binance = MockMessage(exchange="binance", symbol="BTC-USD")
        msg_kraken = MockMessage(exchange="kraken", symbol="BTC-USD")

        key_coinbase = partitioner.get_partition_key(msg_coinbase)
        key_binance = partitioner.get_partition_key(msg_binance)
        key_kraken = partitioner.get_partition_key(msg_kraken)

        # Same symbol from different exchanges → same partition key
        assert key_coinbase == key_binance == key_kraken == b"btc-usd"

    def test_symbol_partitioner_complex_symbols(self):
        """SymbolPartitioner should handle complex symbol formats."""
        partitioner = SymbolPartitioner()

        # Perpetual futures symbols
        msg_perp = MockMessage(exchange="dydx", symbol="BTC-USD-PERP")
        key_perp = partitioner.get_partition_key(msg_perp)
        assert key_perp == b"btc-usd-perp"

        # Spot pair with slashes (normalized)
        msg_slash = MockMessage(exchange="kraken", symbol="BTC/USD")
        key_slash = partitioner.get_partition_key(msg_slash)
        assert key_slash == b"btc/usd"

    def test_symbol_partitioner_whitespace_handling(self):
        """SymbolPartitioner should strip whitespace from symbols."""
        partitioner = SymbolPartitioner()

        msg_spaces = MockMessage(exchange="coinbase", symbol="  BTC-USD  ")
        key = partitioner.get_partition_key(msg_spaces)

        # Should handle whitespace gracefully
        assert key == b"btc-usd"


class TestCompositePartitioner:
    """Test composite partition key strategy.

    Composite partitioner generates keys from exchange + symbol.
    Same exchange-symbol pair always → same partition.
    Different exchanges with same symbol → different partitions.

    Examples:
        - coinbase + BTC-USD → b'coinbase-btc-usd'
        - binance + BTC-USD → b'binance-btc-usd'
        - coinbase + ETH-USD → b'coinbase-eth-usd'
    """

    def test_composite_partitioner_basic_generation(self):
        """CompositePartitioner should combine exchange and symbol."""
        partitioner = CompositePartitioner()
        msg = MockMessage(exchange="coinbase", symbol="BTC-USD")

        key = partitioner.get_partition_key(msg)

        assert isinstance(key, bytes)
        assert key == b"coinbase-btc-usd"

    def test_composite_partitioner_case_normalization(self):
        """CompositePartitioner should normalize both exchange and symbol."""
        partitioner = CompositePartitioner()

        msg_upper = MockMessage(exchange="COINBASE", symbol="BTC-USD")
        msg_lower = MockMessage(exchange="coinbase", symbol="btc-usd")
        msg_mixed = MockMessage(exchange="CoinBase", symbol="BtC-UsD")

        key_upper = partitioner.get_partition_key(msg_upper)
        key_lower = partitioner.get_partition_key(msg_lower)
        key_mixed = partitioner.get_partition_key(msg_mixed)

        assert key_upper == key_lower == key_mixed == b"coinbase-btc-usd"

    def test_composite_partitioner_symbol_normalization(self):
        """CompositePartitioner should normalize symbol underscores to hyphens."""
        partitioner = CompositePartitioner()

        msg_hyphen = MockMessage(exchange="binance", symbol="BTC-USDT")
        msg_underscore = MockMessage(exchange="binance", symbol="BTC_USDT")

        key_hyphen = partitioner.get_partition_key(msg_hyphen)
        key_underscore = partitioner.get_partition_key(msg_underscore)

        assert key_hyphen == key_underscore == b"binance-btc-usdt"

    def test_composite_partitioner_consistency(self):
        """CompositePartitioner should produce consistent keys."""
        partitioner = CompositePartitioner()
        msg = MockMessage(exchange="kraken", symbol="ETH-USD")

        keys = [partitioner.get_partition_key(msg) for _ in range(10)]

        assert all(k == keys[0] for k in keys)
        assert keys[0] == b"kraken-eth-usd"

    def test_composite_partitioner_exchange_sensitivity(self):
        """CompositePartitioner should distinguish different exchanges."""
        partitioner = CompositePartitioner()

        msg_coinbase = MockMessage(exchange="coinbase", symbol="BTC-USD")
        msg_binance = MockMessage(exchange="binance", symbol="BTC-USD")
        msg_kraken = MockMessage(exchange="kraken", symbol="BTC-USD")

        key_coinbase = partitioner.get_partition_key(msg_coinbase)
        key_binance = partitioner.get_partition_key(msg_binance)
        key_kraken = partitioner.get_partition_key(msg_kraken)

        # Different exchanges → different keys
        assert key_coinbase == b"coinbase-btc-usd"
        assert key_binance == b"binance-btc-usd"
        assert key_kraken == b"kraken-btc-usd"
        assert len({key_coinbase, key_binance, key_kraken}) == 3

    def test_composite_partitioner_symbol_sensitivity(self):
        """CompositePartitioner should distinguish different symbols."""
        partitioner = CompositePartitioner()

        msg_btc = MockMessage(exchange="coinbase", symbol="BTC-USD")
        msg_eth = MockMessage(exchange="coinbase", symbol="ETH-USD")
        msg_sol = MockMessage(exchange="coinbase", symbol="SOL-USD")

        key_btc = partitioner.get_partition_key(msg_btc)
        key_eth = partitioner.get_partition_key(msg_eth)
        key_sol = partitioner.get_partition_key(msg_sol)

        # Different symbols → different keys
        assert key_btc == b"coinbase-btc-usd"
        assert key_eth == b"coinbase-eth-usd"
        assert key_sol == b"coinbase-sol-usd"
        assert len({key_btc, key_eth, key_sol}) == 3

    def test_composite_partitioner_different_pairs(self):
        """CompositePartitioner should distinguish all exchange-symbol pairs."""
        partitioner = CompositePartitioner()

        pairs = [
            ("coinbase", "BTC-USD"),
            ("coinbase", "ETH-USD"),
            ("binance", "BTC-USD"),
            ("binance", "ETH-USD"),
            ("kraken", "BTC-USD"),
        ]

        keys = [
            partitioner.get_partition_key(MockMessage(ex, sym))
            for ex, sym in pairs
        ]

        # All different pairs should have unique keys
        assert len(set(keys)) == len(pairs)

    def test_composite_partitioner_complex_symbols(self):
        """CompositePartitioner should handle complex symbol formats."""
        partitioner = CompositePartitioner()

        msg_perp = MockMessage(exchange="dydx", symbol="BTC-USD-PERP")
        key_perp = partitioner.get_partition_key(msg_perp)
        assert key_perp == b"dydx-btc-usd-perp"

        msg_inverse = MockMessage(exchange="bitmex", symbol="XBT-USD")
        key_inverse = partitioner.get_partition_key(msg_inverse)
        assert key_inverse == b"bitmex-xbt-usd"


class TestExchangePartitioner:
    """Test exchange-based partition key strategy.

    Exchange partitioner generates keys based on exchange only.
    Same exchange → same partition across all symbols.
    Different exchanges → different partitions.

    Examples:
        - coinbase (all symbols) → b'coinbase'
        - binance (all symbols) → b'binance'
    """

    def test_exchange_partitioner_basic_generation(self):
        """ExchangePartitioner should encode exchange as partition key."""
        partitioner = ExchangePartitioner()
        msg = MockMessage(exchange="coinbase", symbol="BTC-USD")

        key = partitioner.get_partition_key(msg)

        assert isinstance(key, bytes)
        assert key == b"coinbase"

    def test_exchange_partitioner_case_normalization(self):
        """ExchangePartitioner should normalize exchange to lowercase."""
        partitioner = ExchangePartitioner()

        msg_upper = MockMessage(exchange="COINBASE", symbol="BTC-USD")
        msg_lower = MockMessage(exchange="coinbase", symbol="BTC-USD")
        msg_mixed = MockMessage(exchange="CoinBase", symbol="BTC-USD")

        key_upper = partitioner.get_partition_key(msg_upper)
        key_lower = partitioner.get_partition_key(msg_lower)
        key_mixed = partitioner.get_partition_key(msg_mixed)

        assert key_upper == key_lower == key_mixed == b"coinbase"

    def test_exchange_partitioner_consistency(self):
        """ExchangePartitioner should produce consistent keys."""
        partitioner = ExchangePartitioner()
        msg = MockMessage(exchange="kraken", symbol="ETH-USD")

        keys = [partitioner.get_partition_key(msg) for _ in range(10)]

        assert all(k == keys[0] for k in keys)
        assert keys[0] == b"kraken"

    def test_exchange_partitioner_symbol_independent(self):
        """ExchangePartitioner should ignore symbol."""
        partitioner = ExchangePartitioner()

        msg_btc = MockMessage(exchange="coinbase", symbol="BTC-USD")
        msg_eth = MockMessage(exchange="coinbase", symbol="ETH-USD")
        msg_sol = MockMessage(exchange="coinbase", symbol="SOL-USD")

        key_btc = partitioner.get_partition_key(msg_btc)
        key_eth = partitioner.get_partition_key(msg_eth)
        key_sol = partitioner.get_partition_key(msg_sol)

        # Same exchange, different symbols → same key
        assert key_btc == key_eth == key_sol == b"coinbase"

    def test_exchange_partitioner_exchange_sensitive(self):
        """ExchangePartitioner should distinguish different exchanges."""
        partitioner = ExchangePartitioner()

        msg_coinbase = MockMessage(exchange="coinbase", symbol="BTC-USD")
        msg_binance = MockMessage(exchange="binance", symbol="BTC-USD")
        msg_kraken = MockMessage(exchange="kraken", symbol="BTC-USD")

        key_coinbase = partitioner.get_partition_key(msg_coinbase)
        key_binance = partitioner.get_partition_key(msg_binance)
        key_kraken = partitioner.get_partition_key(msg_kraken)

        # Different exchanges → different keys
        assert key_coinbase == b"coinbase"
        assert key_binance == b"binance"
        assert key_kraken == b"kraken"
        assert len({key_coinbase, key_binance, key_kraken}) == 3

    def test_exchange_partitioner_multiple_exchanges(self):
        """ExchangePartitioner should handle multiple exchanges correctly."""
        partitioner = ExchangePartitioner()

        exchanges = ["coinbase", "binance", "kraken", "bitmex", "dydx"]
        keys = [
            partitioner.get_partition_key(MockMessage(ex, "BTC-USD"))
            for ex in exchanges
        ]

        # All different exchanges should have unique keys
        assert len(set(keys)) == len(exchanges)
        assert keys == [ex.encode() for ex in exchanges]


class TestRoundRobinPartitioner:
    """Test round-robin partition key strategy.

    Round-robin partitioner returns None for partition key,
    allowing Kafka to distribute messages automatically.

    Use case: Maximum parallelism when ordering doesn't matter.
    """

    def test_roundrobin_partitioner_returns_none(self):
        """RoundRobinPartitioner should return None for partition key."""
        partitioner = RoundRobinPartitioner()
        msg = MockMessage(exchange="coinbase", symbol="BTC-USD")

        key = partitioner.get_partition_key(msg)

        assert key is None

    def test_roundrobin_partitioner_always_none(self):
        """RoundRobinPartitioner should always return None regardless of input."""
        partitioner = RoundRobinPartitioner()

        messages = [
            MockMessage("coinbase", "BTC-USD"),
            MockMessage("binance", "ETH-USDT"),
            MockMessage("kraken", "SOL-USD"),
            MockMessage("kraken", "BTC-USD"),
            MockMessage("coinbase", "ETH-USD"),
        ]

        keys = [partitioner.get_partition_key(msg) for msg in messages]

        # All should be None
        assert all(k is None for k in keys)

    def test_roundrobin_partitioner_consistency(self):
        """RoundRobinPartitioner should consistently return None."""
        partitioner = RoundRobinPartitioner()
        msg = MockMessage(exchange="kraken", symbol="ETH-USD")

        keys = [partitioner.get_partition_key(msg) for _ in range(100)]

        assert all(k is None for k in keys)

    def test_roundrobin_partitioner_ignores_all_metadata(self):
        """RoundRobinPartitioner should ignore exchange and symbol."""
        partitioner = RoundRobinPartitioner()

        # All these should produce None
        key1 = partitioner.get_partition_key(MockMessage("ex1", "sym1"))
        key2 = partitioner.get_partition_key(MockMessage("ex2", "sym2"))
        key3 = partitioner.get_partition_key(MockMessage("ex100", "sym100"))

        assert key1 is None
        assert key2 is None
        assert key3 is None


class TestPartitionerFactory:
    """Test factory pattern for partitioner selection.

    Factory should support:
    - Creating partitioners by strategy name
    - Default strategy (composite)
    - Invalid strategy error handling
    - Caching/reuse of partitioners
    """

    def test_factory_creates_symbol_partitioner(self):
        """Factory should create SymbolPartitioner for 'symbol' strategy."""
        partitioner = PartitionerFactory.create("symbol")

        assert isinstance(partitioner, SymbolPartitioner)

    def test_factory_creates_composite_partitioner(self):
        """Factory should create CompositePartitioner for 'composite' strategy."""
        partitioner = PartitionerFactory.create("composite")

        assert isinstance(partitioner, CompositePartitioner)

    def test_factory_creates_exchange_partitioner(self):
        """Factory should create ExchangePartitioner for 'exchange' strategy."""
        partitioner = PartitionerFactory.create("exchange")

        assert isinstance(partitioner, ExchangePartitioner)

    def test_factory_creates_roundrobin_partitioner(self):
        """Factory should create RoundRobinPartitioner for 'round_robin' strategy."""
        partitioner = PartitionerFactory.create("round_robin")

        assert isinstance(partitioner, RoundRobinPartitioner)

    def test_factory_default_is_composite(self):
        """Factory should default to composite partitioner."""
        partitioner = PartitionerFactory.create("composite")

        assert isinstance(partitioner, CompositePartitioner)

    def test_factory_with_no_strategy_defaults_composite(self):
        """Factory should use composite as default when strategy not specified."""
        partitioner = PartitionerFactory.create()

        assert isinstance(partitioner, CompositePartitioner)

    def test_factory_invalid_strategy_raises_error(self):
        """Factory should raise ValueError for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown.*strategy|not.*supported"):
            PartitionerFactory.create("invalid_strategy")

    def test_factory_case_insensitive_strategy(self):
        """Factory should be case-insensitive for strategy names."""
        # Common case variations should all work
        strategies_to_test = [
            ("symbol", SymbolPartitioner),
            ("Symbol", SymbolPartitioner),
            ("SYMBOL", SymbolPartitioner),
            ("composite", CompositePartitioner),
            ("Composite", CompositePartitioner),
            ("COMPOSITE", CompositePartitioner),
            ("exchange", ExchangePartitioner),
            ("Exchange", ExchangePartitioner),
            ("EXCHANGE", ExchangePartitioner),
            ("round_robin", RoundRobinPartitioner),
            ("Round_Robin", RoundRobinPartitioner),
            ("ROUND_ROBIN", RoundRobinPartitioner),
        ]

        for strategy_name, expected_class in strategies_to_test:
            partitioner = PartitionerFactory.create(strategy_name)
            assert isinstance(partitioner, expected_class), \
                f"Strategy {strategy_name} should create {expected_class.__name__}"

    def test_factory_created_partitioners_work_correctly(self):
        """Factory-created partitioners should function correctly."""
        msg = MockMessage("coinbase", "BTC-USD")

        # Test each factory-created partitioner
        symbol_key = PartitionerFactory.create("symbol").get_partition_key(msg)
        assert symbol_key == b"btc-usd"

        composite_key = PartitionerFactory.create("composite").get_partition_key(msg)
        assert composite_key == b"coinbase-btc-usd"

        exchange_key = PartitionerFactory.create("exchange").get_partition_key(msg)
        assert exchange_key == b"coinbase"

        rr_key = PartitionerFactory.create("round_robin").get_partition_key(msg)
        assert rr_key is None


class TestPartitionerIntegration:
    """Integration tests for partitioner system.

    Tests how partitioners work together and with other components.
    """

    def test_partitioner_interface_consistency(self):
        """All partitioners should have consistent interface."""
        msg = MockMessage("coinbase", "BTC-USD")

        partitioners = [
            SymbolPartitioner(),
            CompositePartitioner(),
            ExchangePartitioner(),
            RoundRobinPartitioner(),
        ]

        # All should have get_partition_key method
        for p in partitioners:
            assert hasattr(p, "get_partition_key")
            assert callable(p.get_partition_key)

            # All should return bytes or None
            key = p.get_partition_key(msg)
            assert isinstance(key, (bytes, type(None)))

    def test_multiple_partitioners_independent(self):
        """Multiple partitioner instances should work independently."""
        msg = MockMessage("binance", "ETH-USDT")

        # Create multiple instances of same partitioner type
        p1 = CompositePartitioner()
        p2 = CompositePartitioner()

        key1 = p1.get_partition_key(msg)
        key2 = p2.get_partition_key(msg)

        # Should produce identical keys
        assert key1 == key2 == b"binance-eth-usdt"

    def test_partitioner_deterministic_ordering(self):
        """Partitioner keys should be deterministic for ordering guarantees."""
        messages = [
            MockMessage("coinbase", "BTC-USD"),
            MockMessage("coinbase", "BTC-USD"),
            MockMessage("coinbase", "BTC-USD"),
        ]

        partitioner = SymbolPartitioner()
        keys = [partitioner.get_partition_key(msg) for msg in messages]

        # All identical messages should have identical keys
        assert len(set(keys)) == 1
        assert keys[0] == b"btc-usd"

    def test_all_strategies_with_various_inputs(self):
        """All strategies should handle various input formats."""
        test_cases = [
            ("Coinbase", "BTC-USD"),
            ("binance", "eth-usdt"),
            ("KRAKEN", "SOL_USD"),
            ("bitmex", "XBT-USD"),
        ]

        partitioners = [
            ("symbol", SymbolPartitioner()),
            ("composite", CompositePartitioner()),
            ("exchange", ExchangePartitioner()),
            ("round_robin", RoundRobinPartitioner()),
        ]

        for exchange, symbol in test_cases:
            msg = MockMessage(exchange, symbol)

            for strategy_name, partitioner in partitioners:
                # Should not raise exception
                key = partitioner.get_partition_key(msg)

                # Key should be bytes or None
                assert isinstance(key, (bytes, type(None))), \
                    f"Strategy {strategy_name} returned {type(key)}"


class TestPartitionerEdgeCases:
    """Edge case tests for partition strategies."""

    def test_partitioner_with_empty_symbol(self):
        """Partitioners should handle empty symbol gracefully."""
        partitioner = SymbolPartitioner()
        msg = MockMessage("coinbase", "")

        # Should return bytes even for empty symbol
        key = partitioner.get_partition_key(msg)
        assert isinstance(key, bytes)

    def test_partitioner_with_special_characters(self):
        """Partitioners should handle special characters in symbols."""
        partitioner = CompositePartitioner()

        # Slash-separated symbols
        msg_slash = MockMessage("kraken", "BTC/USD")
        key_slash = partitioner.get_partition_key(msg_slash)
        assert isinstance(key_slash, bytes)

        # Period-separated symbols (rare but possible)
        msg_period = MockMessage("exchange", "BTC.USD")
        key_period = partitioner.get_partition_key(msg_period)
        assert isinstance(key_period, bytes)

    def test_partitioner_with_long_symbols(self):
        """Partitioners should handle long symbol names."""
        partitioner = CompositePartitioner()
        long_symbol = "BTC-USD-PERP-QUARTERLY-MARCH-2025"
        msg = MockMessage("dydx", long_symbol)

        key = partitioner.get_partition_key(msg)
        assert isinstance(key, bytes)
        assert len(key) > 0

    def test_partitioner_with_numeric_exchanges(self):
        """Partitioners should handle numeric exchange identifiers."""
        partitioner = ExchangePartitioner()
        msg = MockMessage("exchange123", "BTC-USD")

        key = partitioner.get_partition_key(msg)
        assert key == b"exchange123"

    def test_partitioner_encoding_is_utf8(self):
        """Partitioner keys should be UTF-8 encoded."""
        partitioner = SymbolPartitioner()
        msg = MockMessage("exchange", "BTC-USD")

        key = partitioner.get_partition_key(msg)

        # Should be valid UTF-8
        assert isinstance(key, bytes)
        try:
            key.decode("utf-8")
        except UnicodeDecodeError:
            pytest.fail("Partition key is not valid UTF-8")

    def test_partitioner_whitespace_in_exchange(self):
        """Partitioners should handle whitespace in exchange names."""
        partitioner = CompositePartitioner()
        msg = MockMessage("  coinbase  ", "BTC-USD")

        key = partitioner.get_partition_key(msg)

        # Should strip whitespace
        assert b"coinbase" in key

    def test_partitioner_whitespace_in_symbol(self):
        """Partitioners should handle whitespace in symbols."""
        partitioner = SymbolPartitioner()
        msg = MockMessage("coinbase", "  BTC-USD  ")

        key = partitioner.get_partition_key(msg)

        # Should strip whitespace
        assert key == b"btc-usd"


class TestPartitionerStatistics:
    """Statistical tests for partitioner behavior.

    Tests to verify partitioner properties like distribution uniformity.
    """

    def test_composite_partitioner_distribution(self):
        """CompositePartitioner should produce diverse keys for different pairs."""
        partitioner = CompositePartitioner()

        # Create diverse set of exchange-symbol pairs
        pairs = [
            ("coinbase", f"PAIR{i:03d}") for i in range(50)
        ] + [
            (f"EXCHANGE{i:02d}", "BTC-USD") for i in range(50)
        ]

        keys = set()
        for exchange, symbol in pairs:
            msg = MockMessage(exchange, symbol)
            key = partitioner.get_partition_key(msg)
            keys.add(key)

        # Should have high diversity
        assert len(keys) == len(pairs)

    def test_symbol_partitioner_different_exchanges_same_symbol(self):
        """Symbol partitioner should map different exchanges to same key."""
        partitioner = SymbolPartitioner()

        exchanges = ["coinbase", "binance", "kraken", "bitmex", "dydx"]
        keys = []

        for exchange in exchanges:
            msg = MockMessage(exchange, "BTC-USD")
            key = partitioner.get_partition_key(msg)
            keys.append(key)

        # All keys should be identical (same symbol)
        assert len(set(keys)) == 1
        assert keys[0] == b"btc-usd"

    def test_exchange_partitioner_different_symbols_same_exchange(self):
        """Exchange partitioner should map different symbols to same key."""
        partitioner = ExchangePartitioner()

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"]
        keys = []

        for symbol in symbols:
            msg = MockMessage("coinbase", symbol)
            key = partitioner.get_partition_key(msg)
            keys.append(key)

        # All keys should be identical (same exchange)
        assert len(set(keys)) == 1
        assert keys[0] == b"coinbase"
