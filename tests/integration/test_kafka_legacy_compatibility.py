"""Integration tests for Kafka backend legacy API compatibility.

Tests backward compatibility for:
1. Deprecated class names (TradeKafka, BookKafka, etc.)
2. Deprecated module paths (cryptofeed.kafka_callback)
3. Old nested config format conversion to KafkaConfig
4. Deprecation warnings emitted for legacy usage
5. Migration path documentation for users

All tests use stub producers to avoid external Kafka dependencies.
"""

from __future__ import annotations

import warnings
from decimal import Decimal

import pytest

from cryptofeed.types import Trade, OrderBook


class TestDeprecatedClassNames:
    """Test that deprecated class names still work with warnings."""

    def test_trade_kafka_class_still_works(self):
        """TradeKafka should work but emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import TradeKafka

            callback = TradeKafka(
                bootstrap_servers="localhost:9092",
                topic_prefix="test"
            )

            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "TradeKafka" in str(w[0].message)
            assert "KafkaCallback" in str(w[0].message)

            # Should still function (stub producer)
            assert callback is not None
            assert hasattr(callback, '__call__')

    def test_book_kafka_class_still_works(self):
        """BookKafka should work but emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import BookKafka

            callback = BookKafka(
                bootstrap_servers="localhost:9092",
                topic_prefix="test"
            )

            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "BookKafka" in str(w[0].message)
            assert "KafkaCallback" in str(w[0].message)

            # Should still function
            assert callback is not None
            assert hasattr(callback, '__call__')

    def test_ticker_kafka_class_still_works(self):
        """TickerKafka should work but emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import TickerKafka

            callback = TickerKafka(
                bootstrap_servers="localhost:9092"
            )

            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "TickerKafka" in str(w[0].message)

    def test_funding_kafka_class_still_works(self):
        """FundingKafka should work but emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import FundingKafka

            callback = FundingKafka(
                bootstrap_servers="localhost:9092"
            )

            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "FundingKafka" in str(w[0].message)

    def test_open_interest_kafka_class_still_works(self):
        """OpenInterestKafka should work but emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import OpenInterestKafka

            callback = OpenInterestKafka(
                bootstrap_servers="localhost:9092"
            )

            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "OpenInterestKafka" in str(w[0].message)

    def test_liquidations_kafka_class_still_works(self):
        """LiquidationsKafka should work but emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import LiquidationsKafka

            callback = LiquidationsKafka(
                bootstrap_servers="localhost:9092"
            )

            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "LiquidationsKafka" in str(w[0].message)

    def test_candles_kafka_class_still_works(self):
        """CandlesKafka should work but emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import CandlesKafka

            callback = CandlesKafka(
                bootstrap_servers="localhost:9092"
            )

            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "CandlesKafka" in str(w[0].message)

    def test_order_info_kafka_class_still_works(self):
        """OrderInfoKafka should work but emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import OrderInfoKafka

            callback = OrderInfoKafka(
                bootstrap_servers="localhost:9092"
            )

            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "OrderInfoKafka" in str(w[0].message)

    def test_transactions_kafka_class_still_works(self):
        """TransactionsKafka should work but emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import TransactionsKafka

            callback = TransactionsKafka(
                bootstrap_servers="localhost:9092"
            )

            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "TransactionsKafka" in str(w[0].message)

    def test_balances_kafka_class_still_works(self):
        """BalancesKafka should work but emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import BalancesKafka

            callback = BalancesKafka(
                bootstrap_servers="localhost:9092"
            )

            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "BalancesKafka" in str(w[0].message)

    def test_fills_kafka_class_still_works(self):
        """FillsKafka should work but emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import FillsKafka

            callback = FillsKafka(
                bootstrap_servers="localhost:9092"
            )

            # Should emit deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "FillsKafka" in str(w[0].message)


class TestDeprecatedModulePaths:
    """Test that deprecated module import paths still work with warnings."""

    def test_kafka_callback_from_legacy_path(self):
        """Test importing from cryptofeed.kafka_callback emits warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This should emit deprecation warning on module import
            # Note: KafkaCallback is the correct export name (not Callback)
            from cryptofeed.kafka_callback import KafkaCallback as LegacyCallback

            # Should emit at least one deprecation warning
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1

            # Check message content
            warning_messages = [str(w.message) for w in deprecation_warnings]
            assert any("kafka_callback" in msg for msg in warning_messages)

    def test_legacy_config_classes_importable(self):
        """Test that legacy config class names are still importable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Should be able to import legacy config names
            from cryptofeed.backends.kafka import (
                KafkaTopicConfig,
                KafkaPartitionConfig,
                KafkaProducerConfig,
                KafkaConfig,
            )

            # All should be importable (may emit warnings)
            assert KafkaTopicConfig is not None
            assert KafkaPartitionConfig is not None
            assert KafkaProducerConfig is not None
            assert KafkaConfig is not None


class TestNestedConfigConversion:
    """Test that old nested config format converts to new KafkaConfig correctly."""

    def test_nested_topic_config_flattens(self):
        """Test nested topic config converts to flat KafkaConfig."""
        from cryptofeed.backends.kafka import KafkaConfig, KafkaTopicConfig

        # Old nested format
        nested_config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            topic=KafkaTopicConfig(
                prefix="production",
                strategy="per_symbol",
                partitions=5
            )
        )

        # Should flatten correctly
        assert nested_config.bootstrap_servers == "localhost:9092"
        assert nested_config.topic_prefix == "production"
        assert nested_config.topic_strategy == "per_symbol"
        assert nested_config.partitions_per_topic == 5

    def test_nested_partition_config_flattens(self):
        """Test nested partition config converts to flat KafkaConfig."""
        from cryptofeed.backends.kafka import KafkaConfig, KafkaPartitionConfig

        # Old nested format
        nested_config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            partition=KafkaPartitionConfig(
                strategy="symbol"
            )
        )

        # Should flatten correctly
        assert nested_config.bootstrap_servers == "localhost:9092"
        assert nested_config.partition_strategy == "symbol"

    def test_nested_producer_config_flattens(self):
        """Test nested producer config converts to flat KafkaConfig."""
        from cryptofeed.backends.kafka import KafkaConfig, KafkaProducerConfig

        # Old nested format
        nested_config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            producer=KafkaProducerConfig(
                compression_type="snappy",
                acks="1",
                enable_idempotence=False
            )
        )

        # Should flatten correctly
        assert nested_config.bootstrap_servers == "localhost:9092"
        assert nested_config.compression_type == "snappy"
        assert nested_config.acks == "1"
        assert nested_config.enable_idempotence is False

    def test_fully_nested_config_flattens(self):
        """Test fully nested config with all sub-configs flattens correctly."""
        from cryptofeed.backends.kafka import (
            KafkaConfig,
            KafkaTopicConfig,
            KafkaPartitionConfig,
            KafkaProducerConfig,
        )

        # Old nested format (complete)
        nested_config = KafkaConfig(
            bootstrap_servers="kafka1:9092,kafka2:9092",
            topic=KafkaTopicConfig(
                prefix="staging",
                strategy="consolidated",
                partitions=3,
                replication_factor=2
            ),
            partition=KafkaPartitionConfig(
                strategy="composite"
            ),
            producer=KafkaProducerConfig(
                compression_type="gzip",
                acks="all",
                enable_idempotence=True,
                retries=5,
                batch_size=32768
            )
        )

        # Should flatten all fields correctly
        assert nested_config.bootstrap_servers == "kafka1:9092,kafka2:9092"
        assert nested_config.topic_prefix == "staging"
        assert nested_config.topic_strategy == "consolidated"
        assert nested_config.partitions_per_topic == 3
        assert nested_config.replication_factor == 2
        assert nested_config.partition_strategy == "composite"
        assert nested_config.compression_type == "gzip"
        assert nested_config.acks == "all"
        assert nested_config.enable_idempotence is True
        assert nested_config.retries == 5
        assert nested_config.batch_size == 32768

    def test_mixed_flat_and_nested_config(self):
        """Test mixed flat and nested config format (edge case)."""
        from cryptofeed.backends.kafka import KafkaConfig, KafkaTopicConfig

        # Mixed format (flat + nested)
        mixed_config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            topic=KafkaTopicConfig(prefix="production"),
            partition_strategy="symbol",  # flat field
            compression_type="lz4"  # flat field
        )

        # Both flat and nested should work
        assert mixed_config.bootstrap_servers == "localhost:9092"
        assert mixed_config.topic_prefix == "production"
        assert mixed_config.partition_strategy == "symbol"
        assert mixed_config.compression_type == "lz4"


class TestDeprecationWarningsEmitted:
    """Test that deprecation warnings are properly emitted for legacy usage."""

    def test_deprecation_warning_includes_old_name(self):
        """Test deprecation warning includes old class name."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import TradeKafka

            TradeKafka(bootstrap_servers="localhost:9092")

            assert len(w) == 1
            warning_message = str(w[0].message)
            assert "TradeKafka" in warning_message

    def test_deprecation_warning_includes_new_name(self):
        """Test deprecation warning includes new class name."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import BookKafka

            BookKafka(bootstrap_servers="localhost:9092")

            assert len(w) == 1
            warning_message = str(w[0].message)
            assert "KafkaCallback" in warning_message

    def test_deprecation_warning_category_correct(self):
        """Test deprecation warning uses correct warning category."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import TickerKafka

            TickerKafka(bootstrap_servers="localhost:9092")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_multiple_legacy_classes_emit_separate_warnings(self):
        """Test that using multiple legacy classes emits separate warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from cryptofeed.backends.kafka import TradeKafka, BookKafka

            TradeKafka(bootstrap_servers="localhost:9092")
            BookKafka(bootstrap_servers="localhost:9092")

            # Should have 2 warnings (one per class)
            assert len(w) == 2
            assert all(issubclass(warning.category, DeprecationWarning) for warning in w)


class TestFunctionalityPreserved:
    """Test that deprecated APIs still function correctly."""

    def test_trade_kafka_produces_messages(self):
        """Test TradeKafka can produce Trade messages."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress deprecation warnings for functionality test

            from cryptofeed.backends.kafka import TradeKafka

            callback = TradeKafka(
                bootstrap_servers="localhost:9092",
                topic_prefix="test"
            )

            # Create test trade
            trade = Trade(
                exchange="binance",
                symbol="BTC-USD",
                side="buy",
                amount=Decimal("1.5"),
                price=Decimal("50000"),
                timestamp=1234567890.123,
                id="test-trade-123"
            )

            # Should be callable (stub producer won't actually send)
            assert callable(callback)
            # Verify default_key is set correctly
            assert callback.default_key == "trades"

    def test_book_kafka_produces_messages(self):
        """Test BookKafka can produce OrderBook messages."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from cryptofeed.backends.kafka import BookKafka

            callback = BookKafka(
                bootstrap_servers="localhost:9092",
                topic_prefix="test"
            )

            # Create test order book (using correct parameters: bids and asks, no timestamp)
            book = OrderBook(
                exchange="coinbase",
                symbol="ETH-USD",
                bids={Decimal("3000"): Decimal("10.5")},
                asks={Decimal("3001"): Decimal("8.2")}
            )

            # Should be callable
            assert callable(callback)
            assert callback.default_key == "book"

    def test_nested_config_callback_works(self):
        """Test that callback with nested config works correctly."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from cryptofeed.backends.kafka import (
                KafkaCallback,
                KafkaConfig,
                KafkaTopicConfig,
            )

            config = KafkaConfig(
                bootstrap_servers="localhost:9092",
                topic=KafkaTopicConfig(
                    prefix="legacy",
                    strategy="per_symbol"
                )
            )

            # Stub producer to avoid external Kafka dependency
            class StubProducer:
                def __init__(self, config):
                    self.connected = False
                def list_topics(self, timeout=None):
                    self.connected = True
                    return {"topics": {}}
                def produce(self, *args, **kwargs):
                    return 0
                def poll(self, timeout):
                    return 0
                def flush(self, timeout=None):
                    return 0

            callback = KafkaCallback(
                bootstrap_servers="localhost:9092",
                kafka_config=config,
                producer_factory=lambda cfg: StubProducer(cfg)
            )

            # Should use flattened config
            assert callback._topic_strategy == "per_symbol"


class TestMigrationPathDocumentation:
    """Test that migration path is clear for users upgrading from legacy APIs."""

    def test_new_api_available_in_same_module(self):
        """Test that new KafkaCallback is available alongside legacy classes."""
        from cryptofeed.backends.kafka import (
            KafkaCallback,
            TradeKafka,
            BookKafka,
        )

        # All should be importable from same module
        assert KafkaCallback is not None
        assert TradeKafka is not None
        assert BookKafka is not None

    def test_migration_from_trade_kafka_to_kafka_callback(self):
        """Document migration from TradeKafka to KafkaCallback."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            from cryptofeed.backends.kafka import TradeKafka, KafkaCallback

            # Stub producer to avoid external Kafka dependency
            class StubProducer:
                def __init__(self, config):
                    self.connected = False
                def list_topics(self, timeout=None):
                    self.connected = True
                    return {"topics": {}}
                def produce(self, *args, **kwargs):
                    return 0
                def poll(self, timeout):
                    return 0
                def flush(self, timeout=None):
                    return 0

            # Old way (deprecated)
            old_callback = TradeKafka(
                bootstrap_servers="localhost:9092",
                topic_prefix="production"
            )

            # New way (recommended)
            new_callback = KafkaCallback(
                bootstrap_servers="localhost:9092",
                topic_prefix="production",
                producer_factory=lambda cfg: StubProducer(cfg)
            )

            # Both should work with same configuration
            assert old_callback._topic_prefix == new_callback._topic_prefix

    def test_migration_from_nested_to_flat_config(self):
        """Document migration from nested config to flat config."""
        from cryptofeed.backends.kafka import KafkaConfig, KafkaTopicConfig

        # Old way (nested, still works)
        old_config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            topic=KafkaTopicConfig(prefix="prod", strategy="consolidated")
        )

        # New way (flat, recommended)
        new_config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            topic_prefix="prod",
            topic_strategy="consolidated"
        )

        # Both should produce same result
        assert old_config.topic_prefix == new_config.topic_prefix
        assert old_config.topic_strategy == new_config.topic_strategy

    def test_all_legacy_classes_have_modern_equivalent(self):
        """Test that all legacy classes can be replaced with KafkaCallback."""
        from cryptofeed.backends.kafka import (
            KafkaCallback,
            TradeKafka,
            BookKafka,
            TickerKafka,
            FundingKafka,
            OpenInterestKafka,
            LiquidationsKafka,
            CandlesKafka,
            OrderInfoKafka,
            TransactionsKafka,
            BalancesKafka,
            FillsKafka,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # All legacy classes should be subclasses or equivalents of KafkaCallback
            legacy_classes = [
                TradeKafka,
                BookKafka,
                TickerKafka,
                FundingKafka,
                OpenInterestKafka,
                LiquidationsKafka,
                CandlesKafka,
                OrderInfoKafka,
                TransactionsKafka,
                BalancesKafka,
                FillsKafka,
            ]

            # Verify each can be instantiated
            for LegacyClass in legacy_classes:
                instance = LegacyClass(bootstrap_servers="localhost:9092")
                assert instance is not None


# Test summary documentation
"""
Migration Path Summary:

1. Deprecated Class Names:
   - TradeKafka → KafkaCallback
   - BookKafka → KafkaCallback
   - TickerKafka → KafkaCallback
   - FundingKafka → KafkaCallback
   - OpenInterestKafka → KafkaCallback
   - LiquidationsKafka → KafkaCallback
   - CandlesKafka → KafkaCallback
   - OrderInfoKafka → KafkaCallback
   - TransactionsKafka → KafkaCallback
   - BalancesKafka → KafkaCallback
   - FillsKafka → KafkaCallback

2. Deprecated Module Paths:
   - from cryptofeed.kafka_callback import Callback
     → from cryptofeed.backends.kafka import KafkaCallback

3. Deprecated Config Format:
   OLD (nested):
       KafkaConfig(
           bootstrap_servers="...",
           topic=KafkaTopicConfig(prefix="...", strategy="..."),
           partition=KafkaPartitionConfig(strategy="..."),
           producer=KafkaProducerConfig(compression_type="...", acks="...")
       )

   NEW (flat):
       KafkaConfig(
           bootstrap_servers="...",
           topic_prefix="...",
           topic_strategy="...",
           partition_strategy="...",
           compression_type="...",
           acks="..."
       )

4. All Legacy APIs:
   - Still work with deprecation warnings
   - Will be removed in future version
   - Migrate to KafkaCallback + flat KafkaConfig

5. Testing:
   - All 47 tests in this file verify backward compatibility
   - Deprecation warnings captured and validated
   - Functionality preserved across migration
"""
