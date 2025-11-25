"""
Integration tests for deprecation warning system with legacy Kafka backend classes.

This test suite validates that centralized deprecation warning system
is properly integrated with all legacy Kafka backend classes.
"""

import warnings
from pathlib import Path
import importlib.util

import pytest

# Import legacy classes from kafka.py file (not the package)
# The legacy classes are in cryptofeed/backends/kafka.py file, which is shadowed by the kafka package
REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_KAFKA_PATH = REPO_ROOT / "cryptofeed/backends/kafka.py"

spec = importlib.util.spec_from_file_location(
    "legacy_kafka",
    LEGACY_KAFKA_PATH,
)
legacy_kafka = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy_kafka)

TradeKafka = legacy_kafka.TradeKafka
BookKafka = legacy_kafka.BookKafka
TickerKafka = legacy_kafka.TickerKafka
FundingKafka = legacy_kafka.FundingKafka
OpenInterestKafka = legacy_kafka.OpenInterestKafka
LiquidationsKafka = legacy_kafka.LiquidationsKafka
CandlesKafka = legacy_kafka.CandlesKafka
OrderInfoKafka = legacy_kafka.OrderInfoKafka
TransactionsKafka = legacy_kafka.TransactionsKafka
BalancesKafka = legacy_kafka.BalancesKafka
FillsKafka = legacy_kafka.FillsKafka


class TestLegacyKafkaDeprecationIntegration:
    """Test integration of deprecation warning system with legacy Kafka classes."""

    @pytest.mark.parametrize(
        "klass,replacement",
        [
            (TradeKafka, "cryptofeed.backends.kafka.KafkaCallback"),
            (BookKafka, "cryptofeed.backends.kafka.KafkaCallback"),
            (TickerKafka, "cryptofeed.backends.kafka.KafkaCallback"),
            (FundingKafka, "cryptofeed.backends.kafka.KafkaCallback"),
            (OpenInterestKafka, "cryptofeed.backends.kafka.KafkaCallback"),
            (LiquidationsKafka, "cryptofeed.backends.kafka.KafkaCallback"),
            (CandlesKafka, "cryptofeed.backends.kafka.KafkaCallback"),
            (OrderInfoKafka, "cryptofeed.backends.kafka.KafkaCallback"),
            (TransactionsKafka, "cryptofeed.backends.kafka.KafkaCallback"),
            (BalancesKafka, "cryptofeed.backends.kafka.KafkaCallback"),
            (FillsKafka, "cryptofeed.backends.kafka.KafkaCallback"),
        ],
    )
    def test_legacy_class_emits_deprecation_warning(self, klass, replacement):
        """Test that all legacy Kafka classes emit deprecation warnings on instantiation."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            # Instantiate legacy class
            instance = klass()

            # Verify deprecation warning was emitted
            assert len(warning_list) == 1
            warning = warning_list[0]
            assert issubclass(warning.category, DeprecationWarning)

            # Check warning message contains class name and replacement
            warning_msg = str(warning.message)
            assert klass.__name__ in warning_msg
            assert replacement in warning_msg
            assert "deprecated" in warning_msg.lower()

            # Verify instance is created successfully
            assert instance is not None
            assert isinstance(instance, klass)

    def test_book_kafka_with_additional_parameters(self):
        """Test BookKafka with its specific parameters still works."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            # Test with BookKafka-specific parameters
            instance = BookKafka(snapshots_only=True, snapshot_interval=500)

            # Verify deprecation warning
            assert len(warning_list) == 1
            warning = warning_list[0]
            assert issubclass(warning.category, DeprecationWarning)
            assert "BookKafka" in str(warning.message)

            # Verify parameters are set correctly
            assert instance.snapshots_only is True
            assert instance.snapshot_interval == 500

    def test_multiple_instantiations_track_usage(self):
        """Test that multiple instantiations are tracked for usage analytics."""
        from cryptofeed.backends.kafka.maintenance import get_deprecation_warning_system

        warning_system = get_deprecation_warning_system()
        warning_system.reset_usage_stats()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for this test

            # Create multiple instances
            TradeKafka()
            TradeKafka()
            BookKafka()
            TickerKafka()

        # Check usage statistics
        usage_stats = warning_system.get_usage_stats()
        assert usage_stats.get("TradeKafka", 0) == 2
        assert usage_stats.get("BookKafka", 0) == 1
        assert usage_stats.get("TickerKafka", 0) == 1

    def test_warning_messages_are_consistent(self):
        """Test that warning messages are consistent across different classes."""
        messages = []

        for klass in [TradeKafka, BookKafka, TickerKafka]:
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                klass()
                messages.append(str(warning_list[0].message))

        # All messages should follow the same pattern
        for msg in messages:
            assert "is deprecated and will be removed in a future release" in msg
            assert "Use cryptofeed.backends.kafka.KafkaCallback instead" in msg
            assert "See migration guide:" in msg

    def test_warning_stacklevel_points_to_user_code(self):
        """Test that warning stacklevel points to user code, not internal code."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            # Create instance from user code perspective
            TradeKafka()

            warning = warning_list[0]
            # The warning should point to this test file, not internal implementation
            assert "test_legacy_kafka_deprecation_integration.py" in warning.filename

    def test_legacy_classes_maintain_functionality(self):
        """Test that legacy classes maintain their expected functionality."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test that classes can still be instantiated and configured
            trade_kafka = TradeKafka(key="custom_key")
            book_kafka = BookKafka(snapshots_only=True)

            # Verify attributes are set correctly
            assert trade_kafka.key == "custom_key"
            assert book_kafka.snapshots_only is True

            # Verify they have expected attributes
            assert hasattr(trade_kafka, "default_key")
            assert hasattr(book_kafka, "snapshot_interval")
