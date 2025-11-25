"""
Tests for the centralized deprecation warning system.

This test suite validates the DeprecationWarningSystem implementation
which provides centralized deprecation warning management for all
Kafka backend components.
"""

import logging
import warnings
import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from cryptofeed.backends.kafka.maintenance import (
    DeprecationWarningSystem,
    _resolve_user_stacklevel,
    get_deprecation_warning_system,
)


class TestDeprecationWarningSystem:
    """Test cases for DeprecationWarningSystem."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset the singleton instance for each test
        DeprecationWarningSystem._instance = None
        self.warning_system = DeprecationWarningSystem()

    def test_singleton_behavior(self):
        """Test that DeprecationWarningSystem follows singleton pattern."""
        instance1 = DeprecationWarningSystem()
        instance2 = DeprecationWarningSystem()
        assert instance1 is instance2

    def test_emit_class_warning_with_migration_guidance(self):
        """Test class deprecation warning emission with proper guidance."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            self.warning_system.emit_class_warning(
                class_name="TradeKafka",
                replacement="cryptofeed.backends.kafka.KafkaCallback",
                stacklevel=_resolve_user_stacklevel(),
            )

            assert len(warning_list) == 1
            warning = warning_list[0]
            assert issubclass(warning.category, DeprecationWarning)
            assert "TradeKafka is deprecated" in str(warning.message)
            assert "cryptofeed.backends.kafka.KafkaCallback" in str(warning.message)
            assert "migration" in str(warning.message).lower()

    def test_emit_import_warning_with_path_guidance(self):
        """Test import deprecation warning emission with path guidance."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            self.warning_system.emit_import_warning(
                old_path="cryptofeed.kafka_callback",
                new_path="cryptofeed.backends.kafka.callback",
                stacklevel=_resolve_user_stacklevel(),
            )

            assert len(warning_list) == 1
            warning = warning_list[0]
            assert issubclass(warning.category, DeprecationWarning)
            assert "cryptofeed.kafka_callback" in str(warning.message)
            assert "deprecated" in str(warning.message).lower()
            assert "cryptofeed.backends.kafka.callback" in str(warning.message)

    def test_track_usage_collects_metrics(self):
        """Test that usage tracking collects and stores metrics."""
        context = {"symbol": "BTC-USD", "exchange": "binance"}

        with patch.object(self.warning_system, "_log_usage") as mock_log:
            self.warning_system.track_usage("TradeKafka", context)

            # Verify that _log_usage is called with enriched context
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0]
            assert call_args[0] == "TradeKafka"
            assert call_args[1]["symbol"] == "BTC-USD"
            assert call_args[1]["exchange"] == "binance"
            assert "timestamp" in call_args[1]
            assert call_args[1]["component"] == "TradeKafka"

    def test_warning_messages_are_actionable(self):
        """Test that all warning messages provide actionable guidance."""
        test_cases = [
            ("TradeKafka", "cryptofeed.backends.kafka.KafkaCallback"),
            ("BookKafka", "cryptofeed.backends.kafka.KafkaCallback"),
            ("cryptofeed.kafka_callback", "cryptofeed.backends.kafka.callback"),
        ]

        for class_name, replacement in test_cases:
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")

                if "." in class_name:  # Import path
                    self.warning_system.emit_import_warning(
                        class_name, replacement, stacklevel=_resolve_user_stacklevel()
                    )
                else:  # Class name
                    self.warning_system.emit_class_warning(
                        class_name, replacement, stacklevel=_resolve_user_stacklevel()
                    )

                warning_msg = str(warning_list[0].message)

                # Check for actionable keywords
                actionable_keywords = ["use", "replace", "migrate", "import", "instead"]
                has_actionable = any(
                    keyword in warning_msg.lower() for keyword in actionable_keywords
                )
                assert has_actionable, (
                    f"Warning for {class_name} lacks actionable guidance: {warning_msg}"
                )

    def test_logging_integration(self):
        """Test integration with cryptofeed logging infrastructure."""
        with patch("cryptofeed.backends.kafka.maintenance.LOG") as mock_logger:
            self.warning_system.emit_class_warning(
                "TestKafka", "NewKafka", stacklevel=_resolve_user_stacklevel()
            )

            # Verify that logging is called
            mock_logger.warning.assert_called()

            # Check the log message contains relevant information
            log_call_args = mock_logger.warning.call_args[0][0]
            assert "TestKafka" in log_call_args
            assert "legacy" in log_call_args.lower()

    def test_usage_tracking_with_context(self):
        """Test usage tracking with various context types."""
        contexts = [
            {"symbol": "ETH-USD", "exchange": "coinbase"},
            {"config_file": "kafka.yaml", "version": "1.0"},
            {"user_agent": "cryptofeed/3.0", "timestamp": "2025-01-01"},
        ]

        for context in contexts:
            with patch.object(self.warning_system, "_log_usage") as mock_log:
                self.warning_system.track_usage("TestComponent", context)

                # Verify that _log_usage is called with enriched context
                mock_log.assert_called_once()
                call_args = mock_log.call_args[0]
                assert call_args[0] == "TestComponent"

                # Check that original context is preserved
                for key, value in context.items():
                    assert call_args[1][key] == value

                # Check that enrichment fields are added
                assert "timestamp" in call_args[1]
                assert call_args[1]["component"] == "TestComponent"

    def test_usage_report_contains_counts_and_last_context(self):
        """Ensure usage report includes counts and last seen context."""
        system = get_deprecation_warning_system()
        system.reset_usage_stats()

        system.track_usage("TradeKafka", {"symbol": "BTC-USD"})
        system.track_usage("TradeKafka", {"symbol": "ETH-USD"})

        report = system.get_usage_report()
        assert report["TradeKafka"]["count"] == 2
        assert report["TradeKafka"]["last_context"]["symbol"] == "ETH-USD"
        assert report["TradeKafka"]["last_timestamp"] is not None

    def test_warning_consistency_across_calls(self):
        """Test that warning messages remain consistent across multiple calls."""
        messages = []

        for _ in range(3):
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")

                self.warning_system.emit_class_warning(
                    "TradeKafka",
                    "cryptofeed.backends.kafka.KafkaCallback",
                    stacklevel=_resolve_user_stacklevel(),
                )

                messages.append(str(warning_list[0].message))

        # All messages should be identical
        assert all(msg == messages[0] for msg in messages)

    def test_warning_system_with_different_components(self):
        """Test warning system works with different Kafka components."""
        components = [
            ("TradeKafka", "cryptofeed.backends.kafka.KafkaCallback"),
            ("BookKafka", "cryptofeed.backends.kafka.KafkaCallback"),
            ("FundingKafka", "cryptofeed.backends.kafka.KafkaCallback"),
        ]

        for class_name, replacement in components:
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")

                self.warning_system.emit_class_warning(
                    class_name, replacement, stacklevel=_resolve_user_stacklevel()
                )

                assert len(warning_list) == 1
                warning = warning_list[0]
                assert class_name in str(warning.message)
                assert replacement in str(warning.message)

    @patch("cryptofeed.backends.kafka.maintenance.time.time")
    def test_usage_tracking_includes_timestamp(self, mock_time):
        """Test that usage tracking includes timestamp information."""
        mock_time.return_value = 1640995200.0  # Fixed timestamp

        with patch.object(self.warning_system, "_log_usage") as mock_log:
            context = {"symbol": "BTC-USD"}
            self.warning_system.track_usage("TradeKafka", context)

            # Verify that the context is enriched with timestamp
            call_args = mock_log.call_args[0]
            component = call_args[0]
            enriched_context = call_args[1]

            assert component == "TradeKafka"
            assert "timestamp" in enriched_context
            assert enriched_context["timestamp"] == 1640995200.0
            assert enriched_context["symbol"] == "BTC-USD"  # Original context preserved
