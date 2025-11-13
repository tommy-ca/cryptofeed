"""Unit tests for Critical Issue #1: TopicManager data type normalization.

Verifies that callback method names (plural, underscored) are correctly mapped
to normalized topic names (singular, no underscores) for TopicManager validation.
"""

import pytest

from cryptofeed.kafka_callback import TopicManager, _SUPPORTED_METHODS


class TestDataTypeNormalization:
    """Test data type normalization from callback methods to topic names."""

    def test_all_supported_methods_map_to_valid_topic_types(self):
        """All _SUPPORTED_METHODS values must be in TopicManager.SUPPORTED_DATA_TYPES."""
        for method_name, topic_type in _SUPPORTED_METHODS.items():
            assert topic_type in TopicManager.SUPPORTED_DATA_TYPES, (
                f"Method '{method_name}' maps to topic type '{topic_type}', "
                f"but '{topic_type}' is not in TopicManager.SUPPORTED_DATA_TYPES. "
                f"This will cause silent fallback to legacy topic builder."
            )

    def test_plural_methods_map_to_singular_topics(self):
        """Plural callback method names should map to singular topic names."""
        plural_mappings = {
            "balances": "balance",
            "transactions": "transaction",
            "fills": "fill",
        }

        for method, expected_topic in plural_mappings.items():
            assert _SUPPORTED_METHODS[method] == expected_topic, (
                f"Plural method '{method}' should map to singular '{expected_topic}'"
            )

    def test_underscored_methods_map_to_normalized_topics(self):
        """Methods with underscores should map to normalized topic names."""
        underscored_mappings = {
            "open_interest": "openinterest",  # underscore removed
            "order_info": "order",            # simplified
        }

        for method, expected_topic in underscored_mappings.items():
            assert _SUPPORTED_METHODS[method] == expected_topic, (
                f"Underscored method '{method}' should map to '{expected_topic}'"
            )

    def test_topic_manager_accepts_all_normalized_types(self):
        """TopicManager.get_topic() should accept all normalized data types."""
        for method_name, topic_type in _SUPPORTED_METHODS.items():
            # Should not raise ValueError
            topic = TopicManager.get_topic(
                data_type=topic_type,
                symbol="BTC-USDT",
                exchange="binance",
                strategy="consolidated"
            )

            assert topic == f"cryptofeed.{topic_type}", (
                f"Method '{method_name}' -> topic type '{topic_type}' "
                f"should generate topic 'cryptofeed.{topic_type}'"
            )

    def test_no_silent_fallback_to_legacy_topic_builder(self):
        """All callback methods should use TopicManager, not legacy fallback."""
        # This test ensures Critical Issue #1 is fixed
        # If a topic type is not in SUPPORTED_DATA_TYPES, TopicManager.get_topic() raises ValueError
        # The fallback in _topic_name() would catch this and use legacy naming

        for method_name, topic_type in _SUPPORTED_METHODS.items():
            # Validate the topic type is supported (should not raise)
            try:
                TopicManager.validate_data_type(topic_type)
            except ValueError as e:
                pytest.fail(
                    f"Method '{method_name}' maps to unsupported topic type '{topic_type}': {e}. "
                    f"This would trigger silent fallback to legacy topic naming."
                )

    def test_all_data_types_have_callback_methods(self):
        """All TopicManager.SUPPORTED_DATA_TYPES should have corresponding callback methods."""
        topic_types = set(TopicManager.SUPPORTED_DATA_TYPES)
        mapped_types = set(_SUPPORTED_METHODS.values())

        # Not all topic types need callback methods (e.g., 'margin', 'position' may not have callbacks yet)
        # But all mapped types should be supported
        assert mapped_types.issubset(topic_types), (
            f"Some callback methods map to unsupported topic types: "
            f"{mapped_types - topic_types}"
        )

    def test_consolidated_topic_naming_for_all_types(self):
        """Consolidated strategy should work for all data types."""
        for method_name, topic_type in _SUPPORTED_METHODS.items():
            topic = TopicManager.get_topic(
                data_type=topic_type,
                symbol="ETH-USD",
                exchange="coinbase",
                strategy="consolidated",
                prefix=None
            )

            # Consolidated topics should be: cryptofeed.{data_type}
            expected = f"cryptofeed.{topic_type}"
            assert topic == expected, (
                f"Method '{method_name}' with consolidated strategy should produce '{expected}', "
                f"got '{topic}'"
            )

    def test_per_symbol_topic_naming_for_all_types(self):
        """Per-symbol strategy should work for all data types."""
        for method_name, topic_type in _SUPPORTED_METHODS.items():
            topic = TopicManager.get_topic(
                data_type=topic_type,
                symbol="BTC-USDT",
                exchange="binance",
                strategy="per_symbol",
                prefix=None
            )

            # Per-symbol topics should be: cryptofeed.{data_type}.{exchange}.{symbol}
            expected = f"cryptofeed.{topic_type}.binance.btc-usdt"
            assert topic == expected, (
                f"Method '{method_name}' with per_symbol strategy should produce '{expected}', "
                f"got '{topic}'"
            )


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_trade_method_still_exists(self):
        """The 'trade' method should still be accessible."""
        assert "trade" in _SUPPORTED_METHODS
        assert _SUPPORTED_METHODS["trade"] == "trade"

    def test_orderbook_method_still_exists(self):
        """The 'orderbook' method should still be accessible."""
        assert "orderbook" in _SUPPORTED_METHODS
        assert _SUPPORTED_METHODS["orderbook"] == "orderbook"

    def test_ticker_method_still_exists(self):
        """The 'ticker' method should still be accessible."""
        assert "ticker" in _SUPPORTED_METHODS
        assert _SUPPORTED_METHODS["ticker"] == "ticker"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
