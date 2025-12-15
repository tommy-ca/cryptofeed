"""
Unit tests for normalization utility module.

Tests normalize_symbol() and normalize_exchange() functions covering all edge cases
including whitespace handling, separator replacement, case conversion, and None/empty
string fallbacks.
"""

import pytest


class TestNormalizeSymbol:
    """Test cases for normalize_symbol() function."""

    def test_slash_to_hyphen_lowercase(self):
        """Verify 'BTC/USD' converts to 'btc-usd'."""
        from cryptofeed.backends.kafka.normalization import normalize_symbol
        assert normalize_symbol("BTC/USD") == "btc-usd"

    def test_underscore_to_hyphen_lowercase(self):
        """Verify 'BTC_USD' converts to 'btc-usd'."""
        from cryptofeed.backends.kafka.normalization import normalize_symbol
        assert normalize_symbol("BTC_USD") == "btc-usd"

    def test_whitespace_stripping(self):
        """Verify ' ETH-BTC ' converts to 'eth-btc'."""
        from cryptofeed.backends.kafka.normalization import normalize_symbol
        assert normalize_symbol(" ETH-BTC ") == "eth-btc"

    def test_none_fallback(self):
        """Verify None converts to 'unknown'."""
        from cryptofeed.backends.kafka.normalization import normalize_symbol
        assert normalize_symbol(None) == "unknown"

    def test_empty_string_fallback(self):
        """Verify empty string converts to 'unknown'."""
        from cryptofeed.backends.kafka.normalization import normalize_symbol
        assert normalize_symbol("") == "unknown"

    def test_whitespace_only_fallback(self):
        """Verify whitespace-only string converts to 'unknown'."""
        from cryptofeed.backends.kafka.normalization import normalize_symbol
        assert normalize_symbol("  ") == "unknown"
        assert normalize_symbol("\t") == "unknown"
        assert normalize_symbol("\n") == "unknown"

    def test_mixed_separators(self):
        """Verify 'BTC/USD_PERP' converts to 'btc-usd-perp'."""
        from cryptofeed.backends.kafka.normalization import normalize_symbol
        assert normalize_symbol("BTC/USD_PERP") == "btc-usd-perp"

    def test_case_variations(self):
        """Verify all case variations normalize to lowercase."""
        from cryptofeed.backends.kafka.normalization import normalize_symbol
        assert normalize_symbol("btc-usd") == "btc-usd"
        assert normalize_symbol("BTC-USD") == "btc-usd"
        assert normalize_symbol("Btc-Usd") == "btc-usd"

    def test_already_normalized(self):
        """Verify already normalized symbols remain unchanged."""
        from cryptofeed.backends.kafka.normalization import normalize_symbol
        assert normalize_symbol("eth-btc") == "eth-btc"

    def test_complex_symbol(self):
        """Verify complex symbols with multiple separators."""
        from cryptofeed.backends.kafka.normalization import normalize_symbol
        assert normalize_symbol("BTC/USDT_PERP") == "btc-usdt-perp"
        assert normalize_symbol("BTC_USDT/PERP") == "btc-usdt-perp"


class TestNormalizeExchange:
    """Test cases for normalize_exchange() function."""

    def test_lowercase_conversion(self):
        """Verify 'Binance' converts to 'binance'."""
        from cryptofeed.backends.kafka.normalization import normalize_exchange
        assert normalize_exchange("Binance") == "binance"

    def test_whitespace_stripping(self):
        """Verify ' OKX ' converts to 'okx'."""
        from cryptofeed.backends.kafka.normalization import normalize_exchange
        assert normalize_exchange(" OKX ") == "okx"

    def test_uppercase_to_lowercase(self):
        """Verify 'COINBASE' converts to 'coinbase'."""
        from cryptofeed.backends.kafka.normalization import normalize_exchange
        assert normalize_exchange("COINBASE") == "coinbase"

    def test_none_fallback(self):
        """Verify None converts to 'unknown'."""
        from cryptofeed.backends.kafka.normalization import normalize_exchange
        assert normalize_exchange(None) == "unknown"

    def test_empty_string_fallback(self):
        """Verify empty string converts to 'unknown'."""
        from cryptofeed.backends.kafka.normalization import normalize_exchange
        assert normalize_exchange("") == "unknown"

    def test_whitespace_only_fallback(self):
        """Verify whitespace-only string converts to 'unknown'."""
        from cryptofeed.backends.kafka.normalization import normalize_exchange
        assert normalize_exchange("  ") == "unknown"
        assert normalize_exchange("\t") == "unknown"
        assert normalize_exchange("\n") == "unknown"

    def test_case_preservation(self):
        """Verify lowercase conversion works for all cases."""
        from cryptofeed.backends.kafka.normalization import normalize_exchange
        assert normalize_exchange("binance") == "binance"
        assert normalize_exchange("BINANCE") == "binance"
        assert normalize_exchange("BiNaNcE") == "binance"

    def test_already_normalized(self):
        """Verify already normalized exchanges remain unchanged."""
        from cryptofeed.backends.kafka.normalization import normalize_exchange
        assert normalize_exchange("kraken") == "kraken"


class TestNormalizationIdempotence:
    """Test idempotence property: normalize(normalize(x)) == normalize(x)."""

    def test_symbol_idempotence(self):
        """Verify normalize_symbol is idempotent."""
        from cryptofeed.backends.kafka.normalization import normalize_symbol
        result = normalize_symbol("BTC/USD")
        assert normalize_symbol(result) == result

    def test_exchange_idempotence(self):
        """Verify normalize_exchange is idempotent."""
        from cryptofeed.backends.kafka.normalization import normalize_exchange
        result = normalize_exchange("Binance")
        assert normalize_exchange(result) == result
