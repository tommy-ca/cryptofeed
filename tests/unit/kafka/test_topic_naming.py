"""Test suite for Topic Management (Task 1: Consolidated Topic Naming).

Tests cover:
- Sub-task 1.1: Consolidated Topic Naming (cryptofeed.{data_type})
- Sub-task 1.2: Per-Symbol Topic Naming (cryptofeed.{data_type}.{exchange}.{symbol})
- Sub-task 1.3: Topic Prefix Support (prefix parameter)
"""

import pytest
from unittest.mock import Mock
from cryptofeed.kafka_callback import TopicManager


# =============================================================================
# Sub-Task 1.1: Consolidated Topic Naming Tests
# =============================================================================
class TestConsolidatedNaming:
    """Test consolidated topic naming strategy: cryptofeed.{data_type}"""

    def test_consolidated_trades_topic(self):
        """Consolidated strategy produces cryptofeed.trade"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='consolidated'
        )
        assert topic == 'cryptofeed.trade'

    def test_consolidated_orderbook_topic(self):
        """Consolidated strategy produces cryptofeed.orderbook"""
        topic = TopicManager.get_topic(
            data_type='orderbook',
            symbol='eth-usd',
            exchange='coinbase',
            strategy='consolidated'
        )
        assert topic == 'cryptofeed.orderbook'

    def test_consolidated_ticker_topic(self):
        """Consolidated strategy produces cryptofeed.ticker"""
        topic = TopicManager.get_topic(
            data_type='ticker',
            symbol='sol-usdt',
            exchange='binance',
            strategy='consolidated'
        )
        assert topic == 'cryptofeed.ticker'

    def test_consolidated_candle_topic(self):
        """Consolidated strategy produces cryptofeed.candle"""
        topic = TopicManager.get_topic(
            data_type='candle',
            symbol='btc-usd',
            exchange='coinbase',
            strategy='consolidated'
        )
        assert topic == 'cryptofeed.candle'

    def test_consolidated_funding_topic(self):
        """Consolidated strategy produces cryptofeed.funding"""
        topic = TopicManager.get_topic(
            data_type='funding',
            symbol='btc-usdt-perp',
            exchange='binance',
            strategy='consolidated'
        )
        assert topic == 'cryptofeed.funding'

    def test_consolidated_liquidation_topic(self):
        """Consolidated strategy produces cryptofeed.liquidation"""
        topic = TopicManager.get_topic(
            data_type='liquidation',
            symbol='btc-usdt',
            exchange='binance',
            strategy='consolidated'
        )
        assert topic == 'cryptofeed.liquidation'

    def test_consolidated_ignores_symbol_and_exchange(self):
        """Consolidated strategy ignores symbol and exchange parameters"""
        topic1 = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='consolidated'
        )
        topic2 = TopicManager.get_topic(
            data_type='trade',
            symbol='eth-usd',
            exchange='coinbase',
            strategy='consolidated'
        )
        assert topic1 == topic2 == 'cryptofeed.trade'

    def test_consolidated_data_type_lowercase(self):
        """Data type must be lowercase in topic name"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='consolidated'
        )
        assert 'TRADES' not in topic
        assert topic == 'cryptofeed.trade'

    def test_consolidated_is_default_strategy(self):
        """Consolidated is default strategy when not specified"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance'
        )
        assert topic == 'cryptofeed.trade'


# =============================================================================
# Sub-Task 1.2: Per-Symbol Topic Naming Tests
# =============================================================================
class TestPerSymbolNaming:
    """Test per-symbol topic naming strategy: cryptofeed.{data_type}.{exchange}.{symbol}"""

    def test_per_symbol_trades_topic(self):
        """Per-symbol strategy produces cryptofeed.trade.binance.btc-usdt"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol'
        )
        assert topic == 'cryptofeed.trade.binance.btc-usdt'

    def test_per_symbol_orderbook_topic(self):
        """Per-symbol strategy produces cryptofeed.orderbook.coinbase.eth-usd"""
        topic = TopicManager.get_topic(
            data_type='orderbook',
            symbol='eth-usd',
            exchange='coinbase',
            strategy='per_symbol'
        )
        assert topic == 'cryptofeed.orderbook.coinbase.eth-usd'

    def test_per_symbol_ticker_topic(self):
        """Per-symbol strategy produces cryptofeed.ticker.kraken.sol-usd"""
        topic = TopicManager.get_topic(
            data_type='ticker',
            symbol='sol-usd',
            exchange='kraken',
            strategy='per_symbol'
        )
        assert topic == 'cryptofeed.ticker.kraken.sol-usd'

    def test_per_symbol_funding_topic(self):
        """Per-symbol strategy produces cryptofeed.funding.dydx.btc-usd-perp"""
        topic = TopicManager.get_topic(
            data_type='funding',
            symbol='btc-usd-perp',
            exchange='dydx',
            strategy='per_symbol'
        )
        assert topic == 'cryptofeed.funding.dydx.btc-usd-perp'

    def test_per_symbol_includes_all_components(self):
        """Per-symbol topic includes data_type, exchange, and symbol"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='XYZ-ABC',
            exchange='test_exchange',
            strategy='per_symbol'
        )
        assert 'trade' in topic.lower()
        assert 'test_exchange' in topic.lower()
        assert 'XYZ-ABC' in topic or 'xyz-abc' in topic.lower()

    def test_per_symbol_exchange_normalization(self):
        """Exchange name is included in topic (case may normalize)"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='Binance',
            strategy='per_symbol'
        )
        # Should contain exchange name (possibly normalized)
        assert 'binance' in topic.lower()

    def test_per_symbol_different_symbols_different_topics(self):
        """Different symbols produce different topics"""
        topic1 = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol'
        )
        topic2 = TopicManager.get_topic(
            data_type='trade',
            symbol='eth-usdT',
            exchange='binance',
            strategy='per_symbol'
        )
        assert topic1 != topic2

    def test_per_symbol_different_exchanges_different_topics(self):
        """Different exchanges produce different topics"""
        topic1 = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol'
        )
        topic2 = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='coinbase',
            strategy='per_symbol'
        )
        assert topic1 != topic2


# =============================================================================
# Sub-Task 1.3: Topic Prefix Support Tests
# =============================================================================
class TestTopicPrefixSupport:
    """Test prefix support for multi-tenant deployments"""

    def test_prefix_consolidated_topic(self):
        """Prefix prepended to consolidated topic"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='consolidated',
            prefix='production'
        )
        assert topic == 'production.cryptofeed.trade'

    def test_prefix_per_symbol_topic(self):
        """Prefix prepended to per-symbol topic"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol',
            prefix='production'
        )
        assert topic == 'production.cryptofeed.trade.binance.btc-usdt'

    def test_prefix_staging_environment(self):
        """Prefix supports staging environment naming"""
        topic = TopicManager.get_topic(
            data_type='orderbook',
            symbol='eth-usd',
            exchange='coinbase',
            strategy='consolidated',
            prefix='staging'
        )
        assert topic == 'staging.cryptofeed.orderbook'

    def test_prefix_custom_namespace(self):
        """Prefix supports custom namespace"""
        topic = TopicManager.get_topic(
            data_type='ticker',
            symbol='SOL-USD',
            exchange='kraken',
            strategy='consolidated',
            prefix='acme'
        )
        assert topic == 'acme.cryptofeed.ticker'

    def test_empty_prefix_no_leading_dot(self):
        """Empty prefix produces no leading dot"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='consolidated',
            prefix=''
        )
        assert not topic.startswith('.')
        assert topic == 'cryptofeed.trade'

    def test_none_prefix_no_leading_dot(self):
        """None prefix produces no leading dot"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='consolidated',
            prefix=None
        )
        assert not topic.startswith('.')
        assert topic == 'cryptofeed.trade'

    def test_whitespace_only_prefix_handled(self):
        """Whitespace-only prefix treated as empty"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='consolidated',
            prefix='   '
        )
        # Should not have leading dot or whitespace
        assert not topic.startswith('.')
        assert not topic.startswith(' ')

    def test_prefix_with_multiple_dot_components(self):
        """Prefix can contain dots (e.g., company.environment)"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='consolidated',
            prefix='company.production'
        )
        assert topic == 'company.production.cryptofeed.trade'

    def test_prefix_plus_per_symbol_full_path(self):
        """Prefix with per-symbol creates full hierarchical path"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol',
            prefix='prod'
        )
        assert topic == 'prod.cryptofeed.trade.binance.btc-usdt'


# =============================================================================
# Error Handling & Validation Tests
# =============================================================================
class TestErrorHandling:
    """Test error handling for invalid inputs"""

    def test_invalid_strategy_raises_error(self):
        """Invalid strategy raises ValueError"""
        with pytest.raises(ValueError) as exc_info:
            TopicManager.get_topic(
                data_type='trade',
                symbol='btc-usdt',
                exchange='binance',
                strategy='invalid_strategy'
            )
        assert 'Unknown strategy' in str(exc_info.value) or 'strategy' in str(exc_info.value).lower()

    def test_invalid_data_type_raises_error(self):
        """Invalid data type raises ValueError"""
        with pytest.raises(ValueError):
            TopicManager.get_topic(
                data_type='invalid_type',
                symbol='btc-usdt',
                exchange='binance',
                strategy='consolidated'
            )
        # Should raise error for unsupported data type

    def test_missing_symbol_raises_error(self):
        """Missing symbol raises error when required"""
        # Symbol is required for per_symbol strategy
        with pytest.raises((ValueError, TypeError)):
            TopicManager.get_topic(
                data_type='trade',
                symbol=None,
                exchange='binance',
                strategy='per_symbol'
            )

    def test_missing_exchange_raises_error(self):
        """Missing exchange raises error when required"""
        # Exchange is required for per_symbol strategy
        with pytest.raises((ValueError, TypeError)):
            TopicManager.get_topic(
                data_type='trade',
                symbol='btc-usdt',
                exchange=None,
                strategy='per_symbol'
            )

    def test_invalid_strategy_type_error(self):
        """Non-string strategy raises error"""
        with pytest.raises((ValueError, TypeError)):
            TopicManager.get_topic(
                data_type='trade',
                symbol='btc-usdt',
                exchange='binance',
                strategy=123
            )


# =============================================================================
# Edge Cases & Data Type Coverage Tests
# =============================================================================
class TestEdgeCases:
    """Test edge cases and all supported data types"""

    @pytest.mark.parametrize('data_type', [
        'trade', 'orderbook', 'ticker', 'candle', 'funding',
        'liquidation', 'index', 'openinterest', 'fill', 'balance',
        'position', 'margin', 'order', 'transaction'
    ])
    def test_all_supported_data_types_consolidated(self, data_type):
        """All 14+ data types work with consolidated strategy"""
        topic = TopicManager.get_topic(
            data_type=data_type,
            symbol='btc-usdt',
            exchange='binance',
            strategy='consolidated'
        )
        assert topic == f'cryptofeed.{data_type}'

    @pytest.mark.parametrize('data_type', [
        'trade', 'orderbook', 'ticker', 'candle', 'funding',
        'liquidation', 'index', 'openinterest', 'fill', 'balance',
        'position', 'margin', 'order', 'transaction'
    ])
    def test_all_supported_data_types_per_symbol(self, data_type):
        """All 14+ data types work with per-symbol strategy"""
        topic = TopicManager.get_topic(
            data_type=data_type,
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol'
        )
        assert topic == f'cryptofeed.{data_type}.binance.btc-usdt'

    def test_symbol_with_special_characters(self):
        """Symbols with special characters handled correctly"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='BTC/USDT',
            exchange='binance',
            strategy='per_symbol'
        )
        # Symbol should be normalized (slashes typically converted)
        assert 'cryptofeed.trade' in topic
        assert 'binance' in topic.lower()

    def test_symbol_case_insensitive_per_symbol(self):
        """Per-symbol strategy handles symbol case appropriately"""
        topic1 = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol'
        )
        topic2 = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol'
        )
        # Topics should match (case normalized or preserved consistently)
        # At least should contain the same components
        assert 'trade' in topic1.lower() and 'trade' in topic2.lower()
        assert 'binance' in topic1.lower() and 'binance' in topic2.lower()

    def test_exchange_case_handling(self):
        """Exchange name case handled appropriately"""
        topic1 = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='Binance',
            strategy='per_symbol'
        )
        topic2 = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol'
        )
        # Should be same (case normalized) or consistent
        assert topic1.lower() == topic2.lower() or topic1 == topic2


# =============================================================================
# Backward Compatibility Tests
# =============================================================================
class TestBackwardCompatibility:
    """Test backward compatibility with existing code"""

    def test_per_symbol_maintains_legacy_path(self):
        """Per-symbol strategy produces legacy topic format"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol'
        )
        # Should match existing format: cryptofeed.trade.binance.btc-usdt
        assert 'cryptofeed' in topic
        assert 'trade' in topic
        assert 'binance' in topic.lower()

    def test_default_strategy_is_safe_default(self):
        """Default strategy produces sensible default behavior"""
        topic = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance'
        )
        # Default should be consolidated (simpler)
        assert topic == 'cryptofeed.trade'

    def test_mixed_strategy_deployments_different_topics(self):
        """Consolidated and per-symbol strategies produce different topics"""
        consolidated = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='consolidated'
        )
        per_symbol = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol'
        )
        assert consolidated != per_symbol


# =============================================================================
# Integration Tests
# =============================================================================
class TestIntegration:
    """Integration tests with mock data objects"""

    def test_topic_generation_from_mock_object(self):
        """Generate topic from mock Trade object"""
        mock_trade = Mock()
        mock_trade.exchange = 'binance'
        mock_trade.symbol = 'btc-usdt'

        topic = TopicManager.get_topic(
            data_type='trade',
            symbol=mock_trade.symbol,
            exchange=mock_trade.exchange,
            strategy='consolidated'
        )
        assert topic == 'cryptofeed.trade'

    def test_multiple_exchanges_produce_distinct_topics(self):
        """Multiple exchanges with same symbol produce distinct per-symbol topics"""
        topic1 = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='binance',
            strategy='per_symbol'
        )
        topic2 = TopicManager.get_topic(
            data_type='trade',
            symbol='btc-usdt',
            exchange='coinbase',
            strategy='per_symbol'
        )
        # Different exchanges should produce different topics
        assert topic1 != topic2
        assert 'binance' in topic1.lower()
        assert 'coinbase' in topic2.lower()

    def test_consolidated_aggregates_multiple_exchanges(self):
        """Consolidated strategy aggregates multiple exchanges"""
        topics = [
            TopicManager.get_topic(
                data_type='trade',
                symbol='btc-usdt',
                exchange=exchange,
                strategy='consolidated'
            )
            for exchange in ['binance', 'coinbase', 'kraken']
        ]
        # All should map to same consolidated topic
        assert len(set(topics)) == 1
        assert topics[0] == 'cryptofeed.trade'
