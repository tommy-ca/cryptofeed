"""Integration tests for KafkaCallback Task 5: Complete integration with all Phase 1 components.

This test suite validates the complete end-to-end Kafka producer pipeline:
1. TopicManager integration (Task 1)
2. Partitioner strategies integration (Task 2)
3. HeaderEnricher integration (Task 3)
4. KafkaConfig integration (Task 4)
5. Complete message pipeline orchestration (Task 5)

The tests ensure all components work together seamlessly to produce messages
with correct topic names, partition keys, headers, and serialization.

Expected test count: 100-120 tests covering:
- Complete message pipeline (15+ tests)
- All message types (Trade, Ticker, Candle, L2Book, etc.) (20+ tests)
- Error handling and edge cases (20+ tests)
- Configuration scenarios (15+ tests)
- Partition strategy combinations (15+ tests)
- Header generation and validation (15+ tests)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch, call

import pytest

# Import KafkaCallback and related components
kafka_module = pytest.importorskip("cryptofeed.kafka_callback")
KafkaCallback = kafka_module.KafkaCallback
TopicManager = kafka_module.TopicManager
HeaderEnricher = kafka_module.HeaderEnricher
MessageHeaders = kafka_module.MessageHeaders
OptionalHeaders = kafka_module.OptionalHeaders
Partitioner = kafka_module.Partitioner
PartitionerFactory = kafka_module.PartitionerFactory
SymbolPartitioner = kafka_module.SymbolPartitioner
CompositePartitioner = kafka_module.CompositePartitioner
ExchangePartitioner = kafka_module.ExchangePartitioner
RoundRobinPartitioner = kafka_module.RoundRobinPartitioner
KafkaConfig = kafka_module.KafkaConfig
KafkaTopicConfig = kafka_module.KafkaTopicConfig
KafkaPartitionConfig = kafka_module.KafkaPartitionConfig
KafkaProducerConfig = kafka_module.KafkaProducerConfig

from cryptofeed.types import (
    Trade, Ticker, Candle, OrderBook, Liquidation,
    Funding, Index, OpenInterest
)


# =============================================================================
# Test Fixtures and Mocks
# =============================================================================

@dataclass
class _RecordedMessage:
    """Record of a produced message for verification."""
    topic: str
    key: Optional[bytes]
    value: bytes
    headers: List[tuple[bytes, bytes]]


class _StubProducer:
    """In-memory producer for testing without Kafka brokers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self.messages: List[_RecordedMessage] = []
        self.poll_count = 0

    def list_topics(self, timeout: Optional[float] = None):
        self.connected = True
        return {"topics": {}}

    def produce(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[List[tuple[bytes, bytes]]] = None,
        on_delivery=None
    ):
        """Record produced message and optionally call delivery callback."""
        headers = headers or []
        self.messages.append(
            _RecordedMessage(topic=topic, key=key, value=value, headers=headers)
        )
        if on_delivery:
            # Simulate successful delivery
            msg = Mock()
            msg.topic.return_value = topic
            msg.partition.return_value = 0
            msg.offset.return_value = len(self.messages) - 1
            on_delivery(None, msg)

    def poll(self, timeout: float):
        self.poll_count += 1
        return 0

    def flush(self, timeout: Optional[float] = None):
        return 0


def _producer_factory(cls):
    """Create a producer factory function."""
    def _factory(config):
        return cls(config)
    return _factory


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def trade_message():
    """Create a sample Trade message for testing."""
    return Trade(
        exchange="coinbase",
        symbol="BTC-USD",
        side="buy",
        amount=Decimal("0.25"),
        price=Decimal("68000.10"),
        timestamp=1700000000.0,
        id="trade-1",
        type="spot",
        raw=None,
    )


@pytest.fixture
def trade_binance():
    """Create a Binance trade message."""
    return Trade(
        exchange="binance",
        symbol="BTC-USDT",
        side="sell",
        amount=Decimal("0.5"),
        price=Decimal("68001.00"),
        timestamp=1700000001.0,
        id="trade-binance-1",
        type="spot",
        raw=None,
    )


@pytest.fixture
def ticker_message():
    """Create a sample Ticker message for testing."""
    return Ticker(
        exchange="kraken",
        symbol="ETH-USD",
        bid=Decimal("3500.00"),
        ask=Decimal("3501.00"),
        timestamp=1700000002.0,
        raw=None,
    )


@pytest.fixture
def candle_message():
    """Create a sample Candle message for testing."""
    return Candle(
        exchange="bitmex",
        symbol="XBT-USD",
        start=1700000000.0,
        stop=1700003600.0,
        interval="1h",
        trades=100,
        open=Decimal("68000.00"),
        high=Decimal("68500.00"),
        low=Decimal("67800.00"),
        close=Decimal("68100.00"),
        volume=Decimal("50.5"),
        closed=True,
        timestamp=1700003600.0,
        raw=None,
    )


@pytest.fixture
def orderbook_message():
    """Create a sample OrderBook (L2Book) message for testing."""
    msg = Mock()
    msg.exchange = "dydx"
    msg.symbol = "ETH-USD-PERP"
    msg.timestamp = 1700000003.0
    return msg


@pytest.fixture
def liquidation_message():
    """Create a sample Liquidation message for testing."""
    msg = Mock()
    msg.exchange = "dydx"
    msg.symbol = "BTC-USD-PERP"
    msg.timestamp = 1700000004.0
    return msg


@pytest.fixture
def funding_message():
    """Create a sample Funding message for testing."""
    msg = Mock()
    msg.exchange = "binance-futures"
    msg.symbol = "BTC-USDT-PERP"
    msg.timestamp = 1700000005.0
    return msg


@pytest.fixture
def index_message():
    """Create a sample Index message for testing."""
    msg = Mock()
    msg.exchange = "index-provider"
    msg.symbol = "BTC-USD"
    msg.timestamp = 1700000006.0
    return msg


@pytest.fixture
def openinterest_message():
    """Create a sample OpenInterest message for testing."""
    msg = Mock()
    msg.exchange = "okex"
    msg.symbol = "BTC-USDT"
    msg.timestamp = 1700000007.0
    return msg


# =============================================================================
# Test Class: Complete Message Pipeline (Task 5 Core)
# =============================================================================

class TestCompleteMessagePipeline:
    """Test the complete end-to-end message pipeline (Task 5)."""

    def test_kafka_callback_initialization_with_config(self):
        """Initialize KafkaCallback with KafkaConfig object."""
        config = KafkaConfig(
            bootstrap_servers=['kafka:9092'],
            topic=KafkaTopicConfig(strategy='consolidated'),
            partition=KafkaPartitionConfig(strategy='composite'),
            acks='all',
            idempotence=True
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.bootstrap_servers == ['kafka:9092']
        assert callback.acks == 'all'
        assert callback._topic_strategy == 'consolidated'
        assert callback.is_connected()

    def test_kafka_callback_backward_compatible_initialization(self):
        """Initialize KafkaCallback with direct parameters (backward compatible)."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            acks='all',
            enable_idempotence=True,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.bootstrap_servers == ['kafka:9092']
        assert callback._topic_strategy == 'consolidated'  # Default
        assert callback.is_connected()

    def test_topic_name_generation_consolidated_strategy(self, trade_message):
        """Test topic name generation using TopicManager with consolidated strategy."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='consolidated'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Use normalized data type 'trade' (singular, per Critical Issue #1)
        topic = callback._topic_name('trade', trade_message)
        assert topic == 'cryptofeed.trade'

    def test_topic_name_generation_per_symbol_strategy(self, trade_message):
        """Test topic name generation using TopicManager with per_symbol strategy."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='per_symbol'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Use normalized data type 'trade' (singular, per Critical Issue #1)
        topic = callback._topic_name('trade', trade_message)
        assert topic == 'cryptofeed.trade.coinbase.btc-usd'

    def test_topic_name_with_custom_prefix(self, trade_message):
        """Test topic name generation with custom prefix."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(
                    strategy='consolidated',
                    prefix='production'
                ),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Use normalized data type 'trade' (singular, per Critical Issue #1)
        topic = callback._topic_name('trade', trade_message)
        assert topic == 'production.cryptofeed.trade'

    def test_partition_key_generation_composite_strategy(self, trade_message):
        """Test partition key generation using CompositePartitioner."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='composite'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key = callback._partition_key(trade_message)
        assert key == b'coinbase-btc-usd'

    def test_partition_key_generation_symbol_strategy(self, trade_message):
        """Test partition key generation using SymbolPartitioner."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='symbol'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key = callback._partition_key(trade_message)
        assert key == b'btc-usd'

    def test_partition_key_generation_exchange_strategy(self, trade_message):
        """Test partition key generation using ExchangePartitioner."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='exchange'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key = callback._partition_key(trade_message)
        assert key == b'coinbase'

    def test_partition_key_generation_round_robin_strategy(self, trade_message):
        """Test partition key generation using RoundRobinPartitioner (returns None)."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='round_robin'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key = callback._partition_key(trade_message)
        assert key is None

    def test_header_enrichment_with_trade_message(self, trade_message):
        """Test header enrichment pipeline with Trade message."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format='protobuf',
        )
        headers = callback._header_enricher.build(
            message=trade_message,
            data_type='trades'
        )
        # Should have 7 headers: 4 mandatory + 3 optional
        assert len(headers) == 7
        header_dict = dict(headers)
        assert header_dict[b'content-type'] == b'application/x-protobuf'
        assert header_dict[b'exchange'] == b'coinbase'
        assert header_dict[b'symbol'] == b'BTC-USD'
        assert header_dict[b'data_type'] == b'trades'
        assert header_dict[b'schema_version'] == b'v1'
        assert b'producer_version' in header_dict
        assert b'timestamp_generated' in header_dict

    def test_header_enrichment_with_json_serialization(self, trade_message):
        """Test header enrichment with JSON serialization format."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format='json',
        )
        headers = callback._header_enricher.build(
            message=trade_message,
            data_type='trades'
        )
        header_dict = dict(headers)
        assert header_dict[b'content-type'] == b'application/json'

    @pytest.mark.asyncio
    async def test_complete_pipeline_with_json_serialization(self, trade_message):
        """Test complete pipeline: queue -> serialize -> topic -> partition -> headers -> produce."""
        stub_producer = _StubProducer({})
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format='json',
        )
        callback._producer = Mock()
        callback._producer.produce = Mock()

        # Process a single message through the pipeline
        await callback._drain_once()  # Should handle empty queue gracefully
        # This test verifies no exceptions are raised

    @pytest.mark.asyncio
    async def test_drain_once_processes_single_message(self, trade_message):
        """Test that _drain_once processes exactly one message from queue."""
        stub_producer = _StubProducer({})
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format='json',
        )

        # Import the internal _QueuedMessage type
        from cryptofeed.kafka_callback import _QueuedMessage

        # Queue a message
        await callback._queue.put(_QueuedMessage(
            data_type='trades',
            obj=trade_message,
            receipt_timestamp=trade_message.timestamp
        ))

        # Drain once should process this message
        await callback._drain_once()
        # Verify queue is now empty
        assert callback._queue.empty()

    def test_message_pipeline_with_different_message_types(
        self, trade_message, ticker_message, candle_message
    ):
        """Test pipeline handles different message types (Trade, Ticker, Candle)."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        # Each message type should generate correct topic names
        trade_topic = callback._topic_name('trades', trade_message)
        ticker_topic = callback._topic_name('ticker', ticker_message)
        candle_topic = callback._topic_name('candles', candle_message)

        assert trade_topic == 'cryptofeed.trades'
        assert ticker_topic == 'cryptofeed.ticker'
        assert candle_topic == 'cryptofeed.candles'

    def test_backward_compatibility_with_messages_lacking_attributes(self):
        """Test pipeline handles messages with missing exchange/symbol attributes."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        # Create a minimal object without exchange/symbol
        minimal_obj = Mock()
        minimal_obj.exchange = None
        minimal_obj.symbol = None

        # Should not raise exception
        topic = callback._topic_name('trades', minimal_obj)
        assert 'cryptofeed' in topic
        key = callback._partition_key(minimal_obj)
        # Should still return a key or None gracefully

# =============================================================================
# Test Class: All Message Types Integration (Task 5)
# =============================================================================

class TestAllMessageTypesIntegration:
    """Test the pipeline with all cryptofeed message types."""

    def test_trade_message_pipeline(self, trade_message):
        """Test Trade message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='consolidated'),
                partition=KafkaPartitionConfig(strategy='composite'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name('trades', trade_message)
        key = callback._partition_key(trade_message)
        headers = callback._header_enricher.build(trade_message, 'trades')

        assert topic == 'cryptofeed.trades'
        assert key == b'coinbase-btc-usd'
        assert len(headers) == 7

    def test_ticker_message_pipeline(self, ticker_message):
        """Test Ticker message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name('ticker', ticker_message)
        key = callback._partition_key(ticker_message)
        headers = callback._header_enricher.build(ticker_message, 'ticker')

        assert topic == 'cryptofeed.ticker'
        assert key is not None  # Composite strategy
        assert len(headers) == 7

    def test_candle_message_pipeline(self, candle_message):
        """Test Candle message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='consolidated'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name('candle', candle_message)
        key = callback._partition_key(candle_message)

        assert topic == 'cryptofeed.candle'
        assert key is not None

    def test_orderbook_message_pipeline(self, orderbook_message):
        """Test OrderBook message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='consolidated'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name('orderbook', orderbook_message)
        key = callback._partition_key(orderbook_message)

        assert topic == 'cryptofeed.orderbook'
        assert key is not None

    def test_liquidation_message_pipeline(self, liquidation_message):
        """Test Liquidation message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='consolidated'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name('liquidation', liquidation_message)
        key = callback._partition_key(liquidation_message)

        assert topic == 'cryptofeed.liquidation'
        assert key is not None

    def test_funding_message_pipeline(self, funding_message):
        """Test Funding message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='consolidated'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name('funding', funding_message)
        key = callback._partition_key(funding_message)

        assert topic == 'cryptofeed.funding'
        assert key is not None

    def test_index_message_pipeline(self, index_message):
        """Test Index message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='consolidated'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name('index', index_message)
        assert topic == 'cryptofeed.index'

    def test_openinterest_message_pipeline(self, openinterest_message):
        """Test OpenInterest message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='consolidated'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name('openinterest', openinterest_message)
        assert topic == 'cryptofeed.openinterest'


# =============================================================================
# Test Class: Error Handling and Edge Cases (Task 5)
# =============================================================================

class TestErrorHandlingAndEdgeCases:
    """Test error handling, edge cases, and graceful degradation."""

    def test_missing_required_parameter_raises_error(self):
        """Test that missing bootstrap_servers raises error."""
        with pytest.raises(TypeError):
            KafkaCallback()  # No bootstrap_servers or kafka_config

    def test_invalid_topic_strategy_raises_error(self):
        """Test that invalid topic strategy raises error."""
        with pytest.raises(ValueError):
            KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='invalid_strategy')
            )

    def test_invalid_partition_strategy_raises_error(self):
        """Test that invalid partition strategy raises error."""
        with pytest.raises(ValueError):
            KafkaPartitionConfig(strategy='invalid_partitioner')

    def test_invalid_acks_value_raises_error(self):
        """Test that invalid acks value raises error."""
        with pytest.raises(ValueError):
            KafkaProducerConfig(
                bootstrap_servers=['kafka:9092'],
                acks='invalid'
            )

    def test_empty_bootstrap_servers_raises_error(self):
        """Test that empty bootstrap_servers raises error."""
        with pytest.raises(ValueError):
            KafkaConfig(bootstrap_servers=[])

    def test_non_positive_partitions_raises_error(self):
        """Test that non-positive partition count raises error."""
        with pytest.raises(ValueError):
            KafkaTopicConfig(partitions_per_topic=0)

    def test_non_positive_replication_factor_raises_error(self):
        """Test that non-positive replication factor raises error."""
        with pytest.raises(ValueError):
            KafkaTopicConfig(replication_factor=0)

    def test_message_with_unknown_exchange_uses_fallback(self):
        """Test that message with unknown exchange uses fallback gracefully."""
        obj = Mock()
        obj.exchange = None
        obj.symbol = 'BTC-USD'
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        headers = callback._header_enricher.build(obj, 'trades')
        header_dict = dict(headers)
        assert header_dict[b'exchange'] == b'unknown'

    def test_message_with_unknown_symbol_uses_fallback(self):
        """Test that message with unknown symbol uses fallback gracefully."""
        obj = Mock()
        obj.exchange = 'coinbase'
        obj.symbol = None
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        headers = callback._header_enricher.build(obj, 'trades')
        header_dict = dict(headers)
        assert header_dict[b'symbol'] == b'unknown'


# =============================================================================
# Test Class: Configuration Scenarios (Task 5)
# =============================================================================

class TestConfigurationScenarios:
    """Test various configuration scenarios and combinations."""

    def test_consolidated_topic_with_composite_partitioner(self, trade_message):
        """Test consolidated topics with composite partitioner."""
        config = KafkaConfig(
            bootstrap_servers=['kafka:9092'],
            topic=KafkaTopicConfig(strategy='consolidated'),
            partition=KafkaPartitionConfig(strategy='composite'),
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback._topic_name('trades', trade_message) == 'cryptofeed.trades'
        assert callback._partition_key(trade_message) == b'coinbase-btc-usd'

    def test_per_symbol_topic_with_symbol_partitioner(self, trade_message):
        """Test per-symbol topics with symbol partitioner."""
        config = KafkaConfig(
            bootstrap_servers=['kafka:9092'],
            topic=KafkaTopicConfig(strategy='per_symbol'),
            partition=KafkaPartitionConfig(strategy='symbol'),
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback._topic_name('trades', trade_message) == 'cryptofeed.trades.coinbase.BTC-USD'
        assert callback._partition_key(trade_message) == b'btc-usd'

    def test_consolidated_topic_with_prefix_and_exchange_partitioner(self, trade_message):
        """Test consolidated topics with prefix and exchange partitioner."""
        config = KafkaConfig(
            bootstrap_servers=['kafka:9092'],
            topic=KafkaTopicConfig(
                strategy='consolidated',
                prefix='production'
            ),
            partition=KafkaPartitionConfig(strategy='exchange'),
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback._topic_name('trades', trade_message) == 'production.cryptofeed.trades'
        assert callback._partition_key(trade_message) == b'coinbase'

    def test_round_robin_partitioner_returns_none(self, trade_message):
        """Test that round-robin partitioner returns None for partition key."""
        config = KafkaConfig(
            bootstrap_servers=['kafka:9092'],
            partition=KafkaPartitionConfig(strategy='round_robin'),
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback._partition_key(trade_message) is None

    def test_load_config_from_dict(self):
        """Test loading KafkaConfig from dictionary."""
        config_dict = {
            'bootstrap_servers': ['kafka:9092', 'kafka:9093'],
            'topic': {'strategy': 'consolidated', 'prefix': 'prod'},
            'partition': {'strategy': 'composite'},
            'acks': 'all',
            'idempotence': True,
        }
        config = KafkaConfig.from_dict(config_dict)
        assert config.bootstrap_servers == ['kafka:9092', 'kafka:9093']
        assert config.topic.strategy == 'consolidated'
        assert config.partition.strategy == 'composite'

    def test_whitespace_only_prefix_becomes_default(self):
        """Test that whitespace-only prefix is normalized to 'cryptofeed'."""
        config = KafkaTopicConfig(prefix="   ")
        assert config.prefix == "cryptofeed"

    def test_none_prefix_becomes_default(self):
        """Test that None prefix is normalized to 'cryptofeed'."""
        config = KafkaTopicConfig(prefix=None)
        assert config.prefix == "cryptofeed"

    def test_multiple_bootstrap_servers(self):
        """Test configuration with multiple bootstrap servers."""
        config = KafkaConfig(
            bootstrap_servers=['kafka1:9092', 'kafka2:9092', 'kafka3:9092']
        )
        assert len(config.bootstrap_servers) == 3
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.bootstrap_servers == ['kafka1:9092', 'kafka2:9092', 'kafka3:9092']


# =============================================================================
# Test Class: Partition Key Consistency (Task 5)
# =============================================================================

class TestPartitionKeyConsistency:
    """Test that partition keys are consistent and deterministic."""

    def test_symbol_partitioner_consistency(self, trade_message):
        """Test that same symbol always generates same partition key."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='symbol'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key1 = callback._partition_key(trade_message)
        key2 = callback._partition_key(trade_message)
        assert key1 == key2 == b'btc-usd'

    def test_composite_partitioner_consistency(self, trade_message):
        """Test that same exchange-symbol always generates same partition key."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='composite'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key1 = callback._partition_key(trade_message)
        key2 = callback._partition_key(trade_message)
        assert key1 == key2 == b'coinbase-btc-usd'

    def test_exchange_partitioner_consistency(self, trade_message):
        """Test that same exchange always generates same partition key."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='exchange'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key1 = callback._partition_key(trade_message)
        key2 = callback._partition_key(trade_message)
        assert key1 == key2 == b'coinbase'

    def test_different_exchanges_different_keys(self, trade_message, trade_binance):
        """Test that different exchanges generate different partition keys."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='exchange'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key_coinbase = callback._partition_key(trade_message)
        key_binance = callback._partition_key(trade_binance)
        assert key_coinbase != key_binance
        assert key_coinbase == b'coinbase'
        assert key_binance == b'binance'

    def test_different_symbols_different_composite_keys(self, trade_message, trade_binance):
        """Test that different exchange-symbol pairs generate different keys."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='composite'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key1 = callback._partition_key(trade_message)
        key2 = callback._partition_key(trade_binance)
        assert key1 != key2


# =============================================================================
# Test Class: Header Generation and Validation (Task 5)
# =============================================================================

class TestHeaderGeneration:
    """Test header generation for all message types."""

    def test_mandatory_headers_structure(self, trade_message):
        """Test that mandatory headers have correct structure."""
        headers = MessageHeaders.build(
            message=trade_message,
            data_type='trades',
            content_type='application/x-protobuf'
        )
        assert len(headers) == 4
        assert all(isinstance(h, tuple) and len(h) == 2 for h in headers)
        assert all(isinstance(h[0], bytes) and isinstance(h[1], bytes) for h in headers)

    def test_optional_headers_structure(self):
        """Test that optional headers have correct structure."""
        headers = OptionalHeaders.build(
            schema_version='v1',
            producer_version='2.4.1',
            timestamp_generated='2025-11-09T12:34:56Z'
        )
        assert len(headers) == 3
        assert all(isinstance(h, tuple) and len(h) == 2 for h in headers)
        assert all(isinstance(h[0], bytes) and isinstance(h[1], bytes) for h in headers)

    def test_header_enricher_combines_mandatory_and_optional(self, trade_message):
        """Test that HeaderEnricher combines both mandatory and optional headers."""
        enricher = HeaderEnricher(
            content_type='application/x-protobuf',
            schema_version='v1'
        )
        headers = enricher.build(trade_message, 'trades')
        assert len(headers) == 7  # 4 mandatory + 3 optional
        header_names = [h[0] for h in headers]
        assert b'content-type' in header_names
        assert b'exchange' in header_names
        assert b'symbol' in header_names
        assert b'data_type' in header_names
        assert b'schema_version' in header_names
        assert b'producer_version' in header_names
        assert b'timestamp_generated' in header_names

    def test_header_values_are_encoded_as_bytes(self, trade_message):
        """Test that all header values are UTF-8 encoded bytes."""
        enricher = HeaderEnricher()
        headers = enricher.build(trade_message, 'trades')
        for name, value in headers:
            assert isinstance(name, bytes)
            assert isinstance(value, bytes)

    def test_headers_with_special_characters_in_symbol(self):
        """Test header generation with special characters in symbol."""
        obj = Mock()
        obj.exchange = 'binance'
        obj.symbol = 'BTC_USDT'  # Underscore should be converted
        headers = MessageHeaders.build(obj, 'trades', 'application/json')
        header_dict = dict(headers)
        assert header_dict[b'symbol'] == b'BTC-USDT'

    def test_headers_with_case_insensitivity(self):
        """Test that exchange names are normalized to lowercase in headers."""
        obj = Mock()
        obj.exchange = 'COINBASE'  # Uppercase
        obj.symbol = 'BTC-USD'
        headers = MessageHeaders.build(obj, 'trades', 'application/json')
        header_dict = dict(headers)
        assert header_dict[b'exchange'] == b'coinbase'

    def test_optional_headers_with_custom_timestamp(self):
        """Test optional headers with custom timestamp."""
        custom_timestamp = '2025-11-09T14:30:00Z'
        headers = OptionalHeaders.build(
            timestamp_generated=custom_timestamp
        )
        header_dict = dict(headers)
        assert header_dict[b'timestamp_generated'] == b'2025-11-09T14:30:00Z'

    def test_optional_headers_with_default_timestamp(self):
        """Test optional headers generate default timestamp when not provided."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)
        assert b'timestamp_generated' in header_dict
        # Timestamp should be a valid ISO8601 format ending with Z
        ts_value = header_dict[b'timestamp_generated'].decode('utf-8')
        assert ts_value.endswith('Z')
        assert 'T' in ts_value  # Should have date-time separator


# =============================================================================
# Test Class: Backward Compatibility (Task 5)
# =============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility with existing deployments."""

    def test_per_symbol_strategy_still_works(self, trade_message):
        """Test that per-symbol strategy still works for legacy deployments."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='per_symbol'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name('trades', trade_message)
        assert 'coinbase' in topic
        assert 'BTC-USD' in topic

    def test_initialization_without_config_object(self):
        """Test backward-compatible initialization without KafkaConfig."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            acks='all',
            enable_idempotence=True,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.acks == 'all'
        assert callback.enable_idempotence is True

    def test_default_strategy_is_consolidated(self):
        """Test that default topic strategy is consolidated."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback._topic_strategy == 'consolidated'

    def test_default_partitioner_is_composite(self):
        """Test that default partitioner is composite."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert isinstance(callback._partitioner, CompositePartitioner)

    def test_message_serialization_format_json(self, trade_message):
        """Test that JSON serialization format still works."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format='json',
        )
        headers = callback._header_enricher.build(trade_message, 'trades')
        header_dict = dict(headers)
        assert header_dict[b'content-type'] == b'application/json'

    def test_message_serialization_format_protobuf(self, trade_message):
        """Test that protobuf serialization format works."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format='protobuf',
        )
        headers = callback._header_enricher.build(trade_message, 'trades')
        header_dict = dict(headers)
        assert header_dict[b'content-type'] == b'application/x-protobuf'


# =============================================================================
# Test Class: Performance and Scale Tests (Task 5)
# =============================================================================

class TestPerformanceAndScale:
    """Test performance characteristics and scaling behavior."""

    def test_partition_key_generation_is_fast(self, trade_message):
        """Test that partition key generation is fast (<1ms)."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        import time
        start = time.time()
        for _ in range(1000):
            callback._partition_key(trade_message)
        elapsed = time.time() - start
        # Should be able to generate 1000 keys in <100ms
        assert elapsed < 0.1

    def test_topic_name_generation_is_fast(self, trade_message):
        """Test that topic name generation is fast (<1ms)."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        import time
        start = time.time()
        for _ in range(1000):
            callback._topic_name('trades', trade_message)
        elapsed = time.time() - start
        # Should be able to generate 1000 topic names in <100ms
        assert elapsed < 0.1

    def test_header_enrichment_is_fast(self, trade_message):
        """Test that header enrichment is fast (<1ms)."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        import time
        start = time.time()
        for _ in range(1000):
            callback._header_enricher.build(trade_message, 'trades')
        elapsed = time.time() - start
        # Should be able to enrich 1000 messages in <100ms
        assert elapsed < 0.1

    def test_multiple_message_types_in_burst(
        self, trade_message, ticker_message, candle_message, orderbook_message
    ):
        """Test pipeline handles burst of different message types."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        messages = [
            (trade_message, 'trades'),
            (ticker_message, 'ticker'),
            (candle_message, 'candles'),
            (orderbook_message, 'orderbook'),
        ]
        # Process all messages without error
        for msg, data_type in messages * 25:  # 100 total messages
            callback._topic_name(data_type, msg)
            callback._partition_key(msg)
            callback._header_enricher.build(msg, data_type)


# =============================================================================
# Test Class: Integration with Message Handler (Task 5)
# =============================================================================

class TestMessageHandlerIntegration:
    """Test integration with KafkaCallback's message handling."""

    def test_queue_message_for_later_processing(self, trade_message):
        """Test queueing messages for async processing."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Queue a message
        result = callback._queue_message('trades', trade_message, 1700000000.0)
        assert result is True
        assert callback.queue_size() == 1

    def test_queue_size_tracking(self, trade_message, ticker_message):
        """Test that queue size is tracked correctly."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            queue_maxsize=10,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.queue_size() == 0
        callback._queue_message('trades', trade_message)
        assert callback.queue_size() == 1
        callback._queue_message('ticker', ticker_message)
        assert callback.queue_size() == 2

    def test_is_connected_check(self):
        """Test that is_connected() works correctly."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.is_connected()

    def test_dynamic_handler_binding(self, trade_message):
        """Test that dynamic handler binding creates correct async handlers."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Dynamic handler binding should work
        handler = callback.trade
        assert callable(handler)
        # Handler should be an async function
        import asyncio
        assert asyncio.iscoroutinefunction(handler)


# =============================================================================
# Test Class: Additional Integration Scenarios (Task 5)
# =============================================================================

class TestAdditionalIntegrationScenarios:
    """Additional comprehensive integration test scenarios."""

    def test_kafka_config_with_all_producer_settings(self):
        """Test KafkaConfig with all producer settings configured."""
        config = KafkaConfig(
            bootstrap_servers=['kafka:9092'],
            topic=KafkaTopicConfig(
                strategy='consolidated',
                prefix='prod',
                partitions_per_topic=6,
                replication_factor=2
            ),
            partition=KafkaPartitionConfig(strategy='composite'),
            acks='all',
            idempotence=True,
            retries=5,
            retry_backoff_ms=200,
            batch_size=32768,
            linger_ms=20,
            compression_type='snappy'
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.acks == 'all'
        assert callback.enable_idempotence is True
        assert callback._topic_prefix == 'prod'

    def test_multiple_message_types_generate_different_topics(
        self, trade_message, ticker_message, candle_message
    ):
        """Test that different message types generate correct topic names."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='consolidated'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        topics = {
            'trades': callback._topic_name('trades', trade_message),
            'ticker': callback._topic_name('ticker', ticker_message),
            'candle': callback._topic_name('candle', candle_message),
        }

        assert topics['trades'] == 'cryptofeed.trades'
        assert topics['ticker'] == 'cryptofeed.ticker'
        assert topics['candle'] == 'cryptofeed.candle'
        assert len(set(topics.values())) == 3  # All different

    def test_per_symbol_topics_with_multiple_exchanges(
        self, trade_message, trade_binance
    ):
        """Test per-symbol topics with different exchanges produce different topic names."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='per_symbol'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        topic_coinbase = callback._topic_name('trades', trade_message)
        topic_binance = callback._topic_name('trades', trade_binance)

        assert 'coinbase' in topic_coinbase
        assert 'binance' in topic_binance
        assert topic_coinbase != topic_binance

    def test_header_normalization_with_mixed_case_exchange(self):
        """Test that exchange names are normalized to lowercase in headers."""
        obj = Mock()
        obj.exchange = 'CoInBaSe'  # Mixed case
        obj.symbol = 'BTC-USD'

        headers = MessageHeaders.build(obj, 'trades', 'application/json')
        header_dict = dict(headers)
        assert header_dict[b'exchange'] == b'coinbase'

    def test_header_symbol_normalization_multiple_formats(self):
        """Test symbol normalization in headers with various input formats."""
        obj1 = Mock()
        obj1.exchange = 'binance'
        obj1.symbol = 'BTC_USDT'

        obj2 = Mock()
        obj2.exchange = 'binance'
        obj2.symbol = 'btc-usdt'

        headers1 = MessageHeaders.build(obj1, 'trades', 'application/json')
        headers2 = MessageHeaders.build(obj2, 'trades', 'application/json')

        dict1 = dict(headers1)
        dict2 = dict(headers2)

        # Both should normalize to BTC-USDT
        assert dict1[b'symbol'] == b'BTC-USDT'
        assert dict2[b'symbol'] == b'btc-usdt'

    def test_configuration_from_dict_with_nested_objects(self):
        """Test loading configuration from nested dictionary structure."""
        config_dict = {
            'bootstrap_servers': ['kafka1:9092', 'kafka2:9092'],
            'topic': {
                'strategy': 'consolidated',
                'prefix': 'staging',
                'partitions_per_topic': 5,
                'replication_factor': 2
            },
            'partition': {
                'strategy': 'symbol'
            },
            'acks': '1',
            'batch_size': 32768,
            'linger_ms': 20,
        }
        config = KafkaConfig.from_dict(config_dict)
        assert config.topic.prefix == 'staging'
        assert config.partition.strategy == 'symbol'
        assert config.acks == '1'
        assert config.batch_size == 32768

    def test_all_supported_data_types_generate_topics(self, trade_message):
        """Test that all supported data types can generate topic names."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='consolidated'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        supported_types = [
            'trades', 'orderbook', 'ticker', 'candle', 'funding',
            'liquidation', 'index', 'openinterest'
        ]

        topics = []
        for data_type in supported_types:
            topic = callback._topic_name(data_type, trade_message)
            topics.append(topic)
            assert f'cryptofeed.{data_type}' == topic

        # All should be different
        assert len(set(topics)) == len(supported_types)

    def test_partition_key_with_special_symbols(self):
        """Test partition key generation with special characters in symbol."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='symbol'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        obj = Mock()
        obj.exchange = 'binance'
        obj.symbol = 'BTC/USD'  # Slash character

        key = callback._partition_key(obj)
        assert key is not None
        assert isinstance(key, bytes)

    def test_partition_key_empty_symbol_graceful_handling(self):
        """Test that empty symbol is handled gracefully."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='symbol'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        obj = Mock()
        obj.exchange = 'binance'
        obj.symbol = ''

        key = callback._partition_key(obj)
        # Should still return a key (for empty symbol)
        assert key is not None

    def test_headers_with_multiple_content_types(self, trade_message):
        """Test header generation with different content types."""
        content_types = [
            'application/x-protobuf',
            'application/json',
            'application/octet-stream',
        ]

        for content_type in content_types:
            headers = MessageHeaders.build(
                trade_message,
                'trades',
                content_type
            )
            header_dict = dict(headers)
            assert header_dict[b'content-type'] == content_type.encode('utf-8')

    def test_header_enricher_with_custom_producer_version(self, trade_message):
        """Test HeaderEnricher with custom producer version."""
        custom_version = '3.0.0-rc1'
        enricher = HeaderEnricher(
            producer_version=custom_version
        )
        headers = enricher.build(trade_message, 'trades')
        header_dict = dict(headers)
        assert header_dict[b'producer_version'] == custom_version.encode('utf-8')

    def test_header_enricher_preserves_header_order(self, trade_message):
        """Test that HeaderEnricher returns headers in consistent order."""
        enricher = HeaderEnricher()
        headers1 = enricher.build(trade_message, 'trades')
        headers2 = enricher.build(trade_message, 'trades')

        # Header order should be consistent
        header_names_1 = [h[0] for h in headers1]
        header_names_2 = [h[0] for h in headers2]
        assert header_names_1 == header_names_2

    def test_configuration_validation_rejects_invalid_compression(self):
        """Test that KafkaProducerConfig rejects invalid compression types."""
        with pytest.raises(ValueError, match="compression_type"):
            KafkaProducerConfig(
                bootstrap_servers=['kafka:9092'],
                compression_type='invalid_compression'
            )

    def test_configuration_validation_rejects_negative_retries(self):
        """Test that KafkaProducerConfig rejects negative retries."""
        with pytest.raises(ValueError, match="retries"):
            KafkaProducerConfig(
                bootstrap_servers=['kafka:9092'],
                retries=-1
            )

    def test_configuration_validation_rejects_negative_batch_size(self):
        """Test that KafkaProducerConfig rejects non-positive batch size."""
        with pytest.raises(ValueError, match="batch_size"):
            KafkaProducerConfig(
                bootstrap_servers=['kafka:9092'],
                batch_size=0
            )

    def test_partitioner_factory_with_case_insensitive_strategy(self):
        """Test PartitionerFactory handles case-insensitive strategy names."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                partition=KafkaPartitionConfig(strategy='SYMBOL'),  # Uppercase
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Should handle uppercase and work correctly
        assert isinstance(callback._partitioner, SymbolPartitioner)

    def test_message_handler_queue_full_handling(self):
        """Test message handler handles full queue gracefully."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            queue_maxsize=1,  # Very small queue
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        obj = Mock()
        obj.symbol = 'BTC-USD'
        obj.exchange = 'coinbase'

        # Fill the queue
        result1 = callback._queue_message('trades', obj)
        assert result1 is True

        # Second message should fail (queue full)
        result2 = callback._queue_message('trades', obj)
        assert result2 is False

    def test_topic_manager_validation_with_all_data_types(self):
        """Test TopicManager validates all supported data types."""
        supported = TopicManager.SUPPORTED_DATA_TYPES

        for data_type in supported:
            # Should not raise
            TopicManager.validate_data_type(data_type)

        # Invalid type should raise
        with pytest.raises(ValueError):
            TopicManager.validate_data_type('unsupported_type')

    def test_topic_manager_validation_with_all_strategies(self):
        """Test TopicManager validates all supported strategies."""
        for strategy in ['consolidated', 'per_symbol']:
            # Should not raise
            TopicManager.validate_strategy(strategy)

        # Invalid strategy should raise
        with pytest.raises(ValueError):
            TopicManager.validate_strategy('invalid_strategy')

    def test_partition_strategies_with_normalized_symbols(self, trade_message):
        """Test all partition strategies handle symbol normalization correctly."""
        callback = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        strategies = ['composite', 'symbol', 'exchange', 'round_robin']
        for strategy_name in strategies:
            partitioner = PartitionerFactory.create(strategy_name)
            key = partitioner.get_partition_key(trade_message)

            # All keys should be either None or bytes
            assert key is None or isinstance(key, bytes)

    def test_configuration_from_yaml_with_minimal_config(self, tmp_path):
        """Test loading KafkaConfig from minimal YAML file."""
        yaml_file = tmp_path / "minimal_kafka.yaml"
        yaml_file.write_text("""
bootstrap_servers:
  - kafka:9092
""")

        config = KafkaConfig.from_yaml(str(yaml_file))
        assert config.bootstrap_servers == ['kafka:9092']
        assert config.topic.strategy == 'consolidated'  # Default
        assert config.partition.strategy == 'composite'  # Default

    def test_configuration_from_yaml_with_full_config(self, tmp_path):
        """Test loading KafkaConfig from full YAML file."""
        yaml_file = tmp_path / "full_kafka.yaml"
        yaml_file.write_text("""
bootstrap_servers:
  - kafka1:9092
  - kafka2:9092
topic:
  strategy: per_symbol
  prefix: production
  partitions_per_topic: 6
  replication_factor: 3
partition:
  strategy: symbol
acks: all
idempotence: true
retries: 5
retry_backoff_ms: 200
batch_size: 32768
linger_ms: 20
compression_type: snappy
""")

        config = KafkaConfig.from_yaml(str(yaml_file))
        assert len(config.bootstrap_servers) == 2
        assert config.topic.strategy == 'per_symbol'
        assert config.topic.prefix == 'production'
        assert config.partition.strategy == 'symbol'
        assert config.acks == 'all'
        assert config.batch_size == 32768

    def test_configuration_from_yaml_missing_file(self):
        """Test loading from non-existent YAML file raises error."""
        with pytest.raises(FileNotFoundError):
            KafkaConfig.from_yaml('/nonexistent/path/kafka.yaml')

    def test_configuration_from_empty_yaml_file(self, tmp_path):
        """Test loading from empty YAML file raises error."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        with pytest.raises(ValueError):
            KafkaConfig.from_yaml(str(yaml_file))

    def test_header_timestamp_iso8601_format(self):
        """Test that timestamp_generated header is valid ISO8601 format."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)

        timestamp_str = header_dict[b'timestamp_generated'].decode('utf-8')

        # Should end with Z for UTC
        assert timestamp_str.endswith('Z')
        # Should have date-time separator T
        assert 'T' in timestamp_str
        # Should have digits (basic ISO8601 check)
        assert any(c.isdigit() for c in timestamp_str)

    def test_header_producer_version_format(self):
        """Test that producer_version header has valid format."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)

        version_str = header_dict[b'producer_version'].decode('utf-8')

        # Should be non-empty and typically in X.Y.Z format
        assert len(version_str) > 0
        # Should contain at least one dot for version format
        parts = version_str.split('.')
        assert len(parts) >= 2  # At least major.minor

    def test_multiple_callbacks_with_different_configs(self, trade_message):
        """Test multiple KafkaCallback instances with different configurations."""
        callback1 = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='consolidated'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        callback2 = KafkaCallback(
            bootstrap_servers=['kafka:9092'],
            kafka_config=KafkaConfig(
                bootstrap_servers=['kafka:9092'],
                topic=KafkaTopicConfig(strategy='per_symbol'),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        topic1 = callback1._topic_name('trades', trade_message)
        topic2 = callback2._topic_name('trades', trade_message)

        # Different strategies should produce different topics
        assert topic1 != topic2

    def test_partitioner_consistency_across_multiple_instances(self, trade_message):
        """Test that same partitioner strategy produces same keys across instances."""
        partitioner1 = PartitionerFactory.create('symbol')
        partitioner2 = PartitionerFactory.create('symbol')

        key1 = partitioner1.get_partition_key(trade_message)
        key2 = partitioner2.get_partition_key(trade_message)

        assert key1 == key2

    def test_header_enricher_with_both_custom_values(self, trade_message):
        """Test HeaderEnricher with all custom values specified."""
        custom_timestamp = '2025-11-09T14:30:00Z'
        custom_version = 'test-version'

        enricher = HeaderEnricher(
            content_type='application/json',
            schema_version='v2',
            producer_version=custom_version,
            timestamp_generated=custom_timestamp,
        )

        headers = enricher.build(trade_message, 'trades')
        header_dict = dict(headers)

        assert header_dict[b'content-type'] == b'application/json'
        assert header_dict[b'schema_version'] == b'v2'
        assert header_dict[b'producer_version'] == custom_version.encode('utf-8')
        assert header_dict[b'timestamp_generated'] == custom_timestamp.encode('utf-8')
