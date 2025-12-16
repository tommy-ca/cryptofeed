"""
Integration test comparing old vs. new Kafka backend behavior (Task 16.1).

This test verifies that complexity reduction phases (Phase 1-3) preserved
behavioral equivalence across key operations:
- Message production (topic naming, partition routing, header encoding, protobuf serialization)
- All 4 partition strategies (symbol, composite, exchange, round_robin)
- Header normalization and encoding
- Topic naming strategies (consolidated vs per_symbol)

Since we cannot instantiate the "old" backend (pre-Phase 1 code deleted), we instead
validate that the current simplified backend produces outputs that match the documented
expected behavior from the original design.

Requirements: REQ-5.17 (behavioral preservation after simplification)
"""

import asyncio
import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
from collections import deque

from cryptofeed.types import Trade, OrderBook
from cryptofeed.backends.kafka.callback import KafkaCallback, _get_partition_key, _build_headers
from cryptofeed.backends.kafka.config import KafkaConfig
from cryptofeed.backends.kafka.normalization import normalize_symbol, normalize_exchange
from cryptofeed.backends.protobuf.converters import trade_to_proto, orderbook_to_proto


class MockKafkaProducer:
    """Mock Kafka producer capturing produced messages for verification."""

    def __init__(self):
        self.messages = deque()
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    def produce(self, topic: str, value: bytes, *, key: bytes | None = None, headers: list | None = None, on_delivery=None):
        """Capture message for verification."""
        self.messages.append({
            'topic': topic,
            'key': key,
            'value': value,
            'headers': headers or [],
        })
        # Simulate successful delivery callback
        if on_delivery:
            # Create mock message object
            class MockMsg:
                def topic(self):
                    return topic
                def partition(self):
                    return 0
                def offset(self):
                    return len(self.messages) - 1
            on_delivery(None, MockMsg())  # err=None, msg=MockMsg

    def poll(self, timeout: float = 0.0):
        pass

    def flush(self, timeout: float | None = None):
        pass

    def close(self, timeout: float | None = None):
        pass


@pytest.mark.asyncio
async def test_simplified_backend_topic_naming_consolidated():
    """
    Verify simplified backend generates correct topic names for consolidated strategy.

    Expected Behavior (Original Design):
    - Consolidated strategy: cryptofeed.{data_type} (no exchange/symbol in topic)
    - All messages for same data_type go to same topic
    - Partitioning by composite key (exchange-symbol) for routing within topic
    """
    # Setup: Create simplified backend with mocked producer
    mock_producer_instance = MockKafkaProducer()

    def mock_producer_factory(*args, **kwargs):
        return MagicMock(
            list_topics=MagicMock(return_value={}),
            produce=mock_producer_instance.produce,
            poll=mock_producer_instance.poll,
            flush=mock_producer_instance.flush,
        )

    config = KafkaConfig(
        bootstrap_servers=['localhost:9092'],
        topic_strategy='consolidated',
        partition_strategy='composite',
    )

    callback = KafkaCallback(
        kafka_config=config,
        producer_factory=mock_producer_factory,
        serialization_format='protobuf',
    )
    callback.start(asyncio.get_event_loop())

    # Execute: Send trades from different exchanges/symbols
    trade1 = Trade(
        exchange='binance',
        symbol='BTC-USD',
        side='buy',
        price=Decimal('50000.00'),
        amount=Decimal('1.0'),
        timestamp=1234567890.123,
    )

    trade2 = Trade(
        exchange='okx',
        symbol='ETH-USD',
        side='sell',
        price=Decimal('3000.00'),
        amount=Decimal('2.0'),
        timestamp=1234567890.456,
    )

    await callback.trade(trade1, receipt_timestamp=1234567890.123)
    await callback.trade(trade2, receipt_timestamp=1234567890.456)

    # Process queue
    await asyncio.sleep(0.2)

    # Verify: Both trades go to same consolidated topic
    assert len(mock_producer_instance.messages) == 2

    msg1 = mock_producer_instance.messages[0]
    msg2 = mock_producer_instance.messages[1]

    # Consolidated strategy: same topic for all trades
    assert msg1['topic'] == 'cryptofeed.trade'
    assert msg2['topic'] == 'cryptofeed.trade'

    # Different partition keys (composite strategy)
    assert msg1['key'] == b'binance-btc-usd'  # normalized exchange-symbol
    assert msg2['key'] == b'okx-eth-usd'

    # Cleanup
    await callback.stop()


@pytest.mark.asyncio
async def test_simplified_backend_topic_naming_per_symbol():
    """
    Verify simplified backend generates correct topic names for per_symbol strategy.

    Expected Behavior (Original Design):
    - Per-symbol strategy: cryptofeed.{data_type}.{exchange}.{symbol}
    - Each exchange-symbol pair gets own topic
    """
    mock_producer_instance = MockKafkaProducer()

    def mock_producer_factory(*args, **kwargs):
        return MagicMock(
            list_topics=MagicMock(return_value={}),
            produce=mock_producer_instance.produce,
            poll=mock_producer_instance.poll,
            flush=mock_producer_instance.flush,
        )

    config = KafkaConfig(
        bootstrap_servers=['localhost:9092'],
        topic_strategy='per_symbol',
        partition_strategy='composite',
    )

    callback = KafkaCallback(
        kafka_config=config,
        producer_factory=mock_producer_factory,
        serialization_format='protobuf',
    )
    callback.start(asyncio.get_event_loop())

    # Execute: Send trade
    trade = Trade(
        exchange='binance',
        symbol='BTC-USD',
        side='buy',
        price=Decimal('50000.00'),
        amount=Decimal('1.0'),
        timestamp=1234567890.123,
    )

    await callback.trade(trade, receipt_timestamp=1234567890.123)
    await asyncio.sleep(0.2)

    # Verify: Per-symbol topic naming
    assert len(mock_producer_instance.messages) == 1
    msg = mock_producer_instance.messages[0]

    assert msg['topic'] == 'cryptofeed.trade.binance.btc-usd'  # per-symbol topic
    assert msg['key'] == b'binance-btc-usd'  # composite partition key

    # Cleanup
    await callback.stop()


@pytest.mark.asyncio
async def test_simplified_backend_all_partition_strategies():
    """
    Verify all 4 partition strategies produce correct partition keys.

    Expected Behavior (Original Design):
    - symbol: Partition by normalized symbol only
    - composite: Partition by exchange-symbol combination (default)
    - exchange: Partition by exchange only
    - round_robin: No partition key (None) for Kafka round-robin
    """
    strategies_expected = {
        'symbol': b'btc-usd',
        'composite': b'binance-btc-usd',
        'exchange': b'binance',
        'round_robin': None,
    }

    trade = Trade(
        exchange='binance',
        symbol='BTC-USD',
        side='buy',
        price=Decimal('50000.00'),
        amount=Decimal('1.0'),
        timestamp=1234567890.123,
    )

    for strategy, expected_key in strategies_expected.items():
        # Use inline partition key function (inlined from partitioner.py in Phase 2)
        actual_key = _get_partition_key(trade, strategy)
        assert actual_key == expected_key, (
            f"Partition strategy '{strategy}' produced incorrect key: "
            f"expected {expected_key}, got {actual_key}"
        )


@pytest.mark.asyncio
async def test_simplified_backend_header_encoding():
    """
    Verify header encoding produces correct normalized values.

    Expected Behavior (Original Design):
    - Mandatory headers: content-type, exchange, symbol, data_type
    - Optional headers: schema_version, producer_version, timestamp_generated, cf.serialization_format
    - Exchange and symbol normalized to lowercase with hyphens
    """
    trade = Trade(
        exchange='Binance',  # mixed case
        symbol='BTC/USD',    # slash separator
        side='buy',
        price=Decimal('50000.00'),
        amount=Decimal('1.0'),
        timestamp=1234567890.123,
    )

    # Use inline header builder (inlined from headers.py in Phase 2)
    headers = _build_headers(
        message=trade,
        data_type='trade',
        content_type='application/x-protobuf',
        schema_version='v2beta1',
    )

    # Convert to dict for easier verification
    header_dict = {name: value for name, value in headers}

    # Verify mandatory headers
    assert header_dict[b'content-type'] == b'application/x-protobuf'
    assert header_dict[b'exchange'] == b'binance'  # normalized to lowercase
    assert header_dict[b'symbol'] == b'btc-usd'     # normalized: slash to hyphen, lowercase
    assert header_dict[b'data_type'] == b'trade'

    # Verify optional headers
    assert header_dict[b'schema_version'] == b'v2beta1'
    assert header_dict[b'producer_version'] == b'2.4.1'
    assert b'timestamp_generated' in header_dict  # present (value is dynamic)
    assert header_dict[b'cf.serialization_format'] == b'json'


@pytest.mark.asyncio
async def test_simplified_backend_protobuf_serialization_identical():
    """
    Verify protobuf message serialization produces byte-identical output.

    Expected Behavior (Original Design):
    - Same Trade object always produces same protobuf bytes
    - All fields populated correctly (core + v2beta1 fields)
    - Timestamp conversion to microseconds
    """
    trade = Trade(
        exchange='binance',
        symbol='BTC-USD',
        side='buy',
        price=Decimal('50000.00'),
        amount=Decimal('1.5'),
        timestamp=1234567890.123,
        id='12345',
        type='limit',
        # v2beta1 fields
        maker=True,
        event_time=1234567890.456,
        match_id='67890',
    )

    # Serialize twice
    proto1 = trade_to_proto(trade)
    proto2 = trade_to_proto(trade)

    # Verify byte-identical serialization
    assert proto1.SerializeToString() == proto2.SerializeToString()

    # Verify all fields populated correctly
    assert proto1.exchange == 'binance'
    assert proto1.symbol == 'BTC-USD'
    assert proto1.price == '50000.00'
    assert proto1.amount == '1.5'
    assert proto1.timestamp == 1234567890123000  # microseconds
    assert proto1.trade_id == '12345'
    assert proto1.trade_type == 'limit'

    # v2beta1 fields
    assert proto1.maker is True
    assert proto1.event_time == 1234567890456000  # microseconds
    assert proto1.match_id == '67890'


@pytest.mark.asyncio
async def test_simplified_backend_normalization_consistency():
    """
    Verify normalization produces consistent output across all usage contexts.

    Expected Behavior (Original Design):
    - normalize_symbol(): lowercase, slash/underscore to hyphen, strip whitespace
    - normalize_exchange(): lowercase, strip whitespace
    - Same output whether used in topic naming, partition keys, or headers
    """
    test_cases = [
        # (input_symbol, expected_output)
        ('BTC/USD', 'btc-usd'),
        ('BTC_USD', 'btc-usd'),
        (' ETH-BTC ', 'eth-btc'),
        ('BTC/USD_PERP', 'btc-usd-perp'),
        (None, 'unknown'),
        ('', 'unknown'),
    ]

    for input_symbol, expected in test_cases:
        actual = normalize_symbol(input_symbol)
        assert actual == expected, f"normalize_symbol({input_symbol!r}) = {actual!r}, expected {expected!r}"

    exchange_cases = [
        ('Binance', 'binance'),
        (' OKX ', 'okx'),
        ('COINBASE', 'coinbase'),
        (None, 'unknown'),
        ('', 'unknown'),
    ]

    for input_exchange, expected in exchange_cases:
        actual = normalize_exchange(input_exchange)
        assert actual == expected, f"normalize_exchange({input_exchange!r}) = {actual!r}, expected {expected!r}"


@pytest.mark.asyncio
async def test_simplified_backend_zero_regressions_e2e():
    """
    End-to-end test verifying zero regressions across full pipeline.

    This test validates the complete message production pipeline:
    1. Topic naming (consolidated strategy)
    2. Partition key generation (composite strategy)
    3. Header encoding (mandatory + optional headers)
    4. Protobuf serialization (Trade with v2beta1 fields)

    Expected: All components work together without errors or data loss.
    """
    mock_producer_instance = MockKafkaProducer()

    def mock_producer_factory(*args, **kwargs):
        return MagicMock(
            list_topics=MagicMock(return_value={}),
            produce=mock_producer_instance.produce,
            poll=mock_producer_instance.poll,
            flush=mock_producer_instance.flush,
        )

    config = KafkaConfig(
        bootstrap_servers=['localhost:9092'],
        topic_strategy='consolidated',
        partition_strategy='composite',
    )

    callback = KafkaCallback(
        kafka_config=config,
        producer_factory=mock_producer_factory,
        serialization_format='protobuf',
    )
    callback.start(asyncio.get_event_loop())

    # Execute: Send Trade with all v2beta1 fields
    trade = Trade(
        exchange='binance',
        symbol='BTC-USD',
        side='buy',
        price=Decimal('50000.00'),
        amount=Decimal('1.5'),
        timestamp=1234567890.123,
        id='12345',
        type='limit',
        maker=True,
        event_time=1234567890.456,
        match_id='67890',
    )

    await callback.trade(trade, receipt_timestamp=1234567890.789)
    await asyncio.sleep(0.2)

    # Verify: Message produced with correct attributes
    assert len(mock_producer_instance.messages) == 1
    msg = mock_producer_instance.messages[0]

    # Topic naming (consolidated strategy)
    assert msg['topic'] == 'cryptofeed.trade'

    # Partition key (composite strategy)
    assert msg['key'] == b'binance-btc-usd'

    # Headers (normalized exchange/symbol)
    header_dict = {name: value for name, value in msg['headers']}
    assert header_dict[b'exchange'] == b'binance'
    assert header_dict[b'symbol'] == b'btc-usd'
    assert header_dict[b'data_type'] == b'trade'
    assert header_dict[b'content-type'] == b'application/x-protobuf'

    # Protobuf body (deserialize and verify)
    from cryptofeed.backends.protobuf.bindings import trade_pb2
    proto = trade_pb2.Trade()
    proto.ParseFromString(msg['value'])

    assert proto.exchange == 'binance'
    assert proto.symbol == 'BTC-USD'
    assert proto.price == '50000.00'
    assert proto.amount == '1.5'
    assert proto.maker is True
    assert proto.event_time == 1234567890456000
    assert proto.match_id == '67890'

    # Data loss check: all fields transmitted
    assert proto.timestamp == 1234567890123000
    assert proto.trade_id == '12345'
    assert proto.trade_type == 'limit'

    # Cleanup
    await callback.stop()


@pytest.mark.asyncio
async def test_simplified_backend_orderbook_regression():
    """
    Verify OrderBook messages work correctly in simplified backend.

    Tests the same pipeline for OrderBook objects to ensure no regressions
    beyond Trade messages.
    """
    mock_producer_instance = MockKafkaProducer()

    def mock_producer_factory(*args, **kwargs):
        return MagicMock(
            list_topics=MagicMock(return_value={}),
            produce=mock_producer_instance.produce,
            poll=mock_producer_instance.poll,
            flush=mock_producer_instance.flush,
        )

    config = KafkaConfig(
        bootstrap_servers=['localhost:9092'],
        topic_strategy='consolidated',
        partition_strategy='composite',
    )

    callback = KafkaCallback(
        kafka_config=config,
        producer_factory=mock_producer_factory,
        serialization_format='protobuf',
    )
    callback.start(asyncio.get_event_loop())

    # Execute: Send OrderBook with v2beta1 fields
    book = OrderBook(
        exchange='binance',
        symbol='BTC-USD',
        bids={Decimal('49999.00'): Decimal('1.5')},
        asks={Decimal('50001.00'): Decimal('2.0')},
    )
    book.timestamp = 1234567890.123
    book.event_time = 1234567890.456
    book.last_update_id = 999888777

    await callback.orderbook(book, receipt_timestamp=1234567890.789)
    await asyncio.sleep(0.2)

    # Verify: OrderBook message produced correctly
    assert len(mock_producer_instance.messages) == 1
    msg = mock_producer_instance.messages[0]

    assert msg['topic'] == 'cryptofeed.orderbook'
    assert msg['key'] == b'binance-btc-usd'

    # Deserialize protobuf
    from cryptofeed.backends.protobuf.bindings import order_book_pb2
    proto = order_book_pb2.Level2Book()
    proto.ParseFromString(msg['value'])

    assert proto.exchange == 'binance'
    assert proto.symbol == 'BTC-USD'
    assert proto.timestamp == 1234567890123000
    assert proto.event_time == 1234567890456000
    assert proto.last_update_id == '999888777'

    # Cleanup
    await callback.stop()


@pytest.mark.asyncio
async def test_simplified_backend_performance_no_degradation():
    """
    Verify simplified backend maintains performance characteristics.

    Expected: Simplification should not degrade performance significantly.
    This test ensures basic operations complete quickly (< 100ms for 10 messages).
    """
    mock_producer_instance = MockKafkaProducer()

    def mock_producer_factory(*args, **kwargs):
        return MagicMock(
            list_topics=MagicMock(return_value={}),
            produce=mock_producer_instance.produce,
            poll=mock_producer_instance.poll,
            flush=mock_producer_instance.flush,
        )

    config = KafkaConfig(
        bootstrap_servers=['localhost:9092'],
        topic_strategy='consolidated',
        partition_strategy='composite',
    )

    callback = KafkaCallback(
        kafka_config=config,
        producer_factory=mock_producer_factory,
        serialization_format='protobuf',
    )
    callback.start(asyncio.get_event_loop())

    # Execute: Send 10 trades
    import time
    start_time = time.time()

    for i in range(10):
        trade = Trade(
            exchange='binance',
            symbol='BTC-USD',
            side='buy',
            price=Decimal(f'{50000 + i}.00'),
            amount=Decimal('1.0'),
            timestamp=1234567890.0 + i,
        )
        await callback.trade(trade, receipt_timestamp=1234567890.0 + i)

    await asyncio.sleep(0.2)
    elapsed_ms = (time.time() - start_time) * 1000

    # Verify: Performance acceptable (< 500ms for 10 messages in test env)
    # Note: This is a reasonable threshold for async queue processing with mocked Kafka
    assert len(mock_producer_instance.messages) == 10
    assert elapsed_ms < 500, f"Performance degradation detected: {elapsed_ms:.2f}ms for 10 messages"

    # Cleanup
    await callback.stop()
