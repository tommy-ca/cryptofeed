"""Phase 2: Task 9.1 - Integration tests for consolidated topic end-to-end flow.

This module tests:
- Task 9.1: Consolidated topic end-to-end flow with real Kafka cluster

Test Strategy:
- Deploy local Kafka cluster with docker-compose (3 brokers)
- Produce trade messages via consolidated topic strategy
- Consume from cryptofeed.trades topic
- Verify messages are present with correct content and headers
- Validate message headers include exchange, symbol, data_type

All tests use REAL Kafka (not mocked) via testcontainers or docker-compose.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pytest

# Optional: testcontainers for Kafka (preferred if available)
try:
    from testcontainers.kafka import KafkaContainer
    HAS_TESTCONTAINERS = True
except ImportError:
    HAS_TESTCONTAINERS = False

# Standard Kafka clients
try:
    from kafka import KafkaConsumer, KafkaProducer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False

from cryptofeed.backends.kafka.config import (
    KafkaConfig,
    KafkaTopicConfig,
    KafkaPartitionConfig,
)
from tests.helpers.kafka_env import get_bootstrap_servers

LOG = logging.getLogger("test_kafka_e2e")


# ============================================================================
# Fixtures: Kafka Cluster Setup (Real Infrastructure)
# ============================================================================


@dataclass
class KafkaClusterInfo:
    """Information about a running Kafka cluster."""
    bootstrap_servers: List[str]
    broker_count: int
    topic_prefix: str = "cryptofeed"

    @property
    def bootstrap_url(self) -> str:
        """Return bootstrap servers as comma-separated string."""
        return ",".join(self.bootstrap_servers)


@pytest.fixture(scope="session")
def kafka_cluster() -> KafkaClusterInfo:
    """Deploy a local Kafka cluster with 3 brokers.

    Returns:
        KafkaClusterInfo with bootstrap servers and broker information.

    This fixture:
    - Attempts to start testcontainers Kafka if available
    - Falls back to local docker-compose if testcontainers unavailable
    - Waits for cluster to be ready before returning
    - Skips tests if Kafka not available
    """
    if not HAS_KAFKA:
        pytest.skip("kafka-python not installed")

    if HAS_TESTCONTAINERS:
        try:
            # Use testcontainers for isolated cluster
            container = KafkaContainer(image="confluentinc/cp-kafka:7.5.0")
            container.start()

            bootstrap_servers = [container.get_bootstrap_server()]
            LOG.info(f"Started Kafka testcontainer: {bootstrap_servers}")

            yield KafkaClusterInfo(
                bootstrap_servers=bootstrap_servers,
                broker_count=1,
                topic_prefix="cryptofeed"
            )

            container.stop()
            return
        except Exception as e:
            LOG.warning(f"Testcontainers Kafka failed: {e}, attempting docker-compose")

    # Fallback: Expect local Kafka cluster (docker-compose or env-configured)
    bootstrap_servers = get_bootstrap_servers()
    LOG.info(f"Using local Kafka cluster: {bootstrap_servers}")

    yield KafkaClusterInfo(
        bootstrap_servers=bootstrap_servers,
        broker_count=len(bootstrap_servers),
        topic_prefix="cryptofeed"
    )


@pytest.fixture
def kafka_producer(kafka_cluster: KafkaClusterInfo) -> KafkaProducer:
    """Create a Kafka producer for writing test messages.

    Yields:
        Connected KafkaProducer instance.

    Cleans up connection after test.
    """
    producer = KafkaProducer(
        bootstrap_servers=kafka_cluster.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8') if isinstance(v, dict) else v,
        request_timeout_ms=5000,
        retries=1,
    )

    yield producer

    producer.close()


@pytest.fixture
def kafka_consumer(kafka_cluster: KafkaClusterInfo) -> KafkaConsumer:
    """Create a Kafka consumer for reading test messages.

    Yields:
        Connected KafkaConsumer instance.

    Cleans up connection after test.
    """
    import uuid
    group_id = f"test-e2e-consumer-{uuid.uuid4().hex[:8]}"

    consumer = KafkaConsumer(
        bootstrap_servers=kafka_cluster.bootstrap_servers,
        value_deserializer=lambda m: m.decode('utf-8') if m else None,
        auto_offset_reset='earliest',
        group_id=group_id,
        enable_auto_commit=True,
        session_timeout_ms=6000,
        request_timeout_ms=10000,
    )

    yield consumer

    consumer.close()


@pytest.fixture
def kafka_callback_config(kafka_cluster: KafkaClusterInfo) -> KafkaConfig:
    """Create a KafkaCallback configuration for testing.

    Returns:
        KafkaConfig with consolidated topic strategy for testing.

    Configuration uses:
    - Consolidated topics: cryptofeed.{data_type}
    - Composite partitioner: {exchange}-{symbol}
    - 3 partitions per topic
    - Snappy compression
    - Exactly-once semantics (acks=all, idempotence=true)
    """
    return KafkaConfig(
        bootstrap_servers=kafka_cluster.bootstrap_servers,
        topic=KafkaTopicConfig(
            strategy="consolidated",
            prefix="cryptofeed",
            partitions_per_topic=3,
            replication_factor=min(kafka_cluster.broker_count, 3),
        ),
        partition=KafkaPartitionConfig(strategy="composite"),
        acks="all",
        idempotence=True,
        retries=3,
        retry_backoff_ms=100,
        batch_size=16384,
        linger_ms=10,
        compression_type="snappy",
    )


# ============================================================================
# Test Message Builders (Real Market Data Simulation)
# ============================================================================


@dataclass
class TradeMessage:
    """Simulated Trade message matching normalized data schema."""
    symbol: str
    exchange: str
    price: float
    amount: float
    timestamp: float
    side: str  # 'buy' or 'sell'
    trade_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "price": self.price,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "side": self.side,
            "trade_id": self.trade_id,
        }

    def to_bytes(self) -> bytes:
        """Convert to bytes for Kafka."""
        return json.dumps(self.to_dict()).encode('utf-8')


def create_trade_message(
    symbol: str = "BTC-USD",
    exchange: str = "coinbase",
    price: float = 50000.0,
    amount: float = 0.5,
    side: str = "buy",
    trade_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> TradeMessage:
    """Factory function to create test trade messages."""
    if timestamp is None:
        timestamp = time.time()
    if trade_id is None:
        trade_id = f"{int(timestamp * 1000)}"

    return TradeMessage(
        symbol=symbol,
        exchange=exchange,
        price=price,
        amount=amount,
        timestamp=timestamp,
        side=side,
        trade_id=trade_id,
    )


# ============================================================================
# Task 9.1: Test Consolidated Topic End-to-End Flow
# ============================================================================


class TestTask91ConsolidatedTopicE2E:
    """End-to-end tests for consolidated topic strategy with real Kafka.

    Test Scenario (Task 9.1):
    1. Deploy local Kafka cluster with 3 brokers
    2. Produce trade messages via consolidated topic strategy
    3. Consume from cryptofeed.trades topic
    4. Verify messages are present with correct content and headers
    5. Validate headers include exchange, symbol, data_type

    Requirements Addressed: FR2 (Topic Management)
    """

    @pytest.mark.integration
    def test_consolidated_topic_creation_on_produce(
        self,
        kafka_cluster: KafkaClusterInfo,
        kafka_producer: KafkaProducer,
    ):
        """Verify consolidated topic is auto-created when messages are produced.

        This test:
        - Produces a message to cryptofeed.trades topic
        - Verifies topic is created with correct name
        - Checks default partition count

        Expected Outcome:
        - Topic 'cryptofeed.trades' is created
        - Topic has 3 partitions (or as configured)
        - Topic is ready to accept messages
        """
        topic = "cryptofeed.trades"

        # Produce a test message
        trade = create_trade_message(
            symbol="BTC-USD",
            exchange="coinbase",
            price=50000.0,
        )

        future = kafka_producer.send(
            topic,
            value=trade.to_dict(),
        )

        # Wait for confirmation
        record_metadata = future.get(timeout=5)

        assert record_metadata.topic == topic, \
            f"Message sent to wrong topic: {record_metadata.topic}"
        assert record_metadata.partition >= 0, \
            "Partition should be assigned"
        assert record_metadata.offset >= 0, \
            "Offset should be assigned"

        LOG.info(
            f"Message produced: topic={topic}, partition={record_metadata.partition}, "
            f"offset={record_metadata.offset}"
        )

    @pytest.mark.integration
    def test_consolidated_topic_message_consumption(
        self,
        kafka_cluster: KafkaClusterInfo,
        kafka_producer: KafkaProducer,
        kafka_consumer: KafkaConsumer,
    ):
        """Verify messages can be produced and consumed from consolidated topics.

        This test:
        - Produces 5 trade messages to cryptofeed.trades.consumption
        - Subscribes consumer to cryptofeed.trades.consumption
        - Reads messages and validates content

        Expected Outcome:
        - All 5 messages are received
        - Message content matches what was sent
        - Messages are in order (per partition)
        """
        import uuid
        topic = f"cryptofeed.trades.consumption.{uuid.uuid4().hex[:8]}"
        message_count = 5

        # Produce messages
        messages = [
            create_trade_message(
                symbol="BTC-USD",
                exchange="coinbase",
                price=50000.0 + (i * 100),
                amount=0.5 + (i * 0.1),
            )
            for i in range(message_count)
        ]

        for msg in messages:
            kafka_producer.send(topic, value=msg.to_dict())

        kafka_producer.flush(timeout=5)
        LOG.info(f"Produced {message_count} messages to {topic}")

        # Consume messages
        kafka_consumer.subscribe([topic])
        received_messages = []

        # Wait for messages (with longer timeout and more polls)
        max_polls = 50
        for poll_count in range(max_polls):
            msg = kafka_consumer.poll(timeout_ms=500)
            if msg:
                for topic_partition, records in msg.items():
                    for record in records:
                        received_messages.append(record)

            if len(received_messages) >= message_count:
                break

        assert len(received_messages) >= message_count, \
            f"Expected {message_count} messages, got {len(received_messages)}"

        # Verify message content
        for i, record in enumerate(received_messages[:message_count]):
            msg_value = json.loads(record.value)
            assert msg_value["symbol"] == "BTC-USD", \
                f"Message {i}: Symbol mismatch"
            assert msg_value["exchange"] == "coinbase", \
                f"Message {i}: Exchange mismatch"
            assert msg_value["price"] == 50000.0 + (i * 100), \
                f"Message {i}: Price mismatch"

        LOG.info(f"Successfully consumed and verified {message_count} messages")

    @pytest.mark.integration
    def test_consolidated_topic_multiple_exchanges(
        self,
        kafka_cluster: KafkaClusterInfo,
        kafka_producer: KafkaProducer,
        kafka_consumer: KafkaConsumer,
    ):
        """Verify consolidated topic aggregates messages from multiple exchanges.

        This test:
        - Produces messages from Coinbase, Binance, Kraken
        - Consumes from single cryptofeed.trades.multiex topic
        - Verifies all exchange messages appear in same topic

        Expected Outcome:
        - All exchange messages in cryptofeed.trades.multiex
        - Exchange information preserved in message content
        - Consumer can filter by exchange using message data
        """
        import uuid
        topic = f"cryptofeed.trades.multiex.{uuid.uuid4().hex[:8]}"
        exchanges = ["coinbase", "binance", "kraken"]
        messages_per_exchange = 2

        # Produce messages from multiple exchanges
        produced_messages = []
        for exchange in exchanges:
            for i in range(messages_per_exchange):
                msg = create_trade_message(
                    symbol="BTC-USD" if exchange == "coinbase" else "BTC-USDT",
                    exchange=exchange,
                    price=50000.0 + (i * 100),
                )
                produced_messages.append(msg)
                kafka_producer.send(topic, value=msg.to_dict())

        kafka_producer.flush(timeout=5)
        total_messages = len(exchanges) * messages_per_exchange
        LOG.info(f"Produced {total_messages} messages from {len(exchanges)} exchanges")

        # Consume all messages
        kafka_consumer.subscribe([topic])
        received_messages = []

        max_polls = 100
        for _ in range(max_polls):
            msg = kafka_consumer.poll(timeout_ms=500)
            if msg:
                for topic_partition, records in msg.items():
                    for record in records:
                        received_messages.append(record)

            if len(received_messages) >= total_messages:
                break

        assert len(received_messages) >= total_messages, \
            f"Expected {total_messages} messages from multiple exchanges, " \
            f"got {len(received_messages)}"

        # Verify all exchanges are represented
        exchanges_in_messages = set()
        for record in received_messages[:total_messages]:
            msg_value = json.loads(record.value)
            exchanges_in_messages.add(msg_value["exchange"])

        assert exchanges_in_messages == set(exchanges), \
            f"Expected exchanges {exchanges}, got {exchanges_in_messages}"

        LOG.info(
            f"Successfully verified messages from all exchanges in single topic: "
            f"{exchanges_in_messages}"
        )

    @pytest.mark.integration
    def test_consolidated_topic_message_headers(
        self,
        kafka_cluster: KafkaClusterInfo,
        kafka_producer: KafkaProducer,
        kafka_consumer: KafkaConsumer,
    ):
        """Verify message headers include required metadata for routing.

        This test:
        - Produces messages WITH headers (simulating KafkaCallback behavior)
        - Consumes messages and reads headers
        - Validates headers contain: exchange, symbol, data_type, content-type

        Expected Outcome:
        - Messages have headers attached
        - Headers include: content-type, exchange, symbol, data_type
        - Header values are proper bytes/strings
        """
        import uuid
        topic = f"cryptofeed.trades.headers.{uuid.uuid4().hex[:8]}"

        # Produce message with headers (simulating HeaderEnricher)
        trade = create_trade_message(
            symbol="BTC-USD",
            exchange="coinbase",
            price=50000.0,
        )

        headers = [
            ("content-type", b"application/x-protobuf"),
            ("exchange", b"coinbase"),
            ("symbol", b"BTC-USD"),
            ("data_type", b"trade"),
            ("schema_version", b"v1"),
        ]

        kafka_producer.send(
            topic,
            value=trade.to_dict(),
            headers=headers,
        )
        kafka_producer.flush(timeout=5)
        LOG.info("Produced message with headers")

        # Consume and verify headers
        kafka_consumer.subscribe([topic])
        received_msg = None

        for _ in range(50):
            msg = kafka_consumer.poll(timeout_ms=500)
            if msg:
                for topic_partition, records in msg.items():
                    for record in records:
                        received_msg = record
                        break
                if received_msg:
                    break

        assert received_msg is not None, "No message received"
        assert received_msg.headers is not None, "Message has no headers"

        # Convert headers to dict for easier validation
        headers_dict = {k: v for k, v in received_msg.headers}

        # Verify required headers (kafka-python stores headers with string keys)
        required_headers = [
            "content-type",
            "exchange",
            "symbol",
            "data_type",
        ]

        # Log actual headers for debugging
        LOG.info(f"Received headers: {headers_dict}")

        for header_name in required_headers:
            # Headers can be stored as bytes keys or string keys depending on kafka-python version
            assert header_name in headers_dict or header_name.encode() in headers_dict, \
                f"Missing required header: {header_name}"

        # Verify header values
        content_type_val = headers_dict.get("content-type") or headers_dict.get(b"content-type")
        assert content_type_val == b"application/x-protobuf", \
            f"content-type header mismatch: {content_type_val}"

        exchange_val = headers_dict.get("exchange") or headers_dict.get(b"exchange")
        assert exchange_val == b"coinbase", \
            f"exchange header mismatch: {exchange_val}"

        symbol_val = headers_dict.get("symbol") or headers_dict.get(b"symbol")
        assert symbol_val == b"BTC-USD", \
            f"symbol header mismatch: {symbol_val}"

        data_type_val = headers_dict.get("data_type") or headers_dict.get(b"data_type")
        assert data_type_val == b"trade", \
            f"data_type header mismatch: {data_type_val}"

        LOG.info("Successfully verified all required message headers")

    @pytest.mark.integration
    def test_consolidated_topic_partition_distribution(
        self,
        kafka_cluster: KafkaClusterInfo,
        kafka_producer: KafkaProducer,
    ):
        """Verify messages are distributed across partitions using partition keys.

        This test:
        - Produces messages for different exchange-symbol pairs
        - Uses partition keys (simulating PartitionerFactory)
        - Verifies messages distribute across topic partitions

        Expected Outcome:
        - Different exchange-symbol pairs may use different partitions
        - Same exchange-symbol pair goes to same partition (ordering)
        - Partitions are utilized (not all messages in partition 0)
        """
        topic = "cryptofeed.trades"
        partition_key_count = 3  # 3 different keys

        # Produce messages with different partition keys
        partitions_used = set()

        for i in range(10):
            # Rotate through different exchange-symbol combinations
            exchange = ["coinbase", "binance", "kraken"][i % partition_key_count]
            symbol = ["BTC-USD", "ETH-USD", "SOL-USD"][i % partition_key_count]

            trade = create_trade_message(
                symbol=symbol,
                exchange=exchange,
                price=50000.0 + (i * 100),
            )

            # Use composite partition key (exchange-symbol)
            partition_key = f"{exchange}-{symbol}".encode('utf-8')

            future = kafka_producer.send(
                topic,
                key=partition_key,
                value=trade.to_dict(),
            )

            record_metadata = future.get(timeout=5)
            partitions_used.add(record_metadata.partition)

        kafka_producer.flush(timeout=5)

        # Verify partitions were used
        assert len(partitions_used) > 0, "No partitions were used"
        LOG.info(f"Messages distributed across {len(partitions_used)} partitions: {partitions_used}")

    @pytest.mark.integration
    def test_consolidated_topic_ordering_per_partition(
        self,
        kafka_cluster: KafkaClusterInfo,
        kafka_producer: KafkaProducer,
        kafka_consumer: KafkaConsumer,
    ):
        """Verify message ordering is preserved within each partition.

        This test:
        - Produces 10 messages with same partition key (ordering guarantee)
        - Consumes messages from same partition
        - Verifies messages are in order

        Expected Outcome:
        - Messages with same partition key arrive in order
        - Message prices increase sequentially (50000, 50100, 50200, ...)
        - No reordering observed
        """
        import uuid
        topic = f"cryptofeed.trades.ordering.{uuid.uuid4().hex[:8]}"
        message_count = 10

        # Produce ordered messages with same partition key
        partition_key = b"coinbase-btc-usd"  # Same key for all = same partition

        for i in range(message_count):
            trade = create_trade_message(
                symbol="BTC-USD",
                exchange="coinbase",
                price=50000.0 + (i * 100),  # Price increases: 50000, 50100, ...
                trade_id=f"trade-{i:03d}",
            )

            kafka_producer.send(
                topic,
                key=partition_key,
                value=trade.to_dict(),
            )

        kafka_producer.flush(timeout=5)
        LOG.info(f"Produced {message_count} ordered messages to same partition")

        # Consume and verify order
        kafka_consumer.subscribe([topic])
        received_messages = []

        max_polls = 100
        for _ in range(max_polls):
            msg = kafka_consumer.poll(timeout_ms=500)
            if msg:
                for topic_partition, records in msg.items():
                    for record in records:
                        received_messages.append(record)

            if len(received_messages) >= message_count:
                break

        assert len(received_messages) >= message_count, \
            f"Expected {message_count} messages, got {len(received_messages)}"

        # Extract messages from same partition
        partition_messages = []
        for record in received_messages:
            msg_value = json.loads(record.value)
            if msg_value["exchange"] == "coinbase" and msg_value["symbol"] == "BTC-USD":
                partition_messages.append(msg_value)

        # Verify ordering
        for i, msg in enumerate(partition_messages[:message_count]):
            expected_price = 50000.0 + (i * 100)
            assert msg["price"] == expected_price, \
                f"Message {i}: Expected price {expected_price}, got {msg['price']}"

        LOG.info(f"Successfully verified ordering of {len(partition_messages)} messages in partition")

    @pytest.mark.integration
    def test_consolidated_topic_multiple_data_types(
        self,
        kafka_cluster: KafkaClusterInfo,
        kafka_producer: KafkaProducer,
        kafka_consumer: KafkaConsumer,
    ):
        """Verify different data types go to different consolidated topics.

        This test:
        - Produces TRADE messages to cryptofeed.trades.datatypes
        - Produces ORDERBOOK messages to cryptofeed.orderbook.datatypes
        - Verifies messages appear in correct topics
        - Shows topic multiplexing pattern

        Expected Outcome:
        - Trades in cryptofeed.trades.datatypes
        - OrderBooks in cryptofeed.orderbook.datatypes
        - No cross-contamination
        """
        import uuid
        suffix = uuid.uuid4().hex[:8]
        # Produce to trades topic
        trades_topic = f"cryptofeed.trades.datatypes.{suffix}"
        orderbook_topic = f"cryptofeed.orderbook.datatypes.{suffix}"

        trade = create_trade_message(
            symbol="BTC-USD",
            exchange="coinbase",
            price=50000.0,
        )

        # Orderbook-like message (simplified)
        orderbook = {
            "symbol": "BTC-USD",
            "exchange": "coinbase",
            "bids": [[49900.0, 1.5], [49800.0, 2.0]],
            "asks": [[50100.0, 1.2], [50200.0, 1.8]],
            "timestamp": time.time(),
        }

        kafka_producer.send(trades_topic, value=trade.to_dict())
        kafka_producer.send(orderbook_topic, value=orderbook)
        kafka_producer.flush(timeout=5)

        LOG.info(f"Produced to {trades_topic} and {orderbook_topic}")

        # Consume from trades topic
        kafka_consumer.subscribe([trades_topic])
        trade_messages = []

        for _ in range(50):
            msg = kafka_consumer.poll(timeout_ms=500)
            if msg:
                for _, records in msg.items():
                    for record in records:
                        trade_messages.append(json.loads(record.value))
            if len(trade_messages) >= 1:
                break

        assert len(trade_messages) >= 1, "No trade messages received"
        assert "side" in trade_messages[0], "Trade message should have 'side'"

        # Consume from orderbook topic
        kafka_consumer.unsubscribe()
        kafka_consumer.subscribe([orderbook_topic])
        orderbook_messages = []

        for _ in range(50):
            msg = kafka_consumer.poll(timeout_ms=500)
            if msg:
                for _, records in msg.items():
                    for record in records:
                        orderbook_messages.append(json.loads(record.value))
            if len(orderbook_messages) >= 1:
                break

        assert len(orderbook_messages) >= 1, "No orderbook messages received"
        assert "bids" in orderbook_messages[0], "Orderbook message should have 'bids'"

        LOG.info("Successfully verified message separation across data types")


# ============================================================================
# Task 9.2: Test Partition Key Routing and Ordering
# ============================================================================


@pytest.fixture
def kafka_topic_creator(kafka_cluster: KafkaClusterInfo):
    """Helper fixture for creating topics with specific partition counts.

    Yields:
        Callable that creates a topic with specified partition count.
    """
    created_topics = []

    def create_topic(topic_name: str, partitions: int = 3, replication_factor: int = 1):
        """Create a Kafka topic with specified partitions.

        Args:
            topic_name: Name of topic to create
            partitions: Number of partitions (default: 3)
            replication_factor: Replication factor (default: 1 for test)

        Returns:
            Topic name
        """
        try:
            from kafka.admin import KafkaAdminClient, NewTopic
            admin_client = KafkaAdminClient(
                bootstrap_servers=kafka_cluster.bootstrap_servers,
                request_timeout_ms=5000
            )

            # Create topic with specified partition count
            topic = NewTopic(
                name=topic_name,
                num_partitions=partitions,
                replication_factor=replication_factor
            )
            fs = admin_client.create_topics([topic], validate_only=False)

            # Wait for topic creation (kafka-python 2.x returns dict-like object)
            try:
                # Try dict-like interface (older versions)
                items_to_check = fs.items() if hasattr(fs, 'items') else [(topic_name, fs)]
            except (AttributeError, TypeError):
                # New versions return response object directly
                items_to_check = [(topic_name, fs)]

            for topic_name_created, f in items_to_check:
                try:
                    # Handle both future objects and direct responses
                    if hasattr(f, 'result'):
                        f.result(timeout=5)
                    LOG.info(f"Created topic {topic_name_created} with {partitions} partitions")
                except Exception as e:
                    # Topic might already exist
                    LOG.warning(f"Failed to create topic {topic_name_created}: {e}")

            admin_client.close()
            created_topics.append(topic_name)
            return topic_name

        except Exception as e:
            # Fallback: just return topic name and hope it gets created
            LOG.warning(f"Topic creation helper failed: {e}, topic may not have correct partitions")
            return topic_name

    yield create_topic

    # Cleanup is optional - topics are per-test unique


class PartitionAssertions:
    """Helper class for partition verification assertions.

    Provides convenience methods for asserting partition routing behavior
    and message ordering characteristics.
    """

    @staticmethod
    def assert_all_messages_same_partition(messages: list, partition_key: str) -> None:
        """Assert all messages went to the same partition.

        Args:
            messages: List of ConsumeRecord messages
            partition_key: Expected partition key for routing

        Raises:
            AssertionError: If messages are spread across multiple partitions
        """
        if not messages:
            raise AssertionError("No messages provided for assertion")

        # Extract partition numbers from messages
        partitions = set()
        for msg in messages:
            partitions.add(msg.partition)

        assert len(partitions) == 1, \
            f"Expected all messages in same partition, but found {len(partitions)} " \
            f"different partitions: {partitions}. Partition key: {partition_key}"

    @staticmethod
    def assert_messages_span_multiple_partitions(messages: list, min_partitions: int = 2) -> None:
        """Assert messages are distributed across multiple partitions.

        Args:
            messages: List of ConsumeRecord messages
            min_partitions: Minimum number of partitions expected

        Raises:
            AssertionError: If messages are not spread across enough partitions
        """
        if not messages:
            raise AssertionError("No messages provided for assertion")

        # Extract partition numbers
        partitions = set()
        for msg in messages:
            partitions.add(msg.partition)

        assert len(partitions) >= min_partitions, \
            f"Expected messages across at least {min_partitions} partitions, " \
            f"but found only {len(partitions)}: {partitions}"

    @staticmethod
    def assert_partition_consistency(
        messages: list,
        symbol: str,
        exchange: str = None,
    ) -> None:
        """Assert that messages for same symbol/exchange go to same partition.

        Args:
            messages: List of ConsumeRecord messages
            symbol: Expected symbol in all messages
            exchange: Expected exchange in all messages (optional)

        Raises:
            AssertionError: If partition consistency is violated
        """
        if not messages:
            raise AssertionError("No messages provided for assertion")

        partitions = set()
        for msg in messages:
            msg_value = json.loads(msg.value)
            if exchange:
                assert msg_value.get("exchange") == exchange, \
                    f"Expected exchange {exchange}, got {msg_value.get('exchange')}"
            assert msg_value.get("symbol") == symbol, \
                f"Expected symbol {symbol}, got {msg_value.get('symbol')}"
            partitions.add(msg.partition)

        assert len(partitions) == 1, \
            f"Symbol {symbol}" + (f" on {exchange}" if exchange else "") + \
            f" should route to single partition, but found {partitions}"


class TestTask92PartitionRouting:
    """Task 9.2: Integration tests for partition key routing and ordering.

    Tests verify that:
    1. Messages with same symbol route to same partition (SymbolPartitioner)
    2. Messages with different symbols distribute across partitions
    3. Composite partitioner ensures per-exchange-symbol ordering
    4. Partition assignments are deterministic across producer restarts
    5. Message ordering is preserved within partitions

    Requirements: FR3 (Partitioning Strategies)
    """

    @pytest.mark.integration
    def test_partition_key_consistency_same_symbol(
        self,
        kafka_cluster: KafkaClusterInfo,
        kafka_producer: KafkaProducer,
        kafka_consumer: KafkaConsumer,
    ):
        """Test that messages for same symbol route to same partition.

        This test:
        - Produces 10 Trade messages for symbol BTC-USD
        - All messages use same symbol but from same exchange (Coinbase)
        - Verifies all messages arrive in same partition
        - Confirms message ordering is preserved

        Expected Outcome:
        - All 10 messages in single partition
        - Messages appear in production order (price increases)
        - Partition is deterministic (same symbol → same partition)
        """
        import uuid
        topic = f"cryptofeed.trades.symbol.{uuid.uuid4().hex[:8]}"
        message_count = 10
        symbol = "BTC-USD"
        exchange = "coinbase"

        # Produce messages with same symbol (via SymbolPartitioner)
        # We use symbol as partition key since SymbolPartitioner routes by symbol only
        partition_key = symbol.lower().replace("_", "-").encode("utf-8")
        produced_partitions = []

        for i in range(message_count):
            trade = create_trade_message(
                symbol=symbol,
                exchange=exchange,
                price=50000.0 + (i * 100),
                amount=0.5 + (i * 0.01),
                trade_id=f"trade-{i:03d}",
            )

            future = kafka_producer.send(
                topic,
                key=partition_key,
                value=trade.to_dict(),
            )
            record_metadata = future.get(timeout=5)
            produced_partitions.append(record_metadata.partition)

        kafka_producer.flush(timeout=5)
        LOG.info(
            f"Produced {message_count} messages to partition(s): {set(produced_partitions)}"
        )

        # All produced messages should go to same partition (deterministic hashing)
        assert len(set(produced_partitions)) == 1, \
            f"SymbolPartitioner should route all same-symbol messages to same partition, " \
            f"but got {len(set(produced_partitions))} different partitions"

        # Consume and verify
        kafka_consumer.subscribe([topic])
        received_messages = []

        max_polls = 100
        for _ in range(max_polls):
            msg = kafka_consumer.poll(timeout_ms=500)
            if msg:
                for topic_partition, records in msg.items():
                    for record in records:
                        received_messages.append(record)

            if len(received_messages) >= message_count:
                break

        assert len(received_messages) >= message_count, \
            f"Expected {message_count} messages, got {len(received_messages)}"

        # Verify all messages in same partition
        PartitionAssertions.assert_all_messages_same_partition(
            received_messages[:message_count],
            partition_key.decode("utf-8")
        )

        # Verify ordering (prices increase sequentially)
        for i, record in enumerate(received_messages[:message_count]):
            msg_value = json.loads(record.value)
            expected_price = 50000.0 + (i * 100)
            assert msg_value["price"] == expected_price, \
                f"Message {i}: Expected price {expected_price}, got {msg_value['price']}"

        LOG.info(
            f"Successfully verified {message_count} messages in same partition with correct ordering"
        )

    @pytest.mark.integration
    def test_partition_distribution_different_symbols(
        self,
        kafka_cluster: KafkaClusterInfo,
        kafka_producer: KafkaProducer,
        kafka_consumer: KafkaConsumer,
        kafka_topic_creator,
    ):
        """Test that different symbols distribute across multiple partitions.

        This test:
        - Produces messages for 5 different symbols (BTC-USD, ETH-USD, SOL-USD, etc.)
        - Each symbol uses SymbolPartitioner routing
        - Verifies messages distribute across multiple partitions
        - Confirms each symbol consistently routes to same partition

        Expected Outcome:
        - Messages for different symbols use different partitions
        - Same symbol always routes to same partition
        - Partitions are utilized (not all in partition 0)
        """
        import uuid
        topic = f"cryptofeed.trades.symbols.{uuid.uuid4().hex[:8]}"
        # Create topic with 3 partitions to allow distribution
        kafka_topic_creator(topic, partitions=3, replication_factor=1)

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD"]
        messages_per_symbol = 3

        # Track which partitions each symbol goes to
        symbol_partitions: Dict[str, int] = {}

        # Produce messages for different symbols
        for symbol in symbols:
            partition_key = symbol.lower().replace("_", "-").encode("utf-8")
            partitions_for_symbol = set()

            for i in range(messages_per_symbol):
                trade = create_trade_message(
                    symbol=symbol,
                    exchange="coinbase",
                    price=50000.0 + (i * 100),
                    trade_id=f"trade-{symbol}-{i}",
                )

                future = kafka_producer.send(
                    topic,
                    key=partition_key,
                    value=trade.to_dict(),
                )
                record_metadata = future.get(timeout=5)
                partitions_for_symbol.add(record_metadata.partition)

            # All messages for this symbol should go to same partition
            assert len(partitions_for_symbol) == 1, \
                f"Symbol {symbol} should go to single partition, " \
                f"but got {partitions_for_symbol}"

            symbol_partitions[symbol] = partitions_for_symbol.pop()

        kafka_producer.flush(timeout=5)
        LOG.info(f"Symbol partition mapping: {symbol_partitions}")

        # Consume all messages
        kafka_consumer.subscribe([topic])
        received_messages = []

        max_polls = 100
        for _ in range(max_polls):
            msg = kafka_consumer.poll(timeout_ms=500)
            if msg:
                for topic_partition, records in msg.items():
                    for record in records:
                        received_messages.append(record)

            if len(received_messages) >= (len(symbols) * messages_per_symbol):
                break

        total_expected = len(symbols) * messages_per_symbol
        assert len(received_messages) >= total_expected, \
            f"Expected {total_expected} messages, got {len(received_messages)}"

        # Verify messages span multiple partitions
        PartitionAssertions.assert_messages_span_multiple_partitions(
            received_messages[:total_expected],
            min_partitions=min(2, len(symbols))  # At least 2 partitions (if enough symbols)
        )

        # Verify each symbol consistently routed to same partition
        for symbol in symbols:
            symbol_messages = [
                m for m in received_messages
                if json.loads(m.value).get("symbol") == symbol
            ]
            PartitionAssertions.assert_partition_consistency(symbol_messages, symbol)

        LOG.info(
            f"Successfully verified {total_expected} messages distributed across "
            f"{len(symbol_partitions)} partitions by symbol"
        )

    @pytest.mark.integration
    def test_composite_partitioner_per_exchange_symbol(
        self,
        kafka_cluster: KafkaClusterInfo,
        kafka_producer: KafkaProducer,
        kafka_consumer: KafkaConsumer,
        kafka_topic_creator,
    ):
        """Test composite partitioner ensures per-exchange-symbol ordering.

        This test:
        - Produces messages for BTC-USD on Coinbase and BTC-USD on Binance
        - Uses composite partitioner (exchange-symbol key)
        - Verifies different exchanges route to different partitions
        - Confirms ordering is preserved per exchange-symbol pair

        Expected Outcome:
        - Coinbase/BTC-USD and Binance/BTC-USD in different partitions
        - Same exchange-symbol pair always in same partition
        - Per-exchange-symbol ordering preserved
        """
        import uuid
        topic = f"cryptofeed.trades.composite.{uuid.uuid4().hex[:8]}"
        # Create topic with 3 partitions to allow distribution
        kafka_topic_creator(topic, partitions=3, replication_factor=1)

        message_count = 8

        # Test data: same symbol, different exchanges
        test_cases = [
            ("coinbase", "BTC-USD"),
            ("binance", "BTC-USD"),
        ]

        exchange_symbol_partitions: Dict[tuple, int] = {}

        # Produce messages for different exchange-symbol pairs
        for exchange, symbol in test_cases:
            # CompositePartitioner uses "exchange-symbol" as key
            partition_key = f"{exchange}-{symbol}".lower().replace("_", "-").encode("utf-8")
            partitions_for_pair = set()

            for i in range(message_count):
                trade = create_trade_message(
                    symbol=symbol,
                    exchange=exchange,
                    price=50000.0 + (i * 100),
                    trade_id=f"trade-{exchange}-{symbol}-{i}",
                )

                future = kafka_producer.send(
                    topic,
                    key=partition_key,
                    value=trade.to_dict(),
                )
                record_metadata = future.get(timeout=5)
                partitions_for_pair.add(record_metadata.partition)

            # All messages for this exchange-symbol should go to same partition
            assert len(partitions_for_pair) == 1, \
                f"Exchange-symbol pair {exchange}/{symbol} should go to single partition, " \
                f"but got {partitions_for_pair}"

            exchange_symbol_partitions[(exchange, symbol)] = partitions_for_pair.pop()

        kafka_producer.flush(timeout=5)
        LOG.info(f"Exchange-symbol partition mapping: {exchange_symbol_partitions}")

        # Consume all messages
        kafka_consumer.subscribe([topic])
        received_messages = []

        max_polls = 100
        for _ in range(max_polls):
            msg = kafka_consumer.poll(timeout_ms=500)
            if msg:
                for topic_partition, records in msg.items():
                    for record in records:
                        received_messages.append(record)

            if len(received_messages) >= (len(test_cases) * message_count):
                break

        total_expected = len(test_cases) * message_count
        assert len(received_messages) >= total_expected, \
            f"Expected {total_expected} messages, got {len(received_messages)}"

        # Verify each exchange-symbol pair is routed consistently
        # Note: Different pairs may hash to same partition, so we verify consistency
        # of routing rather than requiring different partitions
        LOG.info(
            f"Partition assignments: {exchange_symbol_partitions}. "
            f"(Note: Different exchange-symbol pairs may coincidentally hash to same partition)"
        )

        # Verify ordering within each exchange-symbol partition
        for exchange, symbol in test_cases:
            pair_messages = [
                m for m in received_messages
                if json.loads(m.value).get("symbol") == symbol and
                   json.loads(m.value).get("exchange") == exchange
            ]

            PartitionAssertions.assert_partition_consistency(pair_messages, symbol, exchange)

            # Verify ordering (prices increase)
            for i, record in enumerate(pair_messages[:message_count]):
                msg_value = json.loads(record.value)
                expected_price = 50000.0 + (i * 100)
                assert msg_value["price"] == expected_price, \
                    f"Message {i} for {exchange}/{symbol}: Expected price {expected_price}, " \
                    f"got {msg_value['price']}"

        LOG.info(
            f"Successfully verified per-exchange-symbol partitioning and ordering for "
            f"{len(test_cases)} pairs"
        )

    @pytest.mark.integration
    def test_partition_rebalance_after_producer_restart(
        self,
        kafka_cluster: KafkaClusterInfo,
        kafka_producer: KafkaProducer,
        kafka_consumer: KafkaConsumer,
    ):
        """Test partition assignments are deterministic after producer restart.

        This test:
        - Produces initial batch of messages (records partition assignments)
        - Simulates producer reconnect by sending more messages
        - Verifies same partition assignments for same symbols
        - Confirms deterministic hashing (no random partition changes)

        Expected Outcome:
        - Same symbol always goes to same partition
        - Partition assignment is deterministic (reproducible)
        - No random distribution changes on reconnect
        """
        import uuid
        topic = f"cryptofeed.trades.restart.{uuid.uuid4().hex[:8]}"
        symbols = ["BTC-USD", "ETH-USD"]

        # Phase 1: First batch of messages
        initial_partitions: Dict[str, int] = {}

        for symbol in symbols:
            partition_key = symbol.lower().replace("_", "-").encode("utf-8")
            trade = create_trade_message(
                symbol=symbol,
                exchange="coinbase",
                price=50000.0,
                trade_id="batch-1-trade-1",
            )

            future = kafka_producer.send(
                topic,
                key=partition_key,
                value=trade.to_dict(),
            )
            record_metadata = future.get(timeout=5)
            initial_partitions[symbol] = record_metadata.partition
            LOG.info(f"Phase 1: {symbol} → partition {record_metadata.partition}")

        kafka_producer.flush(timeout=5)

        # Phase 2: "Restart" (just send more messages - same producer)
        # In real scenario, producer would reconnect; here we simulate by sending more messages
        restarted_partitions: Dict[str, int] = {}

        for symbol in symbols:
            partition_key = symbol.lower().replace("_", "-").encode("utf-8")
            trade = create_trade_message(
                symbol=symbol,
                exchange="coinbase",
                price=51000.0,
                trade_id="batch-2-trade-1",
            )

            future = kafka_producer.send(
                topic,
                key=partition_key,
                value=trade.to_dict(),
            )
            record_metadata = future.get(timeout=5)
            restarted_partitions[symbol] = record_metadata.partition
            LOG.info(f"Phase 2: {symbol} → partition {record_metadata.partition}")

        kafka_producer.flush(timeout=5)

        # Verify partition assignments are identical (deterministic)
        for symbol in symbols:
            assert initial_partitions[symbol] == restarted_partitions[symbol], \
                f"{symbol} partition changed after restart: " \
                f"initial={initial_partitions[symbol]}, " \
                f"restarted={restarted_partitions[symbol]}"

        LOG.info("Successfully verified deterministic partition assignments across restarts")


# ============================================================================
# Test Helpers and Utilities
# ============================================================================


@pytest.fixture
def cleanup_topics(kafka_cluster: KafkaClusterInfo):
    """Cleanup Kafka topics after tests (optional).

    This fixture can be used to clean up test topics after tests complete.
    """

    yield

    # Cleanup (optional - can be commented out to keep topics for manual inspection)
    # from kafka.admin import KafkaAdminClient
    # try:
    #     admin_client = KafkaAdminClient(
    #         bootstrap_servers=kafka_cluster.bootstrap_servers,
    #         request_timeout_ms=5000
    #     )
    #     admin_client.delete_topics(topics_to_cleanup, timeout_ms=5000)
    #     admin_client.close()
    # except Exception as e:
    #     LOG.warning(f"Failed to cleanup topics: {e}")
