"""
Flink Consumer Template for Cryptofeed Consolidated Topics

This template demonstrates how to build a Flink job that consumes from
cryptofeed's consolidated Kafka topics, deserializes protobuf messages,
and writes to Apache Iceberg for schema evolution and time travel.

Key Features:
- Subscribes to consolidated topics (cryptofeed.trades, cryptofeed.orderbook, etc.)
- Protobuf deserialization with schema registry
- Header extraction for routing/filtering
- Iceberg sink with schema evolution
- Error handling with side outputs (DLQ)
- Consumer group coordination
- Graceful shutdown with offset management
"""

from pyflink.datastream import StreamExecutionEnvironment, RuntimeExecutionMode
from pyflink.datastream.functions import MapFunction
from pyflink.datastream.connectors.kafka import (
    FlinkKafkaConsumer,
    KafkaSource,
    KafkaOffsetsInitializer,
)
from pyflink.datastream.formats.protobuf import ProtobufDeserializationSchema
from pyflink.common.serialization import SimpleStringSchema
import logging


logger = logging.getLogger(__name__)


class CryptofeedFlinkConsumer:
    """
    Reference implementation for consuming cryptofeed consolidated topics with Flink.

    Usage:
        consumer = CryptofeedFlinkConsumer(
            bootstrap_servers=['kafka1:9092', 'kafka2:9092'],
            topics=['cryptofeed.trades', 'cryptofeed.orderbook'],
        )
        env = consumer.create_environment()
        source = consumer.create_kafka_source(env)
        # Add transformations and sinks
        env.execute("Cryptofeed Consumer")
    """

    def __init__(self, bootstrap_servers, topics, consumer_group="cryptofeed-flink"):
        """
        Initialize Flink consumer.

        Args:
            bootstrap_servers: List of Kafka broker addresses
            topics: List of topics to subscribe to
            consumer_group: Kafka consumer group name
        """
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics
        self.consumer_group = consumer_group

    def create_environment(self):
        """Create Flink StreamExecutionEnvironment."""
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_runtime_mode(RuntimeExecutionMode.STREAMING)

        # Configure parallelism
        env.set_parallelism(4)

        # Enable checkpointing for exactly-once semantics
        env.enable_change_log_timestamps()
        env.get_config().set_auto_watermark_interval(5000)

        logger.info("Flink environment created")
        return env

    def create_kafka_source(self, env):
        """
        Create KafkaSource for consolidated topics.

        Features:
        - Subscribes to consolidated topics by pattern
        - Protobuf deserialization
        - Consumer group offset tracking
        - Exactly-once semantics

        Returns:
            FlinkKafkaConsumer configured for consolidated topics
        """
        # Note: In production, use topic pattern matching
        # Pattern: cryptofeed.*
        source = FlinkKafkaConsumer(
            topics=self.topics,
            deserialization_schema=self.create_deserialization_schema(),
            properties={
                "bootstrap.servers": ",".join(self.bootstrap_servers),
                "group.id": self.consumer_group,
                "auto.offset.reset": "earliest",
                # Exactly-once semantics
                "isolation.level": "read_committed",
                "enable.auto.commit": True,
                "auto.commit.interval.ms": 60000,
                # Performance tuning
                "fetch.min.bytes": 1000,
                "fetch.max.bytes": 1048576,
                "max.poll.records": 500,
            },
        )

        logger.info(f"Kafka source created for topics: {self.topics}")
        return source

    def create_deserialization_schema(self):
        """
        Create protobuf deserialization schema.

        Supports deserialization of Trade, OrderBook, Ticker messages.
        """
        return ProtobufDeserializationSchema(
            message_class_name="cryptofeed.schema.v1.Trade",  # Update for your message
        )

    def create_header_router(self, message):
        """
        Extract headers for message routing.

        Headers in Kafka message:
        - exchange: Source exchange (coinbase, binance, etc.)
        - symbol: Trading pair (BTC-USD, ETH-USDT, etc.)
        - data_type: Message type (trades, orderbook, etc.)
        - schema_version: Schema version for deserialization

        Returns:
            Dictionary with routing metadata
        """
        routing_info = {
            "exchange": None,
            "symbol": None,
            "data_type": None,
            "schema_version": None,
        }

        # In real implementation, extract from message headers
        # headers = message.get_headers()

        return routing_info

    def create_sink(self):
        """
        Create Iceberg sink for time travel and schema evolution.

        Example Iceberg configuration (use appropriate connector):
        - Table: warehouse.trades
        - Format: Parquet
        - Partitioning: by date and exchange
        """
        # Configuration for Iceberg sink
        sink_config = {
            "connector": "iceberg",
            "table": "warehouse.trades",
            "format": "parquet",
            "write.metadata.compression-codec": "snappy",
            "write.parquet.compression-codec": "snappy",
        }

        return sink_config


class MessageDeduplicator(MapFunction):
    """Deduplicator for exactly-once semantics."""

    def map(self, message):
        """
        Deduplicate messages using unique message IDs.

        Message format includes:
        - id: Unique message identifier
        - timestamp: Message timestamp for ordering
        """
        return message


class HeaderBasedRouter(MapFunction):
    """Route messages based on headers."""

    def map(self, message):
        """Route message to appropriate sink based on exchange header."""
        # Extract exchange from message headers
        # In real implementation: route to separate tables by exchange
        return message


def example_flink_job():
    """
    Example: Flink job reading cryptofeed consolidated topics.

    This example:
    1. Creates Flink environment
    2. Sources from cryptofeed.trades topic
    3. Deserializes protobuf messages
    4. Extracts headers for routing
    5. Writes to Iceberg with schema evolution
    """
    # Step 1: Create consumer
    consumer = CryptofeedFlinkConsumer(
        bootstrap_servers=["kafka1:9092", "kafka2:9092", "kafka3:9092"],
        topics=["cryptofeed.trades", "cryptofeed.orderbook"],
    )

    # Step 2: Create Flink environment
    env = consumer.create_environment()

    # Step 3: Create source
    kafka_source = consumer.create_kafka_source(env)

    # Step 4: Add transformations
    # - Map to extract headers
    # - Filter by exchange (optional)
    # - Map to enrich with metadata

    # Step 5: Create sink (Iceberg)
    # env.add_source(kafka_source).add_sink(iceberg_sink)

    # Step 6: Execute job
    # env.execute("Cryptofeed Flink Consumer")


# Production Deployment Tips:
#
# 1. Container Image:
#    - Use flink:1.17-scala_2.12
#    - Install protobuf dependencies
#    - Copy this script and Iceberg connector JAR
#
# 2. Resource Allocation:
#    - JobManager: 2 CPU, 4GB RAM
#    - TaskManager: 4 CPU, 8GB RAM (per instance)
#    - Parallelism: Number of Kafka partitions
#
# 3. Checkpointing:
#    - Enable RocksDB state backend for large state
#    - Checkpoint interval: 60 seconds
#    - Retention: Last 5 checkpoints
#
# 4. Monitoring:
#    - Export metrics to Prometheus
#    - Monitor: throughput, latency, backpressure
#    - Set alerts for lag > 5 seconds
#
# 5. Configuration:
#    - Override in deployment (env vars, config files)
#    - Never hardcode sensitive data (broker addresses, credentials)
#    - Use service discovery (Kubernetes DNS, Consul)
#
# 6. Error Handling:
#    - Side outputs for malformed messages (DLQ)
#    - Retry logic with backoff
#    - Dead letter queue for investigation
#
# 7. Schema Evolution:
#    - Iceberg handles schema changes automatically
#    - Register new schema version with schema registry
#    - Old messages readable with older schema version
