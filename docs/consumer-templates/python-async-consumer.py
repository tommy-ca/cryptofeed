"""
Python Async Consumer Template for Cryptofeed Consolidated Topics

This template demonstrates how to build an async Python consumer that:
- Uses aiokafka for async I/O
- Subscribes to consolidated topics (cryptofeed.trades, etc.)
- Deserializes protobuf messages
- Processes messages in batches
- Manages offsets with checkpoints
- Handles errors gracefully

Key Features:
- Async/await patterns for throughput
- Batch processing (100 messages at a time)
- Per-message error handling (failed messages → DLQ)
- Header extraction for filtering
- Offset management with manual commits
- Graceful shutdown with timeout
- Connection pooling and resource cleanup
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from aiokafka import AIOKafkaConsumer
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


class CryptofeedAsyncConsumer:
    """
    Production-ready async Kafka consumer for cryptofeed consolidated topics.

    Usage:
        async def main():
            consumer = CryptofeedAsyncConsumer(
                bootstrap_servers=['kafka1:9092'],
                topics=['cryptofeed.trades', 'cryptofeed.orderbook'],
            )
            async with consumer.create_consumer() as consumer_instance:
                async for message in consumer_instance.consume_messages():
                    await process_message(message)

        asyncio.run(main())
    """

    def __init__(
        self,
        bootstrap_servers: List[str],
        topics: List[str],
        consumer_group: str = "cryptofeed-python-processor",
        batch_size: int = 100,
        batch_timeout_ms: int = 5000,
    ):
        """
        Initialize async consumer.

        Args:
            bootstrap_servers: Kafka broker addresses
            topics: Topics to subscribe to (supports pattern matching)
            consumer_group: Consumer group name
            batch_size: Messages per batch
            batch_timeout_ms: Timeout before processing smaller batch
        """
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics
        self.consumer_group = consumer_group
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.consumer = None

    @asynccontextmanager
    async def create_consumer(self):
        """
        Create and manage AIOKafkaConsumer lifecycle.

        Handles:
        - Consumer creation
        - Subscription to topics
        - Offset tracking
        - Graceful shutdown
        """
        consumer = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            auto_offset_reset="earliest",
            enable_auto_commit=False,  # Manual commits for exactly-once
            max_poll_records=self.batch_size,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            value_deserializer=self._deserialize_protobuf,
            security_protocol="PLAINTEXT",  # Use TLS in production
        )

        try:
            await consumer.start()
            logger.info(f"Consumer started for topics: {self.topics}")
            self.consumer = consumer
            yield consumer
        finally:
            # Commit any pending offsets
            await consumer.commit()
            await consumer.stop()
            logger.info("Consumer stopped gracefully")

    async def consume_messages(self):
        """
        Async generator yielding messages from Kafka.

        Processes messages one at a time with error handling.
        """
        if not self.consumer:
            raise RuntimeError("Consumer not initialized. Use create_consumer context.")

        async for message in self.consumer:
            yield message

    async def consume_batch(self) -> List[Dict[str, Any]]:
        """
        Consume messages in batches for parallel processing.

        Returns:
            List of messages (up to batch_size)
        """
        if not self.consumer:
            raise RuntimeError("Consumer not initialized")

        batch = []
        try:
            while len(batch) < self.batch_size:
                # Try to get messages with timeout
                message = await asyncio.wait_for(
                    self.consumer.__anext__(),
                    timeout=self.batch_timeout_ms / 1000,
                )
                batch.append(message)
        except asyncio.TimeoutError:
            # Timeout is fine, process whatever we have
            pass

        return batch

    def _deserialize_protobuf(self, data: bytes):
        """
        Deserialize protobuf message.

        In real implementation, would:
        1. Determine message type from header
        2. Load appropriate protobuf schema
        3. Parse and return deserialized message

        Args:
            data: Serialized protobuf bytes

        Returns:
            Deserialized message object
        """
        if not data:
            return None

        # Example: Deserialize Trade message
        # from cryptofeed.schema.v1 import trade_pb2
        # message = trade_pb2.Trade()
        # message.ParseFromString(data)
        # return message

        return data

    def extract_routing_headers(self, message) -> Dict[str, str]:
        """
        Extract routing metadata from message headers.

        Headers in Kafka message:
        - exchange: Source exchange
        - symbol: Trading pair
        - data_type: Message type
        - schema_version: Schema version

        Returns:
            Dictionary with routing metadata
        """
        routing = {
            "exchange": None,
            "symbol": None,
            "data_type": None,
            "schema_version": None,
        }

        if message.headers:
            headers = dict(message.headers)
            routing["exchange"] = headers.get(b"exchange", b"").decode()
            routing["symbol"] = headers.get(b"symbol", b"").decode()
            routing["data_type"] = headers.get(b"data_type", b"").decode()
            routing["schema_version"] = headers.get(b"schema_version", b"v1").decode()

        return routing

    async def process_batch(self, messages: List) -> None:
        """
        Process batch of messages in parallel.

        Features:
        - Parallel deserialization
        - Per-message error handling
        - Failed messages sent to DLQ
        - Batch offset commit

        Args:
            messages: Batch of Kafka messages
        """
        if not messages:
            return

        # Process in parallel using asyncio gather
        tasks = [
            self._process_single_message(msg)
            for msg in messages
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        successes = sum(1 for r in results if r is not None)
        failures = sum(1 for r in results if isinstance(r, Exception))

        logger.info(
            f"Batch processed: {successes} successes, {failures} failures "
            f"(total {len(messages)} messages)"
        )

        # Commit offsets after successful processing
        if successes > 0 and self.consumer:
            await self.consumer.commit()

    async def _process_single_message(self, message) -> Optional[bool]:
        """
        Process single message with error handling.

        Returns:
            True if successful, Exception if failed
        """
        try:
            # Extract routing metadata
            routing = self.extract_routing_headers(message)

            # Process message (implement your logic)
            await self._handle_message(message, routing)

            return True
        except Exception as e:
            # Send to DLQ for investigation
            await self._send_to_dlq(message, str(e))
            logger.error(f"Message processing failed: {e}", exc_info=True)
            return e

    async def _handle_message(self, message, routing: Dict[str, str]) -> None:
        """
        Handle message (user implementation).

        Args:
            message: Kafka message
            routing: Routing metadata from headers
        """
        exchange = routing.get("exchange", "unknown")
        symbol = routing.get("symbol", "unknown")
        data_type = routing.get("data_type", "unknown")

        logger.debug(
            f"Processing {data_type} message: {exchange} {symbol} "
            f"@ {datetime.now()}"
        )

        # TODO: Implement message handling
        # - Validate message format
        # - Transform data
        # - Store in database
        # - Send to downstream systems

    async def _send_to_dlq(self, message, error: str) -> None:
        """
        Send failed message to DLQ topic.

        DLQ format:
        - Original message as value
        - Error message in headers
        - Original topic in headers

        Args:
            message: Original Kafka message
            error: Error message
        """
        # In production, would send to cryptofeed.dlq topic
        logger.warning(
            f"Sending to DLQ: {error} "
            f"(partition={message.partition}, offset={message.offset})"
        )


async def example_async_consumer():
    """
    Example: Async consumer with batch processing.

    This example:
    1. Creates consumer
    2. Subscribes to consolidated topics
    3. Processes messages in batches
    4. Handles errors with DLQ
    5. Gracefully shuts down on signal
    """
    consumer = CryptofeedAsyncConsumer(
        bootstrap_servers=["kafka1:9092", "kafka2:9092"],
        topics=["cryptofeed.trades", "cryptofeed.orderbook"],
        batch_size=100,
    )

    try:
        async with consumer.create_consumer():
            # Continuous consumption loop
            while True:
                # Get batch of messages
                batch = await consumer.consume_batch()

                if batch:
                    # Process batch
                    await consumer.process_batch(batch)

                # Small delay to prevent busy-waiting
                await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user")
    except Exception as e:
        logger.error(f"Consumer error: {e}", exc_info=True)


# Production Deployment Tips:
#
# 1. Dependencies:
#    ```bash
#    pip install aiokafka protobuf
#    ```
#
# 2. Configuration:
#    - Use environment variables for broker addresses
#    - Configure TLS/SSL for production
#    - Set heartbeat and session timeouts
#
# 3. Error Handling:
#    - All errors sent to DLQ for investigation
#    - Retry logic for transient failures
#    - Circuit breaker for persistent failures
#
# 4. Performance:
#    - Batch size: 100-500 (tune based on message size)
#    - Parallelism: asyncio.Semaphore to limit concurrent tasks
#    - Connection pooling: One consumer per instance
#
# 5. Monitoring:
#    - Log message count and latency
#    - Track batch processing time
#    - Monitor DLQ message rate
#    - Alert if lag > 5 seconds
#
# 6. Graceful Shutdown:
#    - Commit pending offsets
#    - Close consumer cleanly
#    - Timeout: 30 seconds
#    - Use signal handlers (SIGTERM, SIGINT)
#
# 7. Testing:
#    - Unit test message processing logic
#    - Integration test with docker-compose Kafka
#    - Load test with 10K msg/s
#    - Chaos test with broker failures
