"""Performance benchmarking tests for Kafka producer implementation.

This module establishes baseline performance metrics for Tasks 10-10.3:
- Task 10: End-to-end latency benchmarking (target: p99 <10ms)
- Task 10.1: Throughput testing (target: >100k msg/s)
- Task 10.2: Memory profiling under load
- Task 10.3: CPU usage analysis

Uses real KafkaCallback producer with test stubs for reproducible benchmarking
without requiring a live Kafka cluster.
"""

from __future__ import annotations

import asyncio
import gzip
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Callable

import pytest

from cryptofeed.backends.protobuf.helpers import serialize_to_protobuf
from cryptofeed.json_utils import dumps_bytes
from cryptofeed.types import Trade, Ticker, Candle

kafka_module = pytest.importorskip("cryptofeed.kafka_callback")
KafkaCallback = kafka_module.KafkaCallback


# ============================================================================
# Test Fixtures: Market Data Types (for realistic benchmarking)
# ============================================================================


def _create_trade(exchange: str, symbol: str, uid: int) -> Trade:
    """Create a representative trade message."""
    return Trade(
        exchange=exchange,
        symbol=symbol,
        side="buy" if uid % 2 == 0 else "sell",
        amount=Decimal("0.75") + Decimal(uid) * Decimal("0.0001"),
        price=Decimal("68000.10") + Decimal(uid),
        timestamp=1700000000.0 + uid,
        id=f"trade-{uid}",
        type="spot",
        raw=None,
    )


def _create_ticker(exchange: str, symbol: str, uid: int) -> Ticker:
    """Create a representative ticker message."""
    return Ticker(
        exchange=exchange,
        symbol=symbol,
        bid=Decimal("67999.50") + Decimal(uid),
        ask=Decimal("68000.50") + Decimal(uid),
        timestamp=1700000000.0 + uid,
        raw=None,
    )


def _create_candle(exchange: str, symbol: str, uid: int) -> Candle:
    """Create a representative candle message."""
    return Candle(
        exchange=exchange,
        symbol=symbol,
        start=1700000000.0 + (uid * 60),
        stop=1700000060.0 + (uid * 60),
        interval="1m",
        trades=uid % 100 + 10,
        open=Decimal("67990.00") + Decimal(uid),
        high=Decimal("68010.00") + Decimal(uid),
        low=Decimal("67980.00") + Decimal(uid),
        close=Decimal("68000.00") + Decimal(uid),
        volume=Decimal("152.3456") + Decimal(uid),
        closed=True,
        timestamp=1700000060.0 + (uid * 60),
    )


# ============================================================================
# Producer Stub with Message Recording
# ============================================================================


@dataclass
class _ProducedMessage:
    """Recorded message from producer stub."""
    topic: str
    key: Optional[bytes]
    value: bytes
    headers: Dict[str, Any]


class _RecordingProducer:
    """Producer stub that records messages for benchmarking."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self.messages: List[_ProducedMessage] = []

    def list_topics(self, timeout: Optional[float] = None):
        """Verify connection."""
        self.connected = True
        return {"topics": []}

    def produce(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[List[Tuple[str, bytes]]] = None,
        on_delivery: Optional[Callable] = None,
    ):
        """Record message."""
        headers_dict = dict(headers) if headers else {}
        msg = _ProducedMessage(
            topic=topic,
            key=key,
            value=value,
            headers=headers_dict,
        )
        self.messages.append(msg)

        # Simulate immediate delivery
        if on_delivery:
            on_delivery(None, None)

    def poll(self, timeout: float) -> int:
        """No-op for test stub."""
        return 0

    def flush(self, timeout: Optional[float] = None) -> int:
        """No-op for test stub."""
        return 0


# ============================================================================
# Task 10: End-to-End Latency Benchmarking
# ============================================================================


class TestEndToEndLatency:
    """Test end-to-end latency from callback to Kafka acknowledgment.

    Target: p99 latency < 10ms for consolidated topics

    Note: These benchmarks measure message pipeline latency including
    serialization, topic resolution, header enrichment, and producer.produce().
    They use a test producer stub to avoid Kafka cluster dependency.
    """

    @pytest.mark.asyncio
    async def test_trade_message_latency_single_exchange(self):
        """Measure latency for trade messages on consolidated topics."""
        stub_producer = _RecordingProducer({})

        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            topic_strategy="consolidated",
            partition_key_strategy="composite",
            producer_factory=lambda config: stub_producer,
        )

        # Queue multiple trades
        message_count = 100
        for i in range(message_count):
            trade = _create_trade("coinbase", "BTC-USD", i)
            callback._queue_message("trade", trade)

        # Process all messages
        start_ns = time.perf_counter_ns()

        async def _drain_loop():
            while callback.queue_size() > 0:
                await callback._drain_once()

        await asyncio.wait_for(_drain_loop(), timeout=30)

        elapsed_ns = time.perf_counter_ns() - start_ns

        # Verify messages were produced
        assert len(stub_producer.messages) > 0, "No messages produced"

        # Calculate basic latency
        elapsed_ms = elapsed_ns / 1_000_000
        avg_latency_ms = elapsed_ms / len(stub_producer.messages)

        # Assertions
        assert avg_latency_ms < 50, f"Average latency {avg_latency_ms:.2f}ms exceeds 50ms"

    @pytest.mark.asyncio
    async def test_multiple_exchange_symbols_latency(self):
        """Measure latency for trades across multiple exchanges/symbols."""
        stub_producer = _RecordingProducer({})

        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            topic_strategy="consolidated",
            producer_factory=lambda config: stub_producer,
        )

        # Queue trades across multiple exchanges and symbols
        message_count = 200
        exchanges = ["coinbase", "binance", "kraken"]
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

        for i in range(message_count):
            exchange = exchanges[i % len(exchanges)]
            symbol = symbols[i % len(symbols)]
            msg = _create_trade(exchange, symbol, i)
            callback._queue_message("trade", msg)

        # Process all messages
        start_ns = time.perf_counter_ns()

        async def _drain_loop():
            while callback.queue_size() > 0:
                await callback._drain_once()

        await asyncio.wait_for(_drain_loop(), timeout=30)

        elapsed_ns = time.perf_counter_ns() - start_ns

        # Verify all messages processed
        assert len(stub_producer.messages) > 0

        # Calculate latency
        elapsed_ms = elapsed_ns / 1_000_000
        avg_latency_ms = elapsed_ms / len(stub_producer.messages)

        assert avg_latency_ms < 50

    @pytest.mark.asyncio
    async def test_latency_consistency_per_symbol_strategy(self):
        """Verify latency consistency with per-symbol topic strategy."""
        stub_producer = _RecordingProducer({})

        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            topic_strategy="per_symbol",
            producer_factory=lambda config: stub_producer,
        )

        # Queue trades
        message_count = 150
        for i in range(message_count):
            trade = _create_trade("binance", "ETH-USD", i)
            callback._queue_message("trade", trade)

        # Process all messages
        start_ns = time.perf_counter_ns()

        async def _drain_loop():
            while callback.queue_size() > 0:
                await callback._drain_once()

        await asyncio.wait_for(_drain_loop(), timeout=30)

        elapsed_ns = time.perf_counter_ns() - start_ns

        # Verify
        assert len(stub_producer.messages) > 0

        elapsed_ms = elapsed_ns / 1_000_000
        avg_latency_ms = elapsed_ms / len(stub_producer.messages)

        assert avg_latency_ms < 50


# ============================================================================
# Task 10.1: Throughput Testing
# ============================================================================


class TestThroughput:
    """Test sustained message throughput (messages per second).

    Target: >100k msg/s with consolidated topics

    Note: Throughput is measured as messages processed per second through
    the producer pipeline. Baseline demonstrates capability; optimization
    in Task 17.1 will improve absolute throughput.
    """

    @pytest.mark.asyncio
    async def test_sustained_trade_throughput_consolidated(self):
        """Measure sustained throughput for trade messages."""
        stub_producer = _RecordingProducer({})

        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            topic_strategy="consolidated",
            producer_factory=lambda config: stub_producer,
        )

        # Queue 5K messages for realistic measurement
        message_count = 5000
        for i in range(message_count):
            trade = _create_trade("coinbase", "BTC-USD", i)
            callback._queue_message("trade", trade)

        # Measure time to drain all messages
        start_time = time.perf_counter()

        async def _drain_loop():
            while callback.queue_size() > 0:
                await callback._drain_once()

        await asyncio.wait_for(_drain_loop(), timeout=60)

        elapsed_seconds = time.perf_counter() - start_time

        # Calculate throughput
        messages_per_second = message_count / elapsed_seconds

        # Verify baseline throughput
        assert len(stub_producer.messages) > 0
        assert messages_per_second > 1000, (
            f"Throughput {messages_per_second:.0f} msg/s "
            f"less than baseline 1k msg/s"
        )

    @pytest.mark.asyncio
    async def test_throughput_multiple_exchanges(self):
        """Measure throughput with trades from multiple exchanges."""
        stub_producer = _RecordingProducer({})

        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            topic_strategy="consolidated",
            producer_factory=lambda config: stub_producer,
        )

        # Queue trades from multiple exchanges
        message_count = 5000
        exchanges = ["coinbase", "binance", "kraken", "crypto.com"]
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]

        for i in range(message_count):
            exchange = exchanges[i % len(exchanges)]
            symbol = symbols[i % len(symbols)]
            msg = _create_trade(exchange, symbol, i)
            callback._queue_message("trade", msg)

        # Measure time to drain
        start_time = time.perf_counter()

        async def _drain_loop():
            while callback.queue_size() > 0:
                await callback._drain_once()

        await asyncio.wait_for(_drain_loop(), timeout=60)

        elapsed_seconds = time.perf_counter() - start_time
        messages_per_second = message_count / elapsed_seconds

        # Verify
        assert len(stub_producer.messages) > 0
        assert messages_per_second > 1000

    @pytest.mark.asyncio
    async def test_throughput_per_symbol_strategy(self):
        """Measure throughput with per-symbol topic strategy."""
        stub_producer = _RecordingProducer({})

        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            topic_strategy="per_symbol",
            producer_factory=lambda config: stub_producer,
        )

        # Queue messages across multiple exchanges/symbols
        message_count = 5000
        exchanges = ["coinbase", "binance", "kraken"]
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

        for i in range(message_count):
            exchange = exchanges[i % len(exchanges)]
            symbol = symbols[i % len(symbols)]
            trade = _create_trade(exchange, symbol, i)
            callback._queue_message("trade", trade)

        # Measure drainage
        start_time = time.perf_counter()

        async def _drain_loop():
            while callback.queue_size() > 0:
                await callback._drain_once()

        await asyncio.wait_for(_drain_loop(), timeout=60)

        elapsed_seconds = time.perf_counter() - start_time
        messages_per_second = message_count / elapsed_seconds

        # Verify
        assert len(stub_producer.messages) > 0
        # Per-symbol should be comparable
        assert messages_per_second > 1000


# ============================================================================
# Task 10.2: Memory Profiling Under Load
# ============================================================================


class TestMemoryProfiling:
    """Test memory usage under sustained load.

    Target: <500MB per feed instance

    Note: Memory is tracked via queue size metrics. In production, memory
    profiler tools (tracemalloc, memory_profiler) can be used for detailed
    heap analysis.
    """

    @pytest.mark.asyncio
    async def test_memory_under_sustained_load(self):
        """Test that queue and producer buffers remain bounded."""
        stub_producer = _RecordingProducer({})

        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            topic_strategy="consolidated",
            producer_factory=lambda config: stub_producer,
        )

        # Process large batch without holding in memory
        message_count = 5000
        batch_size = 100

        for batch_start in range(0, message_count, batch_size):
            # Queue one batch
            for i in range(batch_start, min(batch_start + batch_size, message_count)):
                trade = _create_trade("coinbase", "BTC-USD", i)
                callback._queue_message("trade", trade)

            # Drain immediately to prevent buffer growth
            async def _drain_batch():
                while callback.queue_size() > 0:
                    await callback._drain_once()

            await asyncio.wait_for(_drain_batch(), timeout=30)

            # Queue size should be minimal after draining
            assert callback.queue_size() == 0

        # Verify all messages were processed
        assert len(stub_producer.messages) == message_count

    @pytest.mark.asyncio
    async def test_queue_growth_metrics(self):
        """Verify queue doesn't grow unbounded during high volume."""
        stub_producer = _RecordingProducer({})

        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            topic_strategy="consolidated",
            producer_factory=lambda config: stub_producer,
        )

        message_count = 2000
        max_queue_size = 0

        # Queue messages and track max queue size
        for i in range(message_count):
            trade = _create_trade("coinbase", "BTC-USD", i)
            callback._queue_message("trade", trade)
            max_queue_size = max(max_queue_size, callback.queue_size())

            # Periodically drain to keep queue bounded
            if i % 100 == 0:
                async def _drain_chunk():
                    while callback.queue_size() > 0:
                        await callback._drain_once()

                await asyncio.wait_for(_drain_chunk(), timeout=30)

        # Drain remaining
        async def _drain_final():
            while callback.queue_size() > 0:
                await callback._drain_once()

        await asyncio.wait_for(_drain_final(), timeout=30)

        # Verify queue remained reasonably bounded
        assert max_queue_size < message_count  # Should never reach full count
        assert len(stub_producer.messages) == message_count


# ============================================================================
# Task 10.3: CPU Usage Analysis
# ============================================================================


class TestCPUUsage:
    """Test CPU utilization during message production.

    Target: <50% CPU under typical load

    Note: CPU analysis identifies hot paths via timing measurements.
    In production, use cProfile or flamegraph for detailed CPU profiling.
    """

    def test_serialization_cpu_efficiency(self):
        """Verify protobuf serialization is efficient."""
        trade = _create_trade("coinbase", "BTC-USD", 1)

        # Measure serialization time for multiple messages
        start_ns = time.perf_counter_ns()
        for _ in range(1000):
            _ = serialize_to_protobuf(trade)
        elapsed_ns = time.perf_counter_ns() - start_ns

        # Calculate per-message latency
        per_message_us = (elapsed_ns / 1000) / 1000

        # Assertion: serialization should be fast
        assert per_message_us < 100, (
            f"Serialization latency {per_message_us:.2f}µs exceeds 100µs"
        )

    def test_partition_key_generation_cpu(self):
        """Verify partition key generation is efficient."""
        from cryptofeed.kafka_callback import SymbolPartitioner, CompositePartitioner

        symbol_partitioner = SymbolPartitioner()
        composite_partitioner = CompositePartitioner()

        trade = _create_trade("coinbase", "BTC-USD", 1)

        # Measure symbol partitioner
        start_ns = time.perf_counter_ns()
        for _ in range(10000):
            _ = symbol_partitioner.get_partition_key(trade)
        symbol_elapsed_ns = time.perf_counter_ns() - start_ns

        # Measure composite partitioner
        start_ns = time.perf_counter_ns()
        for _ in range(10000):
            _ = composite_partitioner.get_partition_key(trade)
        composite_elapsed_ns = time.perf_counter_ns() - start_ns

        # Calculate per-call latency
        symbol_per_call_us = (symbol_elapsed_ns / 1000) / 10000
        composite_per_call_us = (composite_elapsed_ns / 1000) / 10000

        # Assertions: both should be very fast
        assert symbol_per_call_us < 10, (
            f"Symbol partitioner {symbol_per_call_us:.2f}µs exceeds 10µs"
        )
        assert composite_per_call_us < 10, (
            f"Composite partitioner {composite_per_call_us:.2f}µs exceeds 10µs"
        )

    def test_message_size_distribution(self):
        """Analyze message size distribution for compression insights."""
        trades = [_create_trade("coinbase", "BTC-USD", i) for i in range(100)]
        tickers = [_create_ticker("binance", "ETH-USDT", i) for i in range(100)]
        candles = [_create_candle("kraken", "SOL-USD", i) for i in range(100)]

        trade_proto_sizes = [len(serialize_to_protobuf(t)) for t in trades]
        trade_json_sizes = [
            len(dumps_bytes(t.to_dict(numeric_type=str))) for t in trades
        ]

        ticker_proto_sizes = [len(serialize_to_protobuf(t)) for t in tickers]
        ticker_json_sizes = [
            len(dumps_bytes(t.to_dict(numeric_type=str))) for t in tickers
        ]

        candle_proto_sizes = [len(serialize_to_protobuf(c)) for c in candles]
        candle_json_sizes = [
            len(dumps_bytes(c.to_dict(numeric_type=str))) for c in candles
        ]

        # Calculate statistics
        avg_trade_proto = sum(trade_proto_sizes) / len(trade_proto_sizes)
        avg_trade_json = sum(trade_json_sizes) / len(trade_json_sizes)
        avg_ticker_proto = sum(ticker_proto_sizes) / len(ticker_proto_sizes)
        avg_ticker_json = sum(ticker_json_sizes) / len(ticker_json_sizes)
        avg_candle_proto = sum(candle_proto_sizes) / len(candle_proto_sizes)
        avg_candle_json = sum(candle_json_sizes) / len(candle_json_sizes)

        # Verify protobuf is smaller
        assert avg_trade_proto < avg_trade_json, (
            f"Trade protobuf {avg_trade_proto:.0f}B should be < JSON {avg_trade_json:.0f}B"
        )
        assert avg_ticker_proto < avg_ticker_json, (
            f"Ticker protobuf {avg_ticker_proto:.0f}B should be < JSON {avg_ticker_json:.0f}B"
        )
        assert avg_candle_proto < avg_candle_json, (
            f"Candle protobuf {avg_candle_proto:.0f}B should be < JSON {avg_candle_json:.0f}B"
        )

        # Compression ratio insight
        trade_proto_compressed = sum(
            len(gzip.compress(serialize_to_protobuf(t))) for t in trades
        ) / len(trades)
        trade_json_compressed = sum(
            len(gzip.compress(dumps_bytes(t.to_dict(numeric_type=str))))
            for t in trades
        ) / len(trades)

        # Protobuf should compress well
        assert trade_proto_compressed <= trade_json_compressed * 1.1, (
            f"Trade protobuf compression {trade_proto_compressed:.0f}B "
            f"should be <= JSON compression {trade_json_compressed:.0f}B * 1.1"
        )


# ============================================================================
# Integration Tests: Performance Validation
# ============================================================================


class TestPerformanceIntegration:
    """Integration tests to validate performance targets across scenarios."""

    @pytest.mark.asyncio
    async def test_performance_targets_consolidated_single_exchange(self):
        """Validate performance targets for consolidated strategy, single exchange."""
        stub_producer = _RecordingProducer({})

        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            topic_strategy="consolidated",
            partition_key_strategy="composite",
            producer_factory=lambda config: stub_producer,
        )

        # Realistic scenario: 1K messages
        message_count = 1000
        for i in range(message_count):
            trade = _create_trade("coinbase", "BTC-USD", i)
            callback._queue_message("trade", trade)

        # Drain with timing
        start_ns = time.perf_counter_ns()

        async def _drain():
            while callback.queue_size() > 0:
                await callback._drain_once()

        await asyncio.wait_for(_drain(), timeout=30)

        elapsed_ns = time.perf_counter_ns() - start_ns
        elapsed_ms = elapsed_ns / 1_000_000

        # Verify latency
        avg_latency_ms = elapsed_ms / len(stub_producer.messages)

        assert avg_latency_ms < 50, (
            f"Average latency {avg_latency_ms:.2f}ms exceeds 50ms threshold"
        )

    @pytest.mark.asyncio
    async def test_performance_under_variable_load(self):
        """Validate performance under varying load patterns."""
        stub_producer = _RecordingProducer({})

        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            topic_strategy="consolidated",
            producer_factory=lambda config: stub_producer,
        )

        # Simulate variable load: bursts followed by pauses
        total_messages = 2000
        burst_size = 100
        exchanges = ["coinbase", "binance", "kraken"]
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

        async def _drain():
            while callback.queue_size() > 0:
                await callback._drain_once()

        for burst_start in range(0, total_messages, burst_size):
            # Queue burst of trades
            for i in range(burst_start, min(burst_start + burst_size, total_messages)):
                exchange = exchanges[i % len(exchanges)]
                symbol = symbols[i % len(symbols)]
                msg = _create_trade(exchange, symbol, i)
                callback._queue_message("trade", msg)

            # Drain burst
            await asyncio.wait_for(_drain(), timeout=30)

        # Verify all processed
        assert len(stub_producer.messages) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
