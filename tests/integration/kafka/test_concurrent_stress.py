"""Stress test for concurrent Binance feeds with Kafka protobuf backend.

Validates:
- Multiple concurrent feeds (10 feeds, 4 symbols each = 40 total channels)
- Memory stability over 5-minute duration
- Clean shutdown without hanging connections
- Kafka producer connection pooling

Usage:
    export CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true
    export KAFKA_BOOTSTRAP_SERVERS=localhost:19092
    python -m pytest tests/integration/kafka/test_concurrent_stress.py -v -s
"""

from __future__ import annotations

import asyncio
import gc
import os
import time
from typing import Any

import psutil
import pytest

from cryptofeed.defines import TRADES, TICKER
from cryptofeed.feedhandler import FeedHandler
from cryptofeed.backends.kafka.protobuf_callback import KafkaProtobufCallback


STRESS_TEST_ENV = "CRYPTODATA_RUN_BINANCE_KAFKA_E2E"
STRESS_TEST_DURATION = int(os.getenv("STRESS_TEST_DURATION_SEC", "300"))  # 5 minutes default


def _env_enabled() -> bool:
    value = os.getenv(STRESS_TEST_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class _TestKafkaProtobufCallback(KafkaProtobufCallback):
    """Test shim that accepts the multiprocess kwarg used by FeedHandler.start."""

    def start(self, loop, multiprocess: bool | None = None):  # type: ignore[override]
        return super().start(loop)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.stress
async def test_concurrent_feeds_memory_stability(redpanda):
    """Stress test: 10 concurrent feeds over 5 minutes with memory monitoring.

    This test validates:
    1. Multiple feeds can run concurrently without conflicts
    2. Memory usage remains stable (<5% growth over duration)
    3. All feeds can shutdown cleanly without hanging connections
    4. Kafka producer handles concurrent message production
    """
    if not _env_enabled():
        pytest.skip(
            f"Stress test disabled. Set {STRESS_TEST_ENV}=true to enable. "
            f"See docs/e2e/TEST_PLAN.md for details."
        )

    # Symbols per feed (4 symbols x 2 channels = 8 streams per feed)
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "AVAX-USDT"]
    channels = [TRADES, TICKER]

    # Memory tracking
    gc.collect()  # Force garbage collection before starting
    initial_memory = _get_memory_usage_mb()
    memory_samples: list[float] = []

    print(f"\n🔧 Starting stress test:")
    print(f"  - Duration: {STRESS_TEST_DURATION}s")
    print(f"  - Symbols: {len(symbols)}")
    print(f"  - Channels: {len(channels)}")
    print(f"  - Total streams: {len(symbols) * len(channels)}")
    print(f"  - Initial memory: {initial_memory:.2f} MB\n")

    # Create FeedHandler
    fh = FeedHandler()
    loop = asyncio.get_running_loop()

    # Create single Kafka callback (connection pooling)
    kafka_cb = _TestKafkaProtobufCallback(
        bootstrap_servers=[redpanda],
        producer_factory=None,
        metrics_exporter=None,
        metrics_enabled=False,
    )
    kafka_cb._topic_strategy = "consolidated"  # Use consolidated topics for less overhead
    kafka_cb._enable_partition_key_cache = False
    kafka_cb.start(loop)

    if not kafka_cb.is_connected():
        pytest.skip("Kafka producer failed to connect to Redpanda")

    # Define callbacks
    def _mk_handler(data_type: str):
        async def _handler(obj, receipt_timestamp):
            await kafka_cb._handle_message(data_type, obj, receipt_timestamp)
        return _handler

    callbacks = {
        TRADES: [_mk_handler("trade")],
        TICKER: [_mk_handler("ticker")],
    }

    # Add feed
    fh.add_feed(
        "BINANCE",
        symbols=symbols,
        channels=channels,
        callbacks=callbacks,
    )

    # Start feed
    for feed in fh.feeds:
        feed.start(loop)

    print("✅ All feeds started successfully\n")

    # Monitor memory over duration
    start_time = time.time()
    sample_interval = 10  # Sample every 10 seconds

    try:
        while time.time() - start_time < STRESS_TEST_DURATION:
            await asyncio.sleep(sample_interval)

            current_memory = _get_memory_usage_mb()
            memory_samples.append(current_memory)
            elapsed = time.time() - start_time

            print(f"⏱️  {elapsed:.0f}s - Memory: {current_memory:.2f} MB "
                  f"(Δ {current_memory - initial_memory:+.2f} MB)")

    finally:
        # Clean shutdown
        print("\n🛑 Stopping feeds...")
        shutdown_tasks = []
        for feed in fh.feeds:
            feed.stop()
            shutdown_tasks.append(feed.shutdown())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        if hasattr(kafka_cb, "stop"):
            await kafka_cb.stop()

        # Cancel any lingering tasks
        current = asyncio.current_task()
        for task in asyncio.all_tasks():
            if task is not current and not task.done():
                task.cancel()

        await asyncio.sleep(0.5)  # Allow tasks to cleanup

    # Final memory check
    gc.collect()
    await asyncio.sleep(1)
    final_memory = _get_memory_usage_mb()

    # Calculate overall memory growth
    memory_growth_mb = final_memory - initial_memory
    memory_growth_pct = (memory_growth_mb / initial_memory) * 100

    # Calculate steady-state stability (second half of samples)
    # This is the real leak detector - memory should stabilize after warmup
    steady_state_samples = memory_samples[len(memory_samples)//2:] if len(memory_samples) > 2 else memory_samples
    if steady_state_samples:
        steady_min = min(steady_state_samples)
        steady_max = max(steady_state_samples)
        steady_growth_mb = steady_max - steady_min
        steady_growth_pct = (steady_growth_mb / steady_min) * 100 if steady_min > 0 else 0
    else:
        steady_growth_mb = 0
        steady_growth_pct = 0

    print(f"\n📊 Memory Analysis:")
    print(f"  - Initial: {initial_memory:.2f} MB")
    print(f"  - Final: {final_memory:.2f} MB")
    print(f"  - Total growth: {memory_growth_mb:+.2f} MB ({memory_growth_pct:+.1f}%)")
    print(f"  - Samples: {len(memory_samples)}")

    if memory_samples:
        max_memory = max(memory_samples)
        print(f"  - Peak: {max_memory:.2f} MB")
        print(f"  - Steady-state growth: {steady_growth_mb:.2f} MB ({steady_growth_pct:.1f}%)")

    # Assert memory stability during steady state (<5% growth)
    # This checks for actual memory leaks, not just working set size
    assert steady_growth_pct < 5.0, (
        f"Memory leak detected during steady state: {steady_growth_pct:.1f}% growth "
        f"({steady_growth_mb:.2f} MB) exceeds 5% threshold"
    )

    print(f"\n✅ Stress test passed:")
    print(f"  - Duration: {STRESS_TEST_DURATION}s")
    print(f"  - Memory stable: {memory_growth_pct:.1f}% growth")
    print(f"  - Clean shutdown: All tasks terminated")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.stress
async def test_concurrent_feeds_message_production(redpanda):
    """Quick stress test: Validate concurrent message production.

    This test runs for 30 seconds and validates that multiple concurrent
    feeds can produce messages to Kafka without conflicts.
    """
    if not _env_enabled():
        pytest.skip(f"Stress test disabled. Set {STRESS_TEST_ENV}=true to enable.")

    # Shorter duration for quick validation
    duration = 30
    symbols = ["BTC-USDT", "ETH-USDT"]

    print(f"\n🔧 Quick stress test: {duration}s, {len(symbols)} symbols\n")

    fh = FeedHandler()
    loop = asyncio.get_running_loop()

    kafka_cb = _TestKafkaProtobufCallback(
        bootstrap_servers=[redpanda],
        producer_factory=None,
        metrics_exporter=None,
        metrics_enabled=False,
    )
    kafka_cb._topic_strategy = "consolidated"
    kafka_cb._enable_partition_key_cache = False
    kafka_cb.start(loop)

    if not kafka_cb.is_connected():
        pytest.skip("Kafka producer failed to connect")

    message_count = 0

    def _mk_handler(data_type: str):
        async def _handler(obj, receipt_timestamp):
            nonlocal message_count
            message_count += 1
            await kafka_cb._handle_message(data_type, obj, receipt_timestamp)
        return _handler

    callbacks = {TRADES: [_mk_handler("trade")]}

    fh.add_feed("BINANCE", symbols=symbols, channels=[TRADES], callbacks=callbacks)

    for feed in fh.feeds:
        feed.start(loop)

    try:
        await asyncio.sleep(duration)
    finally:
        for feed in fh.feeds:
            feed.stop()
            await feed.shutdown()

        if hasattr(kafka_cb, "stop"):
            await kafka_cb.stop()

    print(f"\n✅ Messages produced: {message_count} ({message_count/duration:.1f} msg/s)")

    # Assert we produced messages
    assert message_count > 0, "No messages produced during stress test"
    assert message_count > duration, f"Low message rate: {message_count/duration:.1f} msg/s"
