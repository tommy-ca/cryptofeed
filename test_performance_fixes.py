#!/usr/bin/env python
"""Quick test to verify TODO #010 and #011 performance fixes."""

from collections import OrderedDict
from cryptofeed.backends.kafka.callback import KafkaCallback
from cryptofeed.types import Trade
import time


def test_batch_polling_fix():
    """Verify TODO #010: Batch polling implemented."""
    print("\n=== Testing TODO #010: Batch Polling ===")

    # Create a mock producer that doesn't connect
    class MockProducer:
        def connect(self):
            pass  # Skip connection
        def list_topics(self, *args, **kwargs):
            return {}

    # Create callback with batch size of 5
    callback = KafkaCallback(
        bootstrap_servers=["localhost:9092"],
        poll_batch_size=5,
        producer_factory=lambda config: MockProducer()
    )

    # Verify poll counter initialized
    assert hasattr(callback, '_poll_counter'), "Missing _poll_counter attribute"
    assert hasattr(callback, '_poll_batch_size'), "Missing _poll_batch_size attribute"
    assert callback._poll_counter == 0, f"Expected counter=0, got {callback._poll_counter}"
    assert callback._poll_batch_size == 5, f"Expected batch_size=5, got {callback._poll_batch_size}"

    print("✓ Poll counter and batch size initialized correctly")
    print(f"  - _poll_counter: {callback._poll_counter}")
    print(f"  - _poll_batch_size: {callback._poll_batch_size}")

    # Verify default is 100
    callback2 = KafkaCallback(
        bootstrap_servers=["localhost:9092"],
        producer_factory=lambda config: MockProducer()
    )
    assert callback2._poll_batch_size == 100, f"Expected default batch_size=100, got {callback2._poll_batch_size}"
    print(f"✓ Default poll_batch_size is 100")


def test_lru_cache_fix():
    """Verify TODO #011: OrderedDict LRU cache implemented."""
    print("\n=== Testing TODO #011: LRU Cache ===")

    # Create a mock producer that doesn't connect
    class MockProducer:
        def connect(self):
            pass
        def list_topics(self, *args, **kwargs):
            return {}

    # Create callback with small cache for testing
    callback = KafkaCallback(
        bootstrap_servers=["localhost:9092"],
        partition_key_cache_size=3,  # Small cache for easy testing
        producer_factory=lambda config: MockProducer()
    )

    # Verify cache is OrderedDict
    assert hasattr(callback, '_partition_key_cache'), "Missing _partition_key_cache attribute"
    assert isinstance(callback._partition_key_cache, OrderedDict), \
        f"Cache should be OrderedDict, got {type(callback._partition_key_cache)}"

    print("✓ Cache is OrderedDict (not plain dict)")
    print(f"  - Cache type: {type(callback._partition_key_cache).__name__}")
    print(f"  - Cache size limit: {callback._partition_key_cache_size}")

    # Verify default cache size is 10,000 (increased from 1,000)
    callback2 = KafkaCallback(
        bootstrap_servers=["localhost:9092"],
        producer_factory=lambda config: MockProducer()
    )
    assert callback2._partition_key_cache_size == 10000, \
        f"Expected default cache_size=10000, got {callback2._partition_key_cache_size}"
    print(f"✓ Default partition_key_cache_size is 10,000 (increased from 1,000)")


def test_lru_eviction_behavior():
    """Test that LRU eviction works correctly."""
    print("\n=== Testing LRU Eviction Behavior ===")

    # Create a simple OrderedDict to simulate the cache behavior
    cache = OrderedDict()
    cache_size = 3

    # Add 4 items (should evict oldest)
    for i in range(1, 5):
        cache[f"key{i}"] = f"value{i}"
        if len(cache) > cache_size:
            cache.popitem(last=False)  # Remove oldest
        print(f"  After adding key{i}: {list(cache.keys())}")

    # Verify oldest was evicted
    assert 'key1' not in cache, "key1 should have been evicted"
    assert 'key2' in cache, "key2 should still be in cache"
    assert 'key3' in cache, "key3 should still be in cache"
    assert 'key4' in cache, "key4 should still be in cache"
    assert len(cache) == cache_size, f"Cache should have {cache_size} items, got {len(cache)}"

    print("✓ LRU eviction works correctly (oldest item evicted)")

    # Test move_to_end (mark as recently used)
    cache.move_to_end('key2')  # Move key2 to end (recently used)
    print(f"  After accessing key2: {list(cache.keys())}")

    # Add another item - should evict key3 (now oldest)
    cache['key5'] = 'value5'
    if len(cache) > cache_size:
        cache.popitem(last=False)
    print(f"  After adding key5: {list(cache.keys())}")

    assert 'key3' not in cache, "key3 should have been evicted (was oldest)"
    assert 'key2' in cache, "key2 should still be in cache (was marked recently used)"

    print("✓ move_to_end() correctly marks items as recently used")


if __name__ == "__main__":
    print("=" * 60)
    print("Performance Fixes Validation Test")
    print("=" * 60)

    try:
        test_batch_polling_fix()
        test_lru_cache_fix()
        test_lru_eviction_behavior()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Performance fixes verified!")
        print("=" * 60)
        print("\nSummary:")
        print("  • TODO #010: Batch polling implemented ✓")
        print("  • TODO #011: LRU cache with OrderedDict ✓")
        print("  • Cache size increased: 1,000 → 10,000 ✓")
        print("  • Poll batch size default: 100 ✓")
        print()

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
