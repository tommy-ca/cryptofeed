---
title: "Asyncio Queue Contract Violation in Kafka Backend Batch Drain"
problem_type: runtime-errors
component: kafka-backend
subcomponent: batch-draining
symptoms:
  - Missing queue.task_done() calls after queue.get_nowait() retrieval
  - Violates asyncio.Queue protocol contract where every get() must pair with task_done()
  - Queue state tracking becomes inconsistent over time
  - Risk of resource leaks in long-running message processing pipelines
  - Potential deadlocks or hangs when queue synchronization is required
root_cause: |
  The _drain_batch() method in cryptofeed/backends/kafka/base.py was processing messages
  retrieved via queue.get_nowait() without calling queue.task_done() to signal completion.
severity: high
tags:
  - asyncio
  - queue-contract
  - resource-management
  - batch-processing
  - kafka-backend
  - task-synchronization
date_solved: "2025-11-27"
commit: 9730d29e
affected_code_path: cryptofeed/backends/kafka/base.py
affected_method: _drain_batch()
---

# Asyncio Queue Contract Violation in Kafka Backend Batch Drain

## Problem

The `_drain_batch()` method was calling `queue.get_nowait()` to retrieve messages but was NOT calling `queue.task_done()` after processing. This violates the asyncio.Queue contract.

### Observable Symptoms

- Queue internal counter diverges from actual processed messages
- `queue.join()` never completes (hangs indefinitely)
- Memory/resource leaks in long-running pipelines
- Graceful shutdown fails to wait for queue drain

### Why This Matters

The asyncio.Queue class implements task counting semantics to support synchronization patterns. The `queue.join()` method waits for all retrieved items to be marked done. Without matching `task_done()` calls:

1. Queue size tracking becomes inaccurate
2. Graceful shutdown patterns that wait for queue drain fail
3. Any code relying on `queue.join()` for synchronization breaks

## Root Cause

The `_drain_once()` method (single-message path) correctly implemented this pattern with `task_done()` in a finally block, but `_drain_batch()` (batched path) was missing it entirely.

```python
# _drain_once() - CORRECT pattern (existed before fix)
async def _drain_once(self) -> None:
    message = await self._queue.get()
    try:
        if message is _STOP_SENTINEL:
            return
        await self._process_message(message)
    finally:
        try:
            self._queue.task_done()  # Always called
        except Exception as e:
            LOG.error(...)

# _drain_batch() - BROKEN pattern (before fix)
async def _drain_batch(self) -> None:
    while batch_count < max_batch:
        try:
            message = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        if message is _STOP_SENTINEL:
            self._running = False
            return  # task_done() NEVER called!

        await self._process_message(message)
        batch_count += 1  # task_done() NEVER called!
```

## Solution

Wrapped the message processing logic in a try/finally block that ensures `queue.task_done()` is called for every message successfully retrieved, even when processing fails or early returns occur (sentinel detection).

```python
# _drain_batch() - FIXED pattern
async def _drain_batch(self) -> None:
    batch_count = 0
    max_batch = self._batch_drain_size

    while batch_count < max_batch:
        try:
            message = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        try:
            if message is _STOP_SENTINEL:
                self._running = False
                return
            await self._process_message(message)
            batch_count += 1
        finally:
            try:
                self._queue.task_done()
            except Exception as e:
                LOG.error(
                    "%s: Failed to mark task as done: %s",
                    self._log_name,
                    e,
                    extra={
                        "error_type": "task_done_error",
                        "error": str(e),
                    },
                )

    await asyncio.sleep(0)
```

### Key Fix Points

1. **try/finally block** - Ensures `task_done()` is called regardless of success/failure
2. **Nested exception handling** - Logs but doesn't cascade `task_done()` failures
3. **Pattern consistency** - Now matches the correct `_drain_once()` pattern

## Prevention

### Code Review Checklist

- [ ] Every `queue.get()` or `queue.get_nowait()` has a matching `task_done()`
- [ ] `task_done()` is in a finally block to handle early returns/exceptions
- [ ] Batch processing paths match single-item processing patterns

### Testing Guidance

Test that `queue.join()` completes after all messages are processed:

```python
async def test_batch_drain_completes_task_done():
    backend = KafkaBackendBase(...)

    # Queue some messages
    for i in range(10):
        await backend._queue.put(make_test_message(i))

    # Drain the batch
    await backend._drain_batch()

    # This should complete immediately if task_done() was called
    await asyncio.wait_for(backend._queue.join(), timeout=1.0)
```

## Related Documentation

- [Kafka Architecture](../../kafka/architecture.md) - Queue management patterns
- [Knowledge Transfer](../../kafka/KNOWLEDGE_TRANSFER.md) - Kafka backend operations
- [ADR-001](../../kafka/decisions/ADR-001-deprecate-legacy-kafka-backend.md) - Legacy vs modern backend
- [Market Data Kafka Producer Design](../../../.kiro/specs/market-data-kafka-producer/design.md)

## Additional Changes in This Commit

The commit also added pydantic dependencies to base requirements:

```python
# setup.py additions
"pydantic>=2.0.0",
"pydantic-settings>=2.0.0",
```

These were previously optional but are now required by default for consistent behavior across deployments.
