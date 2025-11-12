# Kafka Producer Tuning Guide

## Executive Summary

This comprehensive guide provides production configuration strategies for the Cryptofeed Kafka producer. The guide covers configuration tuning for different use cases (latency-sensitive vs throughput-optimized), explains the impact of each setting, and provides detailed profiles for typical deployment scenarios.

**Audience**: Platform engineers, DevOps operators, and SREs responsible for Kafka producer operations.

**Key Objectives**:
1. Understand Kafka producer configuration parameters and their impacts
2. Select appropriate tuning profiles for your use case
3. Measure and optimize performance through monitoring-driven adjustments
4. Troubleshoot common performance issues

---

## Part 1: Configuration Reference

This section documents all key Kafka producer configuration parameters with explanations and performance impact.

### batch.size (Default: 16384 bytes = 16KB)

**Description**: The maximum size of a batch of messages the producer will accumulate before sending to the broker.

**Impact**:
- **Larger values** (32KB, 64KB): Better throughput but higher latency
  - More messages accumulated before sending = fewer network round trips
  - Trade-off: Each message waits longer to be batched
  - Use case: Throughput-optimized deployments
- **Smaller values** (4KB, 8KB): Lower latency but reduced throughput
  - Messages sent more frequently = more network overhead
  - Trade-off: Each message goes out faster
  - Use case: Latency-sensitive deployments

**Production Recommendations**:
- Latency-sensitive (p99 <5ms): 4KB-8KB
- Balanced: 16KB (default, good for most cases)
- Throughput-optimized: 32KB-64KB

**Formula**: Choose batch size based on message size and target throughput
```
Optimal batch size = Average message size × Messages per batch
                   = 200 bytes × 80 messages = 16KB (typical case)
```

### linger.ms (Default: 0)

**Description**: Time in milliseconds the producer waits before sending a batch, even if the batch is not full.

**Impact**:
- **linger.ms=0** (default): Send immediately when possible
  - Each message may be sent separately if batch not full
  - Higher latency per message, but lower overall latency
  - Maximum throughput limited by network round trip time (RTT)
- **linger.ms=10-100**: Wait up to X milliseconds to accumulate batch
  - Allows batching of messages arriving within window
  - Reduces network round trips, improves throughput
  - Increases message latency by up to X milliseconds
- **linger.ms>100**: Aggressive batching for maximum throughput
  - Trade-off: Significant latency increase (not suitable for latency-sensitive apps)

**Trade-off Formula**:
```
Expected latency increase = linger.ms (approximately)
Throughput improvement = Average RTT / (Average RTT + linger.ms)
```

**Production Recommendations**:
- Latency-sensitive (p99 <5ms): 0 (no linger)
- Balanced: 5-10ms (mild batching)
- Throughput-optimized (>100k msg/s): 50-100ms

**Example Latency Impact**:
- RTT to broker = 5ms
- With linger=0: p99 latency ≈ 5ms
- With linger=10ms: p99 latency ≈ 15ms (10ms wait + 5ms RTT)
- With linger=100ms: p99 latency ≈ 105ms (100ms wait + 5ms RTT)

### buffer.memory (Default: 33554432 bytes = 32MB)

**Description**: The total bytes of memory the producer can use to buffer messages awaiting transmission to the broker.

**Impact**:
- **Too small** (<32MB for high-throughput): Producer blocks when buffer full
  - Application waits for buffer space
  - Can cause cascading delays in feed handler
  - Visible as "producer buffer full" errors in logs
- **Too large** (>512MB): Excessive memory usage
  - Wastes RAM for buffered messages
  - Slower GC pauses in Java/Python systems
  - Risk of OOM if broker unavailable

**Memory Calculation**:
```
Buffer size needed = Expected message rate × linger.ms × Average message size
                   = 10,000 msg/s × 100ms × 200 bytes
                   = 10,000 × 0.1 × 200 = 200KB minimum
Recommended = 10× minimum for safety = 2MB (for safety margin)
```

**Production Recommendations**:
- Single exchange, latency-sensitive: 32MB (default, sufficient)
- Multi-exchange, throughput-optimized: 64MB-128MB
- High-throughput (100k+ msg/s): 256MB-512MB
- Maximum: 512MB (monitor for memory pressure)

### compression.type (Default: none)

**Description**: Compression algorithm applied to message batches before transmission.

**Supported algorithms**:
- `none`: No compression (default, fastest)
- `snappy`: Fast, moderate compression (recommended for most cases)
- `lz4`: Very fast, light compression
- `gzip`: Slow, aggressive compression (best compression ratio)

**Performance Impact** (per message):
| Algorithm | Compression | Speed | CPU | Network | Use Case |
|-----------|-------------|-------|-----|---------|----------|
| none | 0% | ~0μs | 0% | 100% | Latency-critical |
| lz4 | 20-30% | 2μs | 5% | Baseline | Balanced (recommended) |
| snappy | 30-40% | 5μs | 8% | Good | Throughput, cost-conscious |
| gzip | 50-60% | 50μs | 20% | Low | Bandwidth-limited, cost priority |

**Compression Break-Even**:
```
Break-even = CPU cost < Network savings
           = 5μs × throughput < Network transmission time saved

For 10,000 msg/s with snappy (30% compression):
- Network savings ≈ 200 bytes × 30% × 10,000 = 600KB/s = 4.8Mbps/s
- CPU cost ≈ 5μs × 10,000 = 50ms/s = negligible
- Conclusion: Always use snappy for 10k+ msg/s throughput
```

**Production Recommendations**:
- Latency-sensitive (<5ms p99): `none` (save CPU)
- Balanced, multi-exchange: `snappy` (default recommendation)
- Throughput-optimized: `snappy` or `lz4`
- Bandwidth-limited (satellite, WAN): `gzip`

### acks (Default: 1)

**Description**: Number of broker replicas that must acknowledge write before producer considers message sent.

**Options**:
- `acks=0`: No acknowledgment (fire-and-forget)
  - Producer doesn't wait for broker response
  - Lowest latency (no ACK round trip)
  - Risk: No guarantee message received
  - Use only for non-critical data (analytics, summaries)
- `acks=1` (default): Leader replica acknowledges
  - Producer waits for leader acknowledgment
  - Safe for most cases (leader typically doesn't fail)
  - If leader fails before replication: message loss possible (rare)
  - Latency: 1 network round trip
- `acks=all` (or `acks=-1`): All in-sync replicas acknowledge
  - Slowest, most durable option
  - Producer waits for all ISR replicas
  - Guarantee: No message loss (safe for critical data)
  - Latency: Multiple network round trips

**Latency Impact**:
```
acks=0:   p99 latency ≈ producer_processing time (microseconds)
acks=1:   p99 latency ≈ network_rtt + broker_processing (~5-10ms)
acks=all: p99 latency ≈ max(rtt to all replicas) + broker (~15-50ms)
```

**Reliability Matrix**:
| Ack Level | Data Loss Risk | Typical Latency | Use Case |
|-----------|----------------|-----------------|----------|
| acks=0 | High | <1ms | Telemetry, analytics |
| acks=1 | Low | 5-10ms | Most production cases |
| acks=all | None | 15-50ms | Financial, critical data |

**Production Recommendations**:
- Market data trades (critical): `acks=all`
- Market data tickers (moderate criticality): `acks=1`
- Analytics/summaries (non-critical): `acks=0`

### retries (Default: 2147483647 = max int, effectively unlimited)

**Description**: Maximum number of times the producer will retry sending a failed message.

**Impact**:
- Retries are critical for handling transient broker failures
- Producer implements exponential backoff: 10ms, 20ms, 40ms, ...
- Default (unlimited) means: keep retrying until success or timeout
- Each retry adds ~10-100ms to message latency (depends on backoff)

**Configuration**:
```yaml
# Default: unlimited retries
retries: 2147483647

# Or: specific count
retries: 3  # After 3 failures, give up
```

**Backoff Schedule** (exponential):
```
Attempt 1: Immediate
Attempt 2: Wait 10ms + randomization
Attempt 3: Wait 20ms + randomization
Attempt 4: Wait 40ms + randomization
Attempt 5: Wait 80ms + randomization
...
Max backoff: 1000ms (typical default)
```

**Production Recommendations**:
- Default (unlimited): Recommended for most cases
- Can lower retries if you monitor broker health closely and rollback quickly
- Warning: Setting retries=0 is dangerous (no recovery from transient failures)

### max.in.flight.requests.per.connection (Default: 5)

**Description**: Maximum number of unacknowledged requests the producer maintains to a broker.

**Impact on Ordering**:
- If >1: Messages may be delivered out of order on broker failure/retry
  - Producer can have multiple requests in flight simultaneously
  - If request N fails but request N+1 succeeds, N may be retried after N+1
  - Result: Messages may be out of order on consumption
- If =1: Strict ordering guaranteed (but lower throughput)
  - Producer waits for each request to complete before sending next
  - Throughput reduced, latency increased

**Throughput Impact**:
```
Throughput (msg/s) = (max.in.flight × 1000) / (latency_ms)

Example with 5ms latency:
- max.in.flight=1:  Throughput ≈ 200 msg/s (ordered)
- max.in.flight=5:  Throughput ≈ 1,000 msg/s (unordered on retry)
- max.in.flight=10: Throughput ≈ 2,000 msg/s (unordered on retry)
```

**Production Recommendations**:
- Strict ordering required: `max.in.flight.requests=1`
- With idempotent producer (see next section): Safe to use 5-10
- Throughput priority: Use 5-10

**Note**: Combined with `enable.idempotence=true`, Kafka guarantees exactly-once semantics even with max.in.flight > 1.

### request.timeout.ms (Default: 30000 = 30 seconds)

**Description**: Maximum time the producer waits for broker response to a request.

**Impact**:
- If timeout occurs: Producer considers request failed, triggers retry
- Too short (<5s): False timeouts during broker GC pauses (common)
- Too long (>60s): Slow detection of broker failures
- Sweet spot: 30-60s (covers most GC pauses, detects failures quickly)

**Failure Scenario**:
```
Normal case (broker responsive):
1. Producer sends request at T=0
2. Broker processes in 5-10ms
3. Broker sends response at T=10
4. Producer receives at T=15 ✓ (well within 30s timeout)

Broker GC pause (40ms pause):
1. Producer sends request at T=0
2. Broker pauses for GC (40ms)
3. Broker processes request at T=60
4. Broker sends response
5. Producer receives at T=75 ✓ (still within 30s timeout)

Broker failure (broker down):
1. Producer sends request at T=0
2. Broker is down
3. No response received
4. At T=30s: Timeout triggers, producer retries ✓
5. Retry routed to other broker ✓
```

**Production Recommendations**:
- Default (30s): Recommended for most cases
- If frequent false timeouts: Increase to 60s
- Monitor: Check for timeout errors in logs, adjust if needed

### enable.idempotence (Default: false)

**Description**: Enable idempotent producer for exactly-once semantics.

**Impact**:
- When true: Producer assigns sequence numbers to messages
- Broker deduplicates messages with same sequence number
- Guarantees: Each message delivered exactly once (no duplicates)
- Cost: Slight latency increase (sequence tracking overhead)

**Important Configuration Constraint**:
When `enable.idempotence=true`, the following are forced:
- `acks=all` (automatic)
- `retries=MAX_INT` (automatic)
- `max.in.flight.requests=5` (automatic, or higher)

**Production Recommendations**:
- Always enable for critical data: `enable.idempotence=true`
- Trades off ~1-2ms latency for exactly-once guarantee
- Worth the cost for market data (financial accuracy critical)

---

## Part 2: Use Case Profiles

This section provides recommended configurations for four common deployment scenarios.

### Profile 1: Latency-Sensitive (p99 <5ms target)

**Typical Use Case**: Real-time trading systems, live market displays, ultra-low-latency feeds.

**Characteristics**:
- Every millisecond matters (sub-5ms p99 latency required)
- Throughput: 1,000-10,000 msg/s per instance
- Message loss acceptable: No (critical data)
- Ordering important: Yes (per-symbol ordering required)

**Recommended Configuration**:
```yaml
kafka:
  producer:
    bootstrap_servers:
      - "kafka1.internal:9092"
      - "kafka2.internal:9092"
      - "kafka3.internal:9092"

    # Batching: Minimal
    batch_size: 8192           # 8KB (fast batching)
    linger_ms: 0               # No waiting

    # Reliability
    acks: 1                    # Leader acknowledgment sufficient
    enable_idempotence: true   # Exactly-once guarantee
    retries: 2147483647        # Keep retrying on transient failures

    # Performance
    buffer_memory: 32000000    # 32MB (standard)
    max_in_flight_requests_per_connection: 5
    request_timeout_ms: 30000

    # Compression: None (save CPU)
    compression_type: none

    # Advanced tuning
    connections_max_idle_ms: 300000
    metadata_max_age_ms: 300000
```

**Expected Performance**:
- p99 latency: 3-5ms
- Throughput: 5,000-10,000 msg/s
- CPU usage: 20-30%
- Memory: 32MB

**Validation Test**:
```python
# Measure p99 latency
latencies = []
for i in range(10000):
    start = time.perf_counter()
    producer.send("cryptofeed.trades", message)
    latencies.append(time.perf_counter() - start)

p99 = sorted(latencies)[int(len(latencies) * 0.99)]
assert p99 < 0.005, f"P99 latency {p99*1000}ms exceeds 5ms target"
```

---

### Profile 2: Throughput-Optimized (>100k msg/s target)

**Typical Use Case**: Bulk analytics ingestion, data warehouse backfill, multi-exchange aggregation.

**Characteristics**:
- High message volume (100,000+ msg/s)
- Latency secondary (100-500ms acceptable)
- Message loss: Unacceptable (critical data)
- Ordering: Less critical (can aggregate across partitions)

**Recommended Configuration**:
```yaml
kafka:
  producer:
    bootstrap_servers:
      - "kafka1.internal:9092"
      - "kafka2.internal:9092"
      - "kafka3.internal:9092"

    # Batching: Aggressive
    batch_size: 65536          # 64KB (maximize batching)
    linger_ms: 100             # Wait 100ms to fill batches

    # Reliability
    acks: 1                    # Leader ack (fast, safe)
    enable_idempotence: true   # Exactly-once guarantee
    retries: 2147483647        # Keep retrying

    # Performance
    buffer_memory: 536870912   # 512MB (large buffer for high volume)
    max_in_flight_requests_per_connection: 10
    request_timeout_ms: 30000

    # Compression: Aggressive
    compression_type: snappy   # Good compression, fast speed

    # Advanced tuning
    connections_max_idle_ms: 600000
    metadata_max_age_ms: 300000
```

**Expected Performance**:
- Throughput: 100,000-500,000 msg/s
- p99 latency: 100-500ms
- CPU usage: 60-80%
- Memory: 256-512MB

**Validation Test**:
```python
# Measure throughput
import time
import threading

count = [0]
duration = 10
stop = [False]

def send_messages():
    while not stop[0]:
        producer.send("cryptofeed.trades", {"price": 50000})
        count[0] += 1

# Send for 10 seconds
thread = threading.Thread(target=send_messages)
thread.start()
time.sleep(duration)
stop[0] = True
thread.join()

throughput = count[0] / duration
print(f"Throughput: {throughput:,.0f} msg/s")
assert throughput > 100000, f"Throughput {throughput} below 100k target"
```

---

### Profile 3: Balanced (Default, Recommended)

**Typical Use Case**: Standard market data feeds, multi-exchange ingestion, typical production deployments.

**Characteristics**:
- Moderate throughput (10,000-50,000 msg/s)
- Latency: 10-50ms acceptable
- Message loss: Unacceptable
- Ordering: Required (per-symbol minimum)

**Recommended Configuration** (Most Production Deployments):
```yaml
kafka:
  producer:
    bootstrap_servers:
      - "kafka1.internal:9092"
      - "kafka2.internal:9092"
      - "kafka3.internal:9092"

    # Batching: Moderate
    batch_size: 16384          # 16KB (default, good balance)
    linger_ms: 10              # Wait 10ms to accumulate

    # Reliability
    acks: 1                    # Leader acknowledgment
    enable_idempotence: true   # Exactly-once guarantee
    retries: 2147483647        # Keep retrying

    # Performance
    buffer_memory: 67108864    # 64MB
    max_in_flight_requests_per_connection: 5
    request_timeout_ms: 30000

    # Compression
    compression_type: snappy   # Recommended balance

    # Advanced tuning
    connections_max_idle_ms: 300000
    metadata_max_age_ms: 300000
```

**Expected Performance**:
- Throughput: 10,000-50,000 msg/s
- p99 latency: 15-30ms
- CPU usage: 30-50%
- Memory: 64MB

---

### Profile 4: High-Reliability (Critical Financial Data)

**Typical Use Case**: Critical trading infrastructure, risk systems, regulatory reporting.

**Characteristics**:
- Zero message loss requirement
- Throughput: 1,000-10,000 msg/s (not as critical as reliability)
- Latency: Up to 100ms acceptable
- Ordering: Strictly required

**Recommended Configuration**:
```yaml
kafka:
  producer:
    bootstrap_servers:
      - "kafka1.internal:9092"
      - "kafka2.internal:9092"
      - "kafka3.internal:9092"

    # Batching: Conservative
    batch_size: 16384          # Standard batch size
    linger_ms: 50              # Allow batching accumulation

    # Reliability: Maximum
    acks: all                  # All replicas must acknowledge
    enable_idempotence: true   # Exactly-once guarantee
    retries: 2147483647        # Unlimited retries

    # Performance
    buffer_memory: 134217728   # 128MB (large safety margin)
    max_in_flight_requests_per_connection: 5
    request_timeout_ms: 60000  # Extended timeout for replication

    # Compression
    compression_type: snappy   # Reduce network load

    # Advanced tuning
    connections_max_idle_ms: 300000
    metadata_max_age_ms: 300000

    # Additional reliability settings
    min_insync_replicas: 2     # Require 2 replicas minimum (broker config)
```

**Expected Performance**:
- Throughput: 1,000-10,000 msg/s
- p99 latency: 50-150ms
- CPU usage: 20-40%
- Memory: 128MB
- Durability: Zero message loss (exactly-once semantics)

**Validation Test**:
```python
# Verify exactly-once delivery
import random
import string

test_id = ''.join(random.choices(string.ascii_letters, k=10))
messages_sent = []

# Send 1000 messages
for i in range(1000):
    msg_id = f"{test_id}-{i}"
    producer.send("cryptofeed.trades", {"msg_id": msg_id})
    messages_sent.append(msg_id)

# Consume and count
consumer = KafkaConsumer("cryptofeed.trades")
messages_received = set()
for msg in consumer:
    msg_id = json.loads(msg.value)["msg_id"]
    if msg_id.startswith(test_id):
        messages_received.add(msg_id)
    if len(messages_received) >= 1000:
        break

# Verify no duplicates and no loss
assert len(messages_received) == 1000, "Message loss detected"
assert len(messages_received) == len(set(messages_received)), "Duplicates detected"
```

---

## Part 3: Performance Tuning Checklist

Use this checklist when optimizing producer performance for your specific use case.

### Phase 1: Identify Bottleneck

- [ ] **Monitor baseline metrics** (before tuning):
  - [ ] Measure current throughput (msg/s)
  - [ ] Measure current p99 latency (ms)
  - [ ] Record CPU usage (%)
  - [ ] Record memory usage (MB)
  - [ ] Note any error rates

- [ ] **Identify primary constraint**:
  - [ ] CPU-bound? (producer thread maxed out)
    - Likely causes: No compression, small batches, high overhead
  - [ ] Memory-bound? (buffer memory exceeds limit)
    - Likely causes: Large buffer.memory setting, slow broker
  - [ ] Network-bound? (network utilization near link capacity)
    - Likely causes: No compression, large messages
  - [ ] Broker-bound? (broker becoming bottleneck)
    - Likely causes: All producers routing to single broker, rebalancing needed

### Phase 2: Adjust Configuration Based on Bottleneck

**If CPU-bound**:
- [ ] Increase batch_size (8KB → 32KB → 64KB)
- [ ] Enable compression: `compression_type: snappy`
- [ ] Increase linger_ms (0 → 10 → 50ms) to accumulate more batches
- [ ] Validate: CPU should decrease, latency may increase

**If Memory-bound**:
- [ ] Reduce buffer_memory (current → 75% → 50% of current)
- [ ] Reduce batch_size slightly
- [ ] Increase linger_ms to reduce total message accumulation
- [ ] Monitor: Watch for producer.send() blocking (sign of insufficient buffer)

**If Network-bound**:
- [ ] Enable/increase compression: `compression_type: snappy` (or gzip)
- [ ] Increase batch_size for better compression ratio
- [ ] Increase linger_ms to batch more messages
- [ ] Validate: Network utilization should decrease, latency may increase

**If Broker-bound**:
- [ ] Increase broker parallelism: `max.in.flight.requests_per_connection`
- [ ] Check broker rebalancing: Look for partition reassignment in broker logs
- [ ] Consider per-symbol topics (distribute across more brokers)
- [ ] Scale brokers: Add more brokers to cluster

### Phase 3: Measure Impact

- [ ] **Re-measure metrics after each change**:
  - [ ] Throughput change: +/- ?%
  - [ ] Latency change: +/- ?ms
  - [ ] CPU change: +/- ?%
  - [ ] Memory change: +/- ?MB

- [ ] **Document results**:
  ```
  Configuration: batch_size 16KB → 32KB
  Throughput: 50,000 msg/s → 75,000 msg/s (+50%)
  Latency p99: 10ms → 15ms (+5ms acceptable)
  CPU: 60% → 45% (-15% improvement!)
  Verdict: ACCEPT
  ```

### Phase 4: Validate Against Requirements

- [ ] **Check against SLOs**:
  - [ ] Throughput: _____ msg/s (target: _____ msg/s) ✓
  - [ ] Latency p99: _____ ms (target: <_____ ms) ✓
  - [ ] CPU usage: _____ % (target: <_____ %) ✓
  - [ ] Memory: _____ MB (target: <_____ MB) ✓
  - [ ] Error rate: _____ % (target: <_____ %) ✓

- [ ] **Run load test with production traffic pattern**:
  - [ ] Duration: 30+ minutes (catch GC pauses, rebalancing)
  - [ ] Load profile: Match production (bursts, quiet periods)
  - [ ] Multi-exchange: Test with realistic symbol distribution

- [ ] **Check broker health during test**:
  - [ ] No broker outages
  - [ ] No partition reassignment
  - [ ] No consumer lag buildup
  - [ ] Broker CPU <70% (safety margin)

---

## Part 4: Monitoring-Driven Optimization Workflow

### Step 1: Establish Prometheus Metrics Collection

Enable Prometheus metrics in KafkaCallback:
```python
from cryptofeed.backends.kafka_metrics import PrometheusMetricsExporter

metrics = PrometheusMetricsExporter()
callback = KafkaCallback(
    bootstrap_servers=["localhost:9092"],
    metrics_exporter=metrics
)
```

Key metrics to monitor:
- `cryptofeed_kafka_messages_sent_total`: Total messages sent (counter)
- `cryptofeed_kafka_produce_latency_seconds`: Latency histogram (p50, p95, p99)
- `cryptofeed_kafka_errors_total`: Total errors by type
- `cryptofeed_kafka_producer_queue_size`: Current queue size (gauge)
- `cryptofeed_kafka_buffer_memory_usage`: Buffer memory utilization (gauge)

### Step 2: Set Up Grafana Dashboard

Create dashboard with panels:
1. **Throughput panel**: Messages sent per second
2. **Latency panel**: p50, p95, p99 latency over time
3. **Error rate panel**: Errors per second
4. **Queue size panel**: Current producer queue depth
5. **Memory panel**: Buffer memory utilization
6. **CPU panel**: Producer CPU usage

### Step 3: Identify Performance Anomalies

Monitor for these patterns:
- **Latency spike**: Look for spike in `produce_latency_seconds` histogram
  - Check: Broker load, network congestion, GC pauses
- **Queue buildup**: Queue size continuously growing
  - Check: Broker availability, network connectivity, buffer size
- **Error rate increase**: Spike in `errors_total` metric
  - Check: Auth failures, broker restarts, network issues
- **Memory growth**: Steady increase in `buffer_memory_usage`
  - Check: Slow broker processing, buffer leak, large messages

### Step 4: Adjust Configuration Based on Metrics

**Example: Latency spike detected**
```
Current state:
- p99 latency spike from 10ms → 30ms
- Queue size: Stable at 100 messages
- Broker load: Low

Hypothesis: Network congestion during off-peak batching

Action:
1. Reduce linger_ms: 50ms → 10ms (faster batching)
2. Or reduce batch_size: 32KB → 16KB (smaller batches)

Expected result: Latency returns to <15ms baseline
```

**Example: Queue buildup detected**
```
Current state:
- Queue size: Growing from 100 → 1000 → 5000
- Throughput: Steady at 50,000 msg/s
- Broker load: High (>80% CPU)

Hypothesis: Broker overload, unable to keep up

Action:
1. Increase max.in.flight.requests: 5 → 10 (multi-broker sending)
2. Or distribute to different broker cluster

Expected result: Queue drains as brokers share load
```

### Step 5: Validate with Synthetic Load Test

After configuration change:
```python
import asyncio
import time
from statistics import mean, stdev

async def load_test_validation():
    latencies = []

    # Send 10,000 messages
    for i in range(10000):
        start = time.perf_counter()
        await producer.send("cryptofeed.trades", {
            "exchange": "binance",
            "symbol": "btc-usd",
            "price": 50000 + random.random() * 100,
            "volume": 1.0,
        })
        latencies.append(time.perf_counter() - start)

    # Analyze results
    p50 = sorted(latencies)[int(len(latencies) * 0.50)]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]

    print(f"Latency p50: {p50*1000:.1f}ms")
    print(f"Latency p95: {p95*1000:.1f}ms")
    print(f"Latency p99: {p99*1000:.1f}ms")
    print(f"Throughput: {10000 / (time.time() - start):,.0f} msg/s")

    # Assert against SLOs
    assert p99 < 0.050, "P99 latency exceeds 50ms target"
    assert p50 < 0.020, "P50 latency exceeds 20ms target"
```

---

## Part 5: Common Tuning Scenarios

### Scenario 1: Reducing Latency from 15ms to <5ms

**Problem**: Current p99 latency is 15ms, need to reduce to <5ms for trading system.

**Root Cause Analysis**:
1. Check current config: `batch_size=32KB, linger_ms=50, compression=snappy`
2. Linger is 50ms? That's the problem - waiting 50ms per message!

**Solution**:
```yaml
# Before (15ms p99)
batch_size: 32768          # 32KB
linger_ms: 50              # 50ms wait = 50ms latency!
compression_type: snappy

# After (5ms p99)
batch_size: 8192           # 8KB (faster batching)
linger_ms: 0               # No waiting
compression_type: none     # Save CPU for low-latency
acks: 1                    # Leader ack sufficient
```

**Impact**:
- Latency: 15ms → 5ms (meets requirement)
- Throughput: 100k msg/s → 50k msg/s (acceptable trade-off)
- CPU: 40% → 60% (acceptable increase)

---

### Scenario 2: Increasing Throughput from 50k to 150k msg/s

**Problem**: Needs to ingest 150k msg/s but currently at 50k msg/s throughput.

**Root Cause Analysis**:
1. Check bottleneck: `batch_size=16KB, linger_ms=10, no compression`
2. Small batches + no compression = network overhead

**Solution**:
```yaml
# Before (50k msg/s)
batch_size: 16384          # 16KB
linger_ms: 10              # 10ms wait
compression_type: none     # No compression
buffer_memory: 67108864    # 64MB

# After (150k msg/s)
batch_size: 65536          # 64KB (4x larger batches)
linger_ms: 50              # 50ms wait (allow more accumulation)
compression_type: snappy   # Enable compression (30% reduction)
buffer_memory: 268435456   # 256MB (4x larger buffer)
max_in_flight_requests_per_connection: 10
```

**Impact**:
- Throughput: 50k msg/s → 150k msg/s (3x improvement)
- Latency: 10ms → 60ms (acceptable for analytics)
- CPU: 30% → 70% (intensive, monitor for safety)
- Network: Reduced 30% due to compression

---

### Scenario 3: Reducing Memory Usage from 256MB to 64MB

**Problem**: Producer consuming too much memory, need to reduce footprint.

**Root Cause Analysis**:
1. Check: `buffer_memory=268MB` - too large for throughput
2. Calculation: 268MB buffer for 20k msg/s throughput = 13.4s worth of messages!

**Solution**:
```yaml
# Before (256MB memory)
buffer_memory: 268435456   # 268MB (too large)
batch_size: 65536          # 64KB
linger_ms: 100             # 100ms accumulation
max_in_flight_requests: 10

# After (64MB memory)
buffer_memory: 67108864    # 64MB (4x reduction)
batch_size: 16384          # 16KB (4x smaller)
linger_ms: 10              # 10ms (reduced accumulation)
max_in_flight_requests: 5  # Reduce concurrent requests
```

**Calculation**:
```
Memory needed = throughput × linger_ms × avg_message_size
              = 20,000 msg/s × 10ms × 200 bytes
              = 20,000 × 0.01 × 200 = 40MB
Recommended = 40MB × 2.5 (safety) = 100MB
Configured = 64MB (acceptable)
```

**Impact**:
- Memory: 256MB → 64MB (75% reduction)
- Throughput: 100k msg/s → 20k msg/s (reduced, but acceptable)
- Latency: 50ms → 15ms (improvement!)

---

### Scenario 4: Reducing CPU Usage from 80% to 50%

**Problem**: Producer CPU too high, causing jitter in downstream processing.

**Root Cause Analysis**:
1. Check: `compression=snappy, batch_size=8KB, no idempotence disabled`
2. Snappy compression requires CPU; small batches = high overhead

**Solution**:
```yaml
# Before (80% CPU)
batch_size: 8192           # 8KB small batches
compression_type: snappy   # CPU-intensive
linger_ms: 0               # No batching accumulation
enable_idempotence: true   # Extra tracking overhead

# After (50% CPU)
batch_size: 16384          # 16KB (2x larger)
compression_type: lz4      # Faster than snappy
linger_ms: 10              # Allow 10ms batching
enable_idempotence: true   # Keep for exactly-once
```

**Impact**:
- CPU: 80% → 50% (improve headroom)
- Throughput: 50k msg/s → 75k msg/s (improvement)
- Latency: 5ms → 15ms (acceptable)
- Network: Slightly higher (lz4 compresses less than snappy)

---

### Scenario 5: Improving Reliability (Reduce Message Loss Risk)

**Problem**: Currently using `acks=0`, risk of message loss during broker failure.

**Root Cause Analysis**:
1. `acks=0` means fire-and-forget, no reliability
2. No idempotence means duplicates possible on retry

**Solution**:
```yaml
# Before (no reliability)
acks: 0                    # Fire and forget!
enable_idempotence: false  # No duplicate protection
retries: 3                 # Limited retries

# After (reliable)
acks: all                  # All replicas must ack
enable_idempotence: true   # Exactly-once semantics
retries: 2147483647        # Keep retrying
min_insync_replicas: 2     # (broker config) Require 2 replicas
```

**Impact**:
- Reliability: Message loss risk → Zero message loss (exactly-once)
- Latency: 5ms → 50-100ms (trade-off for reliability)
- Throughput: 100k msg/s → 10k msg/s (expected with all-replica ack)
- Network: Increased (more replica communication)

**When to use**: Financial data, risk systems, regulatory reporting

---

## Part 6: Troubleshooting Performance Issues

### Issue: Latency suddenly increases to 100ms+

**Diagnosis Steps**:
1. Check broker metrics:
   ```bash
   # SSH to broker
   jstat -gc -h10 <java_pid>  # Check GC activity
   ```
2. Check Prometheus metrics:
   - Spike in `produce_latency_seconds`?
   - Increase in `producer_queue_size`?
   - Spike in broker CPU?

3. Check logs for:
   - Broker GC pauses ("GC overhead limit exceeded")
   - Network errors ("Connection refused")
   - Timeout errors ("Request timed out")

**Possible Causes**:
- **Broker GC pause**: Broker paused for 50-100ms, all requests delayed
  - Solution: Increase `request.timeout.ms` to 60s (covers GC pauses)
- **Network congestion**: Latency spike during peak traffic
  - Solution: Enable compression to reduce bandwidth
- **Broker overload**: CPU near 100%, throughput hitting limit
  - Solution: Add more brokers or scale existing brokers

---

### Issue: Memory usage grows continuously

**Diagnosis Steps**:
1. Check queue depth:
   ```python
   print(f"Queue size: {producer._queue.qsize()}")  # Growing?
   ```
2. Check if broker is responsive:
   ```bash
   kafka-consumer-groups --bootstrap-server localhost:9092 --list
   ```
3. Check buffer memory setting:
   ```python
   producer.config['buffer.memory']  # How much allocated?
   ```

**Possible Causes**:
- **Broker unavailable**: Messages accumulate in buffer
  - Solution: Check broker health, restart if needed
  - Check: Broker logs for errors
- **Slow broker**: Broker can't keep up, buffer fills
  - Solution: Increase broker resources or add more brokers
  - Check: Broker CPU/memory usage
- **Buffer too large**: Allocation is wasteful
  - Solution: Reduce `buffer_memory` setting to match throughput needs

---

### Issue: High error rate (>1% of messages)

**Diagnosis Steps**:
1. Check error types:
   ```bash
   # Watch logs for error patterns
   tail -f producer.log | grep ERROR
   ```
2. Check specific errors:
   - Authentication errors: Check API keys
   - Broker unavailable: Check broker health
   - Topic errors: Check topic exists and permissions

**Possible Causes**:
- **Authentication failure**: API key expired or invalid
  - Solution: Rotate credentials, check broker auth config
- **Broker unavailable**: Kafka cluster down
  - Solution: Check broker status, failover if needed
- **Topic doesn't exist**: Topic creation failed
  - Solution: Manually create topic, check permissions

---

## Summary: Configuration Decision Matrix

Use this matrix to quickly select configuration for your use case:

| Requirement | Latency Critical | Throughput Critical | Balanced | Reliable |
|-------------|------------------|-------------------|----------|----------|
| **batch.size** | 8KB | 64KB | 16KB | 16KB |
| **linger.ms** | 0 | 100 | 10 | 50 |
| **compression** | none | snappy | snappy | snappy |
| **acks** | 1 | 1 | 1 | all |
| **enable_idempotence** | true | true | true | true |
| **buffer_memory** | 32MB | 512MB | 64MB | 128MB |
| **max.in.flight** | 5 | 10 | 5 | 5 |
| **request.timeout** | 30s | 30s | 30s | 60s |
| **Throughput** | 5k | 500k | 50k | 5k |
| **P99 Latency** | <5ms | 100-500ms | 10-50ms | 50-100ms |

---

## Additional Resources

- [Kafka Producer Tuning Best Practices](https://kafka.apache.org/documentation/#producerconfigs)
- [Confluent Schema Registry Setup](https://docs.confluent.io/kafka-connectors/schema-registry/)
- [Prometheus Metrics Monitoring](https://prometheus.io/docs/introduction/overview/)

---

**Document Version**: 1.0
**Last Updated**: November 12, 2025
**Audience**: Platform engineers, DevOps operators, SREs
**Status**: Production-Ready
