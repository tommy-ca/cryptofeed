# Kafka Producer Troubleshooting Runbook

## Executive Summary

This runbook provides step-by-step procedures for diagnosing and resolving common Kafka producer issues. It covers incident response, diagnostic techniques, alert interpretation, health verification, and escalation procedures.

**Audience**: Operations engineers, on-call SREs, platform support teams.

**Purpose**: Enable fast resolution of Kafka producer incidents with minimal guidance.

**Quick Links**:
- [Common Issues & Solutions](#common-issues) (jump here first)
- [Diagnostic Procedures](#diagnostic-procedures)
- [Alert Decision Tree](#alert-decision-tree)
- [Health Check Verification](#health-check-verification)
- [Escalation Procedures](#escalation-procedures)

---

## Quick Reference: Common Issues at a Glance

| Symptom | Root Cause | Solution | Time to Resolution |
|---------|-----------|----------|-------------------|
| Messages not appearing in Kafka | Broker unavailable | Restart broker or failover | 5-10 min |
| Message loss (count mismatch) | acks setting too low | Check/change acks config | 2-5 min |
| P99 latency >15ms | CPU throttle/batching | Increase batch size or linger | 5-15 min |
| Memory usage growing | Buffer leak or slow broker | Check broker health | 10-30 min |
| Error rate >1% | Schema/auth/connectivity | Check logs, credentials, connectivity | 5-20 min |
| DLQ messages piling up | Message format invalid | Investigate DLQ sample | 15-60 min |
| Circuit breaker OPEN | Broker unreachable | Restore broker connectivity | 5-15 min |

---

## Part 1: Common Issues & Root Cause Analysis

### Issue 1: Broker Unavailable

**Symptoms**:
- Producer connects but no messages appear in Kafka topics
- Logs show "Connection refused" or "Could not connect"
- Producer queue size grows (backlog accumulating)
- Circuit breaker state changes to OPEN

**Root Causes**:
1. **Kafka broker process down**: Broker crashed or stopped
2. **Network connectivity broken**: Firewall rule, DNS, or network issue
3. **Port not listening**: Broker config error, port changed
4. **Broker overload**: CPU/memory maxed out, stop responding to requests

**Expected Behavior During Broker Outage**:
```
Timeline of events:
T=0s:     Producer sends message → No response from broker
T=30s:    Request times out, producer retries (exponential backoff)
T=40s:    Second attempt fails → Circuit breaker opens (protection)
T=45s:    Logs show: "Circuit breaker OPEN: Producer unavailable"
T=50s+:   Messages queued in buffer, waiting for broker recovery
T=60s+:   If broker recovers: Messages start sending, DLQ empty
```

**Impact**:
- Messages are NOT lost (buffered in producer)
- Downstream consumers may fall behind (lag grows)
- Alerting triggers: "Producer queue >10K messages"

---

### Issue 2: Message Loss

**Symptoms**:
- Consumer receives fewer messages than producer sent
- Count mismatch after broker recovery
- No errors in producer logs
- Broker shows healthy

**Root Causes**:
1. **acks=0 setting**: Producer doesn't wait for broker confirmation
   - Message sent but broker fails before persistence = loss
2. **Idempotence disabled**: Duplicates on retry → manual deduplication deletes messages
3. **Broker replication issue**: Leader fails before replica replicates
   - Causes: replication_factor=1, min.insync.replicas too low
4. **Consumer lag**: Consumer hasn't read all messages yet (not actual loss)

**Diagnosis**:
```bash
# Check producer config
grep "acks:" config.yaml
# Expected: acks=1 or acks=all (NOT acks=0)

# Check replication factor
kafka-topics --bootstrap-server localhost:9092 \
  --describe --topic cryptofeed.trades
# Expected: replication factor ≥3, in-sync replicas ≥2

# Check message count
# Produce 1000 test messages
# Consume and count - should match exactly
```

**Recovery**:
```yaml
# Before (unsafe)
acks: 0                      # DANGEROUS!
enable_idempotence: false

# After (safe)
acks: 1                      # or 'all' for critical data
enable_idempotence: true     # Exactly-once semantics
retries: 2147483647          # Keep retrying on transient failure
```

---

### Issue 3: Latency Spikes

**Symptoms**:
- P99 latency suddenly increases from 10ms to 50-100ms
- Spike visible in Prometheus `produce_latency_seconds` histogram
- Temporary (resolves after few minutes)
- Correlates with broker metrics spike

**Root Causes**:
1. **Broker GC pause**: Java garbage collection pausing broker
   - Duration: 20-500ms (blocks all processing)
   - Visible: Check broker logs for "GC pause" messages
2. **CPU throttle**: Producer/broker CPU capped by kernel/container limits
3. **Network congestion**: Latency to Kafka brokers temporarily high
4. **Broker disk I/O**: Slow disk writes to WAL
5. **Broker rebalancing**: Partitions reassigning (metadata update overhead)

**Diagnosis Steps**:

**Step 1: Check broker GC activity**
```bash
# SSH to broker
jstat -gc -h10 <java_pid>

# Output shows GC pause times
# Look for spike in "GC pause" column
# If >100ms pause coincides with producer latency spike → confirmed

# Alternative: Check broker logs
tail -f /var/log/kafka/server.log | grep "GC"
```

**Step 2: Check broker CPU**
```bash
# SSH to broker
top -p <broker_pid>
# Look at CPU% column
# If near 100% → CPU throttled (overloaded broker)

# Check kafka broker metrics
# Prometheus query: kafka_server_brokertopicmetrics_messagesin_total
# Should be steady, not spiking
```

**Step 3: Check network latency**
```bash
# From producer machine to broker
ping -c 10 <broker_host>
# Look for latency spikes in output
# Normal: 1-5ms, Spike: >20ms indicates network issue
```

**Step 4: Check producer queue buildup**
```python
# In producer monitoring
queue_size = producer._queue.qsize()
if queue_size > 1000:
    print("WARN: Producer queue building up - broker lagging")
    # Indicates broker can't keep up with message rate
```

**Expected Behavior**:
```
Broker GC pause scenario:
T=0ms:    Producer sends message
T=5ms:    Broker receives message
T=50ms:   Broker pauses for GC (everything blocked)
T=100ms:  GC finishes, broker resumes processing
T=110ms:  Broker sends ACK
T=115ms:  Producer receives ACK

Total latency: 115ms (spike from normal 10ms)
Duration: Usually 100-500ms, then back to normal
Frequency: Typically every 10-60 seconds (GC pauses)
```

**Solution**:
```yaml
# Increase request timeout to cover GC pauses
request_timeout_ms: 60000    # Increase from 30s to 60s
                             # This covers typical 50-100ms GC pauses

# Or optimize producer to reduce GC pressure
batch_size: 32768            # Larger batches = fewer objects
linger_ms: 50                # Wait for batching
compression_type: snappy     # Reduce message size
```

---

### Issue 4: Memory Growth

**Symptoms**:
- Memory usage grows over time (not stable)
- RSS memory increasing by 10-100MB per hour
- Eventually hits OOM (out of memory)
- `buffer_memory_usage` gauge steadily increasing

**Root Causes**:
1. **Broker unavailable**: Messages accumulate in producer buffer
   - Expected: Buffer fills up to `buffer_memory` limit
   - Problem: If buffer.memory is too large (>512MB)
2. **Slow broker**: Broker can process 1k msg/s but producer sending 10k msg/s
   - Results: Queue grows faster than it drains
3. **Buffer leak**: Bug in code causing memory not to be released
4. **Large messages**: Average message size larger than expected
   - Memory = buffer size / avg message size
5. **Consumer not reading**: Kafka broker accumulating messages on disk

**Diagnosis Steps**:

**Step 1: Check producer queue size**
```python
# Monitor producer queue
queue_size = producer._queue.qsize()
max_queue = producer.config['buffer_memory'] / avg_message_size

if queue_size > max_queue * 0.8:
    print(f"ALERT: Queue at {queue_size/max_queue*100}% capacity")
    # Either broker slow or buffer too small
```

**Step 2: Check broker health**
```bash
# Check broker CPU/memory
ssh <broker_host>
top -p <broker_pid>
# Is broker CPU>80%? Memory high?

# Check broker logs for errors
tail -100 /var/log/kafka/server.log | grep -i error

# Check broker disk space
df -h /var/kafka/data
# Need >10% free space for healthy operation
```

**Step 3: Check message rate vs broker capacity**
```bash
# Producer throughput
# From Prometheus: rate(cryptofeed_kafka_messages_sent_total[1m])
# Expected: X msg/s

# Broker throughput
# From Prometheus: rate(kafka_server_replica_fetcher_client_bytes_in_total[1m])
# Compare: Producer sending faster than broker processing?

# If producer > broker: Queue will grow!
```

**Step 4: Check for buffer leak**
```python
# Monitor over time
import time
import psutil

pid = os.getpid()
for i in range(10):
    process = psutil.Process(pid)
    print(f"Memory: {process.memory_info().rss / 1024 / 1024:.0f}MB")
    time.sleep(10)

# Expected: Stable or slight growth
# Leak: Continuous growth without plateau
```

**Solution by Root Cause**:

**If broker slow/unavailable**:
```bash
# Restart broker or wait for recovery
systemctl restart kafka  # Or failover to replica

# Monitor recovery
watch -n 1 'kafka-consumer-groups --bootstrap-server localhost:9092 --group cryptofeed --describe'
# Lag should decrease as broker processes backlog
```

**If buffer too large**:
```yaml
# Reduce buffer.memory setting
buffer_memory: 67108864      # Reduce from 256MB to 64MB
# Recalculate: buffer = throughput × linger_ms × avg_message_size
#            = 10k msg/s × 50ms × 200 bytes = 100MB max
# So 64MB should be sufficient
```

**If large messages**:
```python
# Check average message size
message_size = len(json.dumps(message).encode('utf-8'))
print(f"Message size: {message_size} bytes")

# If >500 bytes, consider:
# 1. Compression (reduces size before buffering)
# 2. Selective fields (remove non-critical data)
# 3. Protobuf (more efficient than JSON)
```

---

### Issue 5: High Error Rate (>1%)

**Symptoms**:
- `cryptofeed_kafka_errors_total` metric increasing rapidly
- Logs showing ERROR level messages
- Consumer receiving errors
- Some messages may not be processed

**Root Causes**:
1. **Authentication failure**: API credentials invalid or expired
2. **Schema validation failure**: Message format doesn't match schema
3. **Network connectivity**: Transient network errors
4. **Broker unavailable**: All requests failing
5. **Disk full on broker**: Cannot write messages
6. **Malformed message**: Data validation error in producer

**Diagnosis Steps**:

**Step 1: Identify error type**
```bash
# Check recent error logs
tail -100 /var/log/producer.log | grep ERROR

# Examples:
# ERROR: Authentication failed (Invalid API key)
# ERROR: InvalidSchemaException (Missing required field)
# ERROR: ConnectionError (Network unreachable)
# ERROR: BrokerNotAvailable (Leader not available)
```

**Step 2: Check by error type**

**Authentication Error**:
```bash
# Check API credentials
grep -i "api_key\|password" config.yaml
# Verify credentials are not expired

# Rotate credentials if needed
# Restart producer with new credentials
```

**Schema Validation Error**:
```python
# Check what field is missing/invalid
# From error: "Missing required field: 'price'"

# Verify message structure matches schema
from cryptofeed.types import Trade
message = {
    'exchange': 'binance',
    'symbol': 'BTC-USD',
    'price': 50000,  # Required field
    'quantity': 1.0,
    # ... all required fields present?
}
```

**Connectivity Error**:
```bash
# Check network connectivity to brokers
nc -zv <broker_host> 9092  # Should succeed

# Check DNS resolution
nslookup <broker_host>  # Should resolve to IP

# Check firewall
iptables -L -n | grep 9092  # Should allow port 9092
```

**Broker Unavailable**:
```bash
# Check broker status
kafka-broker-api-versions.sh --bootstrap-server localhost:9092
# Should show broker API versions, not error

# If error: Broker is down or unreachable
# Restart or failover to replica
```

---

### Issue 6: DLQ Messages Piling Up

**Symptoms**:
- Messages appearing in Dead Letter Queue (DLQ) topic
- Number of DLQ messages increasing
- These messages not being processed
- Alerts: "DLQ queue size >1000"

**Root Causes**:
1. **Invalid message format**: Message doesn't deserialize
2. **Schema incompatibility**: Message schema doesn't match expected
3. **Serialization error**: to_proto() method failing
4. **Broker-level rejection**: Message too large or other violation

**Diagnosis Steps**:

**Step 1: Check DLQ topic for sample message**
```bash
# Read one message from DLQ
kafka-console-consumer --bootstrap-server localhost:9092 \
  --topic cryptofeed.trades.dlq \
  --max-messages 1 \
  --from-beginning

# Output: One DLQ message (usually includes original error)
```

**Step 2: Understand the message format**
```python
# Try to deserialize the DLQ message
from cryptofeed.types import Trade
import json

dlq_msg = {...}  # The DLQ message

# Try to parse
try:
    trade = Trade.from_dict(dlq_msg)
except Exception as e:
    print(f"Deserialization failed: {e}")
    print(f"Message: {dlq_msg}")
```

**Step 3: Identify the issue**

**If field is missing**:
```python
# Add default value or handle absence
message = {
    'exchange': 'binance',
    'symbol': 'BTC-USD',
    # 'price': None,  # Missing!
}

# Solution: Provide default or skip message
message['price'] = 0.0 if 'price' not in message else message['price']
```

**If format is wrong**:
```python
# Example: timestamp should be float, got string
message = {
    'timestamp': '2025-11-12T10:30:45Z',  # Should be float!
}

# Solution: Convert timestamp to float
import datetime
ts_str = message['timestamp']
ts_float = datetime.datetime.fromisoformat(ts_str).timestamp()
```

---

### Issue 7: Circuit Breaker OPEN

**Symptoms**:
- Logs show: "Circuit breaker OPEN"
- Producer stops sending messages (protective state)
- Queue accumulates messages
- Status endpoint shows: `circuit_breaker_state: OPEN`

**Root Cause**:
- Circuit breaker enters OPEN state when error rate exceeds threshold
- Typically: >5% error rate for >30 seconds

**Expected Circuit Breaker Behavior**:
```
States:
CLOSED       → Normal operation, requests flowing
             → Error rate < threshold: Stay CLOSED

OPEN         → Error rate > threshold
             → Stop sending requests (circuit is open)
             → Wait 60 seconds before trying again

HALF_OPEN    → After 60 second wait
             → Try sending test request
             → If succeeds: Return to CLOSED
             → If fails: Return to OPEN (wait another 60s)
```

**Recovery Steps**:

**Step 1: Identify why circuit opened**
```python
# Check error logs
producer.get_last_errors()  # Get recent errors

# Check error rate
# Prometheus: rate(cryptofeed_kafka_errors_total[5m])
# If >5% → explains OPEN state
```

**Step 2: Fix underlying issue**
```bash
# Common causes (from previous sections):
# 1. Broker down → Restart broker
# 2. Auth failure → Fix credentials
# 3. Schema error → Fix message format
# 4. Network issue → Check connectivity
```

**Step 3: Wait for automatic recovery**
```bash
# Circuit breaker auto-recovers after 60s
# Once underlying issue fixed, wait for recovery

# Manual recovery (if needed):
producer.reset_circuit_breaker()  # Force transition to CLOSED
```

**Validation**:
```bash
# After recovery, verify:
# 1. Status endpoint shows CLOSED
curl http://producer:8080/health
# Response should include: "circuit_breaker_state": "CLOSED"

# 2. Messages flowing again
# Check Prometheus: cryptofeed_kafka_messages_sent_total should increase

# 3. Error rate returned to normal (<0.1%)
# Prometheus: rate(cryptofeed_kafka_errors_total[5m]) < 0.001
```

---

## Part 2: Diagnostic Procedures

### Procedure 1: Check Producer Connectivity

Use this procedure when suspecting connectivity issues.

**Step 1: Verify Kafka broker is reachable**
```bash
# Test network connectivity to broker
nc -zv kafka1.internal 9092
# Expected output: "Connection to kafka1.internal 9092 port [tcp/*] succeeded!"
# Failed output: "Failed to connect to kafka1.internal port 9092"

# Alternative with telnet
telnet kafka1.internal 9092
# Should connect and show Kafka header (^A^...)
```

**Step 2: Verify Kafka broker responds to requests**
```bash
# Use Kafka tools to verify broker
kafka-broker-api-versions.sh --bootstrap-server kafka1.internal:9092
# Expected: Shows ApiVersion API responses
# Failed: Connection refused or timeout

# Alternative: Check cluster metadata
kafka-metadata.sh --bootstrap-server kafka1.internal:9092 --snapshot /tmp/metadata.bin
```

**Step 3: Verify producer configuration**
```python
# Check bootstrap servers
config = load_config('config.yaml')
print(f"Bootstrap servers: {config.bootstrap_servers}")
# Expected: List of 3+ broker addresses

# Verify each broker is in DNS
import socket
for broker in config.bootstrap_servers:
    try:
        ip = socket.gethostbyname(broker.split(':')[0])
        print(f"{broker} → {ip} (OK)")
    except socket.gaierror as e:
        print(f"{broker} → ERROR: {e}")
```

**Step 4: Check producer health endpoint**
```bash
# Query producer health
curl http://producer-host:8080/health/ready

# Expected response:
{
  "status": "ready",
  "kafka": "connected",
  "circuit_breaker": "closed",
  "queue_depth": 15,
  "messages_sent": 1000000
}

# Failed response would show errors in kafka/circuit_breaker fields
```

---

### Procedure 2: Monitor Metrics

Use Prometheus to understand producer behavior.

**Key Metrics to Check**:
```promql
# 1. Messages sent per second
rate(cryptofeed_kafka_messages_sent_total[1m])
# Expected: Should match expected throughput (e.g., 10,000 msg/s)
# If dropping: Indicates producer slowing down

# 2. Latency percentiles
histogram_quantile(0.99, cryptofeed_kafka_produce_latency_seconds_bucket)
# Expected: <50ms for p99 (depends on config)
# If spiking: Indicates broker latency or congestion

# 3. Error rate
rate(cryptofeed_kafka_errors_total[1m])
# Expected: <0.1% (1 error per 1000 messages)
# If >1%: Production issue needing investigation

# 4. Producer queue size
cryptofeed_kafka_producer_queue_size
# Expected: Stable, <10% of buffer_memory
# If growing: Broker can't keep up

# 5. Buffer memory usage
cryptofeed_kafka_buffer_memory_usage
# Expected: Oscillates between 10-80% (not stuck at ceiling)
# If stuck at 100%: Messages accumulating, broker slow
```

**Create Dashboard Queries**:
```yaml
# Prometheus scrape config
global:
  scrape_interval: 10s

scrape_configs:
  - job_name: 'cryptofeed-producer'
    static_configs:
      - targets: ['localhost:8080']  # Producer metrics port
```

---

### Procedure 3: Check Logs

Use structured logging to diagnose issues.

**Log Pattern Interpretation**:

**ERROR Logs (Critical)**:
```
ERROR: Circuit breaker OPEN: Producer unavailable
  → Broker unreachable or error rate too high
  → Action: Check broker health, fix errors

ERROR: Authentication failed: Invalid API key
  → Credentials invalid or expired
  → Action: Rotate credentials, restart producer

ERROR: Schema validation failed: Missing required field 'price'
  → Message doesn't match expected schema
  → Action: Fix message producer code
```

**WARNING Logs (Needs Investigation)**:
```
WARN: Producer queue approaching capacity (95%)
  → Messages accumulating faster than broker can process
  → Action: Check broker throughput, may need scaling

WARN: Broker connection lost, retrying...
  → Transient connection issue
  → Action: Monitor, may recover automatically

WARN: Request timeout waiting for broker
  → Broker slow or network latency high
  → Action: Check broker metrics, network latency
```

**INFO Logs (Informational)**:
```
INFO: Message produced successfully: topic=cryptofeed.trades, partition=3
  → Normal operation
  → No action needed

INFO: Schema registry updated with version 2
  → Schema version changed
  → May need consumer update
```

**Search for specific issues**:
```bash
# Find all errors in last hour
grep "ERROR\|FATAL" producer.log | tail -100

# Find circuit breaker events
grep "circuit breaker" producer.log | grep -i open

# Find timeout errors
grep -i "timeout\|timed out" producer.log

# Find authentication errors
grep -i "auth\|credentials\|api key" producer.log

# Count errors by type
grep "ERROR:" producer.log | awk -F': ' '{print $2}' | sort | uniq -c
```

---

### Procedure 4: Validate Configuration

Use the validation CLI tool to check configuration.

```bash
# Validate configuration file
python -m cryptofeed.tools.kafka_config_validator config.yaml

# Output on success:
# ✓ Configuration syntax valid
# ✓ All required fields present
# ✓ Field values in valid ranges
# ✓ Connectivity test passed

# Output on failure:
# ✗ Invalid value for 'acks': 'invalid'
#   Expected: 0, 1, or 'all'
# ✗ Invalid value for 'batch_size': 0
#   Expected: > 0
```

**Dry-run mode** (test without making changes):
```bash
# Test configuration without deploying
python -m cryptofeed.tools.kafka_config_validator \
  --config config.yaml \
  --dry-run \
  --test-connectivity

# Tests configuration but doesn't deploy
# Verifies producer can connect to brokers
```

---

### Procedure 5: Test Connectivity with Kafka CLI Tools

Use command-line tools to verify topic and broker health.

```bash
# List all topics
kafka-topics.sh --bootstrap-server kafka1:9092 --list

# Check specific topic
kafka-topics.sh --bootstrap-server kafka1:9092 \
  --describe --topic cryptofeed.trades

# Output shows:
# Topic: cryptofeed.trades
# Partitions: 3
# Replication Factor: 3
# In-Sync Replicas: 3 (all healthy)

# Produce test message
echo '{"exchange":"test","symbol":"btc"}' | \
  kafka-console-producer --broker-list kafka1:9092 \
  --topic test.topic

# Consume from topic
kafka-console-consumer --bootstrap-server kafka1:9092 \
  --topic cryptofeed.trades \
  --from-beginning \
  --max-messages 10
```

---

## Part 3: Alert Response Decision Tree

Use this decision tree when alerts fire to diagnose and respond appropriately.

### Alert: Producer Error Rate >1%

```
ALERT FIRES: error_rate(5m) > 0.01
                    |
                    v
    [STEP 1: Check error type]
                    |
    __________________+_______________________
   |                  |                       |
   v                  v                       v
Auth Error      Schema Error            Network Error
   |                  |                       |
   v                  v                       v
Check API keys   Review message       Check broker
Rotate if old    structure in logs    connectivity
Restart          Fix producer code    Verify DNS
               Test deserialization  Check firewall
                    |                       |
                    v                       v
          [STEP 2: Apply fix]      [STEP 2: Wait/retry]
                    |                       |
          Restart producer         Monitor for recovery
          Monitor error rate       Manual intervention
               (should drop)       if persists >5min
                    |                       |
                    v                       v
          Error rate <0.1%?    Connectivity restored?
            ✓ OK to continue       ✓ Messages flowing?
            ✗ Investigate more     ✗ ESCALATE
```

**Decision Point 1: What type of error?**
- Check logs: `tail -100 producer.log | grep ERROR`
- Common error messages:
  - "Invalid credentials" → Auth problem
  - "Invalid schema" → Message format problem
  - "Connection refused" → Network problem
  - "Broker unavailable" → Broker down

**Decision Point 2: Is it transient or persistent?**
- Wait 1-2 minutes
- Check if error rate decreases naturally
- If persistent: Requires manual intervention
- If transient: May auto-recover

**Decision Point 3: Has the issue been resolved?**
- Check error rate trend: `rate(errors_total[5m])`
- Should decrease below 0.1% after fix
- If not decreasing: Deeper investigation needed

---

### Alert: P99 Latency >15ms

```
ALERT FIRES: p99_latency > 15ms
                    |
                    v
    [STEP 1: Check baseline]
    Is this normal for your config?
                    |
        ____________+____________
       |                         |
       v                         v
  Yes, expected       No, latency spike
  (e.g., with           |
  linger_ms=50)        v
       |        [STEP 2: Check what changed?]
       |               |
       |         ______+______
       |        |             |
       v        v             v
    OK   Recent config    Broker metrics
    No action  change?      spiking?
                 |           |
            [STEP 3]    [STEP 3]
            Revert      Check CPU/Memory
            config      Check GC logs
                |           |
                v           v
            Latency    CPU maxed out?
            decreases?      |
             ✓ OK      ______+______
             ✗ More    |            |
               debug   v            v
              Yes     No
               |      |
         Scale up  Check network
         brokers   latency
```

**Decision Point 1: Is latency spike expected?**
- Check config: `linger_ms=50` → Expect 50ms+ latency
- Check recent changes: Any config deployed recently?
- If config is expected: No action needed
- If config changed: Consider reverting

**Decision Point 2: Check broker health**
```bash
# Get broker metrics
ssh <broker> top -p <pid>
# Is CPU near 100%? Memory near limit?

# Check GC activity
jstat -gc -h10 <broker_pid>
# Spike in GC pause time? (100ms+ pause = latency spike)
```

**Decision Point 3: Fix applied successfully?**
- Monitor latency: Should return to <10ms
- If still spiking: Escalate to broker team

---

### Alert: Producer Queue >10K Messages

```
ALERT FIRES: queue_size > 10000
                    |
                    v
    [STEP 1: Check if temporary]
    Is queue decreasing?
                    |
        ____________+____________
       |                         |
       v                         v
  Decreasing      Stable/Growing
  (transient)         |
       |              v
       |      [STEP 2: Check broker]
       |      Broker processing messages?
       |              |
       |         ____+____
       |        |        |
       v        v        v
    Wait  Healthy  Unhealthy
    Monitor Broker      |
            processing  v
             at rate  Broker down?
             OK        Slow?
                       |
                  [STEP 3]
                  Restart broker
                  or scale up
                       |
                       v
                  Queue drains?
                   ✓ OK
                   ✗ ESCALATE
```

**Decision Point 1: Is queue growing or stable?**
```python
# Monitor queue trend
queue_readings = []
for i in range(5):
    queue_readings.append(producer._queue.qsize())
    time.sleep(10)

is_growing = all(queue_readings[i] <= queue_readings[i+1] for i in range(4))
if is_growing:
    print("ALERT: Queue is growing!")
else:
    print("INFO: Queue is decreasing (transient)")
```

**Decision Point 2: Why is broker slow?**
```bash
# Check broker CPU
ssh <broker> top
# Is CPU >80%? Need to scale up

# Check broker disk
ssh <broker> iostat -x 1
# Is disk utilization >80%? Disk I/O bottleneck

# Check broker network
ssh <broker> iftop
# Bandwidth near saturation? Network bottleneck
```

**Decision Point 3: What's the fix?**
- High CPU: Scale broker vertically or add brokers
- High disk I/O: Optimize broker config or add faster disk
- High network: Distribute traffic or add brokers
- Broker down: Restart or failover

---

### Alert: Buffer Memory >80%

```
ALERT FIRES: buffer_memory_usage > 0.80
                    |
                    v
    [STEP 1: Why is buffer filling?]
                    |
        ____________+____________
       |                         |
       v                         v
  Broker slow    Buffer too small
       |              |
       v              v
  Check broker   Reduce batch_size
  throughput     Reduce linger_ms
       |         Reduce buffer_memory
       |              |
       v              v
  Can broker    Throughput adequate?
  handle load?  ✓ OK
   ✓ Broker    ✗ Increase throughput
    scaling      or buffer
   ✗ Overload
     scale up
```

**Decision Point 1: Is broker keeping up?**
```bash
# Measure broker throughput
# Prometheus: rate(kafka_server_messages_in_total[1m])
# vs producer throughput: rate(cryptofeed_kafka_messages_sent_total[1m])

# If producer > broker: Broker can't keep up
# Solution: Add brokers or scale vertical

# If broker > producer: Buffer too small for config
# Solution: Either reduce batch/linger or increase buffer
```

**Decision Point 2: How to fix?**

**Option A: Increase buffer** (if broker can handle it)
```yaml
buffer_memory: 536870912  # Increase from 64MB to 512MB
# Pro: Absorb more transient delays
# Con: More memory usage
```

**Option B: Reduce batching** (if latency is priority)
```yaml
batch_size: 8192          # Reduce from 32KB
linger_ms: 10             # Reduce from 50ms
# Pro: Lower latency, smaller batches → fill less buffer
# Con: Reduced throughput
```

**Option C: Scale brokers** (if buffer is already optimal)
```bash
# Add more brokers to cluster
# Distribute producer traffic across more brokers
# Improves throughput capacity
```

---

### Alert: Circuit Breaker OPEN

```
ALERT FIRES: circuit_breaker_state == OPEN
                    |
                    v
    [STEP 1: Wait 60 seconds for auto-recovery]
    [STEP 2: Check if issue resolved]
                    |
        ____________+____________
       |                         |
       v                         v
  Auto-recovered    Still OPEN
  Circuit CLOSED        |
       |                v
       |        [STEP 3: Fix underlying issue]
       |                |
       v         _______+________
    Done         |              |
                 v              v
           Check logs    Check broker
           What errors?  status
                |         |
                v         v
           Fix issues   Broker down?
                |         |
                v         v
             Wait    [Restart/Failover]
             60sec        |
             auto-        v
             recover    Circuit
                |      recovers
                v
           Circuit
           recovers?
            ✓ OK
            ✗ Manual
              reset
```

**Decision Point 1: Auto-recovery worked?**
```bash
# Check status after 60 seconds
curl http://producer:8080/health
# Should show: "circuit_breaker_state": "CLOSED"

# If still OPEN: Manual intervention needed
```

**Decision Point 2: What caused the OPEN?**
```bash
# Check logs for errors
grep "ERROR\|FATAL" producer.log | tail -50

# Common causes:
# 1. Broker unavailable → Restart broker
# 2. Auth failure → Fix credentials
# 3. Network issue → Check connectivity
```

**Decision Point 3: Manual recovery (if auto-recovery failed)**
```python
# Force circuit breaker reset
producer.reset_circuit_breaker()  # Transition to CLOSED

# Monitor for recovery
# Should start sending messages immediately
# Error rate should drop

# If errors persist: Escalate
```

---

## Part 4: Health Check Verification

Use these procedures after incident to verify recovery.

### Post-Incident Verification Checklist

After resolving any incident, verify that the system is healthy.

**Step 1: Verify Kafka Connectivity**
```bash
# [ ] Brokers are responding to requests
kafka-broker-api-versions.sh --bootstrap-server kafka1:9092

# [ ] Topics are readable
kafka-topics.sh --bootstrap-server kafka1:9092 --list

# [ ] Can produce test message
echo 'test' | kafka-console-producer --broker-list kafka1:9092 \
  --topic test.topic

# [ ] Can consume test message
timeout 5 kafka-console-consumer --bootstrap-server kafka1:9092 \
  --topic test.topic --max-messages 1
```

**Step 2: Verify Producer Health**
```bash
# [ ] Producer process running
ps aux | grep kafka_callback

# [ ] Health check endpoint responds
curl -s http://producer:8080/health | jq .

# [ ] Circuit breaker is CLOSED
curl -s http://producer:8080/health | jq '.circuit_breaker_state'
# Expected: "closed"

# [ ] No messages in DLQ
kafka-console-consumer --bootstrap-server kafka1:9092 \
  --topic cryptofeed.trades.dlq --max-messages 1 \
  --property print.value=false --property print.key=true
# Expected: No output (empty topic)
```

**Step 3: Verify Message Flow**
```bash
# [ ] Messages being produced
# Prometheus: rate(cryptofeed_kafka_messages_sent_total[5m]) > 0

# [ ] Latency returned to normal
# Prometheus: histogram_quantile(0.99, produce_latency_seconds) < 50ms

# [ ] Error rate normal
# Prometheus: rate(cryptofeed_kafka_errors_total[5m]) < 0.001

# [ ] Queue size stable
# Prometheus: cryptofeed_kafka_producer_queue_size < 1000
```

**Step 4: Verify Consumer Processing**
```bash
# [ ] Consumer groups not lagging
kafka-consumer-groups --bootstrap-server kafka1:9092 \
  --group all-consumers \
  --describe
# Lag should be decreasing, not increasing

# [ ] Messages reaching consumers (if applicable)
# Monitor consumer metrics: messages_consumed, lag, etc.

# [ ] No downstream errors reported
# Check downstream application logs for errors
```

**Step 5: Document Incident Resolution**
```yaml
# Create incident report
incident:
  id: INC-2025-1112-001
  title: "Kafka Producer - Broker Unavailability"
  duration: "10:30 - 11:00 UTC (30 minutes)"

  root_cause: "Broker crashed due to out-of-memory condition"

  resolution_steps:
    1. "Identified broker OOM in logs"
    2. "Restarted broker kafka1"
    3. "Verified circuit breaker recovered"
    4. "Confirmed message flow restored"

  prevention:
    - "Set up memory monitoring alert at 80% threshold"
    - "Configure broker heap size to 16GB (increased from 8GB)"
    - "Review message sizes, currently averaging 500 bytes"

  time_to_resolution: "15 minutes"
```

---

## Part 5: Escalation Procedures

Use these procedures when local troubleshooting doesn't resolve the issue.

### When to Escalate

**Escalate IMMEDIATELY (Critical Severity)**:
1. **Zero message throughput** for >10 minutes
   - Messages not appearing in Kafka
   - Producer queue growing indefinitely
   - Circuit breaker OPEN and auto-recovery failed

2. **Message loss confirmed** (messages sent but not in Kafka)
   - Count mismatch after broker recovery
   - Data quality alerts triggered
   - Business impact (trading positions inconsistent)

3. **All brokers down**
   - Cannot connect to any broker
   - Entire Kafka cluster unavailable
   - Affects all producers and consumers

**Escalate when troubleshooting exceeds 30 minutes**:
1. Applied known solutions but issue persists
2. Root cause not identified
3. Multiple systems affected
4. Business SLA at risk

### Escalation Contacts

**Level 1: Team Lead (On-Call)**
- **Time**: First 10 minutes of issue
- **Channel**: Slack #incidents or phone
- **Information to provide**:
  - Symptom (error rate 5%, latency spike, etc.)
  - When it started (T-5 minutes)
  - Recent changes (deployments, config changes)
  - Affected system (producer name, brokers affected)

**Level 2: Kafka Operations Team**
- **Time**: If not resolved in 30 minutes
- **Channel**: Escalation ticket in incident system
- **Information to provide**:
  - All troubleshooting steps already taken
  - Metrics and logs collected
  - Broker status (down, overloaded, etc.)
  - Impact (how many messages affected, users affected)

**Level 3: Infrastructure/Cloud Team**
- **Time**: If Kafka team suspects infrastructure issue
- **Escalation criteria**:
  - All brokers unreachable (network issue)
  - Broker host OOM or out of disk
  - Network connectivity broken
- **Information**: Broker host names, error messages

### Rollback Procedures

If producer deployment caused the issue:

**Step 1: Identify recent changes**
```bash
# Check deployment history
git log --oneline -10
# Identify commit hash of last deployment

# Check if this version is causing issue
# Compare performance metrics before/after deployment
```

**Step 2: Rollback producer deployment**
```bash
# Revert to previous version
git checkout <previous_commit>

# Rebuild and redeploy
make build
make deploy

# Verify recovery
# Circuit breaker should be CLOSED
# Error rate should drop
# Messages should flow
```

**Step 3: Root cause analysis**
```bash
# Review changes in deployment
git diff <previous_commit> HEAD

# Identify problematic code
# Was a configuration changed?
# Was a library upgraded?
# Was new code introduced?

# Fix root cause in dev
# Re-test thoroughly
# Deploy again
```

### Critical Incident Procedures

If incident is critical (message loss, extended outage):

**Immediate Actions** (First 5 minutes):
- [ ] Declare SEV1 incident
- [ ] Page on-call teams (platform, kafka, data)
- [ ] Create incident war room (Slack channel or Zoom)
- [ ] Assign incident commander
- [ ] Start tracking timeline

**Investigation** (Minutes 5-30):
- [ ] Gather metrics and logs
- [ ] Identify root cause
- [ ] Brief incident commander on findings
- [ ] Propose mitigation

**Mitigation** (Minutes 30-60):
- [ ] Execute recovery plan
- [ ] Monitor metrics for improvement
- [ ] Notify stakeholders of progress

**Recovery Verification** (Minutes 60-120):
- [ ] Verify message flow restored
- [ ] Check for message loss
- [ ] Confirm consumer processing normal
- [ ] Run health check procedures (Part 4)

**Post-Incident** (Day+):
- [ ] Root cause analysis document
- [ ] Prevention measures identified
- [ ] Fix tickets created for engineering
- [ ] Incident review meeting scheduled

---

## Summary: Quick Troubleshooting Checklist

Print this checklist and keep handy when responding to producer incidents:

```
PRODUCER INCIDENT RESPONSE CHECKLIST
===================================

[ ] DIAGNOSE (First 5 minutes)
  [ ] Check broker connectivity: nc -zv <broker>
  [ ] Check error rate: Prometheus error_rate query
  [ ] Check circuit breaker: curl /health
  [ ] Identify error type: tail -100 logs | grep ERROR

[ ] APPLY STANDARD FIX (5-15 minutes)
  [ ] Error rate >1%: Check logs, fix message/config
  [ ] Latency spike: Check broker metrics, config
  [ ] Queue growing: Check broker health
  [ ] Memory growing: Check broker throughput
  [ ] Circuit OPEN: Wait 60s for auto-recovery

[ ] VERIFY RECOVERY (15-25 minutes)
  [ ] Error rate back to normal (<0.1%)
  [ ] Circuit breaker CLOSED
  [ ] Messages flowing (throughput > 0)
  [ ] Queue stable (not growing)
  [ ] Health check passes

[ ] IF NOT RESOLVED (25+ minutes)
  [ ] Collect all logs and metrics
  [ ] Document troubleshooting steps
  [ ] ESCALATE to Kafka Operations team
  [ ] Provide incident summary

CRITICAL ISSUES (Escalate Immediately):
[ ] Zero throughput >10 minutes
[ ] Message loss confirmed
[ ] All brokers down
```

---

**Document Version**: 1.0
**Last Updated**: November 12, 2025
**Audience**: Operations engineers, on-call SREs, platform support
**Status**: Production-Ready
**Emergency Contact**: #incidents Slack channel
