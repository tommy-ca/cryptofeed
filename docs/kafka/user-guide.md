# Kafka Backend User Guide

This guide provides practical examples and configuration patterns for using the Kafka backend in production environments.

## Installation

The Kafka backend requires additional dependencies:

```bash
pip install cryptofeed[kafka]
```

For Protocol Buffer support:

```bash
pip install cryptofeed[protobuf]
```

## Basic Setup

### 1. Start Kafka

Using Docker Compose (recommended for development):

```yaml
# docker-compose.yml
version: '3.8'
services:
  kafka:
    image: confluentinc/cp-kafka:7.4.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_INTERNAL:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092,PLAINTEXT_INTERNAL://kafka:29092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1

  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
```

```bash
docker-compose up -d
```

### 2. Basic Producer

```python
from cryptofeed import FeedHandler
from cryptofeed.backends.kafka import KafkaCallback
from cryptofeed.exchanges import Binance

# Simple configuration
callback = KafkaCallback(
    bootstrap_servers=['localhost:9092'],
    topic='crypto-trades'
)

fh = FeedHandler()
fh.add_feed(Binance(channels=['trades'], symbols=['BTC-USDT'], callbacks=[callback]))
fh.run()
```

## Configuration Patterns

### Development Configuration

```python
from cryptofeed.backends.kafka import KafkaCallback

dev_callback = KafkaCallback(
    bootstrap_servers=['localhost:9092'],
    topic='dev-market-data',
    # Development-friendly settings
    acks='1',  # Fire and forget for speed
    compression_type=None,  # No compression for debugging
    batch_size=1024,  # Smaller batches
)
```

### Production Configuration

```python
from cryptofeed.backends.kafka import KafkaProtobufCallback

prod_callback = KafkaProtobufCallback(
    bootstrap_servers=['kafka-1:9092', 'kafka-2:9092', 'kafka-3:9092'],
    topic={'strategy': 'consolidated', 'prefix': 'prod'},
    partition={'strategy': 'composite'},
    # Production-optimized settings
    acks='all',  # Wait for all replicas
    compression_type='gzip',  # Compress for efficiency
    retries=10,
    retry_backoff_ms=500,
    batch_size=32768,  # Larger batches for throughput
    linger_ms=10,  # Wait for batch completion
)
```

### Environment-Based Configuration

```python
import os
from cryptofeed.backends.kafka import KafkaCallback

callback = KafkaCallback(
    bootstrap_servers=os.getenv('KAFKA_SERVERS', 'localhost:9092').split(','),
    topic=os.getenv('KAFKA_TOPIC', 'market-data'),
    acks=os.getenv('KAFKA_ACKS', '1'),
    compression_type=os.getenv('KAFKA_COMPRESSION', 'gzip'),
)
```

## Live Binance → Kafka Protobuf E2E Tests (Opt-In)

- Start Redpanda locally: `make redpanda-up` (port defaults to 19092).
- Enable the live Binance tests: `export CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true`.
- Run the suite: `make test-kafka-binance` (skips if Docker/compose or Binance network access are unavailable).
- Stop Redpanda when finished: `make redpanda-down`.

The tests produce real Binance trades through `FeedHandler` → `KafkaProtobufCallback` into Redpanda and decode protobuf payloads to verify headers, schema version, and routing metadata. They are guarded to avoid CI flakiness and remain out of the default test run.

## Topic Strategies

### Consolidated Topics (Recommended)

All market data goes to a single topic. Best for:
- Simplified consumer architecture
- Better scalability
- Easier monitoring

```python
# All exchanges and symbols → one topic
callback = KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    topic={'strategy': 'consolidated'}
)
# Result: All data → "market-data" topic
```

### Per-Symbol Topics (Legacy)

Each trading symbol gets its own topic. Consider for:
- Legacy system compatibility
- Symbol-specific processing
- Smaller topic sizes

```python
# Each symbol → separate topic
callback = KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    topic={'strategy': 'per_symbol'}
)
# Result: BTC-USDT trades → "BTC-USDT" topic
```

## Partition Strategies

### Composite Partitioning (Recommended)

Partitions by exchange + symbol combination. Provides:
- Even load distribution
- Related data co-location
- Efficient consumer grouping

```python
callback = KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    partition={'strategy': 'composite'}
)
# Key format: "binance:BTC-USDT"
```

### Symbol-Based Partitioning

Partitions by symbol only. Useful when:
- Processing all exchanges for a symbol together
- Exchange-agnostic analytics

```python
callback = KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    partition={'strategy': 'symbol'}
)
# Key format: "BTC-USDT"
```

### Exchange-Based Partitioning

Partitions by exchange. Best for:
- Exchange-specific processing
- Geographic data separation

```python
callback = KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    partition={'strategy': 'exchange'}
)
# Key format: "binance"
```

## Consumer Examples

### Python Consumer (JSON)

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'market-data',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='crypto-consumer',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for message in consumer:
    data = message.value
    print(f"Received {data['type']} from {data['exchange']}: {data['symbol']}")
```

### Python Consumer (Protocol Buffer)

```python
from kafka import KafkaConsumer
from cryptofeed.backends.protobuf.helpers import deserialize_from_protobuf

consumer = KafkaConsumer(
    'market-data',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='crypto-consumer'
)

for message in consumer:
    headers = dict(message.headers)

    if headers.get('cf.serialization_format') == b'protobuf':
        # Deserialize protobuf message
        data_type = headers['cf.data_type'].decode()
        data = deserialize_from_protobuf(message.value, data_type)
        print(f"Received {data_type}: {data}")
```

### Flink Consumer

See [consumer templates](../consumer-templates/flink-consumer.py) for a complete Apache Flink integration.

## Migration Guide

### From Legacy Backend

**Before:**
```python
from cryptofeed.backends.kafka import TradeKafka
```

**After:**
```python
from cryptofeed.backends.kafka import KafkaCallback
```

### From JSON to Protocol Buffers

1. **Update imports:**
```python
# Old
from cryptofeed.backends.kafka import KafkaCallback

# New
from cryptofeed.backends.kafka import KafkaProtobufCallback
```

2. **Update callback instantiation:**
```python
# Old
callback = KafkaCallback(bootstrap_servers=['kafka:9092'])

# New
callback = KafkaProtobufCallback(bootstrap_servers=['kafka:9092'])
```

3. **Update consumer deserialization:**
```python
# Old - JSON
data = json.loads(message.value.decode())

# New - Protocol Buffer
from cryptofeed.backends.protobuf.helpers import deserialize_from_protobuf
data = deserialize_from_protobuf(message.value, data_type)
```

## Troubleshooting

### Connection Issues

**Problem:** `KafkaTimeoutError` or connection refused

**Solutions:**
- Verify Kafka is running: `docker ps | grep kafka`
- Check bootstrap servers configuration
- Ensure network connectivity between producer and Kafka
- Check Kafka logs: `docker logs <kafka-container>`

### Serialization Errors

**Problem:** `ProtobufEncodeError` exceptions

**Solutions:**
- Verify message format matches expected schema
- Check for missing required fields
- Validate enum values are within allowed range
- Review error messages for specific field issues

### Performance Issues

**Problem:** Low throughput or high latency

**Solutions:**
- Increase `batch_size` and `linger_ms` for higher throughput
- Use `compression_type='gzip'` to reduce network overhead
- Consider topic strategy: `consolidated` scales better than `per_symbol`
- Monitor Kafka cluster metrics and scale as needed

### Consumer Lag

**Problem:** Consumers falling behind

**Solutions:**
- Increase consumer instances
- Adjust partition strategy for better load distribution
- Monitor consumer group lag metrics
- Consider increasing `batch_size` on producer side

## Best Practices

### Production Deployment

1. **Use Protocol Buffers** for better performance and type safety
2. **Choose consolidated topics** for scalability
3. **Use composite partitioning** for even load distribution
4. **Configure appropriate acks level** based on durability needs
5. **Enable compression** to reduce network overhead
6. **Monitor performance metrics** and adjust batch settings

### Error Handling

```python
from cryptofeed import FeedHandler
from cryptofeed.backends.kafka import KafkaCallback
from cryptofeed.exceptions import KafkaError

try:
    callback = KafkaCallback(bootstrap_servers=['kafka:9092'])
    fh = FeedHandler()
    fh.add_feed(Binance(callbacks=[callback]))
    fh.run()
except KafkaError as e:
    print(f"Kafka connection failed: {e}")
    # Implement retry logic or fallback
```

### Monitoring

The Kafka backend exposes metrics that can be collected by monitoring systems:

```python
# Access metrics from callback
metrics = callback.metrics
print(f"Messages sent: {metrics.messages_sent}")
print(f"Serialization errors: {metrics.serialization_errors}")
```

See the [monitoring documentation](../monitoring/) for integration with Prometheus and Grafana.</content>
<parameter name="filePath">docs/kafka/user-guide.md
