# Kafka Backend

The Kafka backend provides high-performance, production-ready integration with Apache Kafka for streaming market data. It supports both JSON and Protocol Buffer serialization with configurable topic and partitioning strategies.

## Features

- **Dual Serialization**: JSON (legacy) and Protocol Buffer (recommended) support
- **Flexible Topic Strategies**: Consolidated topics (recommended) or per-symbol topics
- **Advanced Partitioning**: Composite, symbol-based, exchange-based, or round-robin partitioning
- **Schema Validation**: Built-in validation for Protocol Buffer messages
- **Header Enrichment**: Automatic metadata headers for routing and debugging
- **Backward Compatibility**: Legacy API support with deprecation warnings
- **Maintenance Toolkit**: Config translator/validator, health checks, and migration CLI (see `migration-guide-phase2-maintenance.md`)

## Quick Start

### Basic Usage

```python
from cryptofeed import FeedHandler
from cryptofeed.backends.kafka import KafkaCallback
from cryptofeed.exchanges import Binance

# Create Kafka callback with default settings
kafka_callback = KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    topic='market-data'
)

# Use with any exchange
fh = FeedHandler()
fh.add_feed(Binance(channels=['trades'], symbols=['BTC-USDT'], callbacks=[kafka_callback]))
fh.run()
```

### Protocol Buffer Usage (Recommended)

```python
from cryptofeed.backends.kafka import KafkaProtobufCallback

# Use Protocol Buffer serialization for better performance
protobuf_callback = KafkaProtobufCallback(
    bootstrap_servers=['kafka:9092'],
    topic='market-data'
)

fh = FeedHandler()
fh.add_feed(Binance(channels=['trades'], symbols=['BTC-USDT'], callbacks=[protobuf_callback]))
fh.run()
```

## Configuration

### Topic Strategies

```python
# Consolidated topic (recommended for production)
KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    topic={'strategy': 'consolidated'}  # All data goes to one topic
)

# Per-symbol topics (legacy compatibility)
KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    topic={'strategy': 'per_symbol'}  # BTC-USDT -> topic 'BTC-USDT'
)
```

### Partition Strategies

```python
# Composite partitioning (recommended)
KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    partition={'strategy': 'composite'}  # exchange + symbol
)

# Symbol-based partitioning
KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    partition={'strategy': 'symbol'}  # symbol only
)

# Exchange-based partitioning
KafkaCallback(
    bootstrap_servers=['kafka:9092'],
    partition={'strategy': 'exchange'}  # exchange only
)
```

### Advanced Configuration

```python
KafkaCallback(
    bootstrap_servers=['kafka:9092', 'kafka:9093'],
    topic={'strategy': 'consolidated', 'prefix': 'prod'},
    partition={'strategy': 'composite'},
    acks='all',  # Wait for all replicas
    compression_type='gzip',
    batch_size=16384,
    linger_ms=5,
    retries=3
)
```

## Message Format

### JSON Format (Legacy)

```json
{
  "type": "trade",
  "exchange": "binance",
  "symbol": "BTC-USDT",
  "timestamp": 1640995200.123,
  "data": {
    "price": "50000.00",
    "amount": "0.001",
    "side": "buy"
  }
}
```

### Protocol Buffer Format (Recommended)

Messages are serialized using Protocol Buffers for better performance and type safety. See the [schema documentation](../schemas/) for details.

## Headers

All messages include metadata headers:

- `cf.exchange`: Exchange name
- `cf.symbol`: Trading symbol
- `cf.data_type`: Message type (trade, ticker, etc.)
- `cf.serialization_format`: json or protobuf
- `cf.schema_version`: Schema version for protobuf messages
- `cf.timestamp`: Unix timestamp

## Consumer Integration

### Python Consumer

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'market-data',
    bootstrap_servers=['kafka:9092'],
    auto_offset_reset='latest'
)

for message in consumer:
    headers = dict(message.headers)
    data_type = headers.get('cf.data_type', b'').decode()

    if headers.get('cf.serialization_format') == b'json':
        payload = json.loads(message.value.decode())
    else:
        # Handle protobuf deserialization
        payload = deserialize_protobuf(message.value, data_type)

    print(f"Received {data_type}: {payload}")
```

### Flink Consumer

See [consumer templates](../consumer-templates/flink-consumer.py) for a complete Flink integration example.

## Migration Guide

### From Legacy Backend

```python
# Old (deprecated)
from cryptofeed.backends.kafka import TradeKafka

# New (recommended)
from cryptofeed.backends.kafka import KafkaCallback
```

### From JSON to Protocol Buffers

1. Change callback class: `KafkaCallback` → `KafkaProtobufCallback`
2. Update consumer deserialization logic
3. Verify schema compatibility

See the [migration guide](migration-guide.md) for detailed instructions.

## Performance Tuning

- **Topic Strategy**: Use `consolidated` for better scalability
- **Partition Strategy**: Use `composite` for even load distribution
- **Batch Settings**: Increase `batch_size` and `linger_ms` for higher throughput
- **Compression**: Enable `gzip` compression for reduced network usage

See [producer tuning](producer-tuning.md) for detailed performance guidance.

## Monitoring

The Kafka backend includes built-in metrics and monitoring capabilities. See the [monitoring documentation](../monitoring/) for details.

## Troubleshooting

Common issues and solutions are documented in the [troubleshooting guide](troubleshooting.md).</content>
<parameter name="filePath">docs/kafka/README.md
