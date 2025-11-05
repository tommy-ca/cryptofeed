# Protobuf Serialization User Guide

**Version**: 1.0.0  
**Date**: October 31, 2025  
**Status**: Production Ready

---

## Overview

Cryptofeed now supports **Protocol Buffers (protobuf) serialization** for all market data and account data types. This provides significant benefits for high-throughput, low-latency data pipelines.

### Benefits

| Benefit | Description |
|---------|-------------|
| **60% Size Reduction** | Protobuf messages are ~60% smaller than JSON |
| **1.8x Faster** | Binary encoding is faster than JSON text encoding |
| **Type Safety** | Strong typing with schema validation |
| **Backward Compatible** | JSON remains the default format |
| **Multi-Language** | Protobuf schemas work with Python, Go, Java, Rust, etc. |

---

## Quick Start

### Basic Usage

```python
from cryptofeed import FeedHandler
from cryptofeed.exchanges import Coinbase
from cryptofeed.defines import TRADES
from cryptofeed.backends.kafka import TradeKafka

# Create Kafka backend with protobuf serialization
fh = FeedHandler()
fh.add_feed(
    Coinbase(
        channels=[TRADES],
        symbols=['BTC-USD'],
        callbacks={
            TRADES: TradeKafka(
                bootstrap_servers='localhost:9092',
                topic='crypto.trades',
                serialization_format='protobuf'  # <-- Enable protobuf
            )
        }
    )
)

fh.run()
```

### Configuration via YAML

```yaml
# config.yaml
feeds:
  coinbase:
    symbols:
      - BTC-USD
      - ETH-USD
    channels:
      - trades
      - l2_book
    callbacks:
      trades:
        backend: kafka
        bootstrap_servers: localhost:9092
        topic: crypto.trades
        serialization_format: protobuf  # <-- Enable protobuf
      l2_book:
        backend: kafka
        bootstrap_servers: localhost:9092
        topic: crypto.orderbook
        serialization_format: protobuf
```

```python
from cryptofeed import FeedHandler

# Load configuration
fh = FeedHandler(config='config.yaml')
fh.run()
```

---

## Supported Data Types

All 14 Cryptofeed data types support protobuf serialization:

### Market Data Types (8)

| Type | Size (bytes) | Use Case |
|------|--------------|----------|
| **Trade** | 36-68 | Real-time executions |
| **Ticker** | 38 | BBO updates |
| **OrderBook** | 39+ | Level 2 order book |
| **Candle** | 83-125 | OHLCV bars |
| **Funding** | 53 | Perpetual funding rates |
| **Liquidation** | 61 | Forced liquidations |
| **OpenInterest** | 38 | Futures open interest |
| **Index** | 33 | Index prices |

### Account/Order Types (6)

| Type | Size (bytes) | Use Case |
|------|--------------|----------|
| **Balance** | 22 | Account balances |
| **Position** | 52 | Open positions |
| **Fill** | 76 | Trade executions |
| **OrderInfo** | 78 | Order status updates |
| **Order** | 57 | New orders |
| **Transaction** | 45 | Deposits/withdrawals |

---

## Configuration Options

### Serialization Formats

```python
# Protobuf (recommended for production)
TradeKafka(..., serialization_format='protobuf')

# JSON (default, backward compatible)
TradeKafka(..., serialization_format='json')
TradeKafka(...)  # Defaults to JSON
```

### Backend Support

| Backend | Protobuf Support | Notes |
|---------|------------------|-------|
| **Kafka** | ✅ Yes | Recommended for production |
| **Redis** | ✅ Yes | Pub/Sub and streams |
| **PostgreSQL** | ⏳ Planned | JSON for now |
| **InfluxDB** | ⏳ Planned | JSON for now |
| **File** | ✅ Yes | Binary .pb files |

Redis writers store protobuf payloads as base64-encoded bytes alongside
content-type and metadata fields to preserve backward compatibility with
existing stream/ZSET consumers. ZMQ publishers emit multipart messages with a
JSON header (format + metadata) followed by raw protobuf bytes for efficient
fan-out.

---

## Kafka Integration

### Producer Configuration

```python
from cryptofeed.backends.kafka import TradeKafka, BookKafka

# Trade messages
trade_backend = TradeKafka(
    bootstrap_servers='localhost:9092',
    topic='crypto.trades',
    serialization_format='protobuf',
    # Standard Kafka producer options
    acks='all',
    compression_type='zstd',
    batch_size=16384,
    linger_ms=10
)

# OrderBook messages
book_backend = BookKafka(
    bootstrap_servers='localhost:9092',
    topic='crypto.orderbook',
    serialization_format='protobuf',
    max_request_size=10485760  # 10MB for large books
)
```


`TradeKafka` publishes protobuf payloads to the unified `cryptofeed.market.{data_type}.protobuf` topics. JSON callbacks
continue to use the legacy `{key}-{exchange}-{symbol}` names for backward compatibility. Typical mappings include:

- `cryptofeed.market.trades.protobuf`
- `cryptofeed.market.orderbook.protobuf`
- `cryptofeed.market.funding.protobuf`

Downstream systems subscribe per data type and still access exchange/symbol metadata from the protobuf fields.

### Custom Topic Mapping for QuixStreams

QuixStreams often expects organization-specific namespaces (Python-native stack, no JVM runtime) (e.g., `quix.crypto.trades`). Override the Kafka callback
topic selection when the payload is protobuf:

```python
from cryptofeed.backends.kafka import TradeKafka

class QuixTradeKafka(TradeKafka):
    TOPIC_MAP = {
        'trades': 'quix.crypto.trades',
        'orderbook': 'quix.crypto.orderbook',
    }

    def topic(self, data):
        if isinstance(data, bytes):  # protobuf payload
            data_type = getattr(self, 'protobuf_data_type', self.key)
            return self.TOPIC_MAP.get(data_type, f'quix.crypto.{data_type}')
        return super().topic(data)
```

Wire it into `FeedHandler` as usual while keeping JSON fallbacks for existing consumers. Because protobuf messages still carry
`exchange`, `symbol`, and other metadata, Quix pipelines can branch/aggregate using native primitives without parsing topic names.

### QuixStreams (Python-Native) Consumption & Iceberg Sinks

QuixStreams' SDK is Python-first, so you can remain on a JVM-free stack. Example topology:

```python
from quixstreams import Application
from cryptofeed.proto_bindings import trade_pb2

app = Application(broker_address="kafka:9092")

trades = (
    app.topic("cryptofeed.market.trades.protobuf")
       .protobuf(trade_pb2.Trade)
       .key_by(lambda msg: msg.symbol)
)

from quixstreams.logic import Window, Aggregator

vwap = (
    trades
    .window(Window.tumbling("1m"))
    .aggregate(Aggregator.vwap(
        price=lambda m: float(m.price),
        amount=lambda m: float(m.amount),
    ))
)

from pyiceberg.table import Table

iceberg_table = Table("local://lakehouse.crypto.trades")

(
    vwap
    .join(trades, lambda agg, msg: {
        "exchange": msg.exchange,
        "symbol": msg.symbol,
        "event_ts": msg.timestamp,
        "price": str(msg.price),
        "amount": str(msg.amount),
        "vwap_1m": agg.value,
    })
    .foreach(lambda row: iceberg_table.write([row]))
)

app.run()
```

Batch writes (or stage them to Parquet) for higher throughput if needed. Because Kafka partitions and `key_by` share the same symbol key,
state stays consistent per market. PyIceberg/other Python-native clients handle table appends without Spark/Java, keeping the pipeline JVM-free.


### Consumer Example (Python)

```python
from kafka import KafkaConsumer
from cryptofeed.proto_bindings import trade_pb2

# Create consumer
consumer = KafkaConsumer(
    'crypto.trades',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: trade_pb2.Trade().ParseFromString(m)
)

# Process messages
for message in consumer:
    trade = message.value  # Protobuf Trade object
    
    print(f"Symbol: {trade.symbol}")
    print(f"Price: {trade.price}")
    print(f"Amount: {trade.amount}")
    print(f"Side: {trade.side}")  # Enum: TRADE_SIDE_BUY or TRADE_SIDE_SELL
    print(f"Timestamp: {trade.timestamp / 1_000_000}")  # Convert microseconds to seconds
```


### Consumer Example (Go)

```go
package main

import (
    "fmt"
    "github.com/confluentinc/confluent-kafka-go/kafka"
    "google.golang.org/protobuf/proto"
    pb "path/to/generated/protobuf"
)

func main() {
    c, _ := kafka.NewConsumer(&kafka.ConfigMap{
        "bootstrap.servers": "localhost:9092",
        "group.id":          "crypto-consumer",
    })
    
    c.Subscribe("crypto.trades", nil)
    
    for {
        msg, _ := c.ReadMessage(-1)
        
        trade := &pb.Trade{}
        proto.Unmarshal(msg.Value, trade)
        
        fmt.Printf("Symbol: %s, Price: %s, Amount: %s\n",
            trade.Symbol, trade.Price, trade.Amount)
    }
}
```

---

## Performance Characteristics

### Latency

| Data Type | p50 | p95 | p99 |
|-----------|-----|-----|-----|
| Trade | 2.2 µs | ~10 µs | ~40 µs |
| Candle | 3.4 µs | ~15 µs | ~75 µs |
| OrderBook (20 levels) | 13.5 µs | ~50 µs | ~120 µs |

**Comparison**: Protobuf is **1.8x faster** than JSON for equivalent payloads.

### Throughput

**Single-threaded**: 520,000 messages/second  
**Target**: ≥10,000 messages/second ✅ **52x above target**

### Size Reduction

| Type | JSON | Protobuf | Reduction |
|------|------|----------|-----------|
| Trade | 168 bytes | 68 bytes | **59.5%** |
| Candle | 289 bytes | 125 bytes | **56.7%** |
| OrderBook | Variable | Variable | ~60% |

**Bandwidth Savings**: At 1000 msg/s, saves **8-14 MB/day** per feed.

---

## Schema Access

### Python Bindings

All protobuf message classes are available via `cryptofeed.proto_bindings`:

```python
from cryptofeed.proto_bindings import (
    trade_pb2,
    ticker_pb2,
    orderbook_pb2,
    candle_pb2,
    funding_pb2,
    liquidation_pb2,
    open_interest_pb2,
    index_price_pb2,
    balance_pb2,
    position_pb2,
    fill_pb2,
    order_info_pb2,
    order_pb2,
    transaction_pb2,
    trade_side_pb2  # Enum
)

# Create messages
trade = trade_pb2.Trade()
trade.symbol = 'BTC-USD'
trade.price = '50000.123'
trade.amount = '1.5'
trade.side = trade_side_pb2.TRADE_SIDE_BUY
trade.timestamp = 1700000000123456  # microseconds

# Serialize
serialized = trade.SerializeToString()

# Deserialize
parsed = trade_pb2.Trade()
parsed.ParseFromString(serialized)
```

### Schema Files (.proto)

Schema definitions are located in:
```
cryptofeed/proto/
├── cryptofeed/
│   └── normalized/
│       └── v1/
│           ├── trade.proto
│           ├── ticker.proto
│           ├── orderbook.proto
│           ├── candle.proto
│           ├── funding.proto
│           ├── liquidation.proto
│           ├── open_interest.proto
│           ├── index_price.proto
│           ├── balance.proto
│           ├── position.proto
│           ├── fill.proto
│           ├── order_info.proto
│           ├── order.proto
│           ├── transaction.proto
│           └── enums.proto
```

**Accessing schemas**:
```bash
# View schema
cat cryptofeed/proto/cryptofeed/normalized/v1/trade.proto

# Generate bindings for other languages
protoc --go_out=. cryptofeed/normalized/v1/trade.proto
protoc --java_out=. cryptofeed/normalized/v1/trade.proto
protoc --rust_out=. cryptofeed/normalized/v1/trade.proto
```

---

## Data Type Details

### Decimal Precision

Financial data uses **string encoding** to preserve full decimal precision:

```python
# Cryptofeed type
from decimal import Decimal
trade = Trade(
    price=Decimal('50000.123456789012345'),
    amount=Decimal('1.500000000000000001')
)

# Protobuf serialization
proto = trade_pb2.Trade()
proto.price = '50000.123456789012345'  # Exact string
proto.amount = '1.500000000000000001'

# No IEEE 754 float loss!
```

**Consumer**: Parse strings back to Decimal/BigDecimal/BigNum in your language.

### Timestamp Encoding

Timestamps are encoded as **int64 microseconds** since Unix epoch:

```python
# Python (float seconds)
timestamp = 1700000000.123456

# Protobuf (int64 microseconds)
proto.timestamp = int(timestamp * 1_000_000)  # 1700000000123456

# Consumer: Convert back
timestamp_seconds = proto.timestamp / 1_000_000.0
```

### Enum Handling

Side (buy/sell) uses protobuf enums:

```python
from cryptofeed.proto_bindings import trade_side_pb2

# Buy
proto.side = trade_side_pb2.TRADE_SIDE_BUY

# Sell
proto.side = trade_side_pb2.TRADE_SIDE_SELL

# Unknown/unspecified
proto.side = trade_side_pb2.TRADE_SIDE_UNSPECIFIED
```

**Consumer mapping**:
```python
# Python
if proto.side == trade_side_pb2.TRADE_SIDE_BUY:
    side = 'buy'
elif proto.side == trade_side_pb2.TRADE_SIDE_SELL:
    side = 'sell'
```

```go
// Go
switch trade.Side {
case pb.TradeSide_TRADE_SIDE_BUY:
    side = "buy"
case pb.TradeSide_TRADE_SIDE_SELL:
    side = "sell"
}
```

---

## Migration Guide

### From JSON to Protobuf

**Step 1**: Update backend configuration

```python
# Before (JSON)
TradeKafka(topic='trades')

# After (Protobuf)
TradeKafka(topic='trades', serialization_format='protobuf')
```

**Step 2**: Update consumers

```python
# Before (JSON consumer)
import json
for message in consumer:
    data = json.loads(message.value)
    price = data['price']

# After (Protobuf consumer)
from cryptofeed.proto_bindings import trade_pb2
for message in consumer:
    trade = trade_pb2.Trade()
    trade.ParseFromString(message.value)
    price = trade.price  # String, convert as needed
```

**Step 3**: Gradual rollout

Use **separate topics** for gradual migration:

```python
# Legacy JSON
TradeKafka(topic='trades.json', serialization_format='json')

# New Protobuf
TradeKafka(topic='trades.protobuf', serialization_format='protobuf')

# Run both in parallel, switch consumers gradually
```

### Backward Compatibility

**Default behavior unchanged**: JSON is still the default format.

```python
# These are equivalent (both use JSON)
TradeKafka(topic='trades')
TradeKafka(topic='trades', serialization_format='json')
```

Existing configurations continue to work without changes.

---

## Advanced Topics

### Custom Serialization

Implement your own serializer by subclassing `Serializer`:

```python
from cryptofeed.serializers.base import Serializer
import msgpack

class MsgPackSerializer(Serializer):
    """MessagePack serialization."""
    
    def serialize(self, data_obj) -> bytes:
        data_dict = data_obj.to_dict()
        return msgpack.packb(data_dict)
    
    def content_type(self) -> str:
        return 'application/msgpack'

# Use in backend
backend._get_serializer = lambda fmt: MsgPackSerializer()
```

### Compression

Combine protobuf with Kafka compression for maximum efficiency:

```python
TradeKafka(
    topic='trades',
    serialization_format='protobuf',
    compression_type='zstd',  # prefer 'zstd' or 'lz4' for protobuf payloads
    # Achieves 80-90% total size reduction vs uncompressed JSON
)
```

**New (Oct 31, 2025)**: Compression benchmarks live in
`tests/benchmarks/test_compression_serialization.py` and compare protobuf+
Snappy/LZ4 against raw protobuf and JSON payload sizes. The tests assert
compressed protobuf stays below 50% of the JSON payload and validate
round-trip integrity. Install the optional codecs locally to run them:

```bash
pip install lz4 zstandard
pytest tests/benchmarks/test_compression_serialization.py -v
```

If the codecs are missing, Pytest skips the benchmarks automatically. In CI
environments, add the packages so the ratios are recorded in test logs.

Complementary concurrency coverage is available in
`tests/benchmarks/test_concurrency_serialization.py`, which drives protobuf
serialization across eight threads and validates outputs to guarantee
thread-safety in multi-threaded ingestion pipelines:

```bash
pytest tests/benchmarks/test_concurrency_serialization.py -v
```

### Schema Evolution

Protobuf supports backward-compatible schema changes:

```protobuf
// v1 schema
message Trade {
  string symbol = 1;
  string price = 2;
}

// v2 schema (backward compatible)
message Trade {
  string symbol = 1;
  string price = 2;
  string trade_id = 3;  // New optional field
}
```

**Old consumers** ignore new fields.  
**New consumers** handle missing fields gracefully.

---

## Monitoring and Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Shows serialization metrics
logger = logging.getLogger('cryptofeed.serializers')
logger.setLevel(logging.DEBUG)
```

### Metrics to Track

| Metric | Threshold | Action |
|--------|-----------|--------|
| Serialization throughput | <50k msg/s | Investigate bottleneck |
| p99 latency | >1ms | Check system load |
| Error rate | >0.1% | Review error logs |
| Memory growth | >10% / 24h | Check for leaks |

Recent synthetic benchmarks (`tests/benchmarks/test_serialization_performance.py`) show
average protobuf serialization latency of ~26µs for trades and ~320µs for order
book snapshots—comfortably within the 500µs / 2ms budgets defined in the spec.
Compression tests (`tests/benchmarks/test_compression_serialization.py`) confirm
protobuf payloads remain ≤55% of the JSON size uncompressed and ≤45–50% when
compressed with zstd/lz4.

### Common Issues

**Issue**: `SerializationError: to_proto not found`  
**Solution**: Ensure `import cryptofeed.proto_wrappers.registry` is called.

**Issue**: `TypeError: cannot set 'to_proto' attribute`  
**Solution**: Registry pattern handles this automatically (C extension limitation).

**Issue**: Decimal keys in OrderBook JSON  
**Solution**: Use protobuf format (JSON has pre-existing limitation).

---

## Production Deployment

### Recommended Configuration

```python
from cryptofeed.backends.kafka import TradeKafka, BookKafka

# High-throughput production config
config = {
    'bootstrap_servers': 'kafka-1:9092,kafka-2:9092,kafka-3:9092',
    'serialization_format': 'protobuf',
    'acks': 'all',  # Durability
    'compression_type': 'zstd',  # Prefer 'zstd' (higher ratio) or 'lz4' (lower latency)
    'batch_size': 65536,  # Larger batches
    'linger_ms': 5,  # Small delay for batching
    'max_in_flight_requests_per_connection': 5,
    'retries': 10,
    'enable_idempotence': True  # Exactly-once semantics
}

trade_backend = TradeKafka(topic='crypto.trades', **config)
```

### Monitoring Checklist

- [ ] Kafka lag < 1000 messages
- [ ] Producer throughput ≥ feed data rate
- [ ] p99 latency < 1ms
- [ ] Memory usage stable over 24h
- [ ] Error rate < 0.01%
- [ ] Disk I/O within limits

---

## FAQ

**Q: Is protobuf faster than JSON?**  
A: Yes, ~1.8x faster for serialization, ~60% smaller messages.

**Q: Can I mix JSON and protobuf?**  
A: Yes, use different topics or `serialization_format` per feed.

**Q: Do I need to change existing consumers?**  
A: Only if you switch to protobuf. JSON remains the default.

**Q: What about schema versioning?**  
A: Protobuf schemas are in `cryptofeed.normalized.v1`. Future versions will be `v2`, etc.

**Q: Can I use protobuf with PostgreSQL?**  
A: Not yet. PostgreSQL backend uses JSON currently. Protobuf support planned.

**Q: How do I access .proto files for other languages?**  
A: Located in `cryptofeed/proto/`. Use `protoc` to generate bindings.

**Q: What if serialization fails?**  
A: `SerializationError` is raised. Enable logging to debug.

---

## Resources

- **Performance Benchmarks**: `docs/protobuf-performance-baseline.md`
- **Schema Files**: `cryptofeed/proto/cryptofeed/normalized/v1/`
- **Test Examples**: `tests/unit/proto_wrappers/test_all_14_types.py`
- **Kafka Integration**: `tests/integration/test_kafka_serialization_e2e.py`

---

## Support

For issues or questions:
1. Check existing issues: https://github.com/bmoscon/cryptofeed/issues
2. Review test examples in `tests/` directory
3. Enable debug logging for detailed diagnostics
4. Report bugs with serialization logs and data samples

---

**Last Updated**: October 31, 2025  
**Implementation**: protobuf-callback-serialization (Spec 1)  
**Status**: ✅ Production Ready
