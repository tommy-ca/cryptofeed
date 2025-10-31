# Consumer Integration Guide

## Overview

Cryptofeed produces protobuf-serialized market data to Kafka topics. This guide shows how downstream consumers integrate these topics with storage backends (Apache Iceberg, DuckDB, Parquet) and analytics engines (Flink, Spark).

## Architecture

```
Cryptofeed (Ingestion) → Kafka Topics → Consumer (Storage + Analytics)
```

**Cryptofeed Responsibility**: Produce protobuf messages to Kafka
**Consumer Responsibility**: Read Kafka, store data, run analytics

## Topic Schema

Topics follow naming convention: `cryptofeed.{data_type}.{exchange}.{symbol}`

Examples:
- `cryptofeed.trades.coinbase.btc-usd`
- `cryptofeed.l2_book.binance.eth-usdt`
- `cryptofeed.ticker.kraken.sol-usd`

Message format: Protobuf (schemas from `cryptofeed.normalized.v1`)

## Integration Patterns

### Pattern 1: Flink → Apache Iceberg

**Use Case**: Real-time ingestion with schema evolution and time travel

```python
# Flink SQL job
from pyflink.table import EnvironmentSettings, TableEnvironment

env_settings = EnvironmentSettings.in_streaming_mode()
t_env = TableEnvironment.create(env_settings)

# Source: Kafka topic with protobuf
t_env.execute_sql("""
    CREATE TABLE trades_source (
        symbol STRING,
        price DECIMAL(20,8),
        amount DECIMAL(20,8),
        timestamp BIGINT,
        side STRING,
        exchange STRING
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'cryptofeed.trades.*',
        'properties.bootstrap.servers' = 'kafka:9092',
        'format' = 'protobuf',
        'protobuf.message-class' = 'cryptofeed.schema.v1.Trade'
    )
""")

# Sink: Apache Iceberg
t_env.execute_sql("""
    CREATE CATALOG iceberg WITH (
        'type' = 'iceberg',
        'warehouse' = 's3://lakehouse/warehouse'
    )
""")

t_env.execute_sql("""
    CREATE TABLE iceberg.default.trades (
        symbol STRING,
        price DECIMAL(20,8),
        amount DECIMAL(20,8),
        timestamp BIGINT,
        side STRING,
        exchange STRING
    ) PARTITIONED BY (days(timestamp))
""")

# Stream Kafka → Iceberg
t_env.execute_sql("""
    INSERT INTO iceberg.default.trades
    SELECT * FROM trades_source
""")
```

**Benefits**:
- Schema evolution (Iceberg native)
- Time travel queries
- ACID transactions
- S3/GCS/Azure compatible

---

### Pattern 2: DuckDB Direct Consumer

**Use Case**: Local analytics, backtesting, research notebooks

```python
# Python consumer script
import duckdb
from kafka import KafkaConsumer
from cryptofeed.schema.v1.trade_pb2 import Trade

# Setup
consumer = KafkaConsumer(
    'cryptofeed.trades.coinbase.btc-usd',
    bootstrap_servers='kafka:9092',
    value_deserializer=lambda m: Trade().ParseFromString(m)
)

conn = duckdb.connect('lakehouse.db')
conn.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        symbol VARCHAR,
        price DECIMAL(20,8),
        amount DECIMAL(20,8),
        timestamp BIGINT,
        side VARCHAR,
        exchange VARCHAR
    )
""")

# Consume and insert
for msg in consumer:
    trade = msg.value
    conn.execute("""
        INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?)
    """, [
        trade.symbol,
        trade.price,
        trade.amount,
        trade.timestamp,
        trade.side,
        trade.exchange
    ])

# Query
result = conn.execute("""
    SELECT
        symbol,
        AVG(price) as avg_price,
        SUM(amount) as volume
    FROM trades
    WHERE timestamp > ?
    GROUP BY symbol
""", [start_timestamp]).fetchall()
```

**Benefits**:
- Zero infrastructure (SQLite-like)
- Fast analytics (columnar)
- Parquet export for sharing

---

### Pattern 3: Spark Streaming → Parquet

**Use Case**: Large-scale batch processing, historical analytics

```python
# PySpark job
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CryptofeedConsumer") \
    .getOrCreate()

# Read from Kafka
trades_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "cryptofeed.trades.*") \
    .load()

# Deserialize protobuf
from cryptofeed.schema.v1.trade_pb2 import Trade

def deserialize_protobuf(value):
    trade = Trade()
    trade.ParseFromString(value)
    return (
        trade.symbol,
        float(trade.price),
        float(trade.amount),
        trade.timestamp,
        trade.side,
        trade.exchange
    )

trades_parsed = trades_df.selectExpr("value") \
    .rdd.map(lambda row: deserialize_protobuf(row.value)) \
    .toDF(["symbol", "price", "amount", "timestamp", "side", "exchange"])

# Write to Parquet (partitioned by date)
query = trades_parsed \
    .writeStream \
    .format("parquet") \
    .option("path", "s3://lakehouse/trades") \
    .option("checkpointLocation", "s3://lakehouse/checkpoints/trades") \
    .partitionBy("date") \
    .start()

query.awaitTermination()
```

**Benefits**:
- Massive scale (petabytes)
- Parquet compression
- Date partitioning
- S3/HDFS compatible

---

## Schema Registry Integration

### Confluent Schema Registry

```python
from confluent_kafka import DeserializingConsumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.protobuf import ProtobufDeserializer

schema_registry_conf = {'url': 'http://schema-registry:8081'}
schema_registry_client = SchemaRegistryClient(schema_registry_conf)

protobuf_deserializer = ProtobufDeserializer(
    Trade,
    schema_registry_client
)

consumer_conf = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'iceberg-consumer',
    'value.deserializer': protobuf_deserializer
}

consumer = DeserializingConsumer(consumer_conf)
consumer.subscribe(['cryptofeed.trades.*'])

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue

    trade = msg.value()  # Already deserialized
    # Insert into Iceberg/DuckDB/etc.
```

---

## Best Practices

### 1. Consumer Groups
Use Kafka consumer groups for parallel processing:
```
group.id = iceberg-consumer-1  # Multiple instances for scale
```

### 2. Exactly-Once Semantics
Enable idempotent consumers:
```
enable.idempotence = true
isolation.level = read_committed
```

### 3. Checkpointing
Track Kafka offsets in storage backend (Iceberg metadata, DuckDB table):
```sql
CREATE TABLE kafka_offsets (
    topic VARCHAR,
    partition INT,
    offset BIGINT,
    timestamp BIGINT
)
```

### 4. Schema Evolution
Use Protobuf backward compatibility:
- Add fields with defaults
- Deprecate instead of removing
- Version schemas (v1, v2, etc.)

### 5. Monitoring
Track consumer lag:
```bash
kafka-consumer-groups --bootstrap-server kafka:9092 \
    --group iceberg-consumer --describe
```

---

## Troubleshooting

### Issue: Consumer lag increasing
**Cause**: Consumer too slow
**Solution**: Add more consumer instances (scale horizontal)

### Issue: Duplicate records in Iceberg
**Cause**: Non-idempotent writes
**Solution**: Use Iceberg upsert with deduplication key

### Issue: Schema mismatch errors
**Cause**: Protobuf version skew
**Solution**: Pin protobuf version in requirements.txt

---

## Next Steps

1. Choose a storage backend (Iceberg/DuckDB/Parquet)
2. Choose a stream processor (Flink/Spark/custom)
3. Implement consumer based on pattern above
4. Test with cryptofeed Kafka topics
5. Monitor consumer lag and performance
