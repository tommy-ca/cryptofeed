# Requirements Document: Lakehouse Backend Adapter

## Project Description (Input)

Persistent storage layer reactivating disabled cryptofeed-lakehouse-architecture specification. Consume protobuf-serialized market data and aggregated streams from Kafka topics, persist to DuckDB-backed Parquet data lakehouse with columnar storage, date/exchange/symbol partitioning, and efficient query support. Store raw trade/orderbook deltas alongside OHLCV candles, VWAP metrics, cross-exchange correlations in unified data model. Implement streaming buffer with configurable batch sizes and flush intervals, RocksDB-backed state tracking for exactly-once semantics, and partition pruning for rapid time-range queries. Enable analytical queries via DuckDB SQL interface: 24h price movements, volume analysis, liquidity patterns, arb opportunity tracking. Support historical backfill via replay capabilities, incremental updates for new data, and time-travel queries for regulatory compliance. Production operations: automated compaction, retention policies, backup/recovery procedures, monitoring dashboards.

## Specification Overview

This specification establishes the persistent storage and analytics layer for cryptofeed data, enabling:
- Efficient consumption of protobuf-serialized raw and aggregated data from Kafka topics
- Columnar storage via Apache Parquet with date/exchange/symbol partitioning
- SQL-based analytics interface via DuckDB query engine
- Streaming buffer with configurable batch sizes for write optimization
- Exactly-once semantics with distributed state tracking
- Historical backfill and time-travel query support
- Production operations: automated compaction, retention policies, backup/recovery

## Context & Motivation

### Current State
- Disabled `cryptofeed-lakehouse-architecture` specification has design and requirements prepared
- Protobuf callback serialization (Spec 1) enables efficient binary storage format
- QuixStreams integration (Spec 2) provides aggregated analytical streams (OHLCV, VWAP, correlations)
- DuckDB + Parquet technology stack proven in data lakehouse implementations

### Why Now
- Protobuf and QuixStreams foundations enable efficient, type-safe end-to-end data flow
- Users require SQL-queryable analytics on historical and real-time data
- Columnar Parquet storage provides 80%+ compression for time-series data
- Distributed state tracking enables horizontal scaling of ingestion pipeline

## Requirements

### Functional Requirements

**FR1: Kafka Topic Consumption**
- Consume from multiple Kafka topics: raw trades, raw orderbook, OHLCV candles, VWAP metrics, correlations
- Support protobuf deserialization for all message types
- Implement consumer group coordination for multi-worker deployment
- Configure topic subscriptions via YAML (topics, partitions, consumer group)

**FR2: Streaming Buffer & Batching**
- Buffer incoming records in memory with configurable batch size (default 10k records)
- Flush buffer on time interval (default 60s) or size threshold
- Configurable flush strategy: eager (minimize latency) or conservative (maximize batching)
- Atomic batch writes to Parquet for consistency

**FR3: Parquet Partitioning**
- Partition data by: date (YYYY-MM-DD), exchange, symbol
- Partition path: `year=2024/month=01/day=15/exchange=binance/symbol=BTCUSD/`
- Support dynamic partition creation as new symbols/exchanges arrive
- Configurable partition retention and compaction

**FR4: Data Model & Schema**
- Trade data: timestamp, exchange, symbol, side, price, quantity, trade_id, raw fields
- OrderBook delta: timestamp, exchange, symbol, bids, asks, sequence
- OHLCV candle: timestamp, exchange, symbol, open, high, low, close, volume, vwap
- Metric: timestamp, exchange, symbol, metric_name, metric_value
- Correlation: timestamp, symbol_pair, exchange_pair, correlation_value, lookback_window

**FR5: Exactly-Once Semantics**
- RocksDB state store tracking processed message offsets per topic/partition
- Idempotent writes: duplicate detection via (exchange, symbol, timestamp, message_id)
- Checkpoint state periodically to persistent storage
- Graceful recovery from broker/worker failures

**FR6: DuckDB Query Interface**
- Expose SQL query API for analytical queries
- Support parameterized queries with time-range predicates
- Implement view layer for common queries (OHLCV, volume patterns, liquidity)
- Query result caching for hot queries

**FR7: Historical Backfill**
- Replay capability to reprocess Kafka topic ranges by timestamp or offset
- Idempotent reprocessing: existing data updated, not duplicated
- Backfill configuration: topic, start_timestamp, end_timestamp, consumer_group
- Progress tracking and resumable backfill

**FR8: Time-Travel Queries**
- Capability to query data as of specific point in time (for compliance)
- Version tracking at record level or table level
- Support for regulatory audit trails

### Technical Requirements

**TR1: Storage Architecture**
- Base path: configurable (e.g., `/data/lakehouse` or S3 bucket)
- DuckDB catalog: SQL schema definitions for all data models
- Parquet compression: Snappy (default, fast) or Zstd (smaller)
- File size targets: 64-256MB per Parquet file for query optimization

**TR2: State Management**
- RocksDB state store for offset tracking: topic/partition → max_offset
- State store persistence: periodic snapshots to Parquet backup
- Compaction: automatic removal of old offsets (configurable retention: 7 days default)
- Multi-process safety: distributed locks via Kafka offset coordination

**TR3: Streaming Ingest Pipeline**
- Topology: Consumer → Deserializer → Transformer → Buffer → Writer
- Transformer: protobuf → Parquet schema mapping, field validation
- Buffer: in-memory queue with bounded size (max 1GB default)
- Writer: atomic batch writes, partition creation on-demand

**TR4: Query Engine (DuckDB)**
- Connection pool: configurable size (default 10 connections)
- Query timeout: configurable (default 300s for analytical queries)
- Result format: Arrow, pandas DataFrame, JSON
- Query statistics: execution time, rows scanned, partitions pruned

**TR5: Operations & Monitoring**
- Compaction: automatic weekly on partitions older than 24h
- Retention: configurable per partition (default 365 days)
- Backup: incremental snapshot to external storage (S3, GCS, local)
- Recovery: restore from snapshots with point-in-time recovery capability
- Monitoring: ingest throughput, buffer utilization, query latency, storage growth

**TR6: Configuration**
- YAML configuration: topics, partitions, batch sizes, flush intervals, retention
- Environment variable interpolation for paths, credentials, connection strings
- Per-partition overrides: custom retention, compaction strategy
- Runtime reconfiguration without data loss

### Non-Functional Requirements

**NFR1: Performance**
- Ingest throughput: 1M+ trades/sec per worker instance
- Buffer flush latency: <5s p99 for committed writes
- Query latency: <1s p99 for time-range queries (24h, single symbol)
- DuckDB scan throughput: 10M+ rows/sec
- Write amplification: <1.5x (Parquet overhead + compaction)

**NFR2: Storage Efficiency**
- Parquet compression: 80%+ reduction vs raw JSON
- Data deduplication: <0.1% redundancy with exactly-once semantics
- Index size: <5% of data size (partition pruning + statistics)
- 30-day retention: <100GB per 100M trades/day

**NFR3: Scalability**
- Horizontal scaling: 1-10+ worker instances for multi-topic consumption
- Symbol scale: 1000+ active symbols without performance degradation
- Time scale: 1+ year of historical data queryable without degradation
- Partition scale: millions of partitions with efficient pruning

**NFR4: Reliability**
- Data durability: no data loss during worker/broker failures
- Recovery time: <5 minutes from failure to resumed ingest
- Backup frequency: daily incremental snapshots
- Restore verification: automated integrity checks

**NFR5: Maintainability**
- Clear separation: ingest pipeline, storage layer, query API
- Schema versioning: backward-compatible evolution
- Comprehensive logging: per-record tracing for audits
- Documentation: operation runbooks, troubleshooting guides

## Dependencies & Related Specifications

- **Upstream (Blocking)**: `protobuf-callback-serialization` (Spec 1) - Provides protobuf-serialized Kafka topics
- **Upstream (Blocking)**: `quixstreams-integration` (Spec 2) - Provides aggregated stream topics
- **Upstream (Reference)**: `cryptofeed-lakehouse-architecture` (disabled) - Design and requirements foundation
- **Upstream (Reference)**: `normalized-data-schema-crypto` v0.1.0 - Provides protobuf schemas
- **External**: DuckDB, Apache Parquet, RocksDB - Storage and query libraries
- **External**: PyArrow - Serialization and schema management

## Success Criteria

1. ✅ DuckDB database initialized with all required schemas (trades, candles, metrics, correlations)
2. ✅ Streaming buffer implementation: buffer, flush, and write 10k+ record batches correctly
3. ✅ Parquet partitioning: automatic partition creation for new dates/exchanges/symbols
4. ✅ Exactly-once semantics verified: reprocess 1M records, zero duplicates/losses
5. ✅ SQL query interface functional: 100+ queries executed against 1B+ row dataset
6. ✅ Time-range query performance: <1s p99 for 24h single-symbol queries
7. ✅ Historical backfill: replay 30-day dataset in <1 hour per worker instance
8. ✅ Compression ratio: Parquet files 80%+ smaller than JSON source
9. ✅ Failure recovery: restore from backup and resume within 5 minutes
10. ✅ Production integration guide: deployment, monitoring, operations runbook

## Related Architecture

### Data Flow
```
Kafka Topics (Protobuf)
  ├─ Trade topic → Lakehouse trades table
  ├─ OrderBook topic → Lakehouse orderbook table
  ├─ Candle topic (QuixStreams) → Lakehouse candles table
  ├─ VWAP topic (QuixStreams) → Lakehouse metrics table
  └─ Correlation topic (QuixStreams) → Lakehouse correlations table

DuckDB Query Layer
  ├─ OHLCV view: SELECT * FROM candles WHERE exchange=? AND symbol=? AND timestamp >= ?
  ├─ Liquidity view: SELECT * FROM orderbook WHERE exchange=? AND symbol=?
  ├─ Correlation view: SELECT * FROM correlations WHERE timestamp >= ?
  └─ Arb opportunity view: SELECT * FROM metrics WHERE metric_name = 'arb_delta'
```

### Storage Layout
```
/data/lakehouse/
├── .ducksql              # DuckDB catalog and system tables
├── tables/
│   ├── trades/
│   │   └── year=2024/month=01/day=15/exchange=binance/symbol=BTCUSD/part-0.parquet
│   ├── candles/
│   │   └── year=2024/month=01/day=15/exchange=binance/symbol=BTCUSD/part-0.parquet
│   └── correlations/
│       └── year=2024/month=01/day=15/pair=BTCUSD-ETHUSDT/part-0.parquet
├── state/
│   └── rockodb/          # Offset tracking state store
└── backups/
    └── 2025-01-15/      # Daily snapshot backup
```

## Next Steps

1. **Requirements Approval**: Review and approve this requirements document
2. **Design Phase**: Create design.md specifying query schemas, ingest topology, and operational procedures
3. **Tasks Generation**: Generate implementation tasks and timeline
4. **Implementation**: Execute tasks using TDD methodology (must complete Specs 1 & 2 first)

---

**Phase**: Initialized
**Timeline**: 3-4 weeks (after Specs 1 & 2 completion)
**Owner**: Development Team
**Blocked By**: `protobuf-callback-serialization` (Spec 1), `quixstreams-integration` (Spec 2)
