feat: protobuf + buf migration for ingestion, storage, processing, and serving

Summary
- Introduces normalized protobuf schemas for market and account data, event envelope, and Kafka wrappers.
- Establishes Buf-managed schema workflow for safe evolution and multi-language codegen.
- Sets foundation for Kafka-based ingestion, data lake storage, processing pipelines, and real-time + historical serving (WS + Kafka).

Schema Overview
- Common: enums, Decimal string, Symbol, DataChannel.
- Market Data: Trade, Ticker, L1/L2/L3 books, BookDelta, Funding, OI, Liquidation, Index, Candle.
- Account Data: OrderInfo, Balance, Transaction, Fill, Position.
- Events: DataFeedEvent (+ Batch), Subscription, Heartbeat, ErrorMessage.
- Kafka: KafkaMetadata, KafkaHeader, KafkaRecord, KafkaDataFeedEvent.

Transport Design (Kafka)
- Topic naming: crypto.{env}.{exchange}.{channel}.{instrument_type} (e.g., crypto.prod.binance.trades.spot).
- Keys: `{exchange}:{symbol}:{channel}`; stable casing.
- Partitioning: by `{symbol}` for ordering and scaling.
- Headers: `schema.version`, `content.type=application/x-protobuf`, `compression`, `trace.id`, `producer.id`.
- Producer: idempotent=true, acks=all, linger.ms=5–20, batch.size tuned, compression=zstd.
- Consumer: isolation.level=read_committed, backpressure tuned; DLQ topics for failures.

Realtime WS Serving
- `DataFeedEvent`/`DataFeedEventBatch` payloads (protobuf), with optional JSON gateway.
- `SubscriptionRequest`/`Response`, `Heartbeat`, error reporting.
- Snapshot then deltas for order books; batching + compression.

Historical Serving
- Kafka→object store via Connect/Flink/Spark → Delta/Iceberg/Parquet.
- Partitioning: y=YYYY/m=MM/d=DD/h=HH/exchange=x/channel=c/symbol=BASE-QUOTE.
- Store flattened columns + raw payload + KafkaMetadata for replay; query via Trino/Spark.

Processing Pipelines
- Trades→Candles (intervals), Funding/OI snapshots, L2/L3 reconciliation with sequence/checksum.
- Idempotency: event_id + sequence_number dedupe; EOS guarantees with transactional producers.

Compatibility & Evolution
- Buf lint + breaking checks; no tag reuse; deprecations before removals.
- v1 package; major breaking changes → v2 package + new topics.

Operational Notes
- Metrics: serialization size, latency, Kafka lag, DLQ counts; tracing via headers.
- Backfill: bulk loaders; replay strategy documented.

Follow-ups
- CI for Buf + codegen.
- Producer/consumer implementations with headers + DLQ.
- WS gateway; storage sinks; backfill utilities; docs.
