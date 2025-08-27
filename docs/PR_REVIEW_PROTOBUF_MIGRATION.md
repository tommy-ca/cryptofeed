Title: Protobuf Migration – Ingestion, Storage, Processing, and Serving Review

Summary
- Normalizes cryptofeed data with protobuf and Buf-managed schemas to unlock cross-language ingestion, efficient storage, robust streaming pipelines, and real-time + historical serving.
- Schemas: common, market_data, account_data, events (envelope + batch + subscription), kafka (metadata + headers + record).

Key Design Guidance
- Topic naming: `crypto.{env}.{exchange}.{channel}.{instrument_type}`; example: `crypto.prod.binance.trades.spot`.
- Keys/ordering: key by `{exchange}:{symbol}:{channel}`; partition by `symbol` to preserve per-symbol ordering and scale.
- Headers: `schema.version`, `content.type=application/x-protobuf`, `compression=zstd`, `trace.id`, `producer.id`.
- Producer: idempotent=true, acks=all, linger.ms=5–20ms, batch.size tuned by load, compression=zstd, retries with backoff.
- Consumer: isolation.level=read_committed, dead-letter (DLQ) topics per stream, metrics for lag and error budgets.
- Books: periodic snapshots (L2/L3) + continuous BookDelta; reconcile using sequence_number and checksum when available.
- Decimal: string-based precision; convert at edges; avoid binary float.

Ingestion Pipeline
- Exchange connectors → Adapter → DataFeedEvent: enrich with event_id, exchange/symbol/channel, timestamps, optional sequence.
- Validate: ensure non-empty exchange/symbol; clamp/normalize symbol casing; emit parse/validation errors to DLQ.
- Produce: transactional/idempotent producers with stable keys; set headers; include KafkaMetadata at consumer side.

Processing Pipelines
- Stream aggregations: Trade→Candle (intervals), Funding/OI snapshots, L2/L3 order book reconciliation with side outputs for anomalies.
- State and EOS: use Flink/Spark structured streaming with checkpointing; EOS writes to sinks (Delta/Iceberg).
- Derived topics: publish computed metrics (VWAP, spreads), rollups, and snapshot builders.

Storage & Historical Serving
- Lake format: Delta Lake or Apache Iceberg with Parquet; keep raw protobuf payload bytes and flattened columns.
- Partitioning: `/y=YYYY/m=MM/d=DD/h=HH/exchange=x/channel=c/symbol=BASE-QUOTE/`.
- Access: Trino/Presto/Spark SQL for analytics; REST/WS historical API returns `DataFeedEventBatch` over ranges.
- Backfill: bulk loaders for historical data; publish to storage first, to Kafka only if replay needed.

WebSocket Serving
- Contract: `SubscriptionRequest`/`Response`, `Heartbeat`, `ErrorMessage`; serve protobuf; offer JSON gateway for legacy.
- Flow control: batches + compression; initial snapshot then deltas for books.

Evolution & Governance
- Buf lint + breaking checks in CI; never reuse field numbers; deprecate then remove in next major (v2).
- Topic evolution: minor compatible changes keep topics; major breaking → new package (v2) and new topics.

Operational Readiness
- Observability: metrics for serialization size, end-to-end latency, Kafka lag, DLQ rates; trace IDs in headers.
- SLOs: define per-stream error budgets and replay procedures; document backfill and reprocessing.

Action Items (Proposed in Follow-ups)
- [ ] CI: Buf lint/breaking + codegen verification; fail on drift.
- [ ] Producers: idempotent configs and headers; DLQ wiring; tracing.
- [ ] WS Gateway: protobuf-first, JSON bridge; subscription auth if needed.
- [ ] Storage Sinks: Connect/Flink to Delta/Iceberg partitions; compaction.
- [ ] Backfill Tools: historical loaders + validators.
- [ ] Docs: topic conventions, headers, and consumer guidelines.

Requested Review Decisions
- Topic naming and partitioning conventions acceptable for ops?
- Decimal-as-string suitable for target consumers (Go/Rust/Java/Python)?
- Book snapshot+delta approach OK, sequence semantics per exchange?
- Historical serving via Delta/Iceberg vs alternatives (Hudi/Parquet-only)?

