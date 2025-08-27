# Iceberg Table Schemas for Cryptofeed Protobuf Data

This guide defines how each protobuf message maps to Iceberg table schemas and provides reference DDLs for Spark SQL. Adjust catalog/database names, partitioning, and data types to suit your environment (Spark, Flink, Trino, etc.).

General guidelines
- All timestamps are stored in UTC as TIMESTAMP (without timezone) for engine compatibility.
- Decimals use DECIMAL(38, 18). If your prices/sizes need different scales, adjust accordingly.
- Partitioning: The defaults partition by days(event_ts), exchange, and symbol where applicable. Consider bucketing/hash-partitioning for high-cardinality symbols.
- Compression: Parquet + ZSTD by default.

Message → Table mapping
- Trade → trades
  - exchange, symbol, base, quote, instrument_type, side, amount, price, trade_id, trade_type, event_ts, receipt_ts
- Ticker → ticker
  - exchange, symbol, bid, ask, event_ts, receipt_ts
- L1Book → l1_book
  - exchange, symbol, bid_price, bid_size, ask_price, ask_size, event_ts, receipt_ts
- L2Book → l2_book_delta (deltas preferred for efficient storage)
  - exchange, symbol, bid_changes[], ask_changes[], sequence_number, checksum, event_ts, receipt_ts
  - Alternative: create a snapshot table with arrays of levels for periodic snapshots
- Funding → funding
  - exchange, symbol, mark_price, rate, next_funding_time, predicted_rate, event_ts, receipt_ts
- OpenInterest → open_interest
  - exchange, symbol, open_interest, event_ts, receipt_ts
- Liquidation → liquidations
  - exchange, symbol, side, quantity, price, liquidation_id, status, event_ts, receipt_ts
- Index → index_prices
  - exchange, symbol, price, event_ts, receipt_ts
- Candle → candles
  - exchange, symbol, interval, start_time, end_time, trades, open, close, high, low, volume, closed, event_ts, receipt_ts
- Account messages → account.sql
  - OrderInfo → order_info
  - Balance → balances
  - Fill → fills
  - Position → positions
  - Transaction → transactions

DDL location
- See lakehouse/iceberg/ddl/*.sql for each table’s DDL.

Example usage (Spark SQL)
```
spark-sql -f lakehouse/iceberg/ddl/trades.sql \
  --conf spark.sql.catalog.mycat=org.apache.iceberg.spark.SparkCatalog \
  --conf spark.sql.catalog.mycat.type=hive \
  --conf spark.sql.catalog.mycat.uri=thrift://metastore:9083
```

Ingestion notes
- Convert protobuf Decimal.value (strings) to DECIMAL(38, 18) using your ETL/streaming job (Spark/Flink).
- Ensure UTC normalization for timestamps.
- For L2 deltas, arrays-of-structs are supported by Iceberg. Alternatively, model level updates as normalized rows in a separate table with columns (side, price, size, op).

