# Apache Iceberg Tables for Cryptofeed Protobuf Data

This directory contains reference DDLs for creating Iceberg tables to store normalized market and account data derived from the protobuf schemas in proto/cryptofeed/v1.

Assumptions
- Timestamps are UTC and stored as TIMESTAMP (without time zone) for engine compatibility.
- Decimal values use decimal(38, 18). Adjust scale/precision to your needs.
- Partitioning strategy favors days(event_ts), exchange, symbol. Tune for your workloads.
- Storage format defaults to Parquet with zstd compression.

Usage (Spark)
- Replace `<catalog>.<db>` with your Iceberg catalog and database.
- Run with spark-sql or within Spark sessions.

See docs/lakehouse/ICEBERG_TABLES.md for details.

