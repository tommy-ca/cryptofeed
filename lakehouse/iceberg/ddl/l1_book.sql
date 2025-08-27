-- L1 top-of-book table from cryptofeed.v1.L1Book
CREATE TABLE IF NOT EXISTS <catalog>.<db>.l1_book (
  exchange       STRING,
  symbol         STRING,
  bid_price      DECIMAL(38, 18),
  bid_size       DECIMAL(38, 18),
  ask_price      DECIMAL(38, 18),
  ask_size       DECIMAL(38, 18),
  event_ts       TIMESTAMP,
  receipt_ts     TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(event_ts), exchange, symbol)
TBLPROPERTIES (
  'format-version' = '2',
  'write.format.default' = 'parquet',
  'write.parquet.compression-codec' = 'zstd'
);

