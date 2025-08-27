-- Open interest from cryptofeed.v1.OpenInterest
CREATE TABLE IF NOT EXISTS <catalog>.<db>.open_interest (
  exchange       STRING,
  symbol         STRING,
  open_interest  DECIMAL(38, 18),
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

