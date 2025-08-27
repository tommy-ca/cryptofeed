-- Ticker/BBO table derived from cryptofeed.v1.Ticker
CREATE TABLE IF NOT EXISTS <catalog>.<db>.ticker (
  exchange       STRING,
  symbol         STRING,
  bid            DECIMAL(38, 18),
  ask            DECIMAL(38, 18),
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

