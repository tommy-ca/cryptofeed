-- Index prices from cryptofeed.v1.Index
CREATE TABLE IF NOT EXISTS <catalog>.<db>.index_prices (
  exchange       STRING,
  symbol         STRING,
  price          DECIMAL(38, 18),
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

