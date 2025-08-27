-- Liquidations from cryptofeed.v1.Liquidation
CREATE TABLE IF NOT EXISTS <catalog>.<db>.liquidations (
  exchange       STRING,
  symbol         STRING,
  side           STRING,
  quantity       DECIMAL(38, 18),
  price          DECIMAL(38, 18),
  liquidation_id STRING,
  status         STRING,
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

