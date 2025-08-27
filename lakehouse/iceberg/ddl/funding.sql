-- Funding from cryptofeed.v1.Funding
CREATE TABLE IF NOT EXISTS <catalog>.<db>.funding (
  exchange           STRING,
  symbol             STRING,
  mark_price         DECIMAL(38, 18),
  rate               DECIMAL(38, 18),
  next_funding_time  TIMESTAMP,
  predicted_rate     DECIMAL(38, 18),
  event_ts           TIMESTAMP,
  receipt_ts         TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(event_ts), exchange, symbol)
TBLPROPERTIES (
  'format-version' = '2',
  'write.format.default' = 'parquet',
  'write.parquet.compression-codec' = 'zstd'
);

