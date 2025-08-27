-- Candles/OHLCV from cryptofeed.v1.Candle
CREATE TABLE IF NOT EXISTS <catalog>.<db>.candles (
  exchange       STRING,
  symbol         STRING,
  interval       STRING,
  start_time     TIMESTAMP,
  end_time       TIMESTAMP,
  trades         INT,
  open           DECIMAL(38, 18),
  close          DECIMAL(38, 18),
  high           DECIMAL(38, 18),
  low            DECIMAL(38, 18),
  volume         DECIMAL(38, 18),
  closed         BOOLEAN,
  event_ts       TIMESTAMP,
  receipt_ts     TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(start_time), exchange, symbol)
TBLPROPERTIES (
  'format-version' = '2',
  'write.format.default' = 'parquet',
  'write.parquet.compression-codec' = 'zstd'
);

