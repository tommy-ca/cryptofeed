-- Trades table derived from cryptofeed.v1.Trade
-- Adjust <catalog>.<db> as needed
CREATE TABLE IF NOT EXISTS <catalog>.<db>.trades (
  exchange            STRING,
  symbol              STRING,
  base                STRING,
  quote               STRING,
  instrument_type     STRING,
  side                STRING,
  amount              DECIMAL(38, 18),
  price               DECIMAL(38, 18),
  trade_id            STRING,
  trade_type          STRING,
  event_ts            TIMESTAMP,
  receipt_ts          TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(event_ts), exchange, symbol)
TBLPROPERTIES (
  'format-version' = '2',
  'write.format.default' = 'parquet',
  'write.parquet.compression-codec' = 'zstd'
);

