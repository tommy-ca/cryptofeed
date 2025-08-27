-- L2 aggregated order book deltas from cryptofeed.v1.BookDelta
-- Stores changes as an array of structs for bids and asks
CREATE TABLE IF NOT EXISTS <catalog>.<db>.l2_book_delta (
  exchange       STRING,
  symbol         STRING,
  bid_changes    ARRAY<STRUCT<price: DECIMAL(38, 18), size: DECIMAL(38, 18)>>,
  ask_changes    ARRAY<STRUCT<price: DECIMAL(38, 18), size: DECIMAL(38, 18)>>,
  sequence_number BIGINT,
  checksum       STRING,
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

