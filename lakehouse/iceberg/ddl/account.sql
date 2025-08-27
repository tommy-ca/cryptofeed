-- Account data tables

-- Order info from cryptofeed.v1.OrderInfo
CREATE TABLE IF NOT EXISTS <catalog>.<db>.order_info (
  exchange       STRING,
  symbol         STRING,
  order_id       STRING,
  client_order_id STRING,
  side           STRING,
  status         STRING,
  order_type     STRING,
  price          DECIMAL(38, 18),
  amount         DECIMAL(38, 18),
  remaining      DECIMAL(38, 18),
  account        STRING,
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

-- Balances from cryptofeed.v1.Balance
CREATE TABLE IF NOT EXISTS <catalog>.<db>.balances (
  exchange       STRING,
  currency       STRING,
  balance        DECIMAL(38, 18),
  reserved       DECIMAL(38, 18),
  event_ts       TIMESTAMP,
  receipt_ts     TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(event_ts), exchange, currency)
TBLPROPERTIES (
  'format-version' = '2',
  'write.format.default' = 'parquet',
  'write.parquet.compression-codec' = 'zstd'
);

-- Fills from cryptofeed.v1.Fill
CREATE TABLE IF NOT EXISTS <catalog>.<db>.fills (
  exchange       STRING,
  symbol         STRING,
  side           STRING,
  amount         DECIMAL(38, 18),
  price          DECIMAL(38, 18),
  fee            DECIMAL(38, 18),
  fee_currency   STRING,
  fill_id        STRING,
  order_id       STRING,
  liquidity      STRING,
  order_type     STRING,
  account        STRING,
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

-- Positions from cryptofeed.v1.Position
CREATE TABLE IF NOT EXISTS <catalog>.<db>.positions (
  exchange       STRING,
  symbol         STRING,
  position       DECIMAL(38, 18),
  entry_price    DECIMAL(38, 18),
  side           STRING,
  unrealized_pnl DECIMAL(38, 18),
  margin         DECIMAL(38, 18),
  leverage       DECIMAL(38, 18),
  account        STRING,
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

-- Transactions from cryptofeed.v1.Transaction
CREATE TABLE IF NOT EXISTS <catalog>.<db>.transactions (
  exchange       STRING,
  currency       STRING,
  tx_type        STRING,
  status         STRING,
  amount         DECIMAL(38, 18),
  tx_id          STRING,
  fee            DECIMAL(38, 18),
  address        STRING,
  tx_hash        STRING,
  event_ts       TIMESTAMP,
  receipt_ts     TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(event_ts), exchange, currency)
TBLPROPERTIES (
  'format-version' = '2',
  'write.format.default' = 'parquet',
  'write.parquet.compression-codec' = 'zstd'
);

