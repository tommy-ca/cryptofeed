'''
Copyright (C) 2017-2021  Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
import pandas as pd
from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.schema import Schema
from pyiceberg.types import (
    TimestampType,
    StringType,
    FloatType,
    LongType,
    NestedField
)

from cryptofeed.backends.backend import BackendCallback, BackendQueueProto
from cryptofeed.defines import (
    BALANCES, CANDLES, FILLS, FUNDING, OPEN_INTEREST, ORDER_INFO, TICKER, TRADES, LIQUIDATIONS, TRANSACTIONS
)


SCHEMA_MAP = {
    TRADES: Schema(
        NestedField(name="exchange", field_id=1, field_type=StringType(), is_required=True),
        NestedField(name="symbol", field_id=2, field_type=StringType(), is_required=True),
        NestedField(name="timestamp", field_id=3, field_type=TimestampType(), is_required=True),
        NestedField(name="order_id", field_id=4, field_type=StringType(), is_required=True),
        NestedField(name="feed", field_id=5, field_type=StringType(), is_required=True),
        NestedField(name="side", field_id=6, field_type=StringType(), is_required=True),
        NestedField(name="amount", field_id=7, field_type=FloatType(), is_required=True),
        NestedField(name="price", field_id=8, field_type=FloatType(), is_required=True),
        NestedField(name="total", field_id=9, field_type=FloatType(), is_required=True),
        NestedField(name="receipt_timestamp", field_id=10, field_type=TimestampType(), is_required=True),
    ),
    TICKER: Schema(
        NestedField(name="exchange", field_id=1, field_type=StringType(), is_required=True),
        NestedField(name="symbol", field_id=2, field_type=StringType(), is_required=True),
        NestedField(name="timestamp", field_id=3, field_type=TimestampType(), is_required=True),
        NestedField(name="feed", field_id=4, field_type=StringType(), is_required=True),
        NestedField(name="bid", field_id=5, field_type=FloatType(), is_required=True),
        NestedField(name="ask", field_id=6, field_type=FloatType(), is_required=True),
        NestedField(name="receipt_timestamp", field_id=7, field_type=TimestampType(), is_required=True),
    ),
    FUNDING: Schema(
        NestedField(name="exchange", field_id=1, field_type=StringType(), is_required=True),
        NestedField(name="symbol", field_id=2, field_type=StringType(), is_required=True),
        NestedField(name="timestamp", field_id=3, field_type=TimestampType(), is_required=True),
        NestedField(name="feed", field_id=4, field_type=StringType(), is_required=True),
        NestedField(name="delta", field_id=5, field_type=FloatType(), is_required=False),
        NestedField(name="rate", field_id=6, field_type=FloatType(), is_required=False),
        NestedField(name="predicted_rate", field_id=7, field_type=FloatType(), is_required=False),
        NestedField(name="next_funding_time", field_id=8, field_type=TimestampType(), is_required=False),
        NestedField(name="mark_price", field_id=9, field_type=FloatType(), is_required=False),
        NestedField(name="receipt_timestamp", field_id=10, field_type=TimestampType(), is_required=True),
    ),
    OPEN_INTEREST: Schema(
        NestedField(name="exchange", field_id=1, field_type=StringType(), is_required=True),
        NestedField(name="symbol", field_id=2, field_type=StringType(), is_required=True),
        NestedField(name="timestamp", field_id=3, field_type=TimestampType(), is_required=True),
        NestedField(name="feed", field_id=4, field_type=StringType(), is_required=True),
        NestedField(name="open_interest", field_id=5, field_type=FloatType(), is_required=True),
        NestedField(name="receipt_timestamp", field_id=6, field_type=TimestampType(), is_required=True),
    ),
    LIQUIDATIONS: Schema(
        NestedField(name="exchange", field_id=1, field_type=StringType(), is_required=True),
        NestedField(name="symbol", field_id=2, field_type=StringType(), is_required=True),
        NestedField(name="timestamp", field_id=3, field_type=TimestampType(), is_required=True),
        NestedField(name="feed", field_id=4, field_type=StringType(), is_required=True),
        NestedField(name="side", field_id=5, field_type=StringType(), is_required=True),
        NestedField(name="quantity", field_id=6, field_type=FloatType(), is_required=True),
        NestedField(name="price", field_id=7, field_type=FloatType(), is_required=False),
        NestedField(name="order_id", field_id=8, field_type=StringType(), is_required=False),
        NestedField(name="status", field_id=9, field_type=StringType(), is_required=False),
        NestedField(name="receipt_timestamp", field_id=10, field_type=TimestampType(), is_required=True),
    ),
    CANDLES: Schema(
        NestedField(name="exchange", field_id=1, field_type=StringType(), is_required=True),
        NestedField(name="symbol", field_id=2, field_type=StringType(), is_required=True),
        NestedField(name="timestamp", field_id=3, field_type=TimestampType(), is_required=True), # start time
        NestedField(name="feed", field_id=4, field_type=StringType(), is_required=True),
        NestedField(name="period", field_id=5, field_type=StringType(), is_required=True),
        NestedField(name="open", field_id=6, field_type=FloatType(), is_required=True),
        NestedField(name="high", field_id=7, field_type=FloatType(), is_required=True),
        NestedField(name="low", field_id=8, field_type=FloatType(), is_required=True),
        NestedField(name="close", field_id=9, field_type=FloatType(), is_required=True),
        NestedField(name="volume", field_id=10, field_type=FloatType(), is_required=True),
        NestedField(name="num_trades", field_id=11, field_type=LongType(), is_required=False), # Not always present
        NestedField(name="vwap", field_id=12, field_type=FloatType(), is_required=False), # Not always present
        NestedField(name="receipt_timestamp", field_id=13, field_type=TimestampType(), is_required=True),
    ),
    ORDER_INFO: Schema(
        NestedField(name="exchange", field_id=1, field_type=StringType(), is_required=True),
        NestedField(name="symbol", field_id=2, field_type=StringType(), is_required=True),
        NestedField(name="timestamp", field_id=3, field_type=TimestampType(), is_required=True),
        NestedField(name="feed", field_id=4, field_type=StringType(), is_required=True),
        NestedField(name="order_id", field_id=5, field_type=StringType(), is_required=True),
        NestedField(name="client_order_id", field_id=6, field_type=StringType(), is_required=False),
        NestedField(name="side", field_id=7, field_type=StringType(), is_required=True),
        NestedField(name="order_type", field_id=8, field_type=StringType(), is_required=True),
        NestedField(name="price", field_id=9, field_type=FloatType(), is_required=False),
        NestedField(name="amount", field_id=10, field_type=FloatType(), is_required=True),
        NestedField(name="amount_filled", field_id=11, field_type=FloatType(), is_required=False),
        NestedField(name="remaining", field_id=12, field_type=FloatType(), is_required=False),
        NestedField(name="trade_id", field_id=13, field_type=StringType(), is_required=False),
        NestedField(name="status", field_id=14, field_type=StringType(), is_required=True),
        NestedField(name="receipt_timestamp", field_id=15, field_type=TimestampType(), is_required=True),
    ),
    FILLS: Schema(
        NestedField(name="exchange", field_id=1, field_type=StringType(), is_required=True),
        NestedField(name="symbol", field_id=2, field_type=StringType(), is_required=True),
        NestedField(name="timestamp", field_id=3, field_type=TimestampType(), is_required=True),
        NestedField(name="feed", field_id=4, field_type=StringType(), is_required=True),
        NestedField(name="order_id", field_id=5, field_type=StringType(), is_required=True),
        NestedField(name="trade_id", field_id=6, field_type=StringType(), is_required=True),
        NestedField(name="side", field_id=7, field_type=StringType(), is_required=True),
        NestedField(name="price", field_id=8, field_type=FloatType(), is_required=True),
        NestedField(name="amount", field_id=9, field_type=FloatType(), is_required=True),
        NestedField(name="fee", field_id=10, field_type=FloatType(), is_required=False),
        NestedField(name="fee_currency", field_id=11, field_type=StringType(), is_required=False),
        NestedField(name="receipt_timestamp", field_id=12, field_type=TimestampType(), is_required=True),
    ),
    BALANCES: Schema(
        NestedField(name="exchange", field_id=1, field_type=StringType(), is_required=True),
        NestedField(name="timestamp", field_id=2, field_type=TimestampType(), is_required=True),
        NestedField(name="feed", field_id=3, field_type=StringType(), is_required=True),
        NestedField(name="asset", field_id=4, field_type=StringType(), is_required=True),
        NestedField(name="balance", field_id=5, field_type=FloatType(), is_required=True),
        NestedField(name="reserved", field_id=6, field_type=FloatType(), is_required=False),
        NestedField(name="receipt_timestamp", field_id=7, field_type=TimestampType(), is_required=True),
    ),
    TRANSACTIONS: Schema(
        NestedField(name="exchange", field_id=1, field_type=StringType(), is_required=True),
        NestedField(name="timestamp", field_id=2, field_type=TimestampType(), is_required=True),
        NestedField(name="feed", field_id=3, field_type=StringType(), is_required=True),
        NestedField(name="transaction_id", field_id=4, field_type=StringType(), is_required=True), # exchange specific id for the transaction
        NestedField(name="asset", field_id=5, field_type=StringType(), is_required=True),
        NestedField(name="type", field_id=6, field_type=StringType(), is_required=True), # deposit or withdrawal
        NestedField(name="amount", field_id=7, field_type=FloatType(), is_required=True),
        NestedField(name="address", field_id=8, field_type=StringType(), is_required=False), # address for the transaction
        NestedField(name="status", field_id=9, field_type=StringType(), is_required=True), # status of the transaction
        NestedField(name="receipt_timestamp", field_id=10, field_type=TimestampType(), is_required=True),
    )
}


class IcebergCallback(BackendQueueProto):
    def __init__(self, path, gcs_project_id=None, user=None, key=None, numeric_type=float, **kwargs):
        """
        path: str
            Passes to pyiceberg.load_catalog. The s3 or gcs bucket, or local path to the catalog
        gcs_project_id: str
            If using GCS, the project ID. Creds are loaded from env variables
        key: str
            The data type, i.e. trades, funding, etc. This will be the table name
        """
        super().__init__(**kwargs)
        self.path = path
        self.key = key if key else self.default_key
        self.numeric_type = numeric_type
        self.none_to = None  # Iceberg handles None types directly
        self.running = True
        self.base_table_name = self.key
        self.schema = SCHEMA_MAP[self.key]

        if path.startswith("s3://"):
            self.catalog = load_catalog(
                self.path,
                **{
                    "s3.region": kwargs.get("s3_region", "us-west-2"), # TODO make configurable via kwargs
                    "py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO",
                    "s3.access-key-id": kwargs.get("s3_access_key_id"), # if None, anonymous
                    "s3.secret-access-key": kwargs.get("s3_secret_access_key"), # if None, anonymous
                }
            )
        elif path.startswith("gcs://"):
            self.catalog = load_catalog(
                self.path,
                **{
                    "py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO",
                    "gcs.project-id": gcs_project_id, # if None, anonymous
                    "gcs.token": kwargs.get("gcs_token") # if None, anonymous
                }
            )
        else: # local fs
            self.catalog = load_catalog(self.path)


    async def writer(self):
        while self.running:
            async with self.read_queue() as updates:
                if not updates:
                    continue

                df = pd.DataFrame(updates)

                # Convert timestamp columns to datetime64[ns] before converting to Iceberg's TimestampType
                for col in ['timestamp', 'receipt_timestamp', 'next_funding_time']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], unit='s')


                # Create table if it doesn't exist
                table_name = f"{self.path.split('/')[-1]}.{self.base_table_name}" # Use warehouse.tablename format for table name
                try:
                    table = self.catalog.load_table(table_name)
                except NoSuchTableError:
                    table = self.catalog.create_table(table_name, schema=self.schema)

                table.append(df)


class TradeIceberg(IcebergCallback, BackendCallback):
    default_key = TRADES


class FundingIceberg(IcebergCallback, BackendCallback):
    default_key = FUNDING


class TickerIceberg(IcebergCallback, BackendCallback):
    default_key = TICKER


class OpenInterestIceberg(IcebergCallback, BackendCallback):
    default_key = OPEN_INTEREST


class LiquidationsIceberg(IcebergCallback, BackendCallback):
    default_key = LIQUIDATIONS


class CandlesIceberg(IcebergCallback, BackendCallback):
    default_key = CANDLES


class OrderInfoIceberg(IcebergCallback, BackendCallback):
    default_key = ORDER_INFO


class TransactionsIceberg(IcebergCallback, BackendCallback):
    default_key = TRANSACTIONS


class BalancesIceberg(IcebergCallback, BackendCallback):
    default_key = BALANCES


class FillsIceberg(IcebergCallback, BackendCallback):
    default_key = FILLS
