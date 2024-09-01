"""
Copyright (C) 2017-2024 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from deltalake import DeltaTable, write_deltalake

from cryptofeed.backends.backend import (
    BackendBookCallback,
    BackendCallback,
    BackendQueue,
)
from cryptofeed.defines import (
    BALANCES,
    CANDLES,
    FILLS,
    FUNDING,
    LIQUIDATIONS,
    OPEN_INTEREST,
    ORDER_INFO,
    TICKER,
    TRADES,
    TRANSACTIONS,
)


LOG = logging.getLogger("feedhandler")


class DeltaLakeCallback(BackendQueue):
    def __init__(
        self,
        base_path: str,
        key: Optional[str] = None,
        custom_columns: Optional[Dict[str, str]] = None,
        partition_cols: Optional[List[str]] = None,
        optimize_interval: int = 100,
        z_order_cols: Optional[List[str]] = None,
        time_travel: bool = False,
        storage_options: Optional[Dict[str, Any]] = None,
        numeric_type: Union[type, str] = float,
        none_to: Any = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.key = key or self.default_key
        self.base_path = base_path
        self.delta_table_path = f"{self.base_path}/{self.key}"
        self.custom_columns = custom_columns or {}
        self.partition_cols = partition_cols or [
            "exchange",
            "symbol",
            "year",
            "month",
            "day",
        ]
        self.optimize_interval = optimize_interval
        self.z_order_cols = z_order_cols or self._default_z_order_cols()
        self.time_travel = time_travel
        self.storage_options = storage_options or {}
        self.write_count = 0
        self.running = True

        if optimize_interval <= 0:
            raise ValueError("optimize_interval must be a positive integer")

        if not isinstance(self.partition_cols, list):
            raise TypeError("partition_cols must be a list of strings")

        if not isinstance(self.z_order_cols, list):
            raise TypeError("z_order_cols must be a list of strings")

        self.numeric_type = numeric_type
        self.none_to = none_to

    def _default_z_order_cols(self) -> List[str]:
        common_cols = ["timestamp"]
        data_specific_cols = {
            TRADES: ["price", "amount"],
            FUNDING: ["rate"],
            TICKER: ["bid", "ask"],
            OPEN_INTEREST: ["open_interest"],
            LIQUIDATIONS: ["quantity", "price"],
            "book": [],  # Book data is typically queried by timestamp and symbol
            CANDLES: ["open", "close", "high", "low"],
            ORDER_INFO: ["status", "price", "amount"],
            TRANSACTIONS: ["type", "amount"],
            BALANCES: ["balance"],
            FILLS: ["price", "amount"],
        }
        z_order_cols = common_cols + data_specific_cols.get(self.key, [])
        # Remove any columns that are already in partition_cols
        return [col for col in z_order_cols if col not in self.partition_cols]

    async def writer(self):
        while self.running:
            async with self.read_queue() as updates:
                if updates:
                    df = pd.DataFrame(updates)
                    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
                    df["receipt_timestamp"] = pd.to_datetime(
                        df["receipt_timestamp"], unit="s"
                    )
                    df["year"], df["month"], df["day"] = (
                        df["date"].dt.year,
                        df["date"].dt.month,
                        df["date"].dt.day,
                    )

                    # Reorder columns to put exchange and symbol first
                    cols = ["exchange", "symbol"] + [
                        col for col in df.columns if col not in ["exchange", "symbol"]
                    ]
                    df = df[cols]

                    if self.custom_columns:
                        df = df.rename(columns=self.custom_columns)

                    await self._write_batch(df)

    async def _write_batch(self, df: pd.DataFrame):
        if df.empty:
            return

        try:
            # Convert timestamp columns from ns to us
            timestamp_columns = df.select_dtypes(include=["datetime64"]).columns
            for col in timestamp_columns:
                df[col] = df[col].astype("datetime64[us]")

            # Convert numeric columns to the specified numeric type
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = df[col].astype(self.numeric_type)

            # Handle null values
            if self.none_to is not None:
                df = df.fillna(self.none_to)
            else:
                # Replace None with appropriate default values based on column type
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna('')  # Replace None with empty string for object columns
                    elif df[col].dtype in ['float64', 'int64']:
                        df[col] = df[col].fillna(0)  # Replace None with 0 for numeric columns
                    elif df[col].dtype == 'bool':
                        df[col] = df[col].fillna(False)  # Replace None with False for boolean columns
                    elif df[col].dtype == 'datetime64[us]':
                        df[col] = df[col].fillna(pd.Timestamp.min)  # Replace None with minimum timestamp for datetime columns

            LOG.info(f"Writing batch of {len(df)} records to {self.delta_table_path}")
            write_deltalake(
                self.delta_table_path,
                df,
                mode="append",
                partition_by=self.partition_cols,
                schema_mode="merge",
                storage_options=self.storage_options,
            )
            self.write_count += 1

            if self.write_count % self.optimize_interval == 0:
                await self._optimize_table()

            if self.time_travel:
                self._update_metadata()

        except Exception as e:
            LOG.error(f"Error writing to Delta Lake: {e}")

    async def _optimize_table(self):
        LOG.info(f"Running OPTIMIZE on table {self.delta_table_path}")
        dt = DeltaTable(self.delta_table_path, storage_options=self.storage_options)
        dt.optimize.compact()
        if self.z_order_cols:
            dt.optimize.z_order(self.z_order_cols)

    def _update_metadata(self):
        dt = DeltaTable(self.delta_table_path, storage_options=self.storage_options)
        LOG.info(f"Updating metadata for time travel. Current version: {dt.version()}")

    async def stop(self):
        self.running = False

    def get_version(self, timestamp: Optional[int] = None) -> Optional[int]:
        if self.time_travel:
            dt = DeltaTable(self.delta_table_path, storage_options=self.storage_options)
            if timestamp:
                return dt.version_at_timestamp(timestamp)
            else:
                return dt.version()
        else:
            LOG.warning("Time travel is not enabled for this table")
            return None


class TradeDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = TRADES
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - year: int32
    - month: int32
    - day: int32
    - exchange: string
    - symbol: string
    - id: string (nullable)
    - side: string
    - amount: float64
    - price: float64
    - type: string (nullable)
    """


class FundingDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = FUNDING
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - year: int32
    - month: int32
    - day: int32
    - exchange: string
    - symbol: string
    - mark_price: float64 (nullable)
    - rate: float64
    - next_funding_time: datetime64[ns] (nullable)
    - predicted_rate: float64 (nullable)
    """


class TickerDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = TICKER
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - year: int32
    - month: int32
    - day: int32
    - exchange: string
    - symbol: string
    - bid: float64
    - ask: float64
    """


class OpenInterestDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = OPEN_INTEREST
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - year: int32
    - month: int32
    - day: int32
    - exchange: string
    - symbol: string
    - open_interest: float64
    """


class LiquidationsDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = LIQUIDATIONS
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - year: int32
    - month: int32
    - day: int32
    - exchange: string
    - symbol: string
    - side: string
    - quantity: float64
    - price: float64
    - id: string
    - status: string
    """


class BookDeltaLake(DeltaLakeCallback, BackendBookCallback):
    default_key = "book"
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - year: int32
    - month: int32
    - day: int32
    - exchange: string
    - symbol: string
    - delta: dict (nullable, contains 'bid' and 'ask' updates)
    - book: dict (contains full order book snapshot when available)
    """

    def __init__(self, *args, snapshots_only=False, snapshot_interval=1000, **kwargs):
        self.snapshots_only = snapshots_only
        self.snapshot_interval = snapshot_interval
        self.snapshot_count = defaultdict(int)
        super().__init__(*args, **kwargs)


class CandlesDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = CANDLES
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - year: int32
    - month: int32
    - day: int32
    - exchange: string
    - symbol: string
    - start: datetime64[ns]
    - stop: datetime64[ns]
    - interval: string
    - trades: int64 (nullable)
    - open: float64
    - close: float64
    - high: float64
    - low: float64
    - volume: float64
    - closed: bool (nullable)
    """


class OrderInfoDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = ORDER_INFO
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - year: int32
    - month: int32
    - day: int32
    - exchange: string
    - symbol: string
    - id: string
    - client_order_id: string (nullable)
    - side: string
    - status: string
    - type: string
    - price: float64
    - amount: float64
    - remaining: float64 (nullable)
    - account: string (nullable)
    """


class TransactionsDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = TRANSACTIONS
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - year: int32
    - month: int32
    - day: int32
    - exchange: string
    - currency: string
    - type: string
    - status: string
    - amount: float64
    """


class BalancesDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = BALANCES
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - year: int32
    - month: int32
    - day: int32
    - exchange: string
    - currency: string
    - balance: float64
    - reserved: float64 (nullable)
    """


class FillsDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = FILLS
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - year: int32
    - month: int32
    - day: int32
    - exchange: string
    - symbol: string
    - price: float64
    - amount: float64
    - side: string
    - fee: float64 (nullable)
    - id: string
    - order_id: string
    - liquidity: string
    - type: string
    - account: string (nullable)
    """
