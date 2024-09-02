"""
Copyright (C) 2017-2024 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
"""

import asyncio
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
        self.partition_cols = partition_cols or ["exchange", "symbol", "dt"]
        self.optimize_interval = optimize_interval
        self.z_order_cols = z_order_cols or self._default_z_order_cols()
        self.time_travel = time_travel
        self.storage_options = storage_options or {}
        self.write_count = 0
        self.running = True
        self.numeric_type = numeric_type
        self.none_to = none_to
        # Validate configuration parameters
        self._validate_configuration()

    def _validate_configuration(self):
        if self.optimize_interval <= 0:
            raise ValueError("optimize_interval must be a positive integer")

        if not isinstance(self.partition_cols, list) or not all(
            isinstance(col, str) for col in self.partition_cols
        ):
            raise TypeError("partition_cols must be a list of strings")

        if not isinstance(self.z_order_cols, list) or not all(
            isinstance(col, str) for col in self.z_order_cols
        ):
            raise TypeError("z_order_cols must be a list of strings")

        if not isinstance(self.storage_options, dict):
            raise TypeError("storage_options must be a dictionary")

        if not isinstance(self.numeric_type, (type, str)):
            raise TypeError("numeric_type must be a type or a string")

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
                    LOG.info(f"Received {len(updates)} updates for processing.")
                    df = pd.DataFrame(updates)
                    self._convert_fields(df)
                    # Reorder columns to put exchange and symbol first
                    cols = ["exchange", "symbol"] + [
                        col for col in df.columns if col not in ["exchange", "symbol"]
                    ]
                    df = df[cols]

                    if self.custom_columns:
                        df = df.rename(columns=self.custom_columns)

                    await self._write_batch(df)

    def _convert_fields(self, df: pd.DataFrame):
        LOG.debug("Converting fields in DataFrame.")
        self._convert_datetime_fields(df)
        self._convert_category_fields(df)
        self._convert_int_fields(df)

    def _convert_datetime_fields(self, df: pd.DataFrame):
        LOG.debug("Converting datetime fields.")
        datetime_columns = ["timestamp", "receipt_timestamp"]
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit="ns").astype("datetime64[ms]")
        if "timestamp" in df.columns:
            df["dt"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    def _convert_category_fields(self, df: pd.DataFrame):
        LOG.debug("Converting category fields.")
        category_columns = [
            "exchange",
            "symbol",
            "side",
            "type",
            "status",
            "currency",
            "liquidity",
        ]
        for col in category_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")

    def _convert_int_fields(self, df: pd.DataFrame):
        LOG.debug("Converting integer fields.")
        int_columns = ["id", "trade_id", "trades"]
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].astype("int64")

    async def _write_batch(self, df: pd.DataFrame):
        if df.empty:
            LOG.warning("DataFrame is empty. Skipping write operation.")
            return

        # Ensure all partition columns are present in the DataFrame
        for col in self.partition_cols:
            if col not in df.columns:
                if col == "exchange" or col == "symbol":
                    df[col] = ""  # Default to empty string for categorical columns
                elif col == "dt":
                    df[col] = pd.Timestamp.min.strftime(
                        "%Y-%m-%d"
                    )  # Default to min date for date columns
                else:
                    df[col] = 0  # Default to 0 for numeric columns

        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                LOG.info(
                    f"Attempting to write batch to Delta Lake (Attempt {attempt + 1}/{max_retries})."
                )
                # Debug output the schema of the DataFrame
                LOG.debug(f"DataFrame schema:\n{df.dtypes}")

                # Convert timestamp columns to datetime64[ms]
                timestamp_columns = df.select_dtypes(include=["datetime64"]).columns
                for col in timestamp_columns:
                    df[col] = df[col].astype("datetime64[ms]")

                # Convert numeric columns to the specified numeric type
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    df[col] = df[col].astype(self.numeric_type)

                # Handle null values
                df = self._handle_null_values(df)

                LOG.info(
                    f"Writing batch of {len(df)} records to {self.delta_table_path}"
                )
                # Debug output the schema of the DataFrame
                LOG.debug(f"DataFrame schema before write:\n{df.dtypes}")

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

                LOG.info("Batch write successful.")
                break  # Exit the retry loop if write is successful

            except Exception as e:
                # When error is related to timestamp, print the schema of the DataFrame
                LOG.error(f"DataFrame schema:\n{df.dtypes}")

                LOG.error(
                    f"Error writing to Delta Lake on attempt {attempt + 1}/{max_retries}: {e}"
                )

                if attempt < max_retries - 1:
                    LOG.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    LOG.error(
                        "Max retries reached. Failed to write batch to Delta Lake."
                    )

    def _handle_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.none_to is not None:
            return df.fillna(self.none_to)
        else:
            # Replace None with appropriate default values based on column type
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].fillna(
                        ""
                    )  # Replace None with empty string for object columns
                elif df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(
                        0
                    )  # Replace None with 0 for numeric columns
                elif df[col].dtype == "bool":
                    df[col] = df[col].fillna(
                        False
                    )  # Replace None with False for boolean columns
                elif df[col].dtype == "datetime64[ms]":
                    df[col] = df[col].fillna(
                        pd.Timestamp.min.astype("datetime64[ms]")
                    )  # Replace None with minimum timestamp for datetime columns
            return df

    async def _optimize_table(self):
        LOG.info(f"Running OPTIMIZE on table {self.delta_table_path}")
        dt = DeltaTable(self.delta_table_path, storage_options=self.storage_options)
        dt.optimize.compact()
        if self.z_order_cols:
            dt.optimize.z_order(self.z_order_cols)
        LOG.info("OPTIMIZE operation completed.")

    def _update_metadata(self):
        dt = DeltaTable(self.delta_table_path, storage_options=self.storage_options)
        LOG.info(f"Updating metadata for time travel. Current version: {dt.version()}")

    async def stop(self):
        LOG.info("Stopping DeltaLakeCallback writer.")
        self.running = False

    def get_version(self, timestamp: Optional[int] = None) -> Optional[int]:
        if self.time_travel:
            dt = DeltaTable(self.delta_table_path, storage_options=self.storage_options)
            if timestamp:
                version = dt.version_at_timestamp(timestamp)
                LOG.info(f"Retrieved version {version} for timestamp {timestamp}.")
                return version
            else:
                version = dt.version()
                LOG.info(f"Retrieved current version {version}.")
                return version
        else:
            LOG.warning("Time travel is not enabled for this table")
            return None


class TradeDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = TRADES
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - dt: string
    - exchange: category
    - symbol: category
    - id: int64 (nullable)
    - side: category
    - amount: float64
    - price: float64
    - type: category (nullable)
    - trade_id: int64
    """


class FundingDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = FUNDING
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - dt: string
    - exchange: category
    - symbol: category
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
    - dt: string
    - exchange: category
    - symbol: category
    - bid: float64
    - ask: float64
    """


class OpenInterestDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = OPEN_INTEREST
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - dt: string
    - exchange: category
    - symbol: category
    - open_interest: float64
    """


class LiquidationsDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = LIQUIDATIONS
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - dt: string
    - exchange: category
    - symbol: category
    - side: category
    - quantity: float64
    - price: float64
    - id: int64
    - status: category
    """


class BookDeltaLake(DeltaLakeCallback, BackendBookCallback):
    default_key = "book"
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - dt: string
    - exchange: category
    - symbol: category
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
    - dt: string
    - exchange: category
    - symbol: category
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
    - dt: string
    - exchange: category
    - symbol: category
    - id: int64
    - client_order_id: string (nullable)
    - side: category
    - status: category
    - type: category
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
    - dt: string
    - exchange: category
    - currency: category
    - type: category
    - status: category
    - amount: float64
    """


class BalancesDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = BALANCES
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - dt: string
    - exchange: category
    - currency: category
    - balance: float64
    - reserved: float64 (nullable)
    """


class FillsDeltaLake(DeltaLakeCallback, BackendCallback):
    default_key = FILLS
    """
    Schema:
    - timestamp: datetime64[ns] (from 'date' column)
    - receipt_timestamp: datetime64[ns]
    - dt: string
    - exchange: category
    - symbol: category
    - price: float64
    - amount: float64
    - side: category
    - fee: float64 (nullable)
    - id: int64
    - order_id: int64
    - liquidity: category
    - type: category
    - account: string (nullable)
    """
