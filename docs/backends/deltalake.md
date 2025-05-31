# Delta Lake Backend

The Delta Lake backend for Cryptofeed allows you. to store various types of cryptocurrency market data directly into [Delta Lake](https://delta.io/) tables. Delta Lake is an open-source storage layer that brings ACID transactions, scalable metadata handling, and unifies streaming and batch data processing on top of existing data lakes (e.g., S3, HDFS, local file systems).

This backend provides robust data storage with features like schema enforcement, schema evolution, partitioning, and table optimization (compaction and Z-ordering).

## `DeltaLakeCallback` Parameters

The primary callback class `DeltaLakeCallback` is the base for all specific data type writers. It handles the connection, data transformation, batching, and writing to Delta Lake.

Here are the parameters for its constructor:

-   `base_path` (str):
    The root directory where your Delta Lake tables will be stored. Each distinct data feed (e.g., trades, L2 books) will reside in a subdirectory within this `base_path`. For example, if `base_path` is `/data/crypto_deltalake`, trade data might be in `/data/crypto_deltalake/trades_table/`.
    *This parameter is mandatory.*

-   `key` (Optional[str], default: None):
    The name for the specific data table, which also serves as the subdirectory name under `base_path`. If not provided, it defaults to the `default_key` defined in the specific derived callback class (e.g., `TRADES` for `TradeDeltaLake`).

-   `custom_columns` (Optional[Dict[str, str]], default: None):
    A dictionary allowing you to rename columns from the standard Cryptofeed output. Keys are the desired new column names, and values are the original Cryptofeed column names. Example: `{'trade_identifier': 'id'}`.

-   `partition_cols` (Optional[List[str]], default: `["exchange", "symbol", "dt"]`):
    A list of column names by which the data in the Delta table will be partitioned on disk. Effective partitioning can significantly improve query performance by pruning files that don't match query predicates. The default partitioning includes `exchange`, `symbol`, and `dt` (a date string derived from the event timestamp).

-   `optimize_interval` (int, default: 1000):
    Specifies how often (in terms of number of write operations/flushes) to run table optimization tasks (compaction and Z-ordering, if `z_order_cols` are defined). For example, an interval of 1000 means optimization will run after every 1000 batches are written.

-   `z_order_cols` (Optional[List[str]], default: None):
    A list of columns to use for Z-ordering when an OPTIMIZE operation is performed. Z-ordering is a technique to colocate related information in the same set of files, which can drastically improve query speed, especially for columns with high cardinality that are often used in filters. If `None`, a default set of Z-order columns is chosen based on the data type (`key`).

-   `time_travel` (bool, default: True):
    If `True`, enables metadata updates necessary for Delta Lake's time travel feature. This allows you to query previous versions of the table. (Note: The callback's `_update_metadata` method is currently a placeholder for more advanced metadata logging but table versioning is inherent to Delta Lake).

-   `storage_options` (Optional[Dict[str, Any]], default: None):
    A dictionary of parameters to pass to the underlying Delta Lake library, often used for configuring access to cloud storage systems like S3 (e.g., AWS access keys, region). Example: `{'AWS_ACCESS_KEY_ID': 'key', 'AWS_SECRET_ACCESS_KEY': 'secret'}`.

-   `numeric_type` (Union[type, str], default: `float`):
    This parameter is intended to specify a default numeric type for relevant fields (like price, amount). However, in the current implementation, type conversion is primarily handled by Pandas/PyArrow type inference or explicit `custom_dtypes`.

-   `none_to` (Any, default: None):
    A value to use for replacing `None` or `NaN` values in the data before writing. If `None`, type-specific defaults are applied (e.g., 0 for numeric columns, empty string for string columns).

-   `batch_size` (int, default: 10000):
    The maximum number of data records to accumulate in an in-memory batch before flushing them to the Delta Lake table. Larger batches can lead to larger, more optimized Parquet files but consume more memory.

-   `flush_interval` (float, default: 10.0):
    The maximum time in seconds to wait before an in-memory batch is flushed to disk, even if `batch_size` has not yet been reached. This ensures data is persisted in a timely manner.

-   `custom_transformations` (Optional[List[callable]], default: None):
    A list of user-defined functions that will be applied to the Pandas DataFrame just before it's written to Delta Lake. Each function should accept a DataFrame as input and can either modify it in-place or return a new DataFrame. This allows for flexible, custom data manipulation.

-   `schema_mode` (str, default: "append"):
    Defines how to handle schema differences when writing data.
    -   `"append"` (default behavior if not specified, though our callback explicitly sets "merge" for `write_deltalake`): New columns are not allowed. The schema of the data to be written must match the table's schema.
    -   `"overwrite"`: The table's schema will be overwritten with the schema of the new data. Use with caution as this can lead to data loss or corruption if schemas are incompatible.
    -   `"merge"`: The schema of the new data will be merged with the existing table schema. New columns found in the data can be added to the table schema (typically as nullable columns). This is the mode used by the callback when calling `write_deltalake`.

## Data Type Specific Classes

Cryptofeed provides specialized callback classes for different data types, inheriting from `DeltaLakeCallback`:

-   `TradeDeltaLake`
-   `FundingDeltaLake`
-   `TickerDeltaLake`
-   `OpenInterestDeltaLake`
-   `LiquidationsDeltaLake`
-   `BookDeltaLake` (handles L2 order book data)
-   `CandlesDeltaLake`
-   `OrderInfoDeltaLake`
-   `TransactionsDeltaLake`
-   `BalancesDeltaLake`
-   `FillsDeltaLake`

These classes automatically set the `default_key` (e.g., `TRADES` for `TradeDeltaLake`) and may define default schemas or transformations relevant to that data type.

**Example Usage:**

```python
from cryptofeed import FeedHandler
from cryptofeed.exchanges import Coinbase
from cryptofeed.defines import TRADES
from cryptofeed.backends.deltalake import TradeDeltaLake

# Configuration for the DeltaLake backend
delta_config = {
    'base_path': '/path/to/my_delta_lake_data/',  # Root directory for Delta tables
    'partition_cols': ['exchange', 'symbol', 'dt'], # Partition by exchange, symbol, and date
    'z_order_cols': ['timestamp', 'id'],          # Z-Order by timestamp and trade ID
    'optimize_interval': 500,                     # Optimize every 500 writes
    'batch_size': 5000,                           # Write every 5000 records
    'flush_interval': 30.0                        # Or every 30 seconds
}

def main():
    f = FeedHandler()

    # Add the Coinbase feed for BTC-USD trades
    # The TradeDeltaLake callback will store data into:
    # /path/to/my_delta_lake_data/trades/
    f.add_feed(Coinbase(symbols=['BTC-USD'], channels=[TRADES], callbacks=[TradeDeltaLake(**delta_config)]))

    f.run()

if __name__ == '__main__':
    main()
```

## Partitioning

Partitioning your Delta tables can greatly enhance query performance, especially when filtering on the partition columns. The `DeltaLakeCallback` defaults to partitioning by `exchange`, `symbol`, and `dt` (date derived from the timestamp).

-   **Strategy**: Choose partition columns that are frequently used in query `WHERE` clauses. For time-series data like market data, partitioning by date (`dt` or finer granularities like `year`, `month`) is common. Adding `exchange` and `symbol` helps isolate data for specific trading pairs.
-   **Benefits**: Delta Lake can skip reading entire directories (partitions) if they don't match the filter conditions, leading to faster queries and reduced data scanning.
-   **Caution**: Over-partitioning (using too many partition columns or columns with very high cardinality) can lead to too many small files and negatively impact performance. Find a balance based on your query patterns.

## Z-Ordering

Z-ordering is a technique that collocates related data within Parquet files. When you Z-order by certain columns, Delta Lake rearranges the data so that rows with similar values in those columns are physically stored close together.

-   **Impact**: This can significantly improve query performance for queries that filter or join on the Z-ordered columns, as Delta Lake can more effectively skip irrelevant data within files.
-   **Usage**: Specify `z_order_cols` in the `DeltaLakeCallback` configuration. The backend will periodically run `OPTIMIZE ZORDER BY (col1, col2, ...)` on the table.
-   **Defaults**: If `z_order_cols` is not provided, the callback attempts to choose sensible defaults based on the data type (e.g., `timestamp`, `symbol`, `price`, `amount` for trades).

## Schema Evolution

Delta Lake supports schema evolution, allowing you to change a table's schema as your data requirements evolve. The `DeltaLakeCallback` leverages this by using `schema_mode="merge"` when writing data via `deltalake.write_deltalake`.

-   **`schema_mode="merge"` (default for writes by this backend)**:
    -   If the data being written contains new columns not present in the table, these columns will be added to the table's schema as nullable columns.
    -   Existing rows will have null values for these new columns.
    -   This mode ensures that new fields in your data can be captured without interrupting data ingestion.
-   **`schema_mode` constructor parameter**:
    -   The `DeltaLakeCallback` also accepts a `schema_mode` parameter in its constructor (defaulting to "append" conceptually for its own DataFrame operations before writing, but the crucial part is how it interacts with `write_deltalake`).
    -   If you set `schema_mode='overwrite'` in the callback's constructor, this parameter will be passed to `write_deltalake`, instructing it to replace the table's schema with the schema of the current batch being written. This is a more drastic change and should be used carefully.

## Practical Example

For a runnable example demonstrating how to use this backend, please refer to the `examples/demo_deltalake.py` script included in the Cryptofeed repository. This script shows how to set up `FeedHandler` with `DeltaLakeCallback` for various data types.
