"""
Iceberg Backend for Cryptofeed

This backend publishes data from Cryptofeed to Apache Iceberg tables.
It buffers data in memory and writes it in batches to Iceberg.
"""
import asyncio
import logging
from collections import defaultdict
from typing import Dict, Any, List

import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NoSuchTableError, NamespaceNotFoundError

from cryptofeed.backends.backend import BackendQueue, BackendCallback, BackendBookCallback

LOG = logging.getLogger('feedhandler')


class IcebergCallback(BackendQueue):
    """
    Iceberg Backend Base Class

    Parameters
    ----------
    catalog_config : dict
        Configuration for the PyIceberg catalog.
        Example:
        {
            "name": "my_catalog", // Optional, for local tracking
            "uri": "http://rest-catalog-uri:8181", // For REST catalog
            "warehouse": "s3://my-bucket/warehouse/", // For REST/Hive
            // For S3 based catalogs (like REST with S3 warehouse, or Hive on S3)
            // "s3.endpoint": "http://minio:9000",
            // "s3.access-key-id": "YOUR_ACCESS_KEY",
            // "s3.secret-access-key": "YOUR_SECRET_KEY",
            // For Hive catalog
            // "metastore_uri": "thrift://hive-metastore:9083"
            // For SQL catalog (JDBC)
            // "uri": "jdbc:postgresql://user:password@host:port/database",
            // "jdbc.user": "user",
            // "jdbc.password": "password",
            // "jdbc.url": "jdbc:postgresql://host:port/database" (if not in uri)
        }
        The keys in catalog_config should match the keyword arguments for
        pyiceberg.catalog.load_catalog() or specific catalog properties.
    database_name : str, optional
        The default database/namespace in Iceberg where tables will be created/managed.
        Defaults to 'default'.
    table_prefix : str, optional
        A prefix for table names. The full table name will be <table_prefix>_<data_type_key>.
        Defaults to 'cryptofeed'.
    batch_size : int, optional
        Number of records to buffer in memory before writing to Iceberg.
        Defaults to 1000.
    pandas_kwargs : dict, optional
        Dictionary of keyword arguments to pass to pd.DataFrame.from_records when creating
        DataFrames from buffered records. Example: {'index': None}.
    **kwargs :
        Additional keyword arguments for BackendQueue.
    """
    def __init__(self,
                 catalog_config: dict,
                 database_name: str = 'default',
                 table_prefix: str = 'cryptofeed',
                 batch_size: int = 1000,
                 pandas_kwargs: dict = None,
                 **kwargs):
        super().__init__(**kwargs)
        if not catalog_config or 'uri' not in catalog_config and 'type' not in catalog_config : # Basic check
             LOG.warning("IcebergCallback: catalog_config 'uri' or 'type' is recommended for pyiceberg.load_catalog.")

        self.catalog_config = catalog_config
        self.catalog_name = catalog_config.get("name", "default_iceberg_catalog") # For internal reference

        self.database_name = database_name
        self.table_prefix = table_prefix
        self.batch_size = batch_size
        self.pandas_kwargs = pandas_kwargs if pandas_kwargs is not None else {}

        self._buffers = defaultdict(list)
        self._catalog_instance = None # Lazy loaded

    @property
    def catalog(self):
        if self._catalog_instance is None:
            LOG.info(f"IcebergBackend: Initializing catalog '{self.catalog_name}' with config: {self.catalog_config}")
            # pyiceberg.catalog.load_catalog expects properties without the 'name' key if it's passed as first arg
            # However, it's safer to pass all config as kwargs.
            # Let's try to make a copy and remove 'name' if it's not a standard pyiceberg property
            effective_catalog_config = self.catalog_config.copy()
            if "name" in effective_catalog_config and "name" not in load_catalog.__code__.co_varnames : # A bit hacky
                 # 'name' is often used as a local identifier, not a pyiceberg load_catalog arg itself
                 # unless it's a named catalog registered elsewhere.
                 # For direct loading, 'name' kwarg to load_catalog is for specific catalog types.
                 # We assume if 'name' is in config, it might be for our reference or a specific catalog's property.
                 # The most robust way is if pyiceberg's load_catalog ignores unknown kwargs, or if users
                 # provide only valid kwargs for their chosen catalog type.
                 # For now, we pass it as is, pyiceberg should handle it.
                 pass

            self._catalog_instance = load_catalog(**effective_catalog_config)
            LOG.info(f"IcebergBackend: Catalog '{self.catalog_name}' initialized.")
        return self._catalog_instance

    def _get_table_identifier(self, table_key: str) -> tuple[str, str]:
        """Generates the full table identifier (database, table_name)."""
        table_name = f"{self.table_prefix}_{table_key}"
        return (self.database_name, table_name)

    async def _write_batch(self, table_key: str, schema: pa.Schema):
        if not self._buffers[table_key]:
            return

        records_to_write = self._buffers[table_key][:self.batch_size]
        self._buffers[table_key] = self._buffers[table_key][self.batch_size:]

        identifier = self._get_table_identifier(table_key)

        LOG.debug(f"IcebergBackend: Preparing to write {len(records_to_write)} records to {identifier}")

        try:
            df = pd.DataFrame.from_records(records_to_write, **self.pandas_kwargs)
            # Ensure DataFrame schema matches PyArrow schema if possible (pyiceberg does some conversion)
            # For complex types like list<struct<...>>, direct conversion might need care.
            # PyIceberg's append should handle DataFrame to Arrow conversion.
        except Exception as e:
            LOG.error(f"IcebergBackend: Failed to create Pandas DataFrame for {identifier}: {e}")
            # Potentially requeue records or write to a dead-letter queue
            return

        try:
            try:
                table = self.catalog.load_table(identifier)
                LOG.debug(f"IcebergBackend: Loaded existing table {identifier}")
            except NoSuchTableError:
                LOG.info(f"IcebergBackend: Table {identifier} not found. Attempting to create.")
                # Ensure database/namespace exists before creating table
                try:
                    namespaces = self.catalog.list_namespaces() # Returns list of tuples
                    if (self.database_name,) not in namespaces and self.database_name != 'default':
                        LOG.info(f"IcebergBackend: Database {self.database_name} not found. Attempting to create.")
                        self.catalog.create_namespace(self.database_name)
                        LOG.info(f"IcebergBackend: Database {self.database_name} created.")
                except Exception as e_ns: # Broad exception as API varies
                    LOG.warning(f"IcebergBackend: Could not verify/create namespace {self.database_name}: {e_ns}. Assuming it might exist or creation is handled externally for some catalogs.")


                if schema is None:
                    LOG.error(f"IcebergBackend: Cannot create table {identifier} because schema is not defined for this callback type.")
                    return

                # TODO: Add PartitionSpec and SortOrder from callback instance if defined
                table = self.catalog.create_table(identifier, schema)
                LOG.info(f"IcebergBackend: Created new table {identifier} with schema: {schema}")

            # Append data
            table.append(df)
            LOG.info(f"IcebergBackend: Successfully appended {len(records_to_write)} records to {identifier}. Buffer for {table_key} has {len(self._buffers[table_key])} items left.")

        except NamespaceNotFoundError:
            LOG.error(f"IcebergBackend: Namespace (database) {self.database_name} not found for table {identifier}. Please create it or check catalog configuration.")
        except Exception as e:
            LOG.error(f"IcebergBackend: Error writing to Iceberg table {identifier}: {e}")
            # Consider re-adding records_to_write to the front of the buffer for retry,
            # or implementing a more sophisticated retry/dead-letter mechanism.
            # For now, data is lost if write fails after DataFrame creation.
            # self._buffers[table_key] = records_to_write + self._buffers[table_key] # Requeue


    async def writer(self):
        """
        Iceberg writer method.

        Consumes data from the queue, buffers it, and writes to Iceberg tables in batches.
        """
        while True:
            try:
                async with self.read_queue() as updates:
                    if not updates:
                        # Allow other tasks to run if queue is empty for a bit
                        await asyncio.sleep(0.01)
                        continue

                    for update in updates:
                        # 'data_type' key is expected from the base callback's __call__ method
                        # which wraps the raw data. This key is used to find the specific
                        # Iceberg callback instance (e.g., TradeIceberg) which holds the schema.
                        # However, the IcebergCallback itself is instantiated per data type.
                        # So, `self` here is already the specific type (e.g., TradeIceberg).
                        # The `default_table` and `schema` attributes will be on `self`.

                        table_key = self.default_table # Defined in subclasses
                        schema = self.schema         # Defined in subclasses

                        # The `update` here is the raw data dictionary (e.g., a trade dict)
                        # It needs to be enriched with exchange, symbol, receipt_timestamp if not already.
                        # The base BackendCallback.__call__ adds these.
                        # Let's assume `update` is the dict that needs to be written as a row.
                        self._buffers[table_key].append(update)

                        if len(self._buffers[table_key]) >= self.batch_size:
                            await self._write_batch(table_key, schema)

                # After processing a batch of updates from read_queue, check all buffers
                # This ensures data is written even if individual buffer sizes didn't reach batch_size
                # but the queue became empty.
                for table_key in list(self._buffers.keys()): # list() for safe iteration if modified
                    if self._buffers[table_key]: # If any data left after queue processing
                        schema = self.schema # Assuming self is the specific callback here
                        await self._write_batch(table_key, schema)

            except Exception as e:
                LOG.error(f"IcebergBackend: Unhandled exception in writer: {e}", exc_info=True)
                await asyncio.sleep(1) # Wait a bit before retrying the loop


# --- Data Type Specific Subclasses ---

class TradeIceberg(IcebergCallback, BackendCallback):
    default_table = 'trades'
    # PyArrow Schema for Trades
    schema = pa.schema([
        pa.field('exchange', pa.string(), nullable=False),
        pa.field('symbol', pa.string(), nullable=False),
        pa.field('timestamp', pa.float64(), nullable=False), # Cryptofeed uses float for timestamps
        pa.field('receipt_timestamp', pa.float64(), nullable=False),
        pa.field('side', pa.string(), nullable=False),
        pa.field('amount', pa.float64(), nullable=False),
        pa.field('price', pa.float64(), nullable=False),
        pa.field('id', pa.string()), # Exchange specific trade ID, can be null
        pa.field('type', pa.string()), # order type like market, limit. Can be null
    ])

    def __init__(self, *args, **kwargs):
        # Ensure 'id' is a valid pandas_kwarg if users want to exclude it from DataFrame index
        if 'pandas_kwargs' not in kwargs:
            kwargs['pandas_kwargs'] = {}
        if 'columns' not in kwargs['pandas_kwargs']: # Allow users to specify columns
            # Default columns to match schema, helps pd.DataFrame.from_records
             kwargs['pandas_kwargs']['columns'] = [f.name for f in self.schema]
        super().__init__(*args, **kwargs)


class TickerIceberg(IcebergCallback, BackendCallback):
    default_table = 'tickers'
    schema = pa.schema([
        pa.field('exchange', pa.string(), nullable=False),
        pa.field('symbol', pa.string(), nullable=False),
        pa.field('timestamp', pa.float64(), nullable=False),
        pa.field('receipt_timestamp', pa.float64(), nullable=False),
        pa.field('bid', pa.float64()),
        pa.field('ask', pa.float64()),
        # Add other common ticker fields if necessary, mark as nullable
        # pa.field('last', pa.float64()),
        # pa.field('volume_24h', pa.float64()),
    ])
    def __init__(self, *args, **kwargs):
        if 'pandas_kwargs' not in kwargs: kwargs['pandas_kwargs'] = {}
        if 'columns' not in kwargs['pandas_kwargs']:
             kwargs['pandas_kwargs']['columns'] = [f.name for f in self.schema]
        super().__init__(*args, **kwargs)


class BookIceberg(IcebergCallback, BackendBookCallback):
    default_table = 'orderbooks' # Or 'book_snapshots'

    # Schema for order book snapshots with bids and asks as lists of structs
    bid_ask_type = pa.struct([
        pa.field('price', pa.float64(), nullable=False),
        pa.field('size', pa.float64(), nullable=False)
    ])
    schema = pa.schema([
        pa.field('exchange', pa.string(), nullable=False),
        pa.field('symbol', pa.string(), nullable=False),
        pa.field('timestamp', pa.float64(), nullable=False),
        pa.field('receipt_timestamp', pa.float64(), nullable=False),
        pa.field('bids', pa.list_(bid_ask_type)),
        pa.field('asks', pa.list_(bid_ask_type)),
        pa.field('delta', pa.bool_(), nullable=True), # True if this row represents a delta, False/None for snapshot
        pa.field('sequence_number', pa.int64(), nullable=True) # For deltas/snapshots
    ])

    def __init__(self, *args, **kwargs):
        if 'pandas_kwargs' not in kwargs: kwargs['pandas_kwargs'] = {}
        if 'columns' not in kwargs['pandas_kwargs']:
             kwargs['pandas_kwargs']['columns'] = [f.name for f in self.schema]
        super().__init__(*args, **kwargs)

    # Override the process_new_data from BackendBookCallback to transform book data
    # The base BackendBookCallback's __call__ method calls self.process_new_data,
    # which then calls self.write(). self.write() (from BackendQueue) puts data on the queue.
    # We need to ensure the data put on the queue by BookIceberg is in a format
    # that the IcebergCallback.writer can turn into the list<struct> DataFrame.

    async def _format_book_data_for_iceberg(self, data: dict) -> dict:
        """
        Transforms book data from cryptofeed's internal format to a flat dictionary
        suitable for DataFrame creation with list<struct> for bids/asks.
        """
        # data['book'] contains book_t (Dict[Decimal, Decimal])
        # data['delta'] indicates if it's a delta or snapshot
        # For snapshots, data['book'] is the full book.
        # For deltas, data['delta'] contains the changes.
        # PyIceberg expects a list of dicts for list<struct> when creating from Pandas.

        bids_list = []
        asks_list = []

        is_delta = 'delta' in data and data['delta'] is not None

        if not is_delta and 'book' in data and data['book']: # Snapshot
            # data['book'] is a namedtuple Book with .bids and .asks
            # which are price_level_t objects (custom dicts)
            if hasattr(data['book'], 'bids'): # Check if it's the Book type
                 for price, size in data['book'].bids.items():
                    bids_list.append({'price': float(price), 'size': float(size)})
            if hasattr(data['book'], 'asks'):
                for price, size in data['book'].asks.items():
                    asks_list.append({'price': float(price), 'size': float(size)})
        elif is_delta: # Delta processing
            # For deltas, we'll store them similarly. The 'delta' field in schema will mark it.
            # The current schema doesn't fully represent deltas in a way that `table.append` can easily merge.
            # Iceberg itself supports merge/upsert but `table.append` is just append.
            # For simplicity with append, we'll record delta changes as lists too.
            # This might mean deltas are stored as "current state of changes" rather than "how to apply to previous state".
            # This part needs more thought for true delta storage for Iceberg.
            # For now, let's assume deltas are also lists of price/size, representing the changed levels.
            # Or, we focus BookIceberg on snapshots_only=True for now.
            # Based on current BackendBookCallback, deltas are {'bids': [(price, size), ...], 'asks': [...]}
            # where size 0 means delete.
            if 'bids' in data['delta']:
                for price, size in data['delta']['bids']:
                    bids_list.append({'price': float(price), 'size': float(size)})
            if 'asks' in data['delta']:
                 for price, size in data['delta']['asks']:
                    asks_list.append({'price': float(price), 'size': float(size)})


        return {
            'exchange': data['exchange'],
            'symbol': data['symbol'],
            'timestamp': data['timestamp'],
            'receipt_timestamp': data.get('receipt_timestamp', data['timestamp']), # ensure it exists
            'bids': bids_list,
            'asks': asks_list,
            'delta': is_delta,
            'sequence_number': data.get('sequence_number')
        }

    # We need to intercept the data before it goes into the queue via self.write()
    # The __call__ method of BackendBookCallback is where the book object is managed.
    # It calls self.book_delta_convert which then calls self.write().
    # We should override book_delta_convert.

    def book_delta_convert(self, data: dict, timestamp: float, exchange: str, symbol: str) -> dict:
        """
        Overrides BackendBookCallback.book_delta_convert.
        Formats the book data (snapshot or delta) from the internal book object
        into the flat dictionary structure expected by Iceberg writer for list<struct>.
        This data is then put onto the queue by the superclass's write() method.
        """
        # This method is called by BackendBookCallback.__call__
        # `data` here is the raw update from exchange, or a snapshot from self.book
        # `self.book` is the OrderBook object.

        # If it's a delta, data is usually {'delta': {'bids':..., 'asks':...}, 'timestamp':..., 'sequence_number':...}
        # If it's a snapshot, data is usually {'book': self.book, 'timestamp':..., 'sequence_number':...}

        # We need to access self.book for snapshots.
        # The BackendBookCallback already has logic for snapshots_only, snapshot_interval etc.
        # It will call this method with `data` being either a delta or a full book.

        current_book_data = {}
        is_delta_update = 'delta' in data and data['delta'] is not None

        if is_delta_update:
            # Delta update from exchange
            current_book_data = {
                'exchange': exchange,
                'symbol': symbol,
                'timestamp': data.get('timestamp', timestamp), # prefer timestamp from data if available
                'receipt_timestamp': timestamp,
                'delta': data['delta'], # This is like {'bids': [(price, size), ...]}
                'sequence_number': data.get('sequence_number')
            }
        else: # Snapshot
            # data['book'] should be the OrderBook object from self.book
            book_obj = data.get('book', self.book) # self.book is managed by BackendBookCallback
            if not book_obj:
                return {} # Should not happen if called correctly

            current_book_data = {
                'exchange': exchange,
                'symbol': symbol,
                'timestamp': data.get('timestamp', book_obj.timestamp if book_obj.timestamp else timestamp),
                'receipt_timestamp': timestamp,
                'book': {'bids': book_obj.bids.to_dict(json=False), 'asks': book_obj.asks.to_dict(json=False)}, # Get dicts from price_level_t
                'delta': None, # Explicitly mark as not a delta
                'sequence_number': data.get('sequence_number')
            }

        # Now, transform this current_book_data (which resembles the raw data)
        # into the flat structure with list<struct> for bids/asks.
        # This uses the _format_book_data_for_iceberg logic.

        # The _format_book_data_for_iceberg was designed to take the output of book_delta_convert.
        # Let's adjust: book_delta_convert should return the final dict for the queue.

        bids_for_df = []
        asks_for_df = []

        if is_delta_update:
            # current_book_data['delta'] is {'bids': [(price, size), ...], 'asks': [...]}
            for p, s in current_book_data['delta'].get('bids', []):
                bids_for_df.append({'price': float(p), 'size': float(s)})
            for p, s in current_book_data['delta'].get('asks', []):
                asks_for_df.append({'price': float(p), 'size': float(s)})
        elif current_book_data.get('book'): # Snapshot
             # current_book_data['book'] is {'bids': {price: size, ...}, 'asks': {price:size ...}}
            for p, s in current_book_data['book'].get('bids', {}).items():
                bids_for_df.append({'price': float(p), 'size': float(s)})
            for p, s in current_book_data['book'].get('asks', {}).items():
                asks_for_df.append({'price': float(p), 'size': float(s)})

        return {
            'exchange': current_book_data['exchange'],
            'symbol': current_book_data['symbol'],
            'timestamp': current_book_data['timestamp'],
            'receipt_timestamp': current_book_data['receipt_timestamp'],
            'bids': bids_for_df,
            'asks': asks_for_df,
            'delta': is_delta_update, # True if it was a delta, False if it was a snapshot
            'sequence_number': current_book_data.get('sequence_number')
        }


# TODO: Add other subclasses: FundingIceberg, OpenInterestIceberg, LiquidationsIceberg, CandlesIceberg, etc.
# Each will need its `default_table` and `schema` (pyarrow.Schema).

class FundingIceberg(IcebergCallback, BackendCallback):
    default_table = 'funding'
    schema = pa.schema([
        pa.field('exchange', pa.string(), nullable=False),
        pa.field('symbol', pa.string(), nullable=False),
        pa.field('timestamp', pa.float64(), nullable=False),
        pa.field('receipt_timestamp', pa.float64(), nullable=False),
        pa.field('rate', pa.float64()),
        pa.field('next_funding_time', pa.float64()), # Timestamp as float
        pa.field('mark_price', pa.float64(), nullable=True),
        pa.field('funding_rate_prediction', pa.float64(), nullable=True),
        pa.field('predicted_expiration_timestamp', pa.float64(), nullable=True),
        # ... other relevant funding fields
    ])
    def __init__(self, *args, **kwargs):
        if 'pandas_kwargs' not in kwargs: kwargs['pandas_kwargs'] = {}
        if 'columns' not in kwargs['pandas_kwargs']:
             kwargs['pandas_kwargs']['columns'] = [f.name for f in self.schema]
        super().__init__(*args, **kwargs)


class OpenInterestIceberg(IcebergCallback, BackendCallback):
    default_table = 'open_interest'
    schema = pa.schema([
        pa.field('exchange', pa.string(), nullable=False),
        pa.field('symbol', pa.string(), nullable=False),
        pa.field('timestamp', pa.float64(), nullable=False),
        pa.field('receipt_timestamp', pa.float64(), nullable=False),
        pa.field('open_interest', pa.float64()),
    ])
    def __init__(self, *args, **kwargs):
        if 'pandas_kwargs' not in kwargs: kwargs['pandas_kwargs'] = {}
        if 'columns' not in kwargs['pandas_kwargs']:
             kwargs['pandas_kwargs']['columns'] = [f.name for f in self.schema]
        super().__init__(*args, **kwargs)


class LiquidationsIceberg(IcebergCallback, BackendCallback):
    default_table = 'liquidations'
    schema = pa.schema([
        pa.field('exchange', pa.string(), nullable=False),
        pa.field('symbol', pa.string(), nullable=False),
        pa.field('timestamp', pa.float64(), nullable=False),
        pa.field('receipt_timestamp', pa.float64(), nullable=False),
        pa.field('side', pa.string()), # Side of the liquidated order
        pa.field('quantity', pa.float64()), # Amount of the liquidation
        pa.field('price', pa.float64()), # Price of the liquidation. None if not available.
        pa.field('order_id', pa.string(), nullable=True), # Exchange specific ID of the order
        pa.field('status', pa.string(), nullable=True), # 보통 'filled' 또는 상태 정보
    ])
    def __init__(self, *args, **kwargs):
        if 'pandas_kwargs' not in kwargs: kwargs['pandas_kwargs'] = {}
        if 'columns' not in kwargs['pandas_kwargs']:
             kwargs['pandas_kwargs']['columns'] = [f.name for f in self.schema]
        super().__init__(*args, **kwargs)

# For Candles, OrderInfo, Transactions, Balances, Fills, define similarly
# ... (omitted for brevity in this step, will add if time permits or in a subsequent step)

# Example for Candles
class CandlesIceberg(IcebergCallback, BackendCallback):
    default_table = 'candles'
    schema = pa.schema([
        pa.field('exchange', pa.string(), nullable=False),
        pa.field('symbol', pa.string(), nullable=False),
        pa.field('timestamp', pa.float64(), nullable=False), # Start time of the candle
        pa.field('receipt_timestamp', pa.float64(), nullable=False),
        pa.field('period', pa.string()), # e.g., '1m', '5m', '1h'
        pa.field('open', pa.float64()),
        pa.field('high', pa.float64()),
        pa.field('low', pa.float64()),
        pa.field('close', pa.float64()),
        pa.field('volume', pa.float64()),
        pa.field('num_trades', pa.int64(), nullable=True), # Number of trades in candle, if available
    ])
    def __init__(self, *args, **kwargs):
        if 'pandas_kwargs' not in kwargs: kwargs['pandas_kwargs'] = {}
        if 'columns' not in kwargs['pandas_kwargs']:
             kwargs['pandas_kwargs']['columns'] = [f.name for f in self.schema]
        super().__init__(*args, **kwargs)

# OrderInfo, Transactions, Balances, Fills would follow a similar pattern.
# The key is defining their pa.Schema correctly.
# For now, this set of classes provides a good foundation.

class OrderInfoIceberg(IcebergCallback, BackendCallback):
    default_table = 'order_info'
    schema = pa.schema([
        pa.field('exchange', pa.string(), nullable=False),
        pa.field('symbol', pa.string(), nullable=False),
        pa.field('timestamp', pa.float64(), nullable=False),
        pa.field('receipt_timestamp', pa.float64(), nullable=False),
        pa.field('order_id', pa.string()),
        pa.field('client_order_id', pa.string(), nullable=True),
        pa.field('side', pa.string()),
        pa.field('order_type', pa.string()), # e.g. limit, market
        pa.field('price', pa.float64(), nullable=True), # Null for market orders
        pa.field('amount', pa.float64()), # Original amount
        pa.field('amount_filled', pa.float64(), nullable=True),
        pa.field('amount_remaining', pa.float64(), nullable=True),
        pa.field('status', pa.string()), # e.g. open, closed, cancelled
        pa.field('trades', pa.list_(pa.string()), nullable=True) # List of trade IDs if available
    ])
    def __init__(self, *args, **kwargs):
        if 'pandas_kwargs' not in kwargs: kwargs['pandas_kwargs'] = {}
        if 'columns' not in kwargs['pandas_kwargs']:
             kwargs['pandas_kwargs']['columns'] = [f.name for f in self.schema]
        super().__init__(*args, **kwargs)


class TransactionsIceberg(IcebergCallback, BackendCallback):
    default_table = 'transactions'
    schema = pa.schema([
        pa.field('exchange', pa.string(), nullable=False), # Exchange name or 'wallet'
        pa.field('asset', pa.string(), nullable=False), # Asset/currency symbol
        pa.field('timestamp', pa.float64(), nullable=False),
        pa.field('receipt_timestamp', pa.float64(), nullable=False),
        pa.field('transaction_type', pa.string()), # deposit, withdrawal
        pa.field('amount', pa.float64()),
        pa.field('address', pa.string(), nullable=True), # deposit/withdrawal address
        pa.field('tx_id', pa.string(), nullable=True), # transaction hash/id
        pa.field('status', pa.string(), nullable=True), # e.g. pending, completed, failed
        pa.field('fee_amount', pa.float64(), nullable=True),
        pa.field('fee_currency', pa.string(), nullable=True),
    ])
    def __init__(self, *args, **kwargs):
        if 'pandas_kwargs' not in kwargs: kwargs['pandas_kwargs'] = {}
        if 'columns' not in kwargs['pandas_kwargs']:
             kwargs['pandas_kwargs']['columns'] = [f.name for f in self.schema]
        super().__init__(*args, **kwargs)


class BalancesIceberg(IcebergCallback, BackendCallback):
    default_table = 'balances'
    schema = pa.schema([
        pa.field('exchange', pa.string(), nullable=False),
        pa.field('asset', pa.string(), nullable=False),
        pa.field('timestamp', pa.float64(), nullable=False), # Timestamp of balance snapshot
        pa.field('receipt_timestamp', pa.float64(), nullable=False),
        pa.field('balance', pa.float64()),
        pa.field('available', pa.float64(), nullable=True), # Available balance
    ])
    def __init__(self, *args, **kwargs):
        if 'pandas_kwargs' not in kwargs: kwargs['pandas_kwargs'] = {}
        if 'columns' not in kwargs['pandas_kwargs']:
             kwargs['pandas_kwargs']['columns'] = [f.name for f in self.schema]
        super().__init__(*args, **kwargs)


class FillsIceberg(IcebergCallback, BackendCallback):
    default_table = 'fills' # User account fills/trades
    schema = pa.schema([
        pa.field('exchange', pa.string(), nullable=False),
        pa.field('symbol', pa.string(), nullable=False),
        pa.field('timestamp', pa.float64(), nullable=False),
        pa.field('receipt_timestamp', pa.float64(), nullable=False),
        pa.field('order_id', pa.string()),
        pa.field('trade_id', pa.string()), # Exchange's trade ID for this fill
        pa.field('client_order_id', pa.string(), nullable=True),
        pa.field('side', pa.string()),
        pa.field('order_type', pa.string()),
        pa.field('price', pa.float64()),
        pa.field('amount', pa.float64()),
        pa.field('fee_amount', pa.float64(), nullable=True),
        pa.field('fee_currency', pa.string(), nullable=True),
        pa.field('liquidity', pa.string(), nullable=True), # Taker or Maker
    ])
    def __init__(self, *args, **kwargs):
        if 'pandas_kwargs' not in kwargs: kwargs['pandas_kwargs'] = {}
        if 'columns' not in kwargs['pandas_kwargs']:
             kwargs['pandas_kwargs']['columns'] = [f.name for f in self.schema]
        super().__init__(*args, **kwargs)
