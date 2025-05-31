"""
Cryptofeed Iceberg Backend Demo

This script demonstrates how to use the Iceberg backend with Cryptofeed to publish
market data from exchanges to Apache Iceberg tables.

Prerequisites:
1. Install Cryptofeed with Iceberg support:
   pip install cryptofeed[iceberg]
   (This installs pyiceberg, pyarrow, pandas)

2. Run an Iceberg Catalog. A simple way to get started is with a Dockerized REST Catalog.
   The Tabular image provides a REST catalog with S3 (MinIO) backend.

   Docker command for Tabular quickstart (includes MinIO & REST Catalog):
   docker run -p 8080:8080 -p 9000:9000 -p 8181:8181 \\
     --name tabular-iceberg-rest \\
     tabulario/iceberg-rest-runtime:latest

   This makes:
   - MinIO (S3 compatible) UI: http://localhost:9000 (admin/password)
   - Iceberg REST Catalog: http://localhost:8181

   You might need to create a bucket in MinIO (e.g., 'iceberg') and configure
   the warehouse path accordingly. The Tabular image might pre-configure a default
   warehouse and credentials. Refer to its documentation if needed.
   The default credentials for MinIO in this image are often 'admin'/'password'.

   The catalog_config below assumes this Tabular Docker setup.

To run this script:
   python examples/demo_iceberg.py

After the script runs for a bit (and you stop it with CTRL+C), the data will be
in Iceberg tables. You can then inspect the data using PyIceberg or query
engines like Spark, Trino, Dremio, or DuckDB (if they can connect to your catalog).
"""
import asyncio
import logging

from cryptofeed import FeedHandler
from cryptofeed.exchanges import Coinbase
from cryptofeed.defines import TRADES, TICKER, L2_BOOK
from cryptofeed.backends.iceberg import TradeIceberg, TickerIceberg, BookIceberg

# Configure logging for Cryptofeed and PyIceberg (optional, for debugging)
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging.getLogger('pyiceberg').setLevel(logging.DEBUG)


# --- Configuration for Iceberg Catalog ---
# This example uses a REST catalog, assuming the Tabular Docker image mentioned above.
# Adjust these settings based on your Iceberg catalog setup.
CATALOG_NAME = "tabular_rest" # A local name for this catalog configuration
CATALOG_URI = "http://localhost:8181" # REST Catalog endpoint
S3_ENDPOINT_URL = "http://localhost:9000" # MinIO endpoint from Tabular container
S3_ACCESS_KEY = "admin" # Default for Tabular's MinIO
S3_SECRET_KEY = "password" # Default for Tabular's MinIO
ICEBERG_WAREHOUSE_PATH = "s3a://iceberg/cryptofeed_data/" # Bucket 'iceberg', path 'cryptofeed_data/'
# Ensure the 'iceberg' bucket exists in MinIO. You can create it via MinIO UI (localhost:9000).

CATALOG_CONFIG = {
    "name": CATALOG_NAME,
    "type": "rest", # Explicitly setting type, though 'uri' often implies it for REST
    "uri": CATALOG_URI,
    "s3.endpoint": S3_ENDPOINT_URL, # Note: pyiceberg might prefer 's3.endpoint' or 's3.endpoint-url'
                                     # Check pyiceberg docs for exact S3 client factory keys.
                                     # For REST catalog, these S3 settings are usually passed in its server config,
                                     # but pyiceberg client might need them for certain operations if it interacts
                                     # with S3 directly, or if the REST catalog itself needs them passed this way.
                                     # More commonly, for a REST catalog, the 'warehouse' property is key,
                                     # and S3 settings are on the REST server.
                                     # Let's assume REST catalog is configured to use this S3 backend.
                                     # PyIceberg's load_catalog for REST primarily uses 'uri' and 'credential'.
                                     # For warehouse path on S3, it's often just 'warehouse'.
    "warehouse": ICEBERG_WAREHOUSE_PATH,
    # If your REST catalog requires authentication:
    # "credential": "Bearer <your_token>",
    # Or for other auth methods, refer to PyIceberg docs.
    # For the Tabular image, it typically runs without auth by default.

    # For S3 client configuration directly in pyiceberg (might be needed if not using REST server's S3 config):
    # These are typical PyIceberg properties for S3FileIO
    "io-impl": "pyiceberg.io.s3.S3FileIO", # Example if needing to specify S3FileIO explicitly
    "s3.access-key-id": S3_ACCESS_KEY,
    "s3.secret-access-key": S3_SECRET_KEY,
    "s3.endpoint-url": S3_ENDPOINT_URL, # Some PyIceberg versions might use this key
}


# Database and table prefix in Iceberg
DATABASE_NAME = "cryptofeed_db"
TABLE_PREFIX = "demo" # Tables will be like demo_trades, demo_tickers, demo_orderbooks
BATCH_SIZE = 10 # Number of messages to buffer before writing to Iceberg (small for demo)


def main():
    fh = FeedHandler()

    # Define Iceberg callbacks
    trade_cb = TradeIceberg(
        catalog_config=CATALOG_CONFIG,
        database_name=DATABASE_NAME,
        table_prefix=TABLE_PREFIX,
        batch_size=BATCH_SIZE
    )
    ticker_cb = TickerIceberg(
        catalog_config=CATALOG_CONFIG,
        database_name=DATABASE_NAME,
        table_prefix=TABLE_PREFIX,
        batch_size=BATCH_SIZE
    )
    book_cb = BookIceberg(
        catalog_config=CATALOG_CONFIG,
        database_name=DATABASE_NAME,
        table_prefix=TABLE_PREFIX,
        batch_size=BATCH_SIZE, # Book snapshots can be larger, adjust if needed
        # snapshots_only=True # Consider for BookIceberg if delta handling is complex initially
    )

    # Add Coinbase feed for BTC-USD trades, tickers, and L2 book
    fh.add_feed(Coinbase(
        symbols=['BTC-USD'],
        channels=[TRADES, TICKER, L2_BOOK],
        callbacks={
            TRADES: trade_cb,
            TICKER: ticker_cb,
            L2_BOOK: book_cb
        }
    ))

    print(f"Starting FeedHandler. Writing data to Iceberg via catalog: {CATALOG_CONFIG.get('name', CATALOG_CONFIG.get('uri'))}")
    print(f"Target Iceberg Database: {DATABASE_NAME}")
    print(f"Table prefix: {TABLE_PREFIX}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Data will be written to tables like: {DATABASE_NAME}.{TABLE_PREFIX}_trades, etc.")
    print("Run this for a while and then stop with CTRL+C.")
    print("After stopping, you can inspect the Iceberg tables.")

    try:
        fh.run()
    except KeyboardInterrupt:
        print("FeedHandler stopped by user.")
    finally:
        print("\n--- Inspecting Data (Example) ---")
        print("Attempting to load tables using PyIceberg and print some data...")
        print("Note: This inspection part requires the same catalog configuration to be accessible.")

        try:
            from pyiceberg.catalog import load_catalog
            catalog = load_catalog(**CATALOG_CONFIG)

            trades_table_name = f"{DATABASE_NAME}.{TABLE_PREFIX}_trades"
            try:
                print(f"\nLoading table: {trades_table_name}")
                table = catalog.load_table(trades_table_name)
                print(f"Schema: {table.schema()}")
                df_trades = table.scan(row_filter="price > 0", selected_fields=("timestamp", "price", "amount", "side")).to_pandas()
                print("Sample Trades Data (first 5 rows):")
                print(df_trades.head())
            except Exception as e:
                print(f"Could not load or scan trades table '{trades_table_name}': {e}")

            tickers_table_name = f"{DATABASE_NAME}.{TABLE_PREFIX}_tickers"
            try:
                print(f"\nLoading table: {tickers_table_name}")
                table = catalog.load_table(tickers_table_name)
                print(f"Schema: {table.schema()}")
                df_tickers = table.scan(selected_fields=("timestamp", "bid", "ask")).to_pandas()
                print("Sample Tickers Data (first 5 rows):")
                print(df_tickers.head())
            except Exception as e:
                print(f"Could not load or scan tickers table '{tickers_table_name}': {e}")

            # Book table inspection can be more complex due to nested data
            books_table_name = f"{DATABASE_NAME}.{TABLE_PREFIX}_orderbooks" # default_table is 'orderbooks'
            try:
                print(f"\nLoading table: {books_table_name}")
                table = catalog.load_table(books_table_name)
                print(f"Schema: {table.schema()}")
                # Scanning list<struct<...>> can be tricky with to_pandas() if not flattened.
                # PyIceberg's scan should handle it.
                df_books = table.scan(selected_fields=("timestamp", "bids", "asks")).to_pandas()
                print("Sample Book Data (first 2 rows, bids/asks might be truncated):")
                # Pandas display options for wide/nested columns
                pd.set_option('display.max_colwidth', 100)
                print(df_books.head(2))
            except Exception as e:
                print(f"Could not load or scan orderbooks table '{books_table_name}': {e}")

        except Exception as e:
            print(f"Could not initialize PyIceberg catalog for inspection: {e}")
            print("Please ensure your Iceberg catalog is running and configured correctly.")


if __name__ == '__main__':
    # For asyncio event loop management, especially if fh.run() is wrapped or other async ops are added
    # asyncio.run(main()) # This is not needed if fh.run() manages the loop, which it does.
    main()
