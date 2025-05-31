'''
Copyright (C) 2017-2021  Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
from cryptofeed import FeedHandler
from cryptofeed.defines import TICKER, TRADES
from cryptofeed.exchanges import Coinbase

# Import the Iceberg callbacks
from cryptofeed.backends.iceberg import TickerIceberg, TradeIceberg

# Example configuration for a local Iceberg catalog.
# Replace with your actual Iceberg catalog configuration.
# For S3: path = "s3://your-bucket/warehouse/"
# For GCS: path = "gcs://your-bucket/warehouse/"
# For local: path = "/path/to/your/iceberg_warehouse" (ensure this directory exists)
# The catalog name (e.g., "demo_catalog") will be the last part of the path if not specified otherwise.
ICEBERG_CATALOG_PATH = "iceberg_data"  # This will create a catalog named 'iceberg_data' in the local directory.
                                      # Make sure 'iceberg_data' directory exists or can be created.

# Example for local MinIO S3 setup (ensure MinIO is running and bucket exists)
# ICEBERG_CATALOG_PATH = "s3://cryptofeed/iceberg"
# S3_ACCESS_KEY_ID = "minioadmin"
# S3_SECRET_ACCESS_KEY = "minioadmin"
# S3_ENDPOINT_URL = "http://localhost:9000" # Required if not using AWS S3


def main():
    f = FeedHandler()

    # Configuration for the Iceberg backend.
    # The `key` argument in the callback corresponds to the data type (e.g., TICKER, TRADES).
    # This will also be used as the base table name in Iceberg.
    # So, for TICKER, the table will be something like 'iceberg_data.ticker_table'.
    # The exact table name format is catalog_name.table_name.
    # The table name is derived from the `key` (e.g. TICKER -> "ticker")

    # This example uses a local filesystem catalog.
    # Ensure the directory `iceberg_data` exists in the same directory where you run the script,
    # or provide an absolute path.
    iceberg_config_ticker = {
        'path': ICEBERG_CATALOG_PATH,
        # For S3, you might need to add s3_access_key_id, s3_secret_access_key, and potentially s3_region or endpoint overrides
        # 's3_access_key_id': S3_ACCESS_KEY_ID,
        # 's3_secret_access_key': S3_SECRET_ACCESS_KEY,
        # 's3_endpoint_override': S3_ENDPOINT_URL, # if using MinIO or non-AWS S3
        # 's3_region': 'us-east-1'
        # For GCS, you might need gcs_project_id and gcs_token
        # 'gcs_project_id': 'your-gcp-project-id'
    }

    # Add Coinbase feed for Ticker data, writing to Iceberg
    # The table name will be determined by the catalog name (from path) and the default_key of TickerIceberg (which is TICKER)
    # e.g. iceberg_data.ticker
    f.add_feed(Coinbase(channels=[TICKER], symbols=['BTC-USD'], callbacks={TICKER: TickerIceberg(**iceberg_config_ticker)}))

    # Example for TRADES (uncomment to use)
    # iceberg_config_trades = {'path': ICEBERG_CATALOG_PATH}
    # f.add_feed(Coinbase(channels=[TRADES], symbols=['BTC-USD'], callbacks={TRADES: TradeIceberg(**iceberg_config_trades)}))

    # Example for L2_BOOK (uncomment to use)
    # Note: Book data can be voluminous. Ensure your Iceberg setup can handle it.
    # A BookIceberg class would be needed in cryptofeed.backends.iceberg
    # iceberg_config_l2book = {'path': ICEBERG_CATALOG_PATH}
    # f.add_feed(Coinbase(channels=[L2_BOOK], symbols=['BTC-USD'], callbacks={L2_BOOK: BookIceberg(**iceberg_config_l2book)}))

    print(f"Writing data to Iceberg. Catalog path: {ICEBERG_CATALOG_PATH}")
    print("Running for 60 seconds. Check your Iceberg catalog/tables after the script finishes.")
    print("Make sure the catalog directory (e.g., 'iceberg_data/') exists if using a local file system catalog.")

    f.run(duration=60) # Run for 60 seconds

    print("Feed handler stopped.")

if __name__ == '__main__':
    # Create the local directory for the Iceberg catalog if it doesn't exist.
    import os
    if not ICEBERG_CATALOG_PATH.startswith("s3://") and not ICEBERG_CATALOG_PATH.startswith("gcs://"):
        if not os.path.exists(ICEBERG_CATALOG_PATH):
            os.makedirs(ICEBERG_CATALOG_PATH)
            print(f"Created directory: {ICEBERG_CATALOG_PATH}")
    main()
