# examples/demo_influxdb3.py

import logging

from cryptofeed.feedhandler import FeedHandler
# Imports from cryptofeed.backends are preferred if __init__.py is set up
from cryptofeed.backends import BookInflux3, TradeInflux3
from cryptofeed.defines import COINBASE, L2_BOOK, TRADES

# Configure logging to see output from cryptofeed
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-15s %(lineno)d: %(message)s"
)
LOG = logging.getLogger('demo_influxdb3')

# This backend uses the influxdb3-python library.
# Ensure it's installed: pip install influxdb3-python cryptofeed


# --- Configuration Section ---
# IMPORTANT: Replace these placeholder values with your actual InfluxDB v3 connection details.
#
# For InfluxDB Cloud:
#   INFLUXDB_ADDRESS: Your InfluxDB Cloud region URL (e.g., "https://us-east-1-1.aws.cloud2.influxdata.com")
#   INFLUXDB_TOKEN: Your InfluxDB API token with write access to the database.
#   INFLUXDB_ORG: Your InfluxDB Cloud organization ID.
#   INFLUXDB_DATABASE: The name of your database (bucket) in InfluxDB Cloud.
#
# For InfluxDB OSS (v3.x):
#   INFLUXDB_ADDRESS: The address of your InfluxDB OSS instance (e.g., "http://localhost:8086").
#   INFLUXDB_TOKEN: Your InfluxDB API token (if authentication is enabled).
#                    For OSS, this might be in the format "username:password" or a generated token.
#   INFLUXDB_ORG: Your InfluxDB organization name or ID. This might be optional for some OSS setups
#                 if a default organization is used or authentication is simpler.
#   INFLUXDB_DATABASE: The name of your database in InfluxDB OSS.

INFLUXDB_ADDRESS = "http://localhost:8086"  # Replace! Example: "https://<your-region>.cloud2.influxdata.com"
INFLUXDB_DATABASE = "cryptofeed_database"   # Replace! Example: "crypto_data"
INFLUXDB_TOKEN = "your_influx_token"        # Replace! Example: "thisIsMySecretToken"
INFLUXDB_ORG = "your_influx_org"            # Replace! Example: "my_organization" (can be None for some OSS setups)

# Optional: InfluxDB v3 client batching parameters (defaults are usually fine)
# These are passed via kwargs to the callback constructor.
# INFLUXDB_BATCH_SIZE = 5000  # Number of data points to batch before writing
# INFLUXDB_FLUSH_INTERVAL = 10000 # Milliseconds to wait before flushing the batch
# --- End Configuration Section ---


def main():
    LOG.info("Starting InfluxDB v3 demo script.")

    # Check if placeholder values have been changed (basic check)
    if "your_influx_token" in INFLUXDB_TOKEN or "http://localhost:8086" == INFLUXDB_ADDRESS and not LOG.isEnabledFor(logging.DEBUG):
        LOG.warning("###########################################################################")
        LOG.warning("# IMPORTANT: Update InfluxDB connection details in this script before run! #")
        LOG.warning("###########################################################################")
        # Consider exiting if not configured, or add a --force flag for testing with defaults.
        # For this example, we'll proceed but warn the user.

    fh = FeedHandler()

    # Instantiate InfluxDB v3 callbacks
    # For BookInflux3, you can specify snapshots_only and snapshot_interval if needed.
    # Default key for BookInflux3 is 'book'. It will be written to measurement 'book-COINBASE'.
    book_callback = BookInflux3(
        addr=INFLUXDB_ADDRESS,
        database=INFLUXDB_DATABASE,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        snapshots_only=False, # Example: True to store only full snapshots
        snapshot_interval=1000, # Example: if snapshots_only=False, snapshot every 1000 deltas. If True, every 1000 seconds.
        # batch_size=INFLUXDB_BATCH_SIZE, # Uncomment to override default batch_size (5000)
        # flush_interval=INFLUXDB_FLUSH_INTERVAL # Uncomment to override default flush_interval (10000ms)
    )

    # Default key for TradeInflux3 is 'trades'. It will be written to measurement 'trades-COINBASE'.
    trade_callback = TradeInflux3(
        addr=INFLUXDB_ADDRESS,
        database=INFLUXDB_DATABASE,
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        # batch_size=INFLUXDB_BATCH_SIZE, # Uncomment to override default batch_size
        # flush_interval=INFLUXDB_FLUSH_INTERVAL # Uncomment to override default flush_interval
    )

    # Add subscriptions using the InfluxDB v3 callbacks
    # Subscribe to Coinbase BTC-USD L2 Book data
    fh.add_feed(COINBASE, symbols=['BTC-USD'], channels=[L2_BOOK], callbacks=[book_callback])
    LOG.info("Subscribed to COINBASE BTC-USD L2_BOOK with InfluxDB3 backend.")

    # Subscribe to Coinbase BTC-USD Trades data
    fh.add_feed(COINBASE, symbols=['BTC-USD'], channels=[TRADES], callbacks=[trade_callback])
    LOG.info("Subscribed to COINBASE BTC-USD TRADES with InfluxDB3 backend.")

    LOG.info("FeedHandler configured. Starting data ingestion to InfluxDB v3...")
    LOG.info(f"Target InfluxDB: address='{INFLUXDB_ADDRESS}', database='{INFLUXDB_DATABASE}', org='{INFLUXDB_ORG}'")
    LOG.info("Press Ctrl+C to stop the script.")

    fh.run()

    LOG.info("FeedHandler stopped.")


if __name__ == '__main__':
    main()
```

I've added:
- Logging configuration for better output.
- More detailed comments in the configuration section for both Cloud and OSS.
- A basic check to warn the user if default placeholder values are still present.
- Used `from cryptofeed.backends import BookInflux3, TradeInflux3` as decided.
- Clarified how measurement names will be formed (e.g., `book-COINBASE`).
- Ensured `main()` calls `fh.run()` directly.

This script should be a good starting point for users.
