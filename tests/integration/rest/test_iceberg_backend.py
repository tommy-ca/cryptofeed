'''
Copyright (C) 2017-2021  Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''
import asyncio
import os
import shutil
import tempfile
import unittest
from decimal import Decimal

import pandas as pd
from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NoSuchTableError

from cryptofeed import FeedHandler
from cryptofeed.backends.iceberg import TickerIceberg
from cryptofeed.defines import TICKER
from cryptofeed.exchanges import Coinbase


class TestIcebergBackend(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop()
        self.temp_dir = tempfile.mkdtemp()
        self.catalog_path = self.temp_dir
        self.catalog_name = os.path.basename(self.temp_dir) # Catalog name from temp dir name

        # Ensure the catalog directory exists
        if not os.path.exists(self.catalog_path):
            os.makedirs(self.catalog_path)

    def tearDown(self):
        # Clean up: remove the temporary directory and its contents
        # Also ensure tables are dropped if possible, though Iceberg handles this differently.
        # For file-based catalogs, removing the warehouse dir is usually sufficient.
        try:
            catalog = load_catalog(self.catalog_path)
            table_name = f"{self.catalog_name}.{TICKER.lower()}" # e.g. temp_dir_name.ticker
            if catalog.table_exists(table_name):
                catalog.drop_table(table_name)
        except Exception as e:
            print(f"Error during table cleanup: {e}")
        finally:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        # It's good practice to also close the asyncio loop if it was started by the test suite
        # However, if running with other asyncio tests, this might be handled by the test runner.

    async def _run_feed_and_verify(self):
        f = FeedHandler(config={'log': {'disabled': True}}) # Disable verbose logging for tests

        iceberg_config = {
            'path': self.catalog_path,
            'key': TICKER # Explicitly set key for clarity, though TickerIceberg defaults to TICKER
        }

        # Using a well-known exchange and symbol
        f.add_feed(Coinbase(channels=[TICKER], symbols=['BTC-USD'], callbacks={TICKER: TickerIceberg(**iceberg_config)}))

        # Run the feed for a very short duration to get a few messages
        # Enough time for connection, subscription, and a few messages.
        # Needs to be async with other parts of Feedhandler
        await asyncio.wait_for(f.run(start_loop=False, duration=10), timeout=20)


        # Verify data in Iceberg
        catalog = load_catalog(self.catalog_path)
        table_name = f"{self.catalog_name}.{TICKER.lower()}"
        self.assertTrue(catalog.table_exists(table_name), f"Table {table_name} does not exist.")

        table = catalog.load_table(table_name)
        df = table.scan().to_pandas()

        self.assertFalse(df.empty, "No data written to Iceberg table.")
        self.assertIn('exchange', df.columns)
        self.assertIn('symbol', df.columns)
        self.assertIn('bid', df.columns)
        self.assertIn('ask', df.columns)
        self.assertIn('timestamp', df.columns) # Stored as datetime64[ns] by Iceberg
        self.assertIn('receipt_timestamp', df.columns) # Stored as datetime64[ns]

        # Check some values (example)
        self.assertEqual(df['exchange'].iloc[0], 'COINBASE')
        self.assertEqual(df['symbol'].iloc[0], 'BTC-USD')
        self.assertTrue(pd.api.types.is_numeric_dtype(df['bid']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['ask']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['timestamp']))

    def test_iceberg_ticker_write_and_read(self):
        # Since FeedHandler.run() is async and involves an event loop,
        # we need to run it within an asyncio context.
        self.loop.run_until_complete(self._run_feed_and_verify())


if __name__ == '__main__':
    unittest.main()
