import asyncio
import unittest
import pandas as pd
import pyarrow as pa
from decimal import Decimal
import tempfile
import shutil
import os
from datetime import datetime, timezone
from pathlib importPath

from deltalake import DeltaTable, write_deltalake

# Adjust import path as necessary
try:
    from cryptofeed.backends.deltalake import DeltaLakeCallback
    from cryptofeed.defines import TRADES, TICKER, L2_BOOK, FUNDING, CANDLES
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from cryptofeed.backends.deltalake import DeltaLakeCallback
    from cryptofeed.defines import TRADES, TICKER, L2_BOOK, FUNDING, CANDLES


class TestDeltaLakeCallbackIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="cf_deltalake_int_")
        self.loop = asyncio.get_event_loop() # Use one loop for all async calls in a test

    def tearDown(self):
        # Ensure all writer tasks are properly collected if tests create them directly
        # and don't clean up. Forcing GC might help, but explicit cleanup in tests is better.
        # asyncio.gather(*[task for task in asyncio.all_tasks() if task is not asyncio.current_task()])
        shutil.rmtree(self.temp_dir)

    def _get_table_path(self, table_name):
        return os.path.join(self.temp_dir, table_name)

    async def _run_callback_writer_and_flush(self, cb, data_items):
        """Helper to run the callback, feed data, and ensure writer flushes."""
        cb.queue = asyncio.Queue()
        writer_task = self.loop.create_task(cb.writer())
        cb._writer_task = writer_task # For potential cleanup if needed

        for item_type, data_dict in data_items:
            # Simulate how FeedHandler would call the callback
            # The callback's __call__ method puts a dict into the queue
            # The dict should be {'type': item_type, 'data': data_dict, 'timestamp': data_dict['timestamp']}
            # For L2_BOOK, data_dict itself is the list of book entries.
            if item_type == L2_BOOK:
                 await cb.queue.put({'type': item_type, 'data': data_dict, 'timestamp': data_dict[0]['timestamp'] if data_dict else datetime.now(timezone.utc).timestamp()})
            else:
                 await cb.queue.put({'type': item_type, 'data': data_dict, 'timestamp': data_dict['timestamp']})


        # Ensure data is processed and flushed
        # Option 1: Use a sentinel and wait for writer_task
        await cb.queue.put(None) # Sentinel to stop the writer
        try:
            await asyncio.wait_for(writer_task, timeout=5.0) # Wait for writer to finish
        except asyncio.TimeoutError:
            self.fail("Writer task did not finish in time.")

        # Option 2: If cb.flush() is an async method that ensures completion (it is)
        # await cb.flush() # This would require cb.flush() to handle emptying the queue too.
        # The current cb.flush() only processes self.batch, so queue needs to be empty first.

    def _generate_sample_data(self, dtype, count=1, symbol="BTC-USD", exchange="test-exchange"):
        data_list = []
        base_ts = datetime.now(timezone.utc)
        for i in range(count):
            ts = (base_ts + pd.Timedelta(seconds=i)).timestamp()
            dt_ts = base_ts + pd.Timedelta(seconds=i) # datetime object

            data = {'exchange': exchange, 'symbol': symbol, 'timestamp': dt_ts, 'receipt_timestamp': ts + 0.1}
            if dtype == TRADES:
                data.update({'side': 'buy' if i % 2 == 0 else 'sell', 'amount': Decimal(f'{i+1}.1'), 'price': Decimal(f'5000{i}.5'), 'id': f'trade_{i}'})
            elif dtype == TICKER:
                data.update({'bid': Decimal(f'4999{i}.0'), 'ask': Decimal(f'5000{i + 1}.0')})
            elif dtype == L2_BOOK: # For L2_BOOK, we generate a list of book entries for each "update"
                book_entries = [
                    {'exchange': exchange, 'symbol': symbol, 'price': Decimal(f'5000{i}.0'), 'side': 'bid', 'amount': Decimal(f'{i+1}.0'), 'timestamp': dt_ts, 'receipt_timestamp': ts + 0.1},
                    {'exchange': exchange, 'symbol': symbol, 'price': Decimal(f'5000{i}.5'), 'side': 'bid', 'amount': Decimal(f'{i+1}.5'), 'timestamp': dt_ts, 'receipt_timestamp': ts + 0.1},
                    {'exchange': exchange, 'symbol': symbol, 'price': Decimal(f'5001{i}.0'), 'side': 'ask', 'amount': Decimal(f'{i+1}.2'), 'timestamp': dt_ts, 'receipt_timestamp': ts + 0.1}
                ]
                # For L2_BOOK, the callback expects the list of book_delta_convert outputs directly
                return (L2_BOOK, book_entries) # Special case for L2_BOOK data structure
            elif dtype == FUNDING:
                 data.update({'rate': Decimal(f'0.00{i+1}'), 'next_funding_time': (dt_ts + pd.Timedelta(hours=1)).timestamp(), 'mark_price': Decimal(f'5000{i}.0')})
            elif dtype == CANDLES:
                data.update({
                    'period': '1m',
                    'open': Decimal(f'5000{i}'),
                    'high': Decimal(f'5005{i}'),
                    'low': Decimal(f'4995{i}'),
                    'close': Decimal(f'5002{i}'),
                    'volume': Decimal(f'10.{i}'),
                    'trades': i + 5
                })


            data_list.append((dtype, data))
        return data_list

    def test_write_and_read_trades(self):
        table_name = "trades_table"
        table_path = self._get_table_path(table_name)

        config = {
            'table_name': table_path,
            'dtype': TRADES,
            'loop': self.loop,
            'max_batch_size': 5,
            'flush_interval': 1
        }
        cb = DeltaLakeCallback(**config)

        num_items = 10
        data_items = self._generate_sample_data(TRADES, count=num_items)

        self.loop.run_until_complete(self._run_callback_writer_and_flush(cb, data_items))

        # Read back the data
        dt = DeltaTable(table_path)
        df_read = dt.to_pandas()

        self.assertEqual(dt.version(), 0) # Should be first version
        self.assertEqual(len(df_read), num_items)
        self.assertIn('symbol', df_read.columns)
        self.assertIn('price', df_read.columns)
        self.assertIn('amount', df_read.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_read['timestamp']))
        self.assertTrue(pd.api.types.is_float_dtype(df_read['price'])) # Default conversion
        self.assertTrue(pd.api.types.is_float_dtype(df_read['amount']))

        # Check a value
        self.assertEqual(df_read['id'].iloc[0], 'trade_0')
        self.assertEqual(df_read['price'].iloc[1], 50001.5)


    def test_write_and_read_l2_book(self):
        table_name = "l2book_table"
        table_path = self._get_table_path(table_name)

        config = {
            'table_name': table_path,
            'dtype': L2_BOOK, # This will use book_delta_convert internally for schema and processing
            'loop': self.loop,
            'max_batch_size': 2 # L2_BOOK generates multiple rows per "item"
        }
        cb = DeltaLakeCallback(**config)

        # L2_BOOK data is a list of dicts. _generate_sample_data handles the tuple.
        # Let's simulate two book updates. Each update has multiple rows.
        book_update1 = self._generate_sample_data(L2_BOOK, count=1, symbol="BTC-USD")[0] # (L2_BOOK, [entries])
        book_update2 = self._generate_sample_data(L2_BOOK, count=1, symbol="ETH-USD")[0] # (L2_BOOK, [entries])

        all_entries_count = len(book_update1[1]) + len(book_update2[1])

        self.loop.run_until_complete(self._run_callback_writer_and_flush(cb, [book_update1, book_update2]))

        dt = DeltaTable(table_path)
        df_read = dt.to_pandas()

        self.assertEqual(len(df_read), all_entries_count)
        self.assertIn('symbol', df_read.columns)
        self.assertIn('price', df_read.columns)
        self.assertIn('side', df_read.columns)
        self.assertIn('amount', df_read.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_read['timestamp']))

        # Verify symbols
        self.assertIn("BTC-USD", df_read['symbol'].unique())
        self.assertIn("ETH-USD", df_read['symbol'].unique())


    def test_partitioning(self):
        table_name = "partitioned_trades"
        table_path = self._get_table_path(table_name)
        partition_cols = ['symbol', 'year'] # Partition by symbol and year

        config = {
            'table_name': table_path,
            'dtype': TRADES,
            'loop': self.loop,
            'max_batch_size': 2,
            'partition_by': partition_cols,
            'custom_dtypes': {'year': 'int32'} # Ensure year is int
        }
        cb = DeltaLakeCallback(**config)

        data_btc = self._generate_sample_data(TRADES, count=3, symbol="BTC-USD")
        data_eth = self._generate_sample_data(TRADES, count=2, symbol="ETH-USD")
        all_data = data_btc + data_eth

        self.loop.run_until_complete(self._run_callback_writer_and_flush(cb, all_data))

        # Verify directory structure
        # e.g., temp_dir/partitioned_trades/symbol=BTC-USD/year=2024/...
        # e.g., temp_dir/partitioned_trades/symbol=ETH-USD/year=2024/...

        # Get current year for path checking
        current_year = datetime.now(timezone.utc).year

        btc_partition_path = Path(table_path) / f"symbol=BTC-USD" / f"year={current_year}"
        eth_partition_path = Path(table_path) / f"symbol=ETH-USD" / f"year={current_year}"

        self.assertTrue(btc_partition_path.exists() and btc_partition_path.is_dir(), f"BTC partition path {btc_partition_path} not found")
        self.assertTrue(eth_partition_path.exists() and eth_partition_path.is_dir(), f"ETH partition path {eth_partition_path} not found")

        # Verify content by reading partitioned data
        dt_btc = DeltaTable(table_path)
        df_btc_filtered = dt_btc.to_pandas(filters=[("symbol", "=", "BTC-USD")])
        self.assertEqual(len(df_btc_filtered), 3)
        self.assertEqual(df_btc_filtered['symbol'].unique()[0], "BTC-USD")

        df_eth_filtered = dt_btc.to_pandas(filters=[("symbol", "=", "ETH-USD"), ("year", "=", current_year)])
        self.assertEqual(len(df_eth_filtered), 2)
        self.assertEqual(df_eth_filtered['symbol'].unique()[0], "ETH-USD")


    def test_write_and_read_ticker(self):
        table_name = "ticker_table"
        table_path = self._get_table_path(table_name)
        config = {
            'table_name': table_path, 'dtype': TICKER, 'loop': self.loop, 'max_batch_size': 2
        }
        cb = DeltaLakeCallback(**config)
        data_items = self._generate_sample_data(TICKER, count=3)
        self.loop.run_until_complete(self._run_callback_writer_and_flush(cb, data_items))

        dt = DeltaTable(table_path)
        df_read = dt.to_pandas()
        self.assertEqual(len(df_read), 3)
        self.assertIn('bid', df_read.columns)
        self.assertIn('ask', df_read.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_read['timestamp']))


    def test_write_and_read_funding(self):
        table_name = "funding_table"
        table_path = self._get_table_path(table_name)
        config = {
            'table_name': table_path, 'dtype': FUNDING, 'loop': self.loop, 'max_batch_size': 2,
            'custom_dtypes': {'next_funding_time': 'timestamp[ns]'} # Ensure it's a timestamp
        }
        cb = DeltaLakeCallback(**config)
        data_items = self._generate_sample_data(FUNDING, count=3)
        self.loop.run_until_complete(self._run_callback_writer_and_flush(cb, data_items))

        dt = DeltaTable(table_path)
        df_read = dt.to_pandas()
        self.assertEqual(len(df_read), 3)
        self.assertIn('rate', df_read.columns)
        self.assertIn('mark_price', df_read.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_read['timestamp']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_read['next_funding_time']))

    def test_write_and_read_candles(self):
        table_name = "candles_table"
        table_path = self._get_table_path(table_name)
        config = {
            'table_name': table_path, 'dtype': CANDLES, 'loop': self.loop, 'max_batch_size': 2
        }
        cb = DeltaLakeCallback(**config)
        data_items = self._generate_sample_data(CANDLES, count=3)
        self.loop.run_until_complete(self._run_callback_writer_and_flush(cb, data_items))

        dt = DeltaTable(table_path)
        df_read = dt.to_pandas()
        self.assertEqual(len(df_read), 3)
        self.assertIn('period', df_read.columns)
        self.assertIn('open', df_read.columns)
        self.assertIn('high', df_read.columns)
        self.assertIn('low', df_read.columns)
        self.assertIn('close', df_read.columns)
        self.assertIn('volume', df_read.columns)
        self.assertIn('trades', df_read.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_read['timestamp']))


    def test_schema_evolution_overwrite(self):
        table_name = "schema_evo_overwrite"
        table_path = self._get_table_path(table_name)

        # Initial write with schema S1
        config_s1 = {
            'table_name': table_path, 'dtype': TRADES, 'loop': self.loop,
            'max_batch_size': 1,
            'custom_dtypes': {'price': 'float64', 'amount': 'float64'} # Explicitly float64
        }
        cb_s1 = DeltaLakeCallback(**config_s1)
        data_s1 = self._generate_sample_data(TRADES, count=1, symbol="BTC-USD")[0] # Single item tuple
        self.loop.run_until_complete(self._run_callback_writer_and_flush(cb_s1, [data_s1]))

        # Verify S1
        dt_s1 = DeltaTable(table_path)
        df_s1 = dt_s1.to_pandas()
        self.assertEqual(df_s1['price'].dtype, 'float64')

        # New config with schema_mode='overwrite' and modified schema S2
        config_s2 = {
            'table_name': table_path, 'dtype': TRADES, 'loop': self.loop,
            'max_batch_size': 1, 'schema_mode': 'overwrite',
            'custom_dtypes': {
                'price': 'float32', # Change type
                'new_string_col': 'string' # Add new column
            }
        }
        cb_s2 = DeltaLakeCallback(**config_s2)

        # Generate data for S2, including the new column
        data_s2_list = self._generate_sample_data(TRADES, count=1, symbol="ETH-USD")
        # data_s2_list is [(TRADES, data_dict)]. Modify data_dict.
        data_s2_list[0][1]['new_string_col'] = 'hello_delta'
        data_s2_list[0][1]['price'] = Decimal('123.45') # New price to check

        self.loop.run_until_complete(self._run_callback_writer_and_flush(cb_s2, data_s2_list))

        # Verify S2
        dt_s2 = DeltaTable(table_path)
        df_s2 = dt_s2.to_pandas()

        self.assertEqual(df_s2['price'].dtype, 'float32')
        self.assertIn('new_string_col', df_s2.columns)
        self.assertEqual(df_s2['new_string_col'].dtype, 'object') # Pandas uses object for strings by default from Arrow

        # Check that new data is present and old data might have null for new_string_col
        self.assertEqual(len(df_s2), 2) # 1 from S1, 1 from S2 because mode='append' for data

        # First row (old data) should have NaN/None for 'new_string_col'
        self.assertTrue(pd.isna(df_s2[df_s2['symbol'] == 'BTC-USD']['new_string_col'].iloc[0]))
        # Second row (new data) should have the value
        self.assertEqual(df_s2[df_s2['symbol'] == 'ETH-USD']['new_string_col'].iloc[0], 'hello_delta')
        self.assertAlmostEqual(df_s2[df_s2['symbol'] == 'ETH-USD']['price'].iloc[0], 123.45, places=2)


    def test_schema_evolution_merge(self):
        table_name = "schema_evo_merge"
        table_path = self._get_table_path(table_name)

        # Initial write with schema S1
        config_s1 = {
            'table_name': table_path, 'dtype': TRADES, 'loop': self.loop, 'max_batch_size': 1,
            'custom_dtypes': {'id': 'string'} # Ensure id is string
        }
        cb_s1 = DeltaLakeCallback(**config_s1)
        data_s1_list = self._generate_sample_data(TRADES, count=1, symbol="BTC-USD")
        data_s1_list[0][1]['id'] = "trade_s1_001"
        self.loop.run_until_complete(self._run_callback_writer_and_flush(cb_s1, data_s1_list))

        # New config with schema_mode='merge' and additional column
        config_s2 = {
            'table_name': table_path, 'dtype': TRADES, 'loop': self.loop,
            'max_batch_size': 1, 'schema_mode': 'merge', # Key change: schema_mode
            'custom_dtypes': {
                'id': 'string', # Keep existing type consistent
                'venue_id': 'int32' # Add new nullable column
            }
        }
        cb_s2 = DeltaLakeCallback(**config_s2)

        data_s2_list = self._generate_sample_data(TRADES, count=1, symbol="ETH-USD")
        data_s2_list[0][1]['id'] = "trade_s2_002"
        data_s2_list[0][1]['venue_id'] = 999 # Value for the new column

        self.loop.run_until_complete(self._run_callback_writer_and_flush(cb_s2, data_s2_list))

        # Verify merged schema
        dt_merged = DeltaTable(table_path)
        df_merged = dt_merged.to_pandas()

        self.assertEqual(len(df_merged), 2)
        self.assertIn('id', df_merged.columns)
        self.assertIn('venue_id', df_merged.columns)
        self.assertEqual(df_merged['venue_id'].dtype, 'Int32') # Pandas uses nullable Int32 for Arrow int32 with nulls

        # Check S1 data (BTC-USD): venue_id should be null (pd.NA for Int32Dtype)
        s1_row = df_merged[df_merged['id'] == "trade_s1_001"].iloc[0]
        self.assertTrue(pd.isna(s1_row['venue_id']))

        # Check S2 data (ETH-USD): venue_id should have value
        s2_row = df_merged[df_merged['id'] == "trade_s2_002"].iloc[0]
        self.assertEqual(s2_row['venue_id'], 999)


    def test_optimize_compact(self):
        table_name = "optimize_compact_table"
        table_path = self._get_table_path(table_name)

        config = {
            'table_name': table_path, 'dtype': TRADES, 'loop': self.loop,
            'max_batch_size': 1 # Force multiple small files
        }
        cb = DeltaLakeCallback(**config)

        # Write data in multiple batches to create multiple small files
        num_writes = 3
        for i in range(num_writes):
            data_item = self._generate_sample_data(TRADES, count=1, symbol=f"SYM{i}")[0]
            # Need to run the writer for each small batch to ensure multiple parquet files
            self.loop.run_until_complete(self._run_callback_writer_and_flush(cb, [data_item]))

        dt = DeltaTable(table_path)
        initial_files = dt.files()
        self.assertGreaterEqual(len(initial_files), num_writes, "Should have multiple data files before compaction.")

        # Perform compaction
        dt.optimize.compact()

        history = dt.history()
        self.assertEqual(history[0]['operation'], 'OPTIMIZE')
        # operationMetrics for compact might show numFilesAdded (usually 1 for the compacted file)
        # and numFilesRemoved (the number of small files compacted)
        metrics = history[0]['operationMetrics']
        self.assertTrue(int(metrics['numFilesAdded']) < len(initial_files) or int(metrics['numFilesRemoved']) > 1)


    def test_optimize_z_order_via_callback_writer(self):
        table_name = "optimize_z_order_table"
        table_path = self._get_table_path(table_name)
        z_order_cols = ['timestamp', 'symbol']

        config = {
            'table_name': table_path, 'dtype': TRADES, 'loop': self.loop,
            'max_batch_size': 2,
            'z_order_by': z_order_cols # Configure Z-ordering in the callback
        }
        cb = DeltaLakeCallback(**config)

        data_items = self._generate_sample_data(TRADES, count=3)
        self.loop.run_until_complete(self._run_callback_writer_and_flush(cb, data_items))

        # The writer in DeltaLakeCallback calls optimize.z_order after each flush if z_order_by is set.
        dt = DeltaTable(table_path)
        history = dt.history()

        # Expecting at least one OPTIMIZE operation if data was written and flushed.
        # If multiple flushes happened, there might be multiple OPTIMIZE ops.
        optimize_ops = [h for h in history if h['operation'] == 'OPTIMIZE']
        self.assertGreaterEqual(len(optimize_ops), 1, "OPTIMIZE operation should be in history.")

        # Check the last OPTIMIZE operation
        last_optimize_op = optimize_ops[0] # History is in reverse chronological order
        self.assertIn('zOrderBy', last_optimize_op['operationParameters'])
        # The zOrderBy parameter in history is a string, e.g., "[\"timestamp\",\"symbol\"]"
        # We need to compare it carefully.
        import json
        history_z_order_param = json.loads(last_optimize_op['operationParameters']['zOrderBy'])
        self.assertEqual(sorted(history_z_order_param), sorted(z_order_cols))


if __name__ == '__main__':
    unittest.main()
