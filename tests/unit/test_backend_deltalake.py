import asyncio
import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import pyarrow as pa
from decimal import Decimal
import tempfile
import shutil
import os
from datetime import datetime

# Attempt to import the class, adjust path if necessary based on actual structure
try:
    from cryptofeed.backends.deltalake import DeltaLakeCallback
    from cryptofeed.backends._util import book_delta_convert
    from cryptofeed.defines import TRADES, TICKER, L2_BOOK, FUNDING, OPEN_INTEREST, LIQUIDATIONS, CANDLES, ORDER_INFO, TRANSACTIONS, BALANCES, FILLS
except ImportError:
    # This is to help the test run if the environment is not perfectly set up,
    # e.g. when cryptofeed is not installed in editable mode and tests are run directly.
    # For the agent's environment, this might not be strictly necessary if PYTHONPATH is set.
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from cryptofeed.backends.deltalake import DeltaLakeCallback
    from cryptofeed.backends._util import book_delta_convert
    from cryptofeed.defines import TRADES, TICKER, L2_BOOK, FUNDING, OPEN_INTEREST, LIQUIDATIONS, CANDLES, ORDER_INFO, TRANSACTIONS, BALANCES, FILLS


class TestDeltaLakeCallback(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        # Common minimal config
        self.config = {
            'table_name': os.path.join(self.temp_dir, 'test_table'),
            'dtype': TRADES, # Default dtype, will be overridden in specific tests
            'loop': self.mock_loop,
            'max_batch_size': 100, # Default for testing
            'flush_interval': 60 # Default for testing
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        # Ensure all writer tasks are awaited if any test fails to do so
        # This is a bit tricky as writer is a background task.
        # For unit tests, we often mock it out or control its execution directly.

    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    def test_initialization_valid_trades(self, MockDeltaTable, mock_write_deltalake):
        """Test basic initialization with TRADES dtype."""
        config = {**self.config, 'dtype': TRADES, 'table_name': os.path.join(self.temp_dir, 'trades_table')}
        cb = DeltaLakeCallback(**config)
        self.assertEqual(cb.table_path, os.path.join(self.temp_dir, 'trades_table'))
        self.assertEqual(cb.dtype, TRADES)
        self.assertIsInstance(cb.schema, pa.Schema)
        self.assertIn('timestamp', cb.schema.names)
        self.assertIn('symbol', cb.schema.names)
        self.assertIn('side', cb.schema.names)
        self.assertIn('amount', cb.schema.names)
        self.assertIn('price', cb.schema.names)
        self.assertIn('id', cb.schema.names)
        self.assertIn('receipt_timestamp', cb.schema.names)

    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    def test_initialization_valid_l2_book(self, MockDeltaTable, mock_write_deltalake):
        """Test basic initialization with L2_BOOK dtype."""
        config = {**self.config, 'dtype': L2_BOOK, 'table_name': os.path.join(self.temp_dir, 'l2book_table')}
        cb = DeltaLakeCallback(**config)
        self.assertEqual(cb.table_path, os.path.join(self.temp_dir, 'l2book_table'))
        self.assertEqual(cb.dtype, L2_BOOK)
        self.assertIsInstance(cb.schema, pa.Schema)
        # Check some L2_BOOK specific fields
        self.assertIn('symbol', cb.schema.names)
        self.assertIn('price', cb.schema.names)
        self.assertIn('side', cb.schema.names)
        self.assertIn('amount', cb.schema.names) # from book_delta_convert
        self.assertIn('receipt_timestamp', cb.schema.names)

    def test_initialization_missing_table_name(self):
        config = self.config.copy()
        del config['table_name']
        with self.assertRaisesRegex(ValueError, "'table_name' must be specified"):
            DeltaLakeCallback(**config)

    def test_initialization_invalid_dtype(self):
        config = {**self.config, 'dtype': 'INVALID_DTYPE'}
        with self.assertRaisesRegex(ValueError, "Invalid dtype"):
            DeltaLakeCallback(**config)

    def test_initialization_invalid_partition_by_type(self):
        config = {**self.config, 'partition_by': "not_a_list_or_str"}
        with self.assertRaisesRegex(ValueError, "'partition_by' must be a string or a list of strings"):
            DeltaLakeCallback(**config)

    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    def test_validate_configuration_custom_columns(self, MockDeltaTable, mock_write_deltalake):
        """Test validation with custom column names and types."""
        custom_columns = {'my_custom_id': 'id', 'price_usd': 'price'}
        custom_dtypes = {'my_custom_id': 'string', 'price_usd': 'float64', 'amount': 'float32'}
        config = {
            **self.config,
            'dtype': TRADES,
            'table_name': os.path.join(self.temp_dir, 'custom_trades'),
            'custom_columns': custom_columns,
            'custom_dtypes': custom_dtypes
        }
        cb = DeltaLakeCallback(**config)
        self.assertIn('my_custom_id', cb.schema.names)
        self.assertNotIn('id', cb.schema.names)
        self.assertIn('price_usd', cb.schema.names)
        self.assertNotIn('price', cb.schema.names)
        self.assertEqual(cb.schema.field('my_custom_id').type, pa.string())
        self.assertEqual(cb.schema.field('price_usd').type, pa.float64())
        self.assertEqual(cb.schema.field('amount').type, pa.float32()) # Original column, new type

    def test_default_z_order_cols(self):
        """Test _default_z_order_cols for various dtypes."""
        cb = DeltaLakeCallback(**self.config, dtype=TRADES)
        self.assertEqual(cb._default_z_order_cols(), ['timestamp', 'symbol'])

        cb = DeltaLakeCallback(**self.config, dtype=L2_BOOK)
        self.assertEqual(cb._default_z_order_cols(), ['receipt_timestamp', 'symbol']) # or whatever is default for book

        cb = DeltaLakeCallback(**self.config, dtype=TICKER)
        self.assertEqual(cb._default_z_order_cols(), ['timestamp', 'symbol'])

        cb = DeltaLakeCallback(**self.config, dtype=FUNDING)
        self.assertEqual(cb._default_z_order_cols(), ['timestamp', 'symbol'])

    def _get_sample_data(self, dtype, custom_ts=False):
        ts = datetime.utcnow()
        receipt_ts = ts.timestamp()
        data = {
            'exchange': 'test_exchange',
            'symbol': 'BTC-USD',
            'timestamp': receipt_ts if not custom_ts else ts, # some tests need datetime, others float
            'receipt_timestamp': receipt_ts
        }
        if dtype == TRADES:
            data.update({
                'side': 'buy', 'amount': Decimal('1.0'), 'price': Decimal('50000.0'), 'id': 'trade_123'
            })
        elif dtype == L2_BOOK:
            # L2_BOOK data is a list of dicts for book deltas typically
            # For _transform_columns, we expect a list of already processed book entries
            # This sample is more for what goes into book_delta_convert's output list
            return [
                {'exchange': 'test_exchange', 'symbol': 'BTC-USD', 'price': Decimal('50000.0'), 'side': 'bid', 'amount': Decimal('1.0'), 'receipt_timestamp': receipt_ts, 'timestamp': receipt_ts},
                {'exchange': 'test_exchange', 'symbol': 'BTC-USD', 'price': Decimal('50001.0'), 'side': 'ask', 'amount': Decimal('0.5'), 'receipt_timestamp': receipt_ts, 'timestamp': receipt_ts}
            ]
        elif dtype == TICKER:
            data.update({
                'bid': Decimal('49999.0'), 'ask': Decimal('50001.0')
            })
        elif dtype == FUNDING:
            data.update({
                 'rate': Decimal('0.001'), 'next_funding_time': receipt_ts + 3600, 'mark_price': Decimal('50000.0')
            })
        # Add other dtypes as needed for full coverage
        return [data] # _transform_columns expects a list of dicts

    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    def test_rename_custom_columns(self, MockDeltaTable, mock_write_deltalake):
        custom_columns = {'trade_id': 'id', 'trade_price': 'price'}
        config = {**self.config, 'dtype': TRADES, 'custom_columns': custom_columns, 'table_name': os.path.join(self.temp_dir, 'rename_table')}
        cb = DeltaLakeCallback(**config)

        data = self._get_sample_data(TRADES)[0] # single data dict
        original_id = data['id']
        original_price = data['price']

        df = pd.DataFrame([data])
        df = cb._rename_custom_columns(df)

        self.assertIn('trade_id', df.columns)
        self.assertIn('trade_price', df.columns)
        self.assertNotIn('id', df.columns)
        self.assertNotIn('price', df.columns)
        self.assertEqual(df['trade_id'].iloc[0], original_id)
        self.assertEqual(df['trade_price'].iloc[0], original_price)

    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    def test_convert_datetime_columns(self, MockDeltaTable, mock_write_deltalake):
        config = {**self.config, 'dtype': TRADES, 'table_name': os.path.join(self.temp_dir, 'dt_table')}
        cb = DeltaLakeCallback(**config)

        data = self._get_sample_data(TRADES, custom_ts=True)[0] # Needs datetime object for timestamp
        df = pd.DataFrame([data])

        # Manually set a column to be converted if not already datetime-like for testing
        df['custom_dt_col'] = datetime.utcnow()
        # Add a non-datetime column to ensure it's not affected
        df['non_dt_col'] = "some_string"

        # Update schema to include the new column for conversion if needed by _convert_datetime_columns logic
        # cb.schema might need to be adjusted or the method should rely on pre-defined datetime_cols
        # For now, assume 'timestamp' and 'receipt_timestamp' are the targets based on typical usage
        # Or, if custom_dtypes specifies a datetime, it should be converted.
        # Let's test implicit conversion of known datetime columns in the schema.

        # Simulate that 'timestamp' was initially float/object and needs conversion
        df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp() if isinstance(x, datetime) else x)
        df['receipt_timestamp'] = df['receipt_timestamp'].apply(lambda x: x.timestamp() if isinstance(x, datetime) else x)

        # Ensure schema reflects what _convert_datetime_columns expects
        # This part is tricky as schema is finalized early.
        # Let's assume cb.datetime_cols holds the target columns for conversion.
        cb.datetime_cols = ['timestamp', 'receipt_timestamp', 'custom_dt_col'] # Explicitly set for test

        df = cb._convert_datetime_columns(df.copy()) # Pass copy to avoid modifying original test data df

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['timestamp']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['receipt_timestamp']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['custom_dt_col']))
        self.assertFalse(pd.api.types.is_datetime64_any_dtype(df['non_dt_col']))


    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    def test_convert_int_columns(self, MockDeltaTable, mock_write_deltalake):
        config = {**self.config, 'dtype': FUNDING,
                  'custom_dtypes': {'next_funding_time': 'int64'}, # Example of int conversion
                  'table_name': os.path.join(self.temp_dir, 'int_conv_table')}
        cb = DeltaLakeCallback(**config)

        data = self._get_sample_data(FUNDING)[0]
        data['some_float_col'] = 123.0 # A float that can be int
        data['non_int_col'] = "123"   # A string that looks like int, but shouldn't be converted unless specified

        df = pd.DataFrame([data])

        # Assume cb.int_cols is populated based on custom_dtypes or defaults
        cb.int_cols = ['next_funding_time', 'some_float_col'] # Explicitly set for test

        df = cb._convert_int_columns(df.copy())

        self.assertTrue(pd.api.types.is_integer_dtype(df['next_funding_time']))
        self.assertEqual(df['next_funding_time'].iloc[0], int(data['next_funding_time']))
        self.assertTrue(pd.api.types.is_integer_dtype(df['some_float_col']))
        self.assertEqual(df['some_float_col'].iloc[0], 123)
        self.assertFalse(pd.api.types.is_integer_dtype(df['non_int_col'])) # Should remain object/string


    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    def test_ensure_partition_columns(self, MockDeltaTable, mock_write_deltalake):
        partition_cols = ['year', 'month', 'symbol']
        config = {**self.config, 'dtype': TRADES, 'partition_by': partition_cols, 'table_name': os.path.join(self.temp_dir, 'partition_table')}
        cb = DeltaLakeCallback(**config)

        data_list = self._get_sample_data(TRADES, custom_ts=True) # Needs datetime for year/month extraction
        df = pd.DataFrame(data_list)

        # Simulate datetime conversion happened
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['receipt_timestamp'] = pd.to_datetime(df['receipt_timestamp'], unit='s')

        df = cb._ensure_partition_columns(df.copy())

        for p_col in ['year', 'month']: # 'symbol' is already there
            self.assertIn(p_col, df.columns)
            self.assertTrue(pd.api.types.is_integer_dtype(df[p_col]) or pd.api.types.is_string_dtype(df[p_col])) # year/month can be string or int

        # Check values if possible (e.g., year of the timestamp)
        expected_year = data_list[0]['timestamp'].year
        self.assertEqual(df['year'].iloc[0], expected_year)


    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    def test_handle_missing_values(self, MockDeltaTable, mock_write_deltalake):
        config = {**self.config, 'dtype': TRADES, 'table_name': os.path.join(self.temp_dir, 'missing_val_table')}
        cb = DeltaLakeCallback(**config)

        data = self._get_sample_data(TRADES)[0]
        data['price'] = None # Simulate missing numeric value
        data['side'] = None  # Simulate missing string value
        df = pd.DataFrame([data])

        # Manually set dtypes as they would be before this step
        df['price'] = df['price'].astype(float) # Arrow schema expects float/double
        df['amount'] = df['amount'].astype(float)
        df['side'] = df['side'].astype(str)


        # Ensure schema matches what handle_missing_values expects
        # For this test, we assume the schema is already finalized.
        # cb.schema should have types like float64 for price, string for side.

        df = cb._handle_missing_values(df.copy())

        # Numeric columns with None should become NaN (which Arrow handles as null)
        self.assertTrue(pd.isna(df['price'].iloc[0]))
        # String columns with None should become empty string or remain None (Arrow handles as null)
        # Based on current _handle_missing_values, it seems to fill with type-specific defaults
        # For strings, it might fill with "" or allow pd.NA
        if cb.schema.field('side').type == pa.string():
             self.assertTrue(pd.isna(df['side'].iloc[0]) or df['side'].iloc[0] == "")

        # Ensure non-missing values are untouched
        self.assertFalse(pd.isna(df['amount'].iloc[0]))

    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    def test_transform_columns_integration(self, MockDeltaTable, mock_write_deltalake):
        """Test the main _transform_columns orchestrator method."""
        custom_cols = {'TRADE_ID': 'id'}
        partition_cols = ['symbol'] # Simple partition for testing
        custom_dtypes = {'price': 'float32', 'amount': 'float32', 'TRADE_ID': 'string'}

        config = {
            **self.config,
            'dtype': TRADES,
            'table_name': os.path.join(self.temp_dir, 'transform_all_table'),
            'custom_columns': custom_cols,
            'custom_dtypes': custom_dtypes,
            'partition_by': partition_cols
        }
        # Ensure 'dummy_int_col' will be part of the schema and targeted for int conversion
        # Add a dummy column to raw_data that will be converted to int
        raw_data = self._get_sample_data(TRADES, custom_ts=True) # timestamp as datetime
        for item in raw_data:
            item['dummy_int_val'] = 123.0 # float that can be int

        # Update custom_dtypes to include this new column for int conversion
        config['custom_dtypes']['dummy_int_val'] = 'int32'

        cb = DeltaLakeCallback(**config) # Re-initialize with updated config to rebuild schema

        df = pd.DataFrame(raw_data)

        # Mock the individual transformation methods to check if they are called
        cb._rename_custom_columns = MagicMock(wraps=cb._rename_custom_columns)
        cb._convert_datetime_columns = MagicMock(wraps=cb._convert_datetime_columns)
        cb._convert_int_columns = MagicMock(wraps=cb._convert_int_columns) # Now this should be called
        cb._ensure_partition_columns = MagicMock(wraps=cb._ensure_partition_columns)
        cb._handle_missing_values = MagicMock(wraps=cb._handle_missing_values)

        transformed_df = cb._transform_columns(df)

        cb._rename_custom_columns.assert_called_once()
        cb._convert_datetime_columns.assert_called_once()
        cb._convert_int_columns.assert_called_once() # Should be called now
        cb._ensure_partition_columns.assert_called_once()
        cb._handle_missing_values.assert_called_once()

        self.assertIsInstance(transformed_df, pa.Table)
        self.assertIn('TRADE_ID', transformed_df.schema.names)
        self.assertEqual(transformed_df.schema.field('TRADE_ID').type, pa.string())
        self.assertEqual(transformed_df.schema.field('price').type, pa.float32())
        self.assertEqual(transformed_df.schema.field('dummy_int_val').type, pa.int32())
        self.assertTrue(pa.types.is_timestamp(transformed_df.schema.field('timestamp').type))
        self.assertTrue(pa.types.is_timestamp(transformed_df.schema.field('receipt_timestamp').type))
        self.assertIn('symbol', transformed_df.schema.names) # partition column


    async def _run_writer_test_logic(self, cb, num_messages, max_batch_size, mock_write_deltalake):
        # Start the writer task
        writer_task = cb.loop.create_task(cb.writer())
        cb._writer_task = writer_task # Store for cancellation

        # Simulate adding data
        for i in range(num_messages):
            data = self._get_sample_data(TRADES, custom_ts=True)[0]
            data['id'] = f'trade_{i}' # Unique ID
            await cb.queue.put(({'data': data, 'type': TRADES, 'timestamp': data['timestamp']}))

        # Allow the writer to process the first batch
        for _ in range(max_batch_size + 5): # Give it a few cycles
            await asyncio.sleep(0) # Yield control

        self.assertGreaterEqual(mock_write_deltalake.call_count, 1)
        args, kwargs = mock_write_deltalake.call_args_list[0]
        self.assertEqual(args[0], cb.table_path)
        self.assertIsInstance(kwargs['data'], pa.Table)
        self.assertEqual(kwargs['data'].num_rows, max_batch_size)
        self.assertEqual(kwargs['mode'], 'append')

        # Add sentinel to stop writer and await its completion
        await cb.queue.put(None)
        try:
            await asyncio.wait_for(asyncio.shield(writer_task), timeout=1.0)
        except asyncio.TimeoutError:
            writer_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await writer_task

        if num_messages > max_batch_size and num_messages % max_batch_size > 0:
            self.assertGreaterEqual(mock_write_deltalake.call_count, 2, "Should have flushed remaining items")
            if mock_write_deltalake.call_count > 1:
                args_last, kwargs_last = mock_write_deltalake.call_args_list[-1]
                self.assertEqual(kwargs_last['data'].num_rows, num_messages % max_batch_size)
        elif num_messages == max_batch_size : # only one batch
             self.assertEqual(mock_write_deltalake.call_count, 1)


    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable') # Mock DeltaTable as it's used by writer
    @patch('asyncio.sleep', return_value=None) # Mock sleep
    def test_writer_batching_and_flushing_multiple_batches(self, mock_sleep, MockDeltaTable, mock_write_deltalake):
        max_batch_size = 5
        num_messages = max_batch_size + 2 # Should result in two batches
        config = {**self.config, 'dtype': TRADES, 'max_batch_size': max_batch_size, 'table_name': os.path.join(self.temp_dir, 'writer_multi_batch')}
        cb = DeltaLakeCallback(**config)
        cb.queue = asyncio.Queue()
        asyncio.run(self._run_writer_test_logic(cb, num_messages, max_batch_size, mock_write_deltalake))

    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    @patch('asyncio.sleep', return_value=None)
    def test_writer_batching_single_full_batch(self, mock_sleep, MockDeltaTable, mock_write_deltalake):
        max_batch_size = 3
        num_messages = max_batch_size # Exactly one full batch
        config = {**self.config, 'dtype': TRADES, 'max_batch_size': max_batch_size, 'table_name': os.path.join(self.temp_dir, 'writer_one_batch')}
        cb = DeltaLakeCallback(**config)
        cb.queue = asyncio.Queue()
        asyncio.run(self._run_writer_test_logic(cb, num_messages, max_batch_size, mock_write_deltalake))
        self.assertEqual(mock_write_deltalake.call_count, 1) # Should only be one call

    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    @patch('asyncio.sleep', return_value=None) # Mock sleep to speed up writer tests
    def test_writer_schema_evolution(self, mock_sleep, MockDeltaTable, mock_write_deltalake):
        """Test schema evolution handling."""
        config = {
            **self.config,
            'dtype': TRADES,
            'table_name': os.path.join(self.temp_dir, 'schema_evo_table'),
            'max_batch_size': 1,
            'schema_mode': 'overwrite' # Test overwrite schema mode
        }
        cb = DeltaLakeCallback(**config)
        cb.queue = asyncio.Queue()

        async def main_logic():
            writer_task = cb.loop.create_task(cb.writer())
            cb._writer_task = writer_task

            data1 = self._get_sample_data(TRADES, custom_ts=True)[0]
            await cb.queue.put(({'data': data1, 'type': TRADES, 'timestamp': data1['timestamp']}))
            await asyncio.sleep(0) # Yield to allow writer to process

            mock_write_deltalake.assert_called_once()
            _, kwargs1 = mock_write_deltalake.call_args
            self.assertEqual(kwargs1['schema'], cb.schema) # cb.schema is the Arrow schema
            self.assertEqual(kwargs1['mode'], 'append')
            self.assertEqual(kwargs1['overwrite_schema'], True)

            await cb.queue.put(None) # Stop writer
            try:
                await asyncio.wait_for(asyncio.shield(writer_task), timeout=1.0)
            except asyncio.TimeoutError:
                writer_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                   await writer_task

        asyncio.run(main_logic())


    @patch('deltalake.write_deltalake')
    @patch('deltalake.DeltaTable')
    @patch('asyncio.sleep', return_value=None) # Mock sleep
    def test_z_ordering(self, mock_sleep, MockDeltaTable, mock_write_deltalake):
        z_order_cols = ['symbol', 'side'] # Using existing schema fields for simplicity
        config = {
            **self.config,
            'dtype': TRADES,
            'table_name': os.path.join(self.temp_dir, 'zorder_table'),
            'max_batch_size': 1,
            'z_order_by': z_order_cols,
            # Add 'side' to custom_dtypes to ensure it's in the final schema if not already standard
            'custom_dtypes': {'side': 'string'}
        }
        cb = DeltaLakeCallback(**config) # Re-init with z_order_by config
        cb.queue = asyncio.Queue()

        mock_delta_table_instance = MockDeltaTable.return_value

        async def main_logic():
            writer_task = cb.loop.create_task(cb.writer())
            cb._writer_task = writer_task

            data = self._get_sample_data(TRADES, custom_ts=True)[0]
            await cb.queue.put(({'data': data, 'type': TRADES, 'timestamp': data['timestamp']}))
            await asyncio.sleep(0) # Yield

            mock_write_deltalake.assert_called_once()
            mock_delta_table_instance.optimize.assert_called_with(z_order_cols)

            await cb.queue.put(None) # Stop writer
            try:
                await asyncio.wait_for(asyncio.shield(writer_task), timeout=1.0)
            except asyncio.TimeoutError:
                writer_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                   await writer_task

        asyncio.run(main_logic())

if __name__ == '__main__':
    unittest.main()
