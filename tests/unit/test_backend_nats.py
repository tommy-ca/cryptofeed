import asyncio
import json
import unittest
from unittest.mock import patch, AsyncMock, MagicMock, call

from cryptofeed.backends.nats import (
    NATSCallback,
    TradeNATS,
    TickerNATS,
    BookNATS,
    FundingNATS,
    OpenInterestNATS,
    LiquidationsNATS,
    CandlesNATS,
    OrderInfoNATS,
    TransactionsNATS,
    BalancesNATS,
    FillsNATS
)
from cryptofeed.backends.backend import BackendBookCallback # For book_delta_convert
from cryptofeed.defines import TRADES, TICKER, L2_BOOK, FUNDING, OPEN_INTEREST, LIQUIDATIONS, CANDLES, ORDER_INFO, TRANSACTIONS, BALANCES, FILLS
from nats.errors import TimeoutError as NatsTimeoutError, NoServersError


class TestNATSCallbackInitialization(unittest.TestCase):
    def test_nats_callback_default_init(self):
        cb = NATSCallback()
        self.assertEqual(cb.addr, 'nats://localhost:4222')
        self.assertEqual(cb.subject_prefix, 'cryptofeed')
        self.assertFalse(cb.jetstream_mode)
        self.assertEqual(cb.jetstream_timeout, 2.0)
        self.assertIsNone(cb.nc)
        self.assertIsNone(cb.js)

    def test_nats_callback_custom_init(self):
        addr = 'nats://custom:1234'
        prefix = 'myprefix'
        js_mode = True
        js_timeout = 5.5
        cb = NATSCallback(addr=addr, subject_prefix=prefix, jetstream_mode=js_mode, jetstream_timeout=js_timeout)
        self.assertEqual(cb.addr, addr)
        self.assertEqual(cb.subject_prefix, prefix)
        self.assertTrue(cb.jetstream_mode)
        self.assertEqual(cb.jetstream_timeout, js_timeout)

    def test_nats_callback_list_addr_init(self):
        addrs = ['nats://server1:4222', 'nats://server2:4222']
        cb = NATSCallback(addr=addrs)
        self.assertEqual(cb.addr, addrs)


class TestNATSDataPublishing(unittest.IsolatedAsyncioTestCase):

    async def _run_writer_once(self, callback_instance, data_to_put_on_queue):
        """
        Helper to put one item on the queue and run the writer loop once.
        Assumes nats.connect and nc.publish are already mocked.
        """
        # Put data onto the callback's queue
        if data_to_put_on_queue: # Allow None if data is already in queue via __call__
            if isinstance(data_to_put_on_queue, list):
                for item in data_to_put_on_queue:
                    await callback_instance.queue.put(item)
            else:
                await callback_instance.queue.put(data_to_put_on_queue)

        # To ensure writer exits after processing our item(s), we can either:
        # 1. Cancel the writer_task (as done before).
        # 2. Have read_queue() raise a specific exception after yielding the item.
        # 3. Add a sentinel to the queue that the writer recognizes.
        # For simplicity with existing structure, task cancellation is used.
        # If the queue is empty after processing, read_queue() will return an empty list,
        # and the writer's inner loop will finish. The outer `while True` loop of writer
        # means we still need to cancel it to stop the test from hanging.

        writer_task = asyncio.create_task(callback_instance.writer())
        await asyncio.sleep(0.01) # Allow writer to start and process the item(s)
        writer_task.cancel()
        try:
            await writer_task
        except asyncio.CancelledError:
            pass # Expected


    @patch('nats.connect', new_callable=AsyncMock)
    async def test_trade_nats_publishing(self, mock_nats_connect):
        # by making read_queue return only the items we put, then an empty list.
        # This requires a bit of careful mocking if read_queue is an async generator.
        # A simpler way for testing is to ensure the queue has a sentinel or is empty after items.

        # To ensure writer exits after processing our item(s), we'll add a sentinel
        # or rely on the fact that read_queue will yield an empty list if queue is empty.
        # For this test, we'll assume one call to read_queue() is enough.
        # We need to cancel the writer task to make it stop.

        writer_task = asyncio.create_task(callback_instance.writer())
        await asyncio.sleep(0.01) # Allow writer to start and process the item
        writer_task.cancel()
        try:
            await writer_task
        except asyncio.CancelledError:
            pass


    @patch('nats.connect', new_callable=AsyncMock)
    @patch('nats.connect', new_callable=AsyncMock)
    async def test_data_publishing_modes(self, mock_nats_connect):
        sample_trade_data = {'feed': 'COINBASE', 'symbol': 'BTC-USD', 'timestamp': 1678886400.0, 'side': 'buy', 'amount': 0.1, 'price': 30000.0, 'id': 'trade123'}
        expected_subject = "test_trades-trades-COINBASE-BTC-USD"

        # Mock for core NATS connection
        mock_nc_core = AsyncMock()
        mock_nc_core.is_connected = True

        # Mock for JetStream context and its publish method
        mock_js_ctx = AsyncMock()
        mock_nc_jetstream = AsyncMock()
        mock_nc_jetstream.is_connected = True
        mock_nc_jetstream.jetstream.return_value = mock_js_ctx

        for jetstream_enabled_mode in [False, True]:
            with self.subTest(jetstream_mode=jetstream_enabled_mode):
                mock_nats_connect.reset_mock()
                mock_nc_core.publish.reset_mock()
                mock_nc_jetstream.jetstream.reset_mock()
                mock_js_ctx.publish.reset_mock()

                if jetstream_enabled_mode:
                    mock_nats_connect.return_value = mock_nc_jetstream
                else:
                    mock_nats_connect.return_value = mock_nc_core

                # Use a specific timeout for JetStream to test it's passed through
                js_timeout = 5.0 if jetstream_enabled_mode else 2.0 # Default if not JS

                callback = TradeNATS(
                    addr='test_addr',
                    subject_prefix='test_trades',
                    jetstream_mode=jetstream_enabled_mode,
                    jetstream_timeout=js_timeout
                )

                # __call__ puts data into the internal queue.
                await callback(sample_trade_data, 1678886400.1, 'COINBASE', 'BTC-USD')

                # _run_writer_once processes items from the queue.
                # Pass None because __call__ already put data on queue.
                await self._run_writer_once(callback, None)

                expected_payload = callback.serializer(sample_trade_data).encode()

                if jetstream_enabled_mode:
                    mock_nc_jetstream.jetstream.assert_called_once_with(timeout=js_timeout)
                    mock_js_ctx.publish.assert_called_once_with(
                        subject=expected_subject,
                        payload=expected_payload,
                        timeout=js_timeout
                    )
                    mock_nc_core.publish.assert_not_called() # Ensure core publish wasn't called
                else:
                    mock_nc_core.publish.assert_called_once_with(expected_subject, expected_payload)
                    mock_nc_jetstream.jetstream.assert_not_called() # Ensure jetstream wasn't called
                    mock_js_ctx.publish.assert_not_called()


    # The following tests for specific data types can be simplified if the above
    # test_data_publishing_modes is comprehensive.
    # However, keeping one or two to ensure the __call__ methods of subclasses
    # correctly format data for the writer is still useful.
    # For brevity, I will adapt one (e.g. BookNATS) and assume others would follow or be covered by generic test.

    @patch('nats.connect', new_callable=AsyncMock)
    async def test_book_nats_publishing_snapshot_jetstream_aware(self, mock_nats_connect):
        mock_js_ctx = AsyncMock()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.jetstream.return_value = mock_js_ctx
        mock_nats_connect.return_value = mock_nc

        js_timeout = 3.0

        for jetstream_enabled_mode in [False, True]:
            with self.subTest(jetstream_mode=jetstream_enabled_mode):
                mock_nc.publish.reset_mock()
                mock_nc.jetstream.reset_mock()
                mock_js_ctx.publish.reset_mock()

                callback = BookNATS(
                    addr='test_addr',
                    subject_prefix='test_books',
                    jetstream_mode=jetstream_enabled_mode,
                    jetstream_timeout=js_timeout
                )

                sample_snapshot = {'feed': 'COINBASE', 'symbol': 'BTC-USD', 'book': {'bids': {1.0: 2.0}, 'asks': {3.0: 4.0}}, 'delta': None, 'timestamp': 123.0, 'sequence_number': 1}
                converted_data = callback.book_delta_convert(sample_snapshot, 123.1, 'COINBASE', 'BTC-USD')

                # Manually put on queue for _run_writer_once
                await self._run_writer_once(callback, converted_data)

                expected_subject = "test_books-book-COINBASE-BTC-USD"
                expected_payload_dict = {'delta': None, 'book': {'bids': [[1.0, 2.0]], 'asks': [[3.0, 4.0]]}, 'sequence_number': 1, 'timestamp': 123.0}
                expected_payload = callback.serializer(expected_payload_dict).encode()

                if jetstream_enabled_mode:
                    mock_nc.jetstream.assert_called_once_with(timeout=js_timeout)
                    mock_js_ctx.publish.assert_called_once_with(subject=expected_subject, payload=expected_payload, timeout=js_timeout)
                    mock_nc.publish.assert_not_called()
                else:
                    mock_nc.publish.assert_called_once_with(expected_subject, expected_payload)
                    mock_nc.jetstream.assert_not_called()
                    mock_js_ctx.publish.assert_not_called()


    # Simplified generic test for other simple callback types using subtests for jetstream_mode
    async def _test_generic_nats_publishing(self, NatsClass, data_key_name, sample_data_payload, exchange, symbol, mock_nats_connect_parent):

        mock_js_ctx_generic = AsyncMock()
        mock_nc_generic = AsyncMock()
        mock_nc_generic.is_connected = True
        mock_nc_generic.jetstream.return_value = mock_js_ctx_generic

        js_timeout_generic = 2.5

        for jetstream_enabled_mode in [False, True]:
            with self.subTest(data_type=data_key_name, jetstream_mode=jetstream_enabled_mode):
                # Reset mocks for each subtest iteration
                mock_nats_connect_parent.reset_mock() # Reset the top-level mock passed in
                mock_nc_generic.publish.reset_mock()
                mock_nc_generic.jetstream.reset_mock()
                mock_js_ctx_generic.publish.reset_mock()

                # Configure the return value of the top-level mock for this subtest
                mock_nats_connect_parent.return_value = mock_nc_generic

                callback = NatsClass(
                    addr='test_addr',
                    subject_prefix=f'test_{data_key_name}',
                    jetstream_mode=jetstream_enabled_mode,
                    jetstream_timeout=js_timeout_generic
                )

                await callback(sample_data_payload, 1678886400.1, exchange, symbol)
                await self._run_writer_once(callback, None) # Data is on queue via __call__

                expected_subject = f"test_{data_key_name}-{data_key_name}-{exchange}-{symbol}"
                expected_payload_bytes = callback.serializer(sample_data_payload).encode()

                if jetstream_enabled_mode:
                    mock_nc_generic.jetstream.assert_called_once_with(timeout=js_timeout_generic)
                    mock_js_ctx_generic.publish.assert_called_once_with(
                        subject=expected_subject,
                        payload=expected_payload_bytes,
                        timeout=js_timeout_generic
                    )
                    mock_nc_generic.publish.assert_not_called()
                else:
                    mock_nc_generic.publish.assert_called_once_with(expected_subject, expected_payload_bytes)
                    mock_nc_generic.jetstream.assert_not_called()
                    mock_js_ctx_generic.publish.assert_not_called()


    @patch('nats.connect', new_callable=AsyncMock) # This mock is passed to _test_generic_nats_publishing
    async def test_other_simple_callbacks_jetstream_aware(self, mock_nats_connect):
        # Test OpenInterest
        await self._test_generic_nats_publishing(OpenInterestNATS, OPEN_INTEREST,
                                                 {'feed': 'KRAKEN', 'symbol': 'BTC-USD', 'open_interest': 1000.0, 'timestamp': 123.0},
                                                 'KRAKEN', 'BTC-USD', mock_nats_connect)
        mock_nats_connect.reset_mock()

        # Test Liquidations
        await self._test_generic_nats_publishing(LiquidationsNATS, LIQUIDATIONS,
                                                 {'feed': 'BINANCE_FUTURES', 'symbol': 'ETH-USDT', 'side': 'sell', 'quantity': 10.0, 'price': 2000.0, 'order_id': 'liq123', 'status': 'filled', 'timestamp': 123.0},
                                                 'BINANCE_FUTURES', 'ETH-USDT', mock_nats_connect)
        mock_nats_connect.reset_mock()

        # Test Candles
        await self._test_generic_nats_publishing(CandlesNATS, CANDLES,
                                                 {'feed': 'BYBIT', 'symbol': 'ADA-USD', 'timestamp': 1678886400.0, 'period': '1m', 'open': 0.3, 'high': 0.32, 'low': 0.29, 'close': 0.31, 'volume': 100000.0, 'trades': 100},
                                                 'BYBIT', 'ADA-USD', mock_nats_connect)
        mock_nats_connect.reset_mock()

        # Test OrderInfo
        await self._test_generic_nats_publishing(OrderInfoNATS, ORDER_INFO,
                                                 {'feed': 'DERIBIT', 'symbol': 'BTC-PERPETUAL', 'order_id': 'order1', 'side': 'buy', 'amount': 1.0, 'price': 30000, 'status': 'open', 'timestamp': 123.0},
                                                 'DERIBIT', 'BTC-PERPETUAL', mock_nats_connect)
        mock_nats_connect.reset_mock()

        # Test Transactions
        await self._test_generic_nats_publishing(TransactionsNATS, TRANSACTIONS,
                                                 {'feed': 'COINBASE', 'symbol': 'BTC', 'amount': -0.5, 'address': 'addr_to', 'tx_id': 'tx1', 'timestamp': 123.0},
                                                 'COINBASE', 'BTC', mock_nats_connect) # Symbol might be just currency for transactions
        mock_nats_connect.reset_mock()

        # Test Balances
        await self._test_generic_nats_publishing(BalancesNATS, BALANCES,
                                                 {'feed': 'GEMINI', 'asset': 'USD', 'balance': 1000.0, 'available': 800.0, 'timestamp': 123.0},
                                                 'GEMINI', 'USD', mock_nats_connect) # Symbol might be asset name
        mock_nats_connect.reset_mock()

        # Test Fills
        await self._test_generic_nats_publishing(FillsNATS, FILLS,
                                                 {'feed': 'KRAKEN_FUTURES', 'symbol': 'LTC-USD', 'order_id': 'order2', 'trade_id': 'trade_fill1', 'side': 'sell', 'amount': 5.0, 'price': 100.0, 'fee': 0.1, 'fee_currency': 'USD', 'timestamp': 123.0},
                                                 'KRAKEN_FUTURES', 'LTC-USD', mock_nats_connect)


class TestNATSConnectionHandling(unittest.IsolatedAsyncioTestCase):

    @patch('asyncio.sleep', new_callable=AsyncMock) # To check retry delays
    @patch('nats.connect', new_callable=AsyncMock)
    async def test_writer_connects_successfully(self, mock_nats_connect, mock_asyncio_sleep):
        mock_nc = AsyncMock()
        mock_nc.is_connected = True

        mock_js_ctx = AsyncMock() # Mock JetStream context
        mock_nc.jetstream.return_value = mock_js_ctx # nc.jetstream() returns our mock_js_ctx

        mock_nats_connect.return_value = mock_nc

        for jetstream_enabled_mode in [False, True]:
            with self.subTest(jetstream_mode=jetstream_enabled_mode):
                mock_nc.publish.reset_mock()
                mock_nc.jetstream.reset_mock()
                mock_js_ctx.publish.reset_mock()
                mock_asyncio_sleep.reset_mock() # Reset sleep for each subtest
                mock_nats_connect.reset_mock() # Reset nats.connect itself
                mock_nats_connect.return_value = mock_nc # Set it again for this subtest

                callback = NATSCallback(addr='test_addr', jetstream_mode=jetstream_enabled_mode, jetstream_timeout=3.0)
                await callback.queue.put({'data_type': 'test', 'exchange': 'test', 'symbol': 'test', 'data': {}})

                writer_task = asyncio.create_task(callback.writer())
                await asyncio.sleep(0.01) # allow writer to start and process
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass

                mock_nats_connect.assert_called_once() # nats.connect should be called once
                if jetstream_enabled_mode:
                    mock_nc.jetstream.assert_called_once_with(timeout=3.0)
                    mock_js_ctx.publish.assert_called_once() # JS publish was attempted
                    mock_nc.publish.assert_not_called() # Core NATS publish not called
                else:
                    mock_nc.jetstream.assert_not_called()
                    mock_js_ctx.publish.assert_not_called()
                    mock_nc.publish.assert_called_once() # Core NATS publish was attempted

                mock_asyncio_sleep.assert_not_called() # No errors, so no sleep for retries


    @patch('logging.Logger.error')
    @patch('nats.connect', new_callable=AsyncMock)
    async def test_connect_jetstream_context_fails(self, mock_nats_connect, mock_log_error):
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.jetstream.side_effect = Exception("JS context creation failed") # Simulate error
        mock_nats_connect.return_value = mock_nc

        callback = NATSCallback(addr='test_addr', jetstream_mode=True, jetstream_timeout=1.0)

        # _connect is called by writer or directly
        await callback._connect()

        mock_nc.jetstream.assert_called_once_with(timeout=1.0)
        self.assertIsNone(callback.js) # JS context should be None
        mock_log_error.assert_any_call(
            "NATSBackend: Failed to get JetStream context: %s. JetStream publishing will likely fail.",
            "JS context creation failed"
        )

    @patch('asyncio.sleep', new_callable=AsyncMock) # For reconnected_cb potential sleep, not here
    @patch('nats.connect', new_callable=AsyncMock) # Not used directly, but for consistency
    async def test_reconnected_cb_jetstream_handling(self, mock_nats_connect_unused, mock_asyncio_sleep_unused):
        # This test needs to mock the NATS connection (nc) and its jetstream method
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.connected_url.netloc = "testserver:4222" # For log message

        mock_js_ctx = AsyncMock()
        mock_nc.jetstream.return_value = mock_js_ctx

        js_timeout = 4.0
        callback = NATSCallback(addr='test_addr', jetstream_mode=True, jetstream_timeout=js_timeout)
        callback.nc = mock_nc # Simulate an active connection object

        # Call _reconnected_cb
        await callback._reconnected_cb()

        mock_nc.jetstream.assert_called_once_with(timeout=js_timeout)
        self.assertEqual(callback.js, mock_js_ctx) # JS context should be set

        # Test failure case for re-establishing JS context
        mock_nc.jetstream.reset_mock()
        mock_nc.jetstream.side_effect = Exception("JS re-creation failed")
        callback.js = None # Reset js before calling reconnected_cb again

        await callback._reconnected_cb()
        mock_nc.jetstream.assert_called_once_with(timeout=js_timeout)
        self.assertIsNone(callback.js) # Should be None after failure
        # Check for log is also good, but needs @patch('logging.Logger.error')


    @patch('logging.Logger.warning')
    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('nats.connect', new_callable=AsyncMock)
    async def test_writer_handles_natstimeouterror_on_connect(self, mock_nats_connect, mock_asyncio_sleep, mock_log_warning):
        # Simulate NatsTimeoutError on first connect, then success
        mock_nc_success = AsyncMock()
        mock_nc_success.is_connected = True
        mock_nc_success.jetstream = AsyncMock() # For JetStream mode
        mock_nats_connect.side_effect = [NatsTimeoutError("Connection timeout"), mock_nc_success]

        for jetstream_enabled_mode in [False, True]:
            with self.subTest(jetstream_mode=jetstream_enabled_mode):
                mock_nats_connect.reset_mock() # Reset for side_effect
                mock_nats_connect.side_effect = [NatsTimeoutError("Connection timeout"), mock_nc_success]
                mock_asyncio_sleep.reset_mock()
                mock_nc_success.publish.reset_mock()
                mock_nc_success.jetstream.reset_mock() # Reset jetstream mock call count

                callback = NATSCallback(addr='test_addr', jetstream_mode=jetstream_enabled_mode)
                await callback.queue.put({'data_type': 'test', 'exchange': 'test', 'symbol': 'test', 'data': {}})

                writer_task = asyncio.create_task(callback.writer())
                # Let it try, fail, sleep, and then succeed (connect + publish attempt)
                # Sleep duration needs to accommodate initial attempt, sleep, and processing of one item.
                await asyncio.sleep(0.01 + callback.jetstream_timeout + 0.1 + 1.1) # Approximation
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass

                self.assertEqual(mock_nats_connect.call_count, 2) # First attempt failed, second succeeded
                mock_asyncio_sleep.assert_called_once_with(1) # Default retry delay


    @patch('logging.Logger.error')
    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('nats.connect', new_callable=AsyncMock)
    async def test_writer_handles_noserverserror_on_connect(self, mock_nats_connect, mock_asyncio_sleep, mock_log_error):
        mock_nc_success = AsyncMock()
        mock_nc_success.is_connected = True
        mock_nc_success.jetstream = AsyncMock()
        mock_nats_connect.side_effect = [NoServersError("No servers available"), mock_nc_success]

        for jetstream_enabled_mode in [False, True]:
            with self.subTest(jetstream_mode=jetstream_enabled_mode):
                mock_nats_connect.reset_mock()
                mock_nats_connect.side_effect = [NoServersError("No servers available"), mock_nc_success]
                mock_asyncio_sleep.reset_mock()
                mock_nc_success.publish.reset_mock()
                mock_nc_success.jetstream.reset_mock()

                callback = NATSCallback(addr='test_addr', jetstream_mode=jetstream_enabled_mode)
                await callback.queue.put({'data_type': 'test', 'exchange': 'test', 'symbol': 'test', 'data': {}})

                writer_task = asyncio.create_task(callback.writer())
                await asyncio.sleep(0.01 + callback.jetstream_timeout + 0.1 + 1.1)
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass

                self.assertEqual(mock_nats_connect.call_count, 2)
                mock_asyncio_sleep.assert_called_once_with(1)

    @patch('nats.connect', new_callable=AsyncMock)
    async def test_writer_handles_publish_connection_closed(self, mock_nats_connect):
        # This test needs to handle both core NATS and JetStream paths

        sample_data_dict = {'feed': 'COINBASE', 'symbol': 'BTC-USD', 'timestamp': 1.0, 'side': 'buy', 'amount': 1, 'price': 1}
        js_timeout = 2.0

        for jetstream_enabled_mode in [False, True]:
            with self.subTest(jetstream_mode=jetstream_enabled_mode):
                mock_nats_connect.reset_mock()

                # Setup for first connection attempt (will fail during publish)
                mock_nc_fail_publish = AsyncMock()
                mock_nc_fail_publish.is_connected = True
                mock_js_ctx_fail_publish = AsyncMock()
                mock_nc_fail_publish.jetstream.return_value = mock_js_ctx_fail_publish

                if jetstream_enabled_mode:
                    mock_js_ctx_fail_publish.publish.side_effect = ConnectionClosedError("JS Pub Connection Closed")
                else:
                    mock_nc_fail_publish.publish.side_effect = ConnectionClosedError("Core Pub Connection Closed")

                # Setup for second connection attempt (will succeed)
                mock_nc_reconnect_ok = AsyncMock()
                mock_nc_reconnect_ok.is_connected = True
                mock_js_ctx_reconnect_ok = AsyncMock()
                mock_nc_reconnect_ok.jetstream.return_value = mock_js_ctx_reconnect_ok
                # mock_js_ctx_reconnect_ok.publish is an empty AsyncMock, will just record call

                mock_nats_connect.side_effect = [mock_nc_fail_publish, mock_nc_reconnect_ok]

                callback = TradeNATS(addr='test_addr', jetstream_mode=jetstream_enabled_mode, jetstream_timeout=js_timeout)
                await callback(sample_data_dict, 1.1, 'COINBASE', 'BTC-USD') # Puts on queue

                writer_task = asyncio.create_task(callback.writer())
                await asyncio.sleep(0.1) # Allow writer to process, fail, reconnect, retry
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass

                self.assertEqual(mock_nats_connect.call_count, 2) # Initial connect + reconnect attempt

                if jetstream_enabled_mode:
                    mock_js_ctx_fail_publish.publish.assert_called_once() # First attempt
                    mock_nc_reconnect_ok.jetstream.assert_called_once_with(timeout=js_timeout) # JS context on reconnect
                    mock_js_ctx_reconnect_ok.publish.assert_called_once() # Second attempt (retry)
                    mock_nc_fail_publish.publish.assert_not_called() # Core publish not used
                    mock_nc_reconnect_ok.publish.assert_not_called()
                else:
                    mock_nc_fail_publish.publish.assert_called_once() # First attempt
                    mock_nc_reconnect_ok.publish.assert_called_once() # Second attempt (retry)
                    mock_nc_fail_publish.jetstream.assert_not_called() # JS not used
                    mock_nc_reconnect_ok.jetstream.assert_not_called()


    @patch('logging.Logger.warning')
    @patch('nats.connect', new_callable=AsyncMock)
    async def test_writer_handles_jetstream_publish_timeout(self, mock_nats_connect, mock_log_warning):
        # This test is specific to JetStream mode
        mock_js_ctx = AsyncMock()
        mock_js_ctx.publish.side_effect = NatsTimeoutError("JS Ack Timeout") # Simulate ACK timeout

        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.jetstream.return_value = mock_js_ctx
        mock_nats_connect.return_value = mock_nc

        js_timeout = 0.5 # Use a small timeout for the test config
        callback = TradeNATS(addr='test_addr', jetstream_mode=True, jetstream_timeout=js_timeout)
        sample_data = {'feed': 'COINBASE', 'symbol': 'BTC-USD', 'timestamp': 1.0, 'side': 'buy', 'amount': 1, 'price': 1}
        await callback(sample_data, 1.1, 'COINBASE', 'BTC-USD')

        await self._run_writer_once(callback, None) # Process the item

        mock_js_ctx.publish.assert_called_once()
        # Check that a warning about timeout was logged
        # Example: LOG.warning("NATSBackend: Timeout publishing to %s (JetStream: %s): %s. Message may not have been delivered or ack lost.", subject_name, self.jetstream_mode, e_timeout)
        expected_subject = f"{callback.subject_prefix}-trades-COINBASE-BTC-USD"
        mock_log_warning.assert_any_call(
            "NATSBackend: Timeout publishing to %s (JetStream: %s): %s. Message may not have been delivered or ack lost.",
            expected_subject, True, "JS Ack Timeout"
        )
        # Ensure no retry of the same message was attempted automatically by this error handler
        # (js.publish was called only once)


if __name__ == '__main__':
    unittest.main()
