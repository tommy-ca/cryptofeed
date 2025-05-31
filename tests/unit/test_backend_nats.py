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
        self.assertIsNone(cb.nc)

    def test_nats_callback_custom_init(self):
        addr = 'nats://custom:1234'
        prefix = 'myprefix'
        cb = NATSCallback(addr=addr, subject_prefix=prefix)
        self.assertEqual(cb.addr, addr)
        self.assertEqual(cb.subject_prefix, prefix)

    def test_nats_callback_list_addr_init(self):
        addrs = ['nats://server1:4222', 'nats://server2:4222']
        cb = NATSCallback(addr=addrs)
        self.assertEqual(cb.addr, addrs)


class TestNATSDataPublishing(unittest.IsolatedAsyncioTestCase):

    async def _run_writer_once(self, callback_instance, data_to_put):
        """
        Helper to put one item on the queue and run the writer loop once.
        Assumes nats.connect and nc.publish are already mocked.
        """
        # Put data onto the callback's queue
        if isinstance(data_to_put, list):
            for item in data_to_put:
                await callback_instance.queue.put(item)
        else:
            await callback_instance.queue.put(data_to_put)

        # Stop the writer after processing one batch from read_queue
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
    async def test_trade_nats_publishing(self, mock_nats_connect):
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nats_connect.return_value = mock_nc

        callback = TradeNATS(addr='test_addr', subject_prefix='test_trades')
        sample_data = {'feed': 'COINBASE', 'symbol': 'BTC-USD', 'timestamp': 1678886400.0, 'side': 'buy', 'amount': 0.1, 'price': 30000.0, 'id': 'trade123'}

        # Construct the expected data for NATSCallback queue (wrapping)
        # The actual NATSCallback.writer() expects a dict with 'data_type', 'exchange', 'symbol', 'data'
        # This wrapping is usually done by the FeedHandler or the base callback's __call__
        # For direct testing of writer through queue, we need to simulate this structure.
        # However, the subclasses like TradeNATS have their __call__ method that does this.

        # Let's use the __call__ method to put data into queue
        await callback(sample_data, 1678886400.1, 'COINBASE', 'BTC-USD')

        await self._run_writer_once(callback, None) # Data is already in queue via __call__

        expected_subject = "test_trades-trades-COINBASE-BTC-USD"
        expected_payload = callback.serializer(sample_data).encode()
        mock_nc.publish.assert_called_once_with(expected_subject, expected_payload)


    @patch('nats.connect', new_callable=AsyncMock)
    async def test_ticker_nats_publishing(self, mock_nats_connect):
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nats_connect.return_value = mock_nc

        callback = TickerNATS(addr='test_addr', subject_prefix='test_tickers')
        sample_data = {'feed': 'COINBASE', 'symbol': 'BTC-USD', 'bid': 29999.0, 'ask': 30001.0, 'timestamp': 1678886400.0}

        await callback(sample_data, 1678886400.1, 'COINBASE', 'BTC-USD')
        await self._run_writer_once(callback, None)

        expected_subject = "test_tickers-ticker-COINBASE-BTC-USD"
        expected_payload = callback.serializer(sample_data).encode()
        mock_nc.publish.assert_called_once_with(expected_subject, expected_payload)


    @patch('nats.connect', new_callable=AsyncMock)
    async def test_book_nats_publishing_snapshot(self, mock_nats_connect):
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nats_connect.return_value = mock_nc

        callback = BookNATS(addr='test_addr', subject_prefix='test_books')
        # BackendBookCallback creates a book object internally.
        # We need to simulate a book snapshot being passed.
        # The __call__ method of BackendBookCallback handles the book and then calls write()
        # which puts the data into the queue for NATSCallback.writer()
        sample_snapshot = {'feed': 'COINBASE', 'symbol': 'BTC-USD', 'book': {'bids': {1.0: 2.0}, 'asks': {3.0: 4.0}}, 'delta': None, 'timestamp': 123.0, 'sequence_number': 1}

        # Simulate what BackendBookCallback.__call__ does before putting on queue
        # It calls self.book_delta_convert for snapshots
        converted_data = callback.book_delta_convert(sample_snapshot, 123.1, 'COINBASE', 'BTC-USD')
        await callback.queue.put(converted_data) # Manually put the correctly structured item

        await self._run_writer_once(callback, None)

        expected_subject = "test_books-book-COINBASE-BTC-USD"
        # The serializer in BookNATS (from BackendBookCallback) will process the book data.
        # For a snapshot, it should serialize the 'book' field.
        expected_payload_dict = {'delta': None, 'book': {'bids': [[1.0, 2.0]], 'asks': [[3.0, 4.0]]}, 'sequence_number': 1, 'timestamp': 123.0}
        expected_payload = callback.serializer(expected_payload_dict).encode()

        mock_nc.publish.assert_called_once_with(expected_subject, expected_payload)

    @patch('nats.connect', new_callable=AsyncMock)
    async def test_book_nats_publishing_delta(self, mock_nats_connect):
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nats_connect.return_value = mock_nc

        callback = BookNATS(addr='test_addr', subject_prefix='test_books')
        sample_delta = {'feed': 'COINBASE', 'symbol': 'BTC-USD', 'book': None, 'delta': {'bids': [(1.0, 0.0)], 'asks': [(3.0, 5.0)]}, 'timestamp': 124.0, 'sequence_number': 2}

        converted_data = callback.book_delta_convert(sample_delta, 124.1, 'COINBASE', 'BTC-USD')
        await callback.queue.put(converted_data)

        await self._run_writer_once(callback, None)

        expected_subject = "test_books-book-COINBASE-BTC-USD"
        expected_payload_dict = {'delta': {'bids': [['1.0', '0.0']], 'asks': [['3.0', '5.0']]}, 'book': None, 'sequence_number': 2, 'timestamp': 124.0}
        expected_payload = callback.serializer(expected_payload_dict).encode()

        mock_nc.publish.assert_called_once_with(expected_subject, expected_payload)


    @patch('nats.connect', new_callable=AsyncMock)
    async def test_funding_nats_publishing(self, mock_nats_connect):
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nats_connect.return_value = mock_nc

        callback = FundingNATS(addr='test_addr', subject_prefix='test_funding')
        sample_data = {'feed': 'BITMEX', 'symbol': 'XBTUSD', 'timestamp': 1678886400.0, 'rate': 0.0001, 'next_funding_time': 1678890000.0}

        await callback(sample_data, 1678886400.1, 'BITMEX', 'XBTUSD')
        await self._run_writer_once(callback, None)

        expected_subject = "test_funding-funding-BITMEX-XBTUSD"
        expected_payload = callback.serializer(sample_data).encode()
        mock_nc.publish.assert_called_once_with(expected_subject, expected_payload)

    # Simplified generic test for other simple callback types
    async def _test_generic_nats_publishing(self, NatsClass, data_key_name, sample_data_payload, exchange, symbol, mock_nats_connect):
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nats_connect.return_value = mock_nc

        callback = NatsClass(addr='test_addr', subject_prefix=f'test_{data_key_name}')

        await callback(sample_data_payload, 1678886400.1, exchange, symbol)
        await self._run_writer_once(callback, None)

        expected_subject = f"test_{data_key_name}-{data_key_name}-{exchange}-{symbol}"
        expected_payload_bytes = callback.serializer(sample_data_payload).encode()
        mock_nc.publish.assert_called_once_with(expected_subject, expected_payload_bytes)

    @patch('nats.connect', new_callable=AsyncMock)
    async def test_other_simple_callbacks(self, mock_nats_connect):
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
        mock_nats_connect.return_value = mock_nc

        callback = NATSCallback(addr='test_addr')
        # Put a dummy item to make the writer run its loop once
        await callback.queue.put({'data_type': 'test', 'exchange': 'test', 'symbol': 'test', 'data': {}})

        writer_task = asyncio.create_task(callback.writer())
        await asyncio.sleep(0.01) # allow writer to start and process
        writer_task.cancel()
        try:
            await writer_task
        except asyncio.CancelledError:
            pass

        mock_nats_connect.assert_called_once()
        mock_nc.publish.assert_called_once() # Check that publish was attempted
        mock_asyncio_sleep.assert_not_called() # No errors, so no sleep for retries

    @patch('logging.Logger.warning') # To check log messages for retries
    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('nats.connect', new_callable=AsyncMock)
    async def test_writer_handles_natstimeouterror_on_connect(self, mock_nats_connect, mock_asyncio_sleep, mock_log_warning):
        # Simulate NatsTimeoutError on first connect, then success
        mock_nc_success = AsyncMock()
        mock_nc_success.is_connected = True
        mock_nats_connect.side_effect = [NatsTimeoutError("Connection timeout"), mock_nc_success]

        callback = NATSCallback(addr='test_addr')
        await callback.queue.put({'data_type': 'test', 'exchange': 'test', 'symbol': 'test', 'data': {}})

        writer_task = asyncio.create_task(callback.writer())
        # Let it try, fail, sleep, and then succeed
        await asyncio.sleep(0.01 + 1.1) # Initial attempt + first retry delay (default 1s)
        writer_task.cancel()
        try:
            await writer_task
        except asyncio.CancelledError:
            pass

        self.assertEqual(mock_nats_connect.call_count, 2) # First attempt failed, second succeeded
        mock_asyncio_sleep.assert_called_once_with(1) # Default retry delay
        # Check if a warning about timeout was logged (optional, based on actual log messages)
        # mock_log_warning.assert_any_call(something like "NATS connection timeout. Retrying...")


    @patch('logging.Logger.error') # To check log messages for retries
    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('nats.connect', new_callable=AsyncMock)
    async def test_writer_handles_noserverserror_on_connect(self, mock_nats_connect, mock_asyncio_sleep, mock_log_error):
        mock_nc_success = AsyncMock()
        mock_nc_success.is_connected = True
        mock_nats_connect.side_effect = [NoServersError("No servers available"), mock_nc_success]

        callback = NATSCallback(addr='test_addr')
        await callback.queue.put({'data_type': 'test', 'exchange': 'test', 'symbol': 'test', 'data': {}})

        writer_task = asyncio.create_task(callback.writer())
        await asyncio.sleep(0.01 + 1.1)
        writer_task.cancel()
        try:
            await writer_task
        except asyncio.CancelledError:
            pass

        self.assertEqual(mock_nats_connect.call_count, 2)
        mock_asyncio_sleep.assert_called_once_with(1)
        # mock_log_error.assert_any_call(something like "No NATS servers available. Retrying...")

    @patch('nats.connect', new_callable=AsyncMock)
    async def test_writer_handles_publish_connection_closed(self, mock_nats_connect):
        mock_nc = AsyncMock()
        mock_nc.is_connected = True # Initially connected
        # First publish raises ConnectionClosedError, second succeeds
        # Need connect to be called again after error
        mock_nc.publish.side_effect = [NatsTimeoutError("Publish timeout"), AsyncMock()] # Using NatsTimeoutError as ConnectionClosedError is similar for this test path

        mock_nc_reconnect = AsyncMock() # Simulate a new connection object after reconnect
        mock_nc_reconnect.is_connected = True

        mock_nats_connect.side_effect = [mock_nc, mock_nc_reconnect] # Initial connect, then reconnect

        callback = TradeNATS(addr='test_addr')
        sample_data = {'feed': 'COINBASE', 'symbol': 'BTC-USD', 'timestamp': 1.0, 'side': 'buy', 'amount': 1, 'price': 1}
        await callback(sample_data, 1.1, 'COINBASE', 'BTC-USD') # Puts on queue

        writer_task = asyncio.create_task(callback.writer())
        await asyncio.sleep(0.1) # Allow writer to process, fail publish, reconnect, and retry publish
        writer_task.cancel()
        try:
            await writer_task
        except asyncio.CancelledError:
            pass

        self.assertEqual(mock_nats_connect.call_count, 2) # Initial connect + reconnect
        self.assertEqual(mock_nc.publish.call_count, 1) # First attempt on original connection
        self.assertEqual(mock_nc_reconnect.publish.call_count, 1) # Second attempt on new connection


if __name__ == '__main__':
    unittest.main()
