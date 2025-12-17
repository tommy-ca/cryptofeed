'''
Unit tests for Binance field extraction (REQ-1: Schema Field Population)

Tests verify extraction of new protobuf v2beta1 fields from Binance WebSocket messages:
- Trade: maker, event_time, match_id
- OrderBook: event_time, last_update_id
'''
from decimal import Decimal
import pytest

from cryptofeed.exchanges.binance import Binance
from cryptofeed.types import Trade, OrderBook


class TestBinanceTradeFieldExtraction:
    """Test Trade field extraction from Binance WebSocket messages"""

    @pytest.mark.asyncio
    async def test_trade_extracts_maker_field_true(self):
        """Verify maker field extracted when 'm' is True (buyer is maker)"""
        binance = Binance()

        # Mock Binance trade message with 'm': True
        msg = {
            'e': 'aggTrade',
            'E': 1234567890000,  # Event time in milliseconds
            's': 'BTCUSDT',
            'a': 12345,  # Aggregate trade ID
            'p': '50000.00',
            'q': '1.0',
            'f': 100,
            'l': 105,
            'T': 1234567885000,  # Trade time
            'm': True,  # Buyer is maker
            'M': True
        }

        # Call _trade handler
        trade_obj = None
        async def capture_trade(trade, timestamp):
            nonlocal trade_obj
            trade_obj = trade

        binance.callbacks = {'trades': [capture_trade]}
        await binance._trade(msg, timestamp=1234567890.0)

        # Verify maker field is True
        assert trade_obj is not None
        assert trade_obj.maker is True

    @pytest.mark.asyncio
    async def test_trade_extracts_maker_field_false(self):
        """Verify maker field extracted when 'm' is False (buyer is taker)"""
        binance = Binance()

        msg = {
            'e': 'aggTrade',
            'E': 1234567890000,
            's': 'BTCUSDT',
            'a': 12345,
            'p': '50000.00',
            'q': '1.0',
            'f': 100,
            'l': 105,
            'T': 1234567885000,
            'm': False,  # Buyer is taker
            'M': False
        }

        trade_obj = None
        async def capture_trade(trade, timestamp):
            nonlocal trade_obj
            trade_obj = trade

        binance.callbacks = {'trades': [capture_trade]}
        await binance._trade(msg, timestamp=1234567890.0)

        assert trade_obj is not None
        assert trade_obj.maker is False

    @pytest.mark.asyncio
    async def test_trade_extracts_event_time(self):
        """Verify event_time extracted from 'E' field and converted to seconds"""
        binance = Binance()

        msg = {
            'e': 'aggTrade',
            'E': 1234567890000,  # Milliseconds
            's': 'BTCUSDT',
            'a': 12345,
            'p': '50000.00',
            'q': '1.0',
            'f': 100,
            'l': 105,
            'T': 1234567885000,
            'm': True,
            'M': True
        }

        trade_obj = None
        async def capture_trade(trade, timestamp):
            nonlocal trade_obj
            trade_obj = trade

        binance.callbacks = {'trades': [capture_trade]}
        await binance._trade(msg, timestamp=1234567890.0)

        assert trade_obj is not None
        # Verify conversion from milliseconds to seconds
        assert trade_obj.event_time == 1234567890.0  # 1234567890000 / 1000

    @pytest.mark.asyncio
    async def test_trade_extracts_match_id(self):
        """Verify match_id extracted from 'a' field (aggregate trade ID)"""
        binance = Binance()

        msg = {
            'e': 'aggTrade',
            'E': 1234567890000,
            's': 'BTCUSDT',
            'a': 98765,  # Aggregate trade ID
            'p': '50000.00',
            'q': '1.0',
            'f': 100,
            'l': 105,
            'T': 1234567885000,
            'm': True,
            'M': True
        }

        trade_obj = None
        async def capture_trade(trade, timestamp):
            nonlocal trade_obj
            trade_obj = trade

        binance.callbacks = {'trades': [capture_trade]}
        await binance._trade(msg, timestamp=1234567890.0)

        assert trade_obj is not None
        assert trade_obj.match_id == "98765"  # Converted to string

    @pytest.mark.asyncio
    async def test_trade_handles_missing_maker_field(self):
        """Verify graceful degradation when 'm' field is missing"""
        binance = Binance()

        # Message without 'm' field
        msg = {
            'e': 'aggTrade',
            'E': 1234567890000,
            's': 'BTCUSDT',
            'a': 12345,
            'p': '50000.00',
            'q': '1.0',
            'f': 100,
            'l': 105,
            'T': 1234567885000,
            'M': True
        }

        trade_obj = None
        async def capture_trade(trade, timestamp):
            nonlocal trade_obj
            trade_obj = trade

        binance.callbacks = {'trades': [capture_trade]}
        await binance._trade(msg, timestamp=1234567890.0)

        assert trade_obj is not None
        # Field should be None when not in message
        assert trade_obj.maker is None

    @pytest.mark.asyncio
    async def test_trade_handles_missing_event_time(self):
        """Verify graceful degradation when 'E' field is missing"""
        binance = Binance()

        msg = {
            'e': 'aggTrade',
            's': 'BTCUSDT',
            'a': 12345,
            'p': '50000.00',
            'q': '1.0',
            'f': 100,
            'l': 105,
            'T': 1234567885000,
            'm': True,
            'M': True
        }

        trade_obj = None
        async def capture_trade(trade, timestamp):
            nonlocal trade_obj
            trade_obj = trade

        binance.callbacks = {'trades': [capture_trade]}
        await binance._trade(msg, timestamp=1234567890.0)

        assert trade_obj is not None
        assert trade_obj.event_time is None

    @pytest.mark.asyncio
    async def test_trade_handles_missing_match_id(self):
        """Verify graceful degradation when 'a' field is missing"""
        binance = Binance()

        msg = {
            'e': 'aggTrade',
            'E': 1234567890000,
            's': 'BTCUSDT',
            'p': '50000.00',
            'q': '1.0',
            'f': 100,
            'l': 105,
            'T': 1234567885000,
            'm': True,
            'M': True
        }

        trade_obj = None
        async def capture_trade(trade, timestamp):
            nonlocal trade_obj
            trade_obj = trade

        binance.callbacks = {'trades': [capture_trade]}
        await binance._trade(msg, timestamp=1234567890.0)

        assert trade_obj is not None
        assert trade_obj.match_id is None

    @pytest.mark.asyncio
    async def test_trade_all_fields_populated(self):
        """Verify all new fields populated together with complete message"""
        binance = Binance()

        msg = {
            'e': 'aggTrade',
            'E': 1609459200000,  # 2021-01-01 00:00:00 UTC
            's': 'ETHUSDT',
            'a': 555555,
            'p': '1234.56',
            'q': '10.5',
            'f': 100,
            'l': 105,
            'T': 1609459195000,
            'm': False,
            'M': False
        }

        trade_obj = None
        async def capture_trade(trade, timestamp):
            nonlocal trade_obj
            trade_obj = trade

        binance.callbacks = {'trades': [capture_trade]}
        await binance._trade(msg, timestamp=1609459200.0)

        assert trade_obj is not None
        # Verify all new fields
        assert trade_obj.maker is False
        assert trade_obj.event_time == 1609459200.0
        assert trade_obj.match_id == "555555"
        # Verify existing fields still work
        assert trade_obj.exchange == 'BINANCE'  # Exchange constant is uppercase
        assert trade_obj.price == Decimal('1234.56')
        assert trade_obj.amount == Decimal('10.5')


class TestBinanceOrderBookFieldExtraction:
    """Test OrderBook field extraction from Binance WebSocket messages"""

    @pytest.mark.asyncio
    async def test_book_extracts_event_time(self):
        """Verify event_time extracted from order book 'E' field"""
        binance = Binance()
        binance._reset()  # Initialize state

        # Mock order book update message
        msg = {
            'e': 'depthUpdate',
            'E': 1234567890000,  # Event time in milliseconds
            's': 'BTCUSDT',
            'U': 157,  # First update ID
            'u': 160,  # Final update ID
            'b': [['50000.00', '1.5']],
            'a': [['50100.00', '2.0']]
        }

        book_obj = None
        async def capture_book(book, timestamp):
            nonlocal book_obj
            book_obj = book

        binance.callbacks = {'l2_book': [capture_book]}

        # Need to create initial snapshot first
        binance._l2_book['BTC-USDT'] = OrderBook(
            exchange='binance',
            symbol='BTC-USDT',
            bids={Decimal('50000.00'): Decimal('1.0')},
            asks={Decimal('50100.00'): Decimal('1.0')}
        )
        binance.last_update_id['BTC-USDT'] =156

        await binance._book(msg, pair='BTCUSDT', timestamp=1234567890.0)

        assert book_obj is not None
        # Verify event_time converted to seconds
        assert book_obj.event_time == 1234567890.0

    @pytest.mark.asyncio
    async def test_book_extracts_last_update_id(self):
        """Verify last_update_id extracted from 'u' field"""
        binance = Binance()
        binance._reset()

        msg = {
            'e': 'depthUpdate',
            'E': 1234567890000,
            's': 'BTCUSDT',
            'U': 200,
            'u': 250,  # Final update ID in event
            'b': [['50000.00', '1.5']],
            'a': [['50100.00', '2.0']]
        }

        book_obj = None
        async def capture_book(book, timestamp):
            nonlocal book_obj
            book_obj = book

        binance.callbacks = {'l2_book': [capture_book]}

        binance._l2_book['BTC-USDT'] = OrderBook(
            exchange='binance',
            symbol='BTC-USDT',
            bids={Decimal('50000.00'): Decimal('1.0')},
            asks={Decimal('50100.00'): Decimal('1.0')}
        )
        binance.last_update_id['BTC-USDT'] =199

        await binance._book(msg, pair='BTCUSDT', timestamp=1234567890.0)

        assert book_obj is not None
        assert book_obj.last_update_id == 250

    @pytest.mark.asyncio
    async def test_book_handles_missing_event_time(self):
        """Verify graceful degradation when 'E' field is missing from order book"""
        binance = Binance()
        binance._reset()

        # Message without 'E' field
        msg = {
            'e': 'depthUpdate',
            's': 'BTCUSDT',
            'U': 157,
            'u': 160,
            'b': [['50000.00', '1.5']],
            'a': [['50100.00', '2.0']]
        }

        book_obj = None
        async def capture_book(book, timestamp):
            nonlocal book_obj
            book_obj = book

        binance.callbacks = {'l2_book': [capture_book]}

        binance._l2_book['BTC-USDT'] = OrderBook(
            exchange='binance',
            symbol='BTC-USDT',
            bids={Decimal('50000.00'): Decimal('1.0')},
            asks={Decimal('50100.00'): Decimal('1.0')}
        )
        binance.last_update_id['BTC-USDT'] =156

        await binance._book(msg, pair='BTCUSDT', timestamp=1234567890.0)

        assert book_obj is not None
        # Should be None when field missing
        assert book_obj.event_time is None

    @pytest.mark.asyncio
    async def test_book_all_fields_populated(self):
        """Verify both new fields populated together in order book update"""
        binance = Binance()
        binance._reset()

        msg = {
            'e': 'depthUpdate',
            'E': 1609459200000,  # Event time
            's': 'ETHUSDT',
            'U': 1000,
            'u': 1005,  # Last update ID
            'b': [['1200.00', '5.0']],
            'a': [['1205.00', '3.0']]
        }

        book_obj = None
        async def capture_book(book, timestamp):
            nonlocal book_obj
            book_obj = book

        binance.callbacks = {'l2_book': [capture_book]}

        binance._l2_book['ETH-USDT'] = OrderBook(
            exchange='binance',
            symbol='ETH-USDT',
            bids={Decimal('1200.00'): Decimal('4.0')},
            asks={Decimal('1205.00'): Decimal('2.0')}
        )
        binance.last_update_id['ETH-USDT'] =999

        await binance._book(msg, pair='ETHUSDT', timestamp=1609459200.0)

        assert book_obj is not None
        # Verify both new fields
        assert book_obj.event_time == 1609459200.0
        assert book_obj.last_update_id == 1005
        # Verify existing fields
        assert book_obj.exchange == 'binance'
        assert book_obj.symbol == 'ETH-USDT'
