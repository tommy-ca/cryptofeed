'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Integration tests for all protobuf wrappers with ProtobufSerializer
'''
import pytest
from decimal import Decimal
from cryptofeed.types import Trade, Ticker, Candle, Funding, OrderBook
from cryptofeed.serializers import ProtobufSerializer
from cryptofeed.proto_bindings import trade_pb2, ticker_pb2, candle_pb2, funding_pb2, order_book_pb2
import cryptofeed.proto_wrappers.registry  # Ensure converters registered


def test_trade_with_serializer():
    """Trade serialization end-to-end."""
    trade = Trade(
        symbol='BTC-USD',
        side='buy',
        amount=Decimal('1.5'),
        price=Decimal('50000'),
        timestamp=1700000000.0,
        exchange='test'
    )
    
    serializer = ProtobufSerializer()
    bytes_data = serializer.serialize(trade)
    
    # Deserialize and verify
    proto = trade_pb2.Trade()
    proto.ParseFromString(bytes_data)
    assert proto.symbol == 'BTC-USD'
    assert proto.price == '50000'


def test_ticker_with_serializer():
    """Ticker serialization end-to-end."""
    ticker = Ticker(
        symbol='BTC-USD',
        bid=Decimal('50000'),
        ask=Decimal('50001'),
        timestamp=1700000000.123,
        exchange='test'
    )
    
    serializer = ProtobufSerializer()
    bytes_data = serializer.serialize(ticker)
    
    # Deserialize and verify
    proto = ticker_pb2.Ticker()
    proto.ParseFromString(bytes_data)
    assert proto.symbol == 'BTC-USD'
    assert proto.bid == '50000'
    assert proto.ask == '50001'
    assert proto.timestamp == 1700000000123000


def test_orderbook_with_serializer():
    """OrderBook serialization end-to-end."""
    book = OrderBook(
        exchange='test',
        symbol='BTC-USD',
        bids={Decimal('50000'): Decimal('1.5'), Decimal('49999'): Decimal('2.0')},
        asks={Decimal('50001'): Decimal('1.0'), Decimal('50002'): Decimal('1.5')}
    )
    book.timestamp = 1700000000.123
    
    serializer = ProtobufSerializer()
    bytes_data = serializer.serialize(book)
    
    # Deserialize and verify
    proto = order_book_pb2.Level2Book()
    proto.ParseFromString(bytes_data)
    assert proto.symbol == 'BTC-USD'
    assert len(proto.bids) == 2
    assert len(proto.asks) == 2
    assert proto.bids[0].price in ['50000', '49999']
    assert proto.timestamp == 1700000000123000


def test_candle_with_serializer():
    """Candle serialization end-to-end."""
    candle = Candle(
        exchange='test',
        symbol='BTC-USD',
        start=1700000000.0,
        stop=1700000060.0,
        interval='1m',
        trades=100,
        open=Decimal('50000'),
        close=Decimal('50050'),
        high=Decimal('50100'),
        low=Decimal('49900'),
        volume=Decimal('100.5'),
        closed=True,
        timestamp=1700000060.0
    )
    
    serializer = ProtobufSerializer()
    bytes_data = serializer.serialize(candle)
    
    # Deserialize and verify
    proto = candle_pb2.Candle()
    proto.ParseFromString(bytes_data)
    assert proto.symbol == 'BTC-USD'
    assert proto.open == '50000'
    assert proto.close == '50050'
    assert proto.high == '50100'
    assert proto.low == '49900'
    assert proto.volume == '100.5'
    assert proto.closed == True
    assert proto.interval == '1m'


def test_funding_with_serializer():
    """Funding serialization end-to-end."""
    funding = Funding(
        exchange='test',
        symbol='BTC-USD-PERP',
        mark_price=Decimal('50000'),
        rate=Decimal('0.0001'),
        next_funding_time=1700000100.0,
        timestamp=1700000000.0
    )
    
    serializer = ProtobufSerializer()
    bytes_data = serializer.serialize(funding)
    
    # Deserialize and verify
    proto = funding_pb2.Funding()
    proto.ParseFromString(bytes_data)
    assert proto.symbol == 'BTC-USD-PERP'
    assert proto.mark_price == '50000'
    assert proto.rate == '0.0001'
    assert proto.next_funding_time == 1700000100000000
    assert proto.timestamp == 1700000000000000


def test_all_types_with_same_serializer():
    """All data types work with the same serializer instance."""
    serializer = ProtobufSerializer()
    
    # Trade
    trade = Trade(
        symbol='BTC-USD', side='buy', amount=Decimal('1'), price=Decimal('50000'),
        timestamp=1700000000.0, exchange='test'
    )
    trade_bytes = serializer.serialize(trade)
    assert len(trade_bytes) > 0
    
    # Ticker
    ticker = Ticker(
        symbol='BTC-USD', bid=Decimal('50000'), ask=Decimal('50001'),
        timestamp=1700000000.0, exchange='test'
    )
    ticker_bytes = serializer.serialize(ticker)
    assert len(ticker_bytes) > 0
    
    # OrderBook
    book = OrderBook(
        exchange='test', symbol='BTC-USD',
        bids={Decimal('50000'): Decimal('1')},
        asks={Decimal('50001'): Decimal('1')}
    )
    book_bytes = serializer.serialize(book)
    assert len(book_bytes) > 0
    
    # All should produce different bytes
    assert trade_bytes != ticker_bytes
    assert trade_bytes != book_bytes
    assert ticker_bytes != book_bytes
