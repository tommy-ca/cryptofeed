'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Integration test for all 14 data types with protobuf serialization
'''
import pytest
from decimal import Decimal
from cryptofeed.types import (
    Trade, Ticker, OrderBook, Candle, Funding,
    Liquidation, OpenInterest, Index,
    Balance, Position, Fill, OrderInfo, Order, Transaction
)
from cryptofeed.serializers import ProtobufSerializer
from cryptofeed.proto_bindings import (
    liquidation_pb2, open_interest_pb2, index_price_pb2,
    balance_pb2, position_pb2, fill_pb2, order_info_pb2, order_pb2, transaction_pb2
)
import cryptofeed.proto_wrappers.registry  # Ensure all converters registered


def test_all_14_types_serialize():
    """Verify all 14 data types can be serialized successfully."""
    serializer = ProtobufSerializer()
    
    # Market data types (8)
    types_serialized = []
    
    # 1. Trade
    trade = Trade(
        symbol='BTC-USD', side='buy', amount=Decimal('1'), price=Decimal('50000'),
        timestamp=1700000000.0, exchange='test'
    )
    types_serialized.append(('Trade', serializer.serialize(trade)))
    
    # 2. Ticker
    ticker = Ticker(
        symbol='BTC-USD', bid=Decimal('50000'), ask=Decimal('50001'),
        timestamp=1700000000.0, exchange='test'
    )
    types_serialized.append(('Ticker', serializer.serialize(ticker)))
    
    # 3. OrderBook
    book = OrderBook(
        exchange='test', symbol='BTC-USD',
        bids={Decimal('50000'): Decimal('1')},
        asks={Decimal('50001'): Decimal('1')}
    )
    types_serialized.append(('OrderBook', serializer.serialize(book)))
    
    # 4. Candle
    candle = Candle(
        exchange='test', symbol='BTC-USD',
        start=1700000000.0, stop=1700000060.0, interval='1m', trades=100,
        open=Decimal('50000'), close=Decimal('50050'), high=Decimal('50100'),
        low=Decimal('49900'), volume=Decimal('100'), closed=True, timestamp=1700000060.0
    )
    types_serialized.append(('Candle', serializer.serialize(candle)))
    
    # 5. Funding
    funding = Funding(
        exchange='test', symbol='BTC-USD-PERP',
        mark_price=Decimal('50000'), rate=Decimal('0.0001'),
        next_funding_time=1700000100.0, timestamp=1700000000.0
    )
    types_serialized.append(('Funding', serializer.serialize(funding)))
    
    # 6. Liquidation
    liquidation = Liquidation(
        exchange='test', symbol='BTC-USD-PERP', side='sell',
        quantity=Decimal('10'), price=Decimal('50000'), id='liq123',
        status='completed', timestamp=1700000000.0
    )
    types_serialized.append(('Liquidation', serializer.serialize(liquidation)))
    
    # 7. OpenInterest
    oi = OpenInterest(
        exchange='test', symbol='BTC-USD-PERP',
        open_interest=Decimal('1000000'), timestamp=1700000000.0
    )
    types_serialized.append(('OpenInterest', serializer.serialize(oi)))
    
    # 8. Index
    index = Index(
        exchange='test', symbol='BTC-INDEX',
        price=Decimal('50000'), timestamp=1700000000.0
    )
    types_serialized.append(('Index', serializer.serialize(index)))
    
    # Account/Order types (6)
    
    # 9. Balance
    balance = Balance(
        exchange='test', currency='BTC',
        balance=Decimal('10.5'), reserved=Decimal('0.5')
    )
    types_serialized.append(('Balance', serializer.serialize(balance)))
    
    # 10. Position
    position = Position(
        exchange='test', symbol='BTC-USD-PERP',
        position=Decimal('100'), entry_price=Decimal('50000'),
        side='long', unrealised_pnl=Decimal('500'), timestamp=1700000000.0
    )
    types_serialized.append(('Position', serializer.serialize(position)))
    
    # 11. Fill
    fill = Fill(
        exchange='test', symbol='BTC-USD', side='buy',
        amount=Decimal('1'), price=Decimal('50000'),
        fee=Decimal('0.001'), id='fill123', order_id='order456',
        type='limit', liquidity='taker', timestamp=1700000000.0
    )
    types_serialized.append(('Fill', serializer.serialize(fill)))
    
    # 12. OrderInfo
    order_info = OrderInfo(
        exchange='test', symbol='BTC-USD', id='order456',
        client_order_id='client123', side='buy', status='filled',
        type='limit', price=Decimal('50000'), amount=Decimal('1'),
        remaining=Decimal('0'), timestamp=1700000000.0
    )
    types_serialized.append(('OrderInfo', serializer.serialize(order_info)))
    
    # 13. Order
    order = Order(
        exchange='test', symbol='BTC-USD', client_order_id='client123',
        side='buy', type='limit', price=Decimal('50000'), amount=Decimal('1'),
        timestamp=1700000000.0
    )
    types_serialized.append(('Order', serializer.serialize(order)))
    
    # 14. Transaction
    transaction = Transaction(
        exchange='test', currency='BTC', type='deposit', status='completed',
        amount=Decimal('1.0'), timestamp=1700000000.0
    )
    types_serialized.append(('Transaction', serializer.serialize(transaction)))
    
    # Verify all serialized successfully
    assert len(types_serialized) == 14
    
    for type_name, serialized_bytes in types_serialized:
        assert isinstance(serialized_bytes, bytes), f"{type_name} did not serialize to bytes"
        assert len(serialized_bytes) > 0, f"{type_name} serialized to empty bytes"
    
    print(f"✅ All 14 data types serialized successfully")
    for type_name, serialized_bytes in types_serialized:
        print(f"  {type_name}: {len(serialized_bytes)} bytes")


def test_all_types_roundtrip():
    """Verify all types can be deserialized after serialization."""
    serializer = ProtobufSerializer()
    
    # Test a few key types for roundtrip
    
    # Liquidation
    liquidation = Liquidation(
        exchange='test', symbol='BTC-USD-PERP', side='sell',
        quantity=Decimal('10.5'), price=Decimal('50000.123'), id='liq123',
        status='completed', timestamp=1700000000.123
    )
    liq_bytes = serializer.serialize(liquidation)
    liq_proto = liquidation_pb2.Liquidation()
    liq_proto.ParseFromString(liq_bytes)
    assert liq_proto.symbol == 'BTC-USD-PERP'
    assert liq_proto.quantity == '10.5'
    assert liq_proto.price == '50000.123'
    
    # Balance
    balance = Balance(
        exchange='test', currency='BTC',
        balance=Decimal('10.5'), reserved=Decimal('0.5')
    )
    bal_bytes = serializer.serialize(balance)
    bal_proto = balance_pb2.Balance()
    bal_proto.ParseFromString(bal_bytes)
    assert bal_proto.currency == 'BTC'
    assert bal_proto.balance == '10.5'
    assert bal_proto.reserved == '0.5'
    
    # OrderInfo
    order_info = OrderInfo(
        exchange='test', symbol='BTC-USD', id='order456',
        side='buy', status='filled', type='limit',
        price=Decimal('50000'), amount=Decimal('1'), remaining=Decimal('0'),
        timestamp=1700000000.0
    )
    oi_bytes = serializer.serialize(order_info)
    oi_proto = order_info_pb2.OrderInfo()
    oi_proto.ParseFromString(oi_bytes)
    assert oi_proto.order_id == 'order456'
    assert oi_proto.status == 'filled'
    assert oi_proto.price == '50000'


if __name__ == '__main__':
    test_all_14_types_serialize()
    test_all_types_roundtrip()
    print("\n✅ All tests passed!")
