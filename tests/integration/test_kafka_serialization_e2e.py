'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

End-to-end integration test for Kafka with protobuf serialization.

This is a mock-based validation test. For full Kafka E2E testing with
docker-compose, see docs/kafka-integration-guide.md
'''
import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch, call
from cryptofeed.types import Trade, Ticker, Candle
from cryptofeed.backends.backend import BackendCallback
from cryptofeed.serializers import ProtobufSerializer
from cryptofeed.proto_bindings import trade_pb2, ticker_pb2, candle_pb2
import cryptofeed.proto_wrappers.registry


class MockKafkaBackend(BackendCallback):
    """Mock Kafka backend for testing serialization integration."""
    
    def __init__(self, format='protobuf'):
        self.format = format
        self.messages = []
        self.producer = MagicMock()
    
    async def write(self, data):
        """Simulate Kafka producer write."""
        # Serialize the data
        serializer = self._get_serializer(self.format)
        serialized = serializer.serialize(data)
        
        # Simulate Kafka send
        topic = f"{data.exchange}-{type(data).__name__}"
        self.producer.send(topic=topic, value=serialized)
        
        # Store for verification
        self.messages.append({
            'topic': topic,
            'value': serialized,
            'original': data
        })
        
        return serialized


@pytest.mark.asyncio
async def test_kafka_protobuf_trade_roundtrip():
    """Test Trade message through Kafka serialization roundtrip."""
    backend = MockKafkaBackend(format='protobuf')
    
    # Create trade
    trade = Trade(
        symbol='BTC-USD',
        side='buy',
        amount=Decimal('1.5'),
        price=Decimal('50000.123'),
        timestamp=1700000000.123,
        exchange='coinbase'
    )
    
    # Write to backend
    serialized = await backend.write(trade)
    
    # Verify Kafka producer was called
    assert backend.producer.send.called
    call_args = backend.producer.send.call_args
    assert call_args.kwargs['topic'] == 'coinbase-Trade'
    assert call_args.kwargs['value'] == serialized
    
    # Verify deserialization
    from cryptofeed.proto_bindings import trade_side_pb2
    proto = trade_pb2.Trade()
    proto.ParseFromString(serialized)
    
    assert proto.symbol == 'BTC-USD'
    assert proto.side == trade_side_pb2.TRADE_SIDE_BUY
    assert proto.price == '50000.123'
    assert proto.amount == '1.5'
    assert proto.exchange == 'coinbase'
    
    print(f"✅ Kafka Trade roundtrip: {len(serialized)} bytes")


@pytest.mark.asyncio
async def test_kafka_protobuf_ticker_roundtrip():
    """Test Ticker message through Kafka serialization roundtrip."""
    backend = MockKafkaBackend(format='protobuf')
    
    ticker = Ticker(
        symbol='ETH-USD',
        bid=Decimal('3000.50'),
        ask=Decimal('3000.75'),
        timestamp=1700000000.0,
        exchange='binance'
    )
    
    serialized = await backend.write(ticker)
    
    # Verify Kafka send
    assert backend.producer.send.called
    assert backend.producer.send.call_args.kwargs['topic'] == 'binance-Ticker'
    
    # Verify deserialization
    proto = ticker_pb2.Ticker()
    proto.ParseFromString(serialized)
    
    assert proto.symbol == 'ETH-USD'
    assert proto.bid == '3000.50'
    assert proto.ask == '3000.75'
    
    print(f"✅ Kafka Ticker roundtrip: {len(serialized)} bytes")


@pytest.mark.asyncio
async def test_kafka_protobuf_candle_roundtrip():
    """Test Candle message through Kafka serialization roundtrip."""
    backend = MockKafkaBackend(format='protobuf')
    
    candle = Candle(
        exchange='kraken',
        symbol='BTC-EUR',
        start=1700000000.0,
        stop=1700000060.0,
        interval='1m',
        trades=150,
        open=Decimal('45000.00'),
        close=Decimal('45050.00'),
        high=Decimal('45100.00'),
        low=Decimal('44950.00'),
        volume=Decimal('100.0'),
        closed=True,
        timestamp=1700000060.0
    )
    
    serialized = await backend.write(candle)
    
    # Verify Kafka send
    assert backend.producer.send.called
    assert backend.producer.send.call_args.kwargs['topic'] == 'kraken-Candle'
    
    # Verify deserialization
    proto = candle_pb2.Candle()
    proto.ParseFromString(serialized)
    
    assert proto.symbol == 'BTC-EUR'
    assert proto.open == '45000.00'
    assert proto.close == '45050.00'
    assert proto.high == '45100.00'
    assert proto.low == '44950.00'
    assert proto.volume == '100.0'
    
    print(f"✅ Kafka Candle roundtrip: {len(serialized)} bytes")


@pytest.mark.asyncio
async def test_kafka_multiple_messages_batch():
    """Test multiple messages to same topic."""
    backend = MockKafkaBackend(format='protobuf')
    
    # Send 100 trades
    trades = [
        Trade(
            symbol='BTC-USD',
            side='buy' if i % 2 == 0 else 'sell',
            amount=Decimal('1.0'),
            price=Decimal(f'{50000 + i}'),
            timestamp=1700000000.0 + i,
            exchange='test'
        )
        for i in range(100)
    ]
    
    for trade in trades:
        await backend.write(trade)
    
    # Verify all messages stored
    assert len(backend.messages) == 100
    
    # Verify all sent to same topic
    topics = [msg['topic'] for msg in backend.messages]
    assert all(topic == 'test-Trade' for topic in topics)
    
    # Verify each message is unique
    prices = []
    for msg in backend.messages:
        proto = trade_pb2.Trade()
        proto.ParseFromString(msg['value'])
        prices.append(proto.price)
    
    assert len(set(prices)) == 100  # All unique
    
    print(f"✅ Kafka batch: {len(backend.messages)} messages")


@pytest.mark.asyncio
async def test_kafka_json_fallback():
    """Test fallback to JSON serialization."""
    backend = MockKafkaBackend(format='json')
    
    trade = Trade(
        symbol='BTC-USD',
        side='buy',
        amount=Decimal('1.0'),
        price=Decimal('50000'),
        timestamp=1700000000.0,
        exchange='test'
    )
    
    serialized = await backend.write(trade)
    
    # Verify JSON format
    import json
    data = json.loads(serialized.decode('utf-8'))
    assert data['symbol'] == 'BTC-USD'
    assert data['side'] == 'buy'
    
    print(f"✅ Kafka JSON fallback: {len(serialized)} bytes")


@pytest.mark.asyncio
async def test_kafka_topic_routing():
    """Test messages route to correct topics."""
    backend = MockKafkaBackend(format='protobuf')
    
    # Different exchanges and types
    messages = [
        Trade(symbol='BTC-USD', side='buy', amount=Decimal('1'), price=Decimal('50000'), timestamp=1.0, exchange='coinbase'),
        Trade(symbol='ETH-USD', side='sell', amount=Decimal('10'), price=Decimal('3000'), timestamp=2.0, exchange='binance'),
        Ticker(symbol='BTC-USD', bid=Decimal('50000'), ask=Decimal('50001'), timestamp=3.0, exchange='kraken'),
        Candle(exchange='gemini', symbol='BTC-USD', start=1.0, stop=2.0, interval='1m', trades=100,
               open=Decimal('50000'), close=Decimal('50001'), high=Decimal('50002'),
               low=Decimal('49999'), volume=Decimal('100'), closed=True, timestamp=2.0)
    ]
    
    for msg in messages:
        await backend.write(msg)
    
    # Verify topics
    topics = [m['topic'] for m in backend.messages]
    assert 'coinbase-Trade' in topics
    assert 'binance-Trade' in topics
    assert 'kraken-Ticker' in topics
    assert 'gemini-Candle' in topics
    
    print(f"✅ Kafka topic routing: {len(set(topics))} unique topics")


if __name__ == '__main__':
    import asyncio
    
    async def run_tests():
        await test_kafka_protobuf_trade_roundtrip()
        await test_kafka_protobuf_ticker_roundtrip()
        await test_kafka_protobuf_candle_roundtrip()
        await test_kafka_multiple_messages_batch()
        await test_kafka_json_fallback()
        await test_kafka_topic_routing()
        print("\n✅ All Kafka E2E tests passed!")
    
    asyncio.run(run_tests())
