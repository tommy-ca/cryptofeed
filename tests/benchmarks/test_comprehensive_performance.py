'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Comprehensive performance benchmarks: JSON vs Protobuf for all 14 data types.

Run with: pytest tests/benchmarks/test_comprehensive_performance.py --benchmark-only -v
'''
import pytest
from decimal import Decimal
from cryptofeed.types import (
    Trade, Ticker, OrderBook, Candle, Funding,
    Liquidation, OpenInterest, Index,
    Balance, Position, Fill, OrderInfo, Order, Transaction
)
from cryptofeed.serializers import ProtobufSerializer, JSONSerializer
import cryptofeed.proto_wrappers.registry


# Fixtures for all 14 data types
@pytest.fixture
def sample_trade():
    return Trade(
        symbol='BTC-USD', side='buy', amount=Decimal('1.5'),
        price=Decimal('50000.12345678'), timestamp=1700000000.123,
        exchange='coinbase', id='trade123', type='limit'
    )


@pytest.fixture
def sample_ticker():
    return Ticker(
        symbol='BTC-USD', bid=Decimal('50000.50'), ask=Decimal('50000.75'),
        timestamp=1700000000.0, exchange='binance'
    )


@pytest.fixture
def sample_orderbook():
    bids = {Decimal(f'{50000 - i}'): Decimal(f'{1 + i * 0.1}') for i in range(20)}
    asks = {Decimal(f'{50001 + i}'): Decimal(f'{1 + i * 0.1}') for i in range(20)}
    book = OrderBook(exchange='kraken', symbol='BTC-USD', bids=bids, asks=asks)
    book.timestamp = 1700000000.0
    book.sequence_number = 12345
    return book


@pytest.fixture
def sample_candle():
    return Candle(
        exchange='coinbase', symbol='BTC-USD', start=1700000000.0, stop=1700000060.0,
        interval='1m', trades=100, open=Decimal('50000.00'), close=Decimal('50050.00'),
        high=Decimal('50100.00'), low=Decimal('49900.00'), volume=Decimal('100.5'),
        closed=True, timestamp=1700000060.0
    )


@pytest.fixture
def sample_funding():
    return Funding(
        exchange='binance', symbol='BTC-USD-PERP', mark_price=Decimal('50000.00'),
        rate=Decimal('0.0001'), next_funding_time=1700000100.0, timestamp=1700000000.0
    )


@pytest.fixture
def sample_liquidation():
    return Liquidation(
        exchange='binance', symbol='BTC-USD-PERP', side='sell',
        quantity=Decimal('10.5'), price=Decimal('50000.00'), id='liq123',
        status='completed', timestamp=1700000000.0
    )


@pytest.fixture
def sample_open_interest():
    return OpenInterest(
        exchange='binance', symbol='BTC-USD-PERP',
        open_interest=Decimal('1000000.0'), timestamp=1700000000.0
    )


@pytest.fixture
def sample_index():
    return Index(
        exchange='binance', symbol='BTC-INDEX',
        price=Decimal('50000.00'), timestamp=1700000000.0
    )


@pytest.fixture
def sample_balance():
    return Balance(
        exchange='binance', currency='BTC',
        balance=Decimal('10.5'), reserved=Decimal('0.5')
    )


@pytest.fixture
def sample_position():
    return Position(
        exchange='binance', symbol='BTC-USD-PERP', position=Decimal('100.0'),
        entry_price=Decimal('50000.00'), side='long',
        unrealised_pnl=Decimal('500.00'), timestamp=1700000000.0
    )


@pytest.fixture
def sample_fill():
    return Fill(
        exchange='binance', symbol='BTC-USD', side='buy', amount=Decimal('1.0'),
        price=Decimal('50000.00'), fee=Decimal('0.001'), id='fill123',
        order_id='order456', type='limit', liquidity='taker', timestamp=1700000000.0
    )


@pytest.fixture
def sample_order_info():
    return OrderInfo(
        exchange='binance', symbol='BTC-USD', id='order456',
        side='buy', status='filled', type='limit', price=Decimal('50000.00'),
        amount=Decimal('1.0'), remaining=Decimal('0.0'), timestamp=1700000000.0
    )


@pytest.fixture
def sample_order():
    return Order(
        exchange='binance', symbol='BTC-USD', client_order_id='client123',
        side='buy', type='limit', price=Decimal('50000.00'), amount=Decimal('1.0'),
        timestamp=1700000000.0
    )


@pytest.fixture
def sample_transaction():
    return Transaction(
        exchange='binance', currency='BTC', type='deposit', status='completed',
        amount=Decimal('1.0'), timestamp=1700000000.0
    )


# ============================================================================
# PART 1: Individual Type Latency Benchmarks
# ============================================================================

class TestLatencyBenchmarks:
    """Latency comparison: JSON vs Protobuf for each type."""

    def test_trade_protobuf(self, benchmark, sample_trade):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_trade)

    def test_trade_json(self, benchmark, sample_trade):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_trade)

    def test_ticker_protobuf(self, benchmark, sample_ticker):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_ticker)

    def test_ticker_json(self, benchmark, sample_ticker):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_ticker)

    def test_candle_protobuf(self, benchmark, sample_candle):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_candle)

    def test_candle_json(self, benchmark, sample_candle):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_candle)

    def test_funding_protobuf(self, benchmark, sample_funding):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_funding)

    def test_funding_json(self, benchmark, sample_funding):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_funding)

    def test_liquidation_protobuf(self, benchmark, sample_liquidation):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_liquidation)

    def test_liquidation_json(self, benchmark, sample_liquidation):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_liquidation)

    def test_open_interest_protobuf(self, benchmark, sample_open_interest):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_open_interest)

    def test_open_interest_json(self, benchmark, sample_open_interest):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_open_interest)

    def test_index_protobuf(self, benchmark, sample_index):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_index)

    def test_index_json(self, benchmark, sample_index):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_index)

    def test_balance_protobuf(self, benchmark, sample_balance):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_balance)

    def test_balance_json(self, benchmark, sample_balance):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_balance)

    def test_position_protobuf(self, benchmark, sample_position):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_position)

    def test_position_json(self, benchmark, sample_position):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_position)

    def test_fill_protobuf(self, benchmark, sample_fill):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_fill)

    def test_fill_json(self, benchmark, sample_fill):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_fill)

    def test_order_info_protobuf(self, benchmark, sample_order_info):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_order_info)

    def test_order_info_json(self, benchmark, sample_order_info):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_order_info)

    def test_order_protobuf(self, benchmark, sample_order):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_order)

    def test_order_json(self, benchmark, sample_order):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_order)

    def test_transaction_protobuf(self, benchmark, sample_transaction):
        serializer = ProtobufSerializer()
        benchmark(serializer.serialize, sample_transaction)

    def test_transaction_json(self, benchmark, sample_transaction):
        serializer = JSONSerializer()
        benchmark(serializer.serialize, sample_transaction)


# ============================================================================
# PART 2: Size Comparison for All Types
# ============================================================================

class TestSizeComparison:
    """Size comparison: JSON vs Protobuf for all types."""

    def test_all_types_size_comparison(
        self, sample_trade, sample_ticker, sample_candle, sample_funding,
        sample_liquidation, sample_open_interest, sample_index,
        sample_balance, sample_position, sample_fill, sample_order_info,
        sample_order, sample_transaction
    ):
        """Compare sizes for all 14 types."""
        proto_serializer = ProtobufSerializer()
        json_serializer = JSONSerializer()

        types_data = [
            ('Trade', sample_trade),
            ('Ticker', sample_ticker),
            ('Candle', sample_candle),
            ('Funding', sample_funding),
            ('Liquidation', sample_liquidation),
            ('OpenInterest', sample_open_interest),
            ('Index', sample_index),
            ('Balance', sample_balance),
            ('Position', sample_position),
            ('Fill', sample_fill),
            ('OrderInfo', sample_order_info),
            ('Order', sample_order),
            ('Transaction', sample_transaction),
        ]

        print("\n" + "=" * 80)
        print("SIZE COMPARISON: JSON vs Protobuf (All 14 Types)")
        print("=" * 80)
        print(f"{'Type':<15} {'JSON':>10} {'Protobuf':>10} {'Reduction':>12} {'Speedup':>10}")
        print("-" * 80)

        total_json = 0
        total_proto = 0

        for type_name, sample in types_data:
            proto_bytes = proto_serializer.serialize(sample)
            json_bytes = json_serializer.serialize(sample)

            reduction = (1 - len(proto_bytes) / len(json_bytes)) * 100
            speedup = len(json_bytes) / len(proto_bytes)

            total_json += len(json_bytes)
            total_proto += len(proto_bytes)

            print(f"{type_name:<15} {len(json_bytes):>10} {len(proto_bytes):>10} "
                  f"{reduction:>11.1f}% {speedup:>9.2f}x")

        overall_reduction = (1 - total_proto / total_json) * 100
        overall_speedup = total_json / total_proto

        print("-" * 80)
        print(f"{'TOTAL':<15} {total_json:>10} {total_proto:>10} "
              f"{overall_reduction:>11.1f}% {overall_speedup:>9.2f}x")
        print("=" * 80)

        # Assertions
        assert total_proto < total_json, "Protobuf should be smaller overall"
        assert overall_reduction > 50, f"Expected >50% reduction, got {overall_reduction:.1f}%"


# ============================================================================
# PART 3: Throughput Tests
# ============================================================================

class TestThroughput:
    """Throughput comparison for different types."""

    def test_trade_throughput_10k_protobuf(self):
        """Benchmark 10k Trade messages with Protobuf."""
        import time
        serializer = ProtobufSerializer()

        trades = [
            Trade(
                symbol='BTC-USD', side='buy', amount=Decimal('1.0'),
                price=Decimal(f'{50000 + i}'), timestamp=1700000000.0 + i,
                exchange='test'
            )
            for i in range(10000)
        ]

        start = time.perf_counter()
        for trade in trades:
            serializer.serialize(trade)
        elapsed = time.perf_counter() - start

        throughput = 10000 / elapsed
        print(f"\n✅ Protobuf Trade Throughput: {throughput:,.0f} msg/s ({elapsed:.3f}s)")
        assert throughput >= 100000, f"Throughput {throughput:.0f} below 100k msg/s"

    def test_trade_throughput_10k_json(self):
        """Benchmark 10k Trade messages with JSON."""
        import time
        serializer = JSONSerializer()

        trades = [
            Trade(
                symbol='BTC-USD', side='buy', amount=Decimal('1.0'),
                price=Decimal(f'{50000 + i}'), timestamp=1700000000.0 + i,
                exchange='test'
            )
            for i in range(10000)
        ]

        start = time.perf_counter()
        for trade in trades:
            serializer.serialize(trade)
        elapsed = time.perf_counter() - start

        throughput = 10000 / elapsed
        print(f"\n✅ JSON Trade Throughput: {throughput:,.0f} msg/s ({elapsed:.3f}s)")

    def test_mixed_workload_throughput(
        self, sample_trade, sample_ticker, sample_candle
    ):
        """Test throughput with mixed message types (realistic workload)."""
        import time

        proto_serializer = ProtobufSerializer()
        json_serializer = JSONSerializer()

        # Mixed workload: 70% trades, 20% tickers, 10% candles
        messages = []
        for i in range(1000):
            if i % 10 < 7:  # 70% trades
                messages.append(sample_trade)
            elif i % 10 < 9:  # 20% tickers
                messages.append(sample_ticker)
            else:  # 10% candles
                messages.append(sample_candle)

        # Protobuf
        start = time.perf_counter()
        for msg in messages:
            proto_serializer.serialize(msg)
        proto_elapsed = time.perf_counter() - start
        proto_throughput = 1000 / proto_elapsed

        # JSON
        start = time.perf_counter()
        for msg in messages:
            json_serializer.serialize(msg)
        json_elapsed = time.perf_counter() - start
        json_throughput = 1000 / json_elapsed

        speedup = proto_throughput / json_throughput

        print(f"\n{'='*60}")
        print("MIXED WORKLOAD THROUGHPUT (70% trades, 20% tickers, 10% candles)")
        print(f"{'='*60}")
        print(f"Protobuf: {proto_throughput:,.0f} msg/s ({proto_elapsed:.3f}s)")
        print(f"JSON:     {json_throughput:,.0f} msg/s ({json_elapsed:.3f}s)")
        print(f"Speedup:  {speedup:.2f}x")
        print(f"{'='*60}")

        assert speedup > 1.0, "Protobuf should be faster than JSON"


# ============================================================================
# PART 4: Memory Efficiency
# ============================================================================

class TestMemoryEfficiency:
    """Memory usage comparison."""

    def test_memory_overhead_comparison(self):
        """Compare memory overhead for 100k messages."""
        import gc
        import sys

        proto_serializer = ProtobufSerializer()
        json_serializer = JSONSerializer()

        # Force GC before test
        gc.collect()

        # Create 100k trades
        trades = [
            Trade(
                symbol='BTC-USD', side='buy', amount=Decimal('1.0'),
                price=Decimal('50000'), timestamp=1700000000.0,
                exchange='test'
            )
            for _ in range(100000)
        ]

        # Serialize with protobuf
        proto_messages = []
        for trade in trades:
            proto_messages.append(proto_serializer.serialize(trade))
        proto_total_bytes = sum(len(msg) for msg in proto_messages)

        gc.collect()

        # Serialize with JSON
        json_messages = []
        for trade in trades:
            json_messages.append(json_serializer.serialize(trade))
        json_total_bytes = sum(len(msg) for msg in json_messages)

        gc.collect()

        reduction = (1 - proto_total_bytes / json_total_bytes) * 100

        print(f"\n{'='*60}")
        print("MEMORY EFFICIENCY (100k Trade messages)")
        print(f"{'='*60}")
        print(f"Protobuf Total: {proto_total_bytes:,} bytes ({proto_total_bytes/1024/1024:.2f} MB)")
        print(f"JSON Total:     {json_total_bytes:,} bytes ({json_total_bytes/1024/1024:.2f} MB)")
        print(f"Reduction:      {reduction:.1f}%")
        print(f"Saved:          {(json_total_bytes - proto_total_bytes):,} bytes "
              f"({(json_total_bytes - proto_total_bytes)/1024/1024:.2f} MB)")
        print(f"{'='*60}")

        assert proto_total_bytes < json_total_bytes
        assert reduction > 50


if __name__ == '__main__':
    pytest.main([__file__, '--benchmark-only', '-v'])
