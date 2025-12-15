#!/usr/bin/env python
"""
Field Population Validation Tool

Task 9.3: REQ-1.19 - Measure and validate zero silent data loss

This script monitors field population rates across exchanges to verify
that all extracted fields are being transmitted through the protobuf pipeline.

Usage:
    python tools/validate_field_population.py

Features:
- Tracks field population rates per exchange
- Calculates silent data loss percentage
- Validates that Binance fields have 100% population
- Documents field availability matrix

Output:
- Field population statistics
- Data loss percentage per exchange
- Field availability matrix
"""

from decimal import Decimal
from typing import Dict, List, Tuple
from collections import defaultdict

from cryptofeed.types import Trade, OrderBook
from cryptofeed.backends.protobuf.converters import trade_to_proto, orderbook_to_proto


class FieldPopulationValidator:
    """
    Validates field population in protobuf messages to detect silent data loss.

    Tracks statistics on which fields are populated for which exchanges,
    enabling detection of missing field extraction or converter bugs.
    """

    def __init__(self):
        self.trade_stats = defaultdict(lambda: {
            'total': 0,
            'maker': 0,
            'event_time': 0,
            'match_id': 0,
            'liquidity_flag': 0,
        })
        self.orderbook_stats = defaultdict(lambda: {
            'total': 0,
            'event_time': 0,
            'last_update_id': 0,
        })

    def validate_trade(self, trade_obj: Trade) -> Dict[str, bool]:
        """
        Validate Trade field population and update statistics.

        Args:
            trade_obj: Trade object to validate

        Returns:
            Dictionary mapping field names to population status (True if populated)
        """
        exchange = trade_obj.exchange
        self.trade_stats[exchange]['total'] += 1

        # Convert to protobuf
        proto = trade_to_proto(trade_obj)

        # Check field population
        populated = {
            'maker': proto.HasField('maker'),
            'event_time': proto.HasField('event_time'),
            'match_id': proto.HasField('match_id'),
            'liquidity_flag': proto.HasField('liquidity_flag'),
        }

        # Update statistics
        for field, is_populated in populated.items():
            if is_populated:
                self.trade_stats[exchange][field] += 1

        return populated

    def validate_orderbook(self, orderbook_obj: OrderBook) -> Dict[str, bool]:
        """
        Validate OrderBook field population and update statistics.

        Args:
            orderbook_obj: OrderBook object to validate

        Returns:
            Dictionary mapping field names to population status (True if populated)
        """
        exchange = orderbook_obj.exchange
        self.orderbook_stats[exchange]['total'] += 1

        # Convert to protobuf
        proto = orderbook_to_proto(orderbook_obj)

        # Check field population
        populated = {
            'event_time': proto.HasField('event_time'),
            'last_update_id': proto.HasField('last_update_id'),
        }

        # Update statistics
        for field, is_populated in populated.items():
            if is_populated:
                self.orderbook_stats[exchange][field] += 1

        return populated

    def calculate_population_rate(self, exchange: str, data_type: str, field: str) -> float:
        """
        Calculate population rate for a specific field.

        Args:
            exchange: Exchange name
            data_type: 'trade' or 'orderbook'
            field: Field name

        Returns:
            Population rate as percentage (0-100)
        """
        stats = self.trade_stats if data_type == 'trade' else self.orderbook_stats

        total = stats[exchange]['total']
        if total == 0:
            return 0.0

        populated = stats[exchange][field]
        return (populated / total) * 100

    def calculate_data_loss(self, exchange: str, data_type: str) -> float:
        """
        Calculate silent data loss percentage for an exchange.

        Data loss is calculated as the percentage of expected fields
        that are NOT being populated.

        Args:
            exchange: Exchange name
            data_type: 'trade' or 'orderbook'

        Returns:
            Data loss percentage (0-100, where 0 means no data loss)
        """
        if data_type == 'trade':
            fields = ['maker', 'event_time', 'match_id', 'liquidity_flag']
        else:
            fields = ['event_time', 'last_update_id']

        total_fields = len(fields)
        populated_fields = sum(
            1 for field in fields
            if self.calculate_population_rate(exchange, data_type, field) > 0
        )

        return ((total_fields - populated_fields) / total_fields) * 100

    def get_field_availability_matrix(self) -> Dict[str, Dict[str, str]]:
        """
        Generate field availability matrix showing which exchanges support which fields.

        Returns:
            Dictionary mapping exchanges to field availability status.
            Status values: 'SUPPORTED', 'NOT_AVAILABLE', 'PARTIAL'
        """
        matrix = {}

        # Trade fields
        for exchange in self.trade_stats:
            if exchange not in matrix:
                matrix[exchange] = {}

            for field in ['maker', 'event_time', 'match_id', 'liquidity_flag']:
                rate = self.calculate_population_rate(exchange, 'trade', field)
                if rate >= 95:
                    matrix[exchange][f'trade.{field}'] = 'SUPPORTED'
                elif rate > 5:
                    matrix[exchange][f'trade.{field}'] = 'PARTIAL'
                else:
                    matrix[exchange][f'trade.{field}'] = 'NOT_AVAILABLE'

        # OrderBook fields
        for exchange in self.orderbook_stats:
            if exchange not in matrix:
                matrix[exchange] = {}

            for field in ['event_time', 'last_update_id']:
                rate = self.calculate_population_rate(exchange, 'orderbook', field)
                if rate >= 95:
                    matrix[exchange][f'orderbook.{field}'] = 'SUPPORTED'
                elif rate > 5:
                    matrix[exchange][f'orderbook.{field}'] = 'PARTIAL'
                else:
                    matrix[exchange][f'orderbook.{field}'] = 'NOT_AVAILABLE'

        return matrix

    def print_report(self):
        """Print comprehensive field population report."""
        print("=" * 80)
        print("FIELD POPULATION VALIDATION REPORT")
        print("=" * 80)
        print()

        # Trade statistics
        print("TRADE FIELD POPULATION:")
        print("-" * 80)
        for exchange in sorted(self.trade_stats.keys()):
            total = self.trade_stats[exchange]['total']
            if total == 0:
                continue

            print(f"\nExchange: {exchange.upper()}")
            print(f"  Total Trades: {total}")

            for field in ['maker', 'event_time', 'match_id', 'liquidity_flag']:
                rate = self.calculate_population_rate(exchange, 'trade', field)
                populated = self.trade_stats[exchange][field]
                print(f"  {field:20s}: {populated:5d}/{total:5d} ({rate:6.2f}%)")

            data_loss = self.calculate_data_loss(exchange, 'trade')
            print(f"  Data Loss: {data_loss:.2f}%")

        # OrderBook statistics
        print("\n" + "=" * 80)
        print("ORDERBOOK FIELD POPULATION:")
        print("-" * 80)
        for exchange in sorted(self.orderbook_stats.keys()):
            total = self.orderbook_stats[exchange]['total']
            if total == 0:
                continue

            print(f"\nExchange: {exchange.upper()}")
            print(f"  Total OrderBooks: {total}")

            for field in ['event_time', 'last_update_id']:
                rate = self.calculate_population_rate(exchange, 'orderbook', field)
                populated = self.orderbook_stats[exchange][field]
                print(f"  {field:20s}: {populated:5d}/{total:5d} ({rate:6.2f}%)")

            data_loss = self.calculate_data_loss(exchange, 'orderbook')
            print(f"  Data Loss: {data_loss:.2f}%")

        # Field availability matrix
        print("\n" + "=" * 80)
        print("FIELD AVAILABILITY MATRIX:")
        print("-" * 80)
        matrix = self.get_field_availability_matrix()
        for exchange in sorted(matrix.keys()):
            print(f"\n{exchange.upper()}:")
            for field, status in sorted(matrix[exchange].items()):
                print(f"  {field:30s}: {status}")

        print("\n" + "=" * 80)


def run_sample_validation():
    """
    Run sample validation with mock data to demonstrate usage.

    This is a demonstration of how to use the validator in production.
    """
    validator = FieldPopulationValidator()

    # Sample Binance trades with full field population
    print("Processing sample Binance trades (full field population)...")
    for i in range(100):
        trade = Trade(
            exchange='binance',
            symbol='BTC-USD',
            side='buy' if i % 2 == 0 else 'sell',
            price=Decimal('50000.00'),
            amount=Decimal('1.0'),
            timestamp=1234567890.0 + i,
            maker=i % 2 == 0,
            event_time=1234567890.0 + i + 0.1,
            match_id=str(i),
        )
        validator.validate_trade(trade)

    # Sample OKX trades without new fields (simulating unimplemented exchange)
    print("Processing sample OKX trades (no new fields - simulating NOT_YET_IMPLEMENTED)...")
    for i in range(50):
        trade = Trade(
            exchange='okx',
            symbol='BTC-USD',
            side='buy',
            price=Decimal('50000.00'),
            amount=Decimal('1.0'),
            timestamp=1234567890.0 + i,
            # No new fields
        )
        validator.validate_trade(trade)

    # Sample Binance order books
    print("Processing sample Binance order books...")
    for i in range(50):
        book = OrderBook(
            exchange='binance',
            symbol='BTC-USD',
            bids={Decimal('49999.00'): Decimal('1.5')},
            asks={Decimal('50001.00'): Decimal('2.0')},
        )
        book.timestamp = 1234567890.0 + i
        book.event_time = 1234567890.0 + i + 0.1
        book.last_update_id = i
        validator.validate_orderbook(book)

    # Print comprehensive report
    validator.print_report()

    # Validation checks
    print("\nVALIDATION CHECKS:")
    print("-" * 80)

    # Note: Binance provides maker, event_time, match_id but NOT liquidity_flag
    # This is documented in REQ-1.18 field availability matrix
    binance_trade_loss = validator.calculate_data_loss('binance', 'trade')
    print(f"Binance Trade Data Loss: {binance_trade_loss:.2f}%")
    print(f"  Note: Binance supports 3/4 fields (no liquidity_flag - exchange limitation)")

    # Verify core Binance fields have 100% population
    maker_rate = validator.calculate_population_rate('binance', 'trade', 'maker')
    event_time_rate = validator.calculate_population_rate('binance', 'trade', 'event_time')
    match_id_rate = validator.calculate_population_rate('binance', 'trade', 'match_id')

    assert maker_rate == 100.0, "FAIL: Binance maker field should be 100%"
    assert event_time_rate == 100.0, "FAIL: Binance event_time field should be 100%"
    assert match_id_rate == 100.0, "FAIL: Binance match_id field should be 100%"
    print("  ✓ PASS: All Binance-supported fields at 100% (maker, event_time, match_id)")

    binance_orderbook_loss = validator.calculate_data_loss('binance', 'orderbook')
    print(f"Binance OrderBook Data Loss: {binance_orderbook_loss:.2f}% (EXPECTED: 0.00%)")
    assert binance_orderbook_loss == 0.0, "FAIL: Binance should have zero orderbook data loss"
    print("  ✓ PASS: Zero data loss for Binance order books")

    okx_trade_loss = validator.calculate_data_loss('okx', 'trade')
    print(f"OKX Trade Data Loss: {okx_trade_loss:.2f}% (EXPECTED: 100.00% - not implemented)")
    assert okx_trade_loss == 100.0, "FAIL: OKX should show 100% data loss (not yet implemented)"
    print("  ✓ PASS: OKX correctly shows not implemented")

    print("\n" + "=" * 80)
    print("ALL VALIDATION CHECKS PASSED")
    print("=" * 80)


if __name__ == '__main__':
    run_sample_validation()
