# Data Integrity Validation Script

## Purpose

Validates data integrity between legacy JSON messages and new protobuf messages during the Blue-Green Kafka migration.

## Features

- **Message Normalization**: Handles both JSON dict and protobuf Message types
- **Float Precision**: Normalizes floats to 8 decimals for consistent hashing
- **Hash-Based Comparison**: SHA256 hash validation for data integrity
- **O(1) Message Counting**: Uses Kafka offsets instead of slow wc -l
- **Per-Exchange/Symbol Filtering**: Granular validation by exchange/symbol
- **Detailed Reports**: Clear comparison reports with mismatch details

## Installation

```bash
# Script requires protobuf and Kafka dependencies
pip install google-protobuf kafka-python
```

## Usage

### CLI Usage

```bash
# Basic validation
python scripts/validate_data_integrity.py \
    --legacy-topic cryptofeed.trades.coinbase.btc-usd \
    --new-topic cryptofeed.trades \
    --exchange coinbase \
    --symbol BTC-USD \
    --sample-size 1000

# Output comparison report
python scripts/validate_data_integrity.py \
    --legacy-topic cryptofeed.trades.coinbase.btc-usd \
    --new-topic cryptofeed.trades \
    --exchange coinbase \
    --symbol BTC-USD \
    --output reports/coinbase-btc-usd-validation.json
```

### Library Usage

```python
from validate_data_integrity import (
    normalize_message,
    hash_message,
    validate_message_count,
    compare_messages,
    generate_comparison_report,
    filter_messages
)

# Example: Compare legacy JSON with new protobuf messages
legacy_msgs = [
    {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 50000.12345678,
     'amount': 1.0, 'timestamp': 1699876543.0, 'side': 'buy'}
]

new_msgs = [
    {'exchange': 'coinbase', 'symbol': 'BTC-USD', 'price': 50000.12345678,
     'amount': 1.0, 'timestamp': 1699876543.0, 'side': 'buy'}
]

# Run comparison
result = compare_messages(legacy_msgs, new_msgs, format_types=('json', 'json'))

# Generate report
report = generate_comparison_report(result, 'coinbase', 'BTC-USD')

print(f"Status: {report['status']}")
print(f"Match Rate: {report['match_rate']:.2%}")
print(f"Mismatches: {report['mismatches']}")
```

## Implementation Details

### Message Normalization

Messages are normalized before hashing:

1. **Metadata Removal**: Headers, partition, offset, producer_version removed
2. **Float Precision**: Price, amount, timestamp normalized to 8 decimals using ROUND_HALF_UP
3. **Timestamp Conversion**: Microseconds converted to seconds automatically
4. **Field Ordering**: Canonical JSON ordering (sorted keys) for consistent hashing

### Float Normalization

```python
# Example: 50123.456789123 → "50123.45678912" (8 decimals)
normalize_float(50123.456789123, decimals=8)
# Returns: "50123.45678912"

# Timestamp conversion: microseconds → seconds
normalize_float(1699876543123456 / 1e6, decimals=8)
# Returns: "1699876543.12345600"
```

### Hash-Based Comparison

SHA256 hashes generated from normalized messages:

```python
hash_message(msg, format_type='json')
# Returns: "a1b2c3d4..." (64-char hex digest)
```

### Offset-Based Counting

O(1) message counting using Kafka consumer offsets:

```python
# Traditional approach (slow, O(N))
count = sum(1 for _ in consumer)  # Iterates all messages

# Offset-based approach (fast, O(1))
count = validate_message_count(consumer, topic, partitions)
# Uses: end_offsets - beginning_offsets
```

## Test Coverage

21 comprehensive unit tests covering:

- Message normalization (JSON and protobuf)
- Float precision normalization
- Metadata field removal
- Hash generation and consistency
- Offset-based message counting
- Message comparison logic
- Report generation
- Filtering by exchange/symbol

### Run Tests

```bash
python -m pytest tests/unit/test_validate_data_integrity.py -v
```

Expected output:
```
21 passed in 0.19s
```

## Example Reports

### Success Report

```json
{
  "exchange": "coinbase",
  "symbol": "BTC-USD",
  "status": "PASS",
  "total_compared": 1000,
  "matches": 1000,
  "mismatches": 0,
  "match_rate": 1.0,
  "length_mismatch": false,
  "legacy_count": 1000,
  "new_count": 1000,
  "mismatch_samples": []
}
```

### Failure Report

```json
{
  "exchange": "binance",
  "symbol": "ETH-USDT",
  "status": "FAIL",
  "total_compared": 1000,
  "matches": 998,
  "mismatches": 2,
  "match_rate": 0.998,
  "length_mismatch": false,
  "legacy_count": 1000,
  "new_count": 1000,
  "mismatch_samples": [
    {
      "index": 100,
      "legacy_hash": "abc123...",
      "new_hash": "def456...",
      "legacy_msg": {...},
      "new_msg": {...}
    }
  ]
}
```

## Troubleshooting

### Common Issues

**Issue**: TypeError: Expected protobuf Message, got dict

**Solution**: Ensure protobuf messages are actual Message instances, not dicts. Use `MessageToDict()` for conversion.

---

**Issue**: Hash mismatch due to float precision

**Solution**: Normalization uses 8 decimal precision. Verify source data precision matches.

---

**Issue**: Length mismatch between legacy and new topics

**Solution**: Check if migration is complete. Use `--sample-size` to limit comparison range.

---

**Issue**: Timestamp comparison fails

**Solution**: Script auto-detects microseconds (>1e12) and converts to seconds. Verify timestamp format.

## Integration with Phase 5 Migration

This script is used during Week 3 (Gradual Consumer Migration) for per-exchange validation:

```bash
# Day 1: Validate Coinbase migration
python scripts/validate_data_integrity.py \
    --legacy-topic cryptofeed.trades.coinbase.btc-usd \
    --new-topic cryptofeed.trades \
    --exchange coinbase \
    --sample-size 10000 \
    --output reports/day1-coinbase-validation.json

# Day 2: Validate Binance migration
python scripts/validate_data_integrity.py \
    --legacy-topic cryptofeed.trades.binance.btc-usdt \
    --new-topic cryptofeed.trades \
    --exchange binance \
    --sample-size 10000 \
    --output reports/day2-binance-validation.json
```

## References

- Specification: `.kiro/specs/market-data-kafka-producer/tasks.md` (Task 19.5)
- Tests: `tests/unit/test_validate_data_integrity.py`
- Multi-Agent Review: CRIT-4 blocker (2025-11-26)

## License

Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

See LICENSE file for terms and conditions.
