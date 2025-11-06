# Python-Proto Alignment Test Plan

**Date**: 2025-10-25  
**Purpose**: Test suite to validate bidirectional conversion between Python types and Proto schemas  
**Status**: 📋 Planning

---

## Test Strategy

### Approach

1. **Round-Trip Testing**: Python → Proto → Python should preserve all data
2. **Type Conversion**: Validate Decimal, timestamp, and enum conversions
3. **Optionality**: Ensure `None` values handled correctly
4. **Precision**: Verify Decimal scale (1e-8) doesn't lose precision
5. **Edge Cases**: Test boundary conditions (None, zero, very large numbers)
6. **Schema Regression Tooling**: Run `python tools/schema_regression.py --events <fixture> --output <report> --strict` to verify Decimal parity; use `--no-strict` only when triaging malformed fixtures. The new report fields `tolerance` and `difference` highlight Decimal comparisons.

---

## Test Suite Structure

```
tests/proto_integration/
├── test_python_proto_alignment.py       # Main test file
├── fixtures/
│   ├── python_samples.py                # Sample Python objects
│   └── proto_samples.py                 # Sample proto messages
└── converters/
    ├── trade_converter.py               # Trade conversion logic
    ├── ticker_converter.py              # Ticker conversion
    └── ...                              # Other converters
```

---

## Test Cases

### TC1: Trade Round-Trip Conversion

**Objective**: Validate Trade Python → Proto → Python preserves all fields

```python
def test_trade_round_trip():
    # Given: Python Trade object
    python_trade = Trade(
        exchange="BINANCE",
        symbol="BTC-USDT",
        side="buy",
        amount=Decimal("1.5"),
        price=Decimal("50000.12345678"),
        timestamp=1729868400.123456,
        id="12345",
        type="market"
    )
    
    # When: Convert to proto
    proto_trade = python_to_proto_trade(python_trade)
    
    # Then: Proto fields match
    assert proto_trade.exchange == "BINANCE"
    assert proto_trade.symbol == "BTC-USDT"
    assert proto_trade.side == TradeSide.BUY
    assert proto_trade.price == "50000.12345678"
    assert proto_trade.amount == "1.50000000"
    assert proto_trade.timestamp == 1729868400123456  # microseconds
    assert proto_trade.trade_id == "12345"
    # ISSUE: trade_type field missing in proto!
    
    # When: Convert back to Python
    python_trade_2 = proto_to_python_trade(proto_trade)
    
    # Then: Fields match original (except raw)
    assert python_trade_2.exchange == python_trade.exchange
    assert python_trade_2.symbol == python_trade.symbol
    assert python_trade_2.price == python_trade.price
    assert python_trade_2.amount == python_trade.amount
    assert python_trade_2.side == python_trade.side
    assert python_trade_2.timestamp == python_trade.timestamp
    assert python_trade_2.id == python_trade.id
    # ISSUE: type field lost!
```

**Expected Issues**:
- ❌ `type` field not in proto (data loss)
- ❌ `raw` field not persisted

---

### TC2: Funding Optional Fields

**Objective**: Validate optional fields (mark_price, rate, predicted_rate)

```python
def test_funding_optional_fields():
    # Given: Funding with None values
    funding = Funding(
        exchange="BINANCE",
        symbol="BTC-USDT-PERP",
        mark_price=None,  # Can be None in Python
        rate=Decimal("0.0001"),
        next_funding_time=None,
        timestamp=1729868400.0,
        predicted_rate=None
    )
    
    # When: Convert to proto
    proto_funding = python_to_proto_funding(funding)
    
    # Then: Should handle None values
    # ISSUE: Proto mark_price is required, not optional!
    assert proto_funding.mark_price == ""  # Empty string? Or error?
    assert proto_funding.rate == "0.00010000"
    assert not proto_funding.HasField("predicted_rate")
    assert not proto_funding.HasField("next_funding_time")
```

**Expected Issues**:
- ❌ `mark_price` required in proto but can be None in Python
- ❌ `rate` required in proto but can be None in Python

---

### TC3: Timestamp Precision

**Objective**: Validate timestamp conversion doesn't lose precision

```python
def test_timestamp_precision():
    # Given: Python timestamp with microsecond precision
    timestamp_float = 1729868400.123456  # seconds with µs
    
    # When: Convert to proto microseconds
    timestamp_proto = int(timestamp_float * 1_000_000)
    
    # Then: Convert back to Python float
    timestamp_back = timestamp_proto / 1_000_000
    
    # Should preserve precision
    assert abs(timestamp_back - timestamp_float) < 1e-9
    
    # Edge case: Very large timestamp (year 2286)
    large_timestamp = 9999999999.999999
    large_proto = int(large_timestamp * 1_000_000)
    assert large_proto == 9999999999999999
```

**Expected Issues**:
- ⚠️ Limited to ~2^53 microseconds precision (float64 limit)
- ⚠️ Dates beyond 2286 may overflow int64 microseconds

---

### TC4: Decimal Precision (1e-8 scale)

**Objective**: Validate Decimal → string doesn't lose precision

```python
def test_decimal_precision():
    # Test cases for various price scales
    test_cases = [
        ("Bitcoin", Decimal("50000.12345678")),    # BTC price
        ("Satoshi", Decimal("0.00000001")),        # Minimum BTC unit
        ("High precision", Decimal("1.123456789")), # Exceeds 1e-8
        ("Zero", Decimal("0")),
        ("Very large", Decimal("99999999.99999999"))
    ]
    
    for name, price in test_cases:
        # When: Convert to proto string
        proto_price = decimal_to_proto_string(price, scale=8)
        
        # Then: Convert back to Decimal
        price_back = proto_string_to_decimal(proto_price)
        
        # Check precision
        if price.as_tuple().exponent < -8:
            # Precision loss expected
            assert abs(price_back - price) < Decimal("0.00000001")
        else:
            # No precision loss
            assert price_back == price
```

**Expected Issues**:
- ⚠️ Values with >8 decimal places truncated
- ⚠️ Need to document scale per asset class

---

### TC5: OrderBook Structure

**Objective**: Validate OrderBook Python wrapper → Proto flattening

```python
def test_orderbook_structure():
    # Given: Python OrderBook with wrapped order_book.OrderBook
    ob = OrderBook(
        exchange="BINANCE",
        symbol="BTC-USDT",
        bids=[(Decimal("50000"), Decimal("1.5")), 
              (Decimal("49999"), Decimal("2.0"))],
        asks=[(Decimal("50001"), Decimal("1.0")),
              (Decimal("50002"), Decimal("0.5"))]
    )
    ob.sequence_number = 12345
    ob.checksum = "abc123"
    ob.timestamp = 1729868400.0
    
    # When: Convert to proto
    proto_ob = python_to_proto_orderbook(ob)
    
    # Then: Bids/asks flattened to PriceLevel repeated fields
    assert len(proto_ob.bids) == 2
    assert proto_ob.bids[0].price == "50000.00000000"
    assert proto_ob.bids[0].size == "1.50000000"
    assert proto_ob.sequence == 12345
    assert proto_ob.checksum == "abc123"
    
    # When: Convert back to Python
    ob_back = proto_to_python_orderbook(proto_ob)
    
    # Then: Should reconstruct OrderBook wrapper
    assert ob_back.book.to_dict()["bids"] == ob.book.to_dict()["bids"]
    # ISSUE: delta field lost!
```

**Expected Issues**:
- ❌ `delta` field not in proto (incremental updates unsupported)
- ⚠️ Structural conversion required (wrapper → flat)

---

### TC6: Side Enum Conversion

**Objective**: Validate string side → TradeSide enum mapping

```python
def test_side_enum_conversion():
    # Python uses string values
    test_cases = [
        ("buy", TradeSide.BUY),
        ("sell", TradeSide.SELL),
        ("BUY", TradeSide.BUY),  # Case insensitive?
        ("SELL", TradeSide.SELL),
    ]
    
    for python_side, expected_proto_side in test_cases:
        # When: Convert to proto
        proto_side = side_to_proto_enum(python_side)
        
        # Then: Should map correctly
        assert proto_side == expected_proto_side
        
        # Round trip
        python_side_back = proto_enum_to_side(proto_side)
        assert python_side_back.lower() == python_side.lower()
```

**Expected Issues**:
- ⚠️ Need to document canonical case (lowercase vs uppercase)
- ⚠️ Invalid side strings should raise error

---

### TC7: Missing Optional Fields

**Objective**: Validate None handling across all types

```python
def test_missing_optional_fields():
    # Test each type with all optional fields set to None
    
    # Ticker with None timestamp
    ticker = Ticker(
        exchange="BINANCE",
        symbol="BTC-USDT",
        bid=Decimal("50000"),
        ask=Decimal("50001"),
        timestamp=None
    )
    proto_ticker = python_to_proto_ticker(ticker)
    # ISSUE: proto timestamp required, not optional!
    
    # Liquidation with None timestamp
    liquidation = Liquidation(
        exchange="BINANCE",
        symbol="BTC-USDT",
        side="buy",
        quantity=Decimal("1.0"),
        price=Decimal("50000"),
        id=None,
        status=None,
        timestamp=None
    )
    proto_liq = python_to_proto_liquidation(liquidation)
    assert not proto_liq.HasField("liquidation_id")
    assert not proto_liq.HasField("status")
    # ISSUE: timestamp required in proto!
```

**Expected Issues**:
- ❌ Several timestamp fields required in proto but can be None in Python
- ⚠️ Need policy for required vs optional fields

---

## Implementation Plan

### Phase 1: Converter Library

```python
# cryptofeed/converters/python_to_proto.py

from decimal import Decimal
from cryptofeed.types import Trade, Ticker, Funding
from cryptofeed.gen.python.cryptofeed.normalized.v1 import trade_pb2

def decimal_to_proto_string(value: Decimal, scale: int = 8) -> str:
    """Convert Decimal to fixed-point string with specified scale."""
    quantized = value.quantize(Decimal(10) ** -scale)
    return str(quantized)

def timestamp_to_proto_micros(timestamp: float) -> int:
    """Convert float seconds to int64 microseconds."""
    return int(timestamp * 1_000_000)

def side_to_proto_enum(side: str) -> int:
    """Convert side string to TradeSide enum."""
    mapping = {
        "buy": trade_pb2.TradeSide.BUY,
        "sell": trade_pb2.TradeSide.SELL,
    }
    return mapping[side.lower()]

def python_to_proto_trade(trade: Trade) -> trade_pb2.Trade:
    """Convert Python Trade to Proto Trade."""
    return trade_pb2.Trade(
        exchange=trade.exchange,
        symbol=trade.symbol,
        side=side_to_proto_enum(trade.side),
        trade_id=trade.id or "",
        price=decimal_to_proto_string(trade.price),
        amount=decimal_to_proto_string(trade.amount),
        timestamp=timestamp_to_proto_micros(trade.timestamp),
        # ISSUE: trade.type not mapped!
    )
```

### Phase 2: Test Implementation

```python
# tests/proto_integration/test_python_proto_alignment.py

import pytest
from decimal import Decimal
from cryptofeed.types import Trade, Ticker, Funding
from cryptofeed.converters.python_to_proto import python_to_proto_trade
from cryptofeed.converters.proto_to_python import proto_to_python_trade

class TestTradeAlignment:
    def test_trade_round_trip_complete_data(self):
        """Test TC1: Full trade with all fields"""
        # Implementation here
        
    def test_trade_optional_fields_none(self):
        """Test trade with id=None, type=None"""
        # Implementation here
        
    def test_trade_decimal_precision(self):
        """Test various decimal precisions"""
        # Implementation here

class TestFundingAlignment:
    def test_funding_optional_fields(self):
        """Test TC2: Funding with None mark_price/rate"""
        # Implementation here
        
    @pytest.mark.xfail(reason="mark_price not optional in proto")
    def test_funding_none_mark_price(self):
        # This should fail until proto updated
        pass

class TestTimestampConversion:
    def test_timestamp_precision(self):
        """Test TC3: Timestamp precision"""
        # Implementation here
        
    def test_timestamp_edge_cases(self):
        """Test very large, very small, zero timestamps"""
        # Implementation here

# ... more test classes
```

### Phase 3: CI Integration

```yaml
# .github/workflows/schema-alignment.yml
name: Schema Alignment Tests

on: [push, pull_request]

jobs:
  test-alignment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest
      - name: Run alignment tests
        run: pytest tests/proto_integration/test_python_proto_alignment.py -v
```

---

## Success Criteria

- [ ] All 20+ Python types have round-trip tests
- [ ] Converter library handles all type conversions
- [ ] Tests document all known precision/data loss issues
- [ ] CI fails if new alignment issues introduced
- [ ] Migration guide references test examples

---

## Blockers & Dependencies

1. **Proto Schema Updates Needed**:
   - Add `trade_type` to Trade
   - Make `mark_price`/`rate` optional in Funding
   - Make several `timestamp` fields optional
   - Decide on `raw` field strategy

2. **Code Generation**:
   - Need Python bindings from proto files
   - Requires `buf generate` to run first

3. **Converter Library**:
   - Need bidirectional converter functions
   - Should be production-ready, not just for tests

---

## Timeline

| Phase | Tasks | Duration | Status |
|-------|-------|----------|--------|
| Phase 1 | Converter library implementation | 2-3 days | 🔄 Not Started |
| Phase 2 | Test suite implementation | 3-5 days | 🔄 Not Started |
| Phase 3 | CI integration & documentation | 1-2 days | 🔄 Not Started |
| **Total** | | **6-10 days** | |

---

**Next Steps**:
1. Create GitHub issue for proto schema updates
2. Implement converter library (start with Trade/Ticker/Funding)
3. Write first round-trip tests
4. Document discovered issues

**Owner**: TBD  
**Priority**: High (blocks v0.1.0 production usage)
