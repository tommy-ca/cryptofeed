# Python Types to Proto Schema Alignment Review

**Date**: 2025-10-25  
**Purpose**: Comprehensive review of Protocol Buffer schemas against original Python Cython types  
**Status**: 🔍 In Progress

---

## Overview

This document maps each Python data type from `cryptofeed/types.pyx` to its corresponding Protocol Buffer schema in `proto/cryptofeed/normalized/v1/`, identifying alignment issues, field mappings, and migration considerations.

## Methodology

1. **Field-by-Field Comparison**: Map Python class attributes to Proto message fields
2. **Type Conversion Analysis**: Identify precision/representation differences
3. **Optionality Review**: Ensure nullable fields correctly marked as `optional`
4. **Timestamp Normalization**: Verify consistent timestamp handling (float seconds → int64 microseconds)
5. **Decimal Precision**: Confirm Decimal → string conversion with documented scale

---

## 1. Trade

### Python Type (`types.pyx`)
```python
cdef class Trade:
    cdef readonly str exchange
    cdef readonly str symbol
    cdef readonly object price      # Decimal
    cdef readonly object amount     # Decimal
    cdef readonly str side
    cdef readonly str id
    cdef readonly str type
    cdef readonly double timestamp  # float seconds
    cdef readonly object raw        # dict or list
```

### Proto Schema (`trade.proto`)
```protobuf
message Trade {
  string exchange = 1;
  string symbol = 2;
  TradeSide side = 3;
  string trade_id = 4;        // Was: id
  string price = 5;           // Decimal as string
  string amount = 6;          // Decimal as string
  int64 timestamp = 7;        // microseconds
  string raw_id = 8;          // Was: type (?)
}
```

### ✅ Alignment Status: **MOSTLY ALIGNED**

| Python Field | Proto Field | Status | Notes |
|--------------|-------------|--------|-------|
| `exchange` | `exchange` | ✅ Match | Both `string` |
| `symbol` | `symbol` | ✅ Match | Both `string` |
| `price` | `price` | ✅ Match | Decimal → string with scale 1e-8 |
| `amount` | `amount` | ✅ Match | Decimal → string with scale 1e-8 |
| `side` | `side` | ⚠️ Type Change | Python: `str` → Proto: `TradeSide` enum |
| `id` | `trade_id` | ✅ Match | Field renamed for clarity |
| `type` | ❌ **MISSING** | ⚠️ Not Mapped | Python `type` field not in proto |
| `timestamp` | `timestamp` | ✅ Match | float seconds → int64 microseconds |
| `raw` | ❌ **MISSING** | ⚠️ Not Mapped | Raw data not persisted in proto |

### 🔍 Discrepancies

1. **`type` Field Missing**: Python has `type` (e.g., "market", "limit"), proto doesn't include it
   - **Impact**: Loss of trade type information
   - **Recommendation**: Add `optional string trade_type = 9;` to proto

2. **`raw` Field Not Persisted**: Proto doesn't include raw exchange data
   - **Impact**: Cannot reconstruct original exchange message
   - **Recommendation**: Consider adding `optional bytes raw = 10;` if needed for debugging

3. **`raw_id` Unclear Mapping**: Proto has `raw_id` but Python doesn't have equivalent
   - **Impact**: Unclear what this field represents
   - **Recommendation**: Clarify field purpose or remove if unused

---

## 2. Ticker

### Python Type (`types.pyx`)
```python
cdef class Ticker:
    cdef readonly str exchange
    cdef readonly str symbol
    cdef readonly object bid        # Decimal
    cdef readonly object ask        # Decimal
    cdef readonly object timestamp  # None or float
    cdef readonly object raw
```

### Proto Schema (`ticker.proto`)
```protobuf
message Ticker {
  string exchange = 1;
  string symbol = 2;
  string bid = 3;           // Decimal as string
  string ask = 4;           // Decimal as string
  int64 timestamp = 5;      // microseconds
}
```

### ✅ Alignment Status: **FULLY ALIGNED**

| Python Field | Proto Field | Status | Notes |
|--------------|-------------|--------|-------|
| `exchange` | `exchange` | ✅ Match | Both `string` |
| `symbol` | `symbol` | ✅ Match | Both `string` |
| `bid` | `bid` | ✅ Match | Decimal → string with scale 1e-8 |
| `ask` | `ask` | ✅ Match | Decimal → string with scale 1e-8 |
| `timestamp` | `timestamp` | ✅ Match | float seconds → int64 microseconds |
| `raw` | ❌ **MISSING** | ⚠️ Not Mapped | Raw data not persisted in proto |

### 🔍 Discrepancies

1. **Timestamp Optionality**: Python allows `None`, proto `int64` defaults to `0`
   - **Impact**: Cannot distinguish missing timestamp from epoch 0
   - **Recommendation**: Consider `optional int64 timestamp = 5;` if `None` is meaningful

---

## 3. Funding

### Python Type (`types.pyx`)
```python
cdef class Funding:
    cdef readonly str exchange
    cdef readonly str symbol
    cdef readonly object mark_price        # None or Decimal
    cdef readonly object rate              # None or Decimal
    cdef readonly object next_funding_time # None or float
    cdef readonly object predicted_rate    # None or Decimal
    cdef readonly double timestamp
    cdef readonly object raw
```

### Proto Schema (`funding.proto`)
```protobuf
message Funding {
  string exchange = 1;
  string symbol = 2;
  string mark_price = 3;
  string rate = 4;
  optional string predicted_rate = 5;
  optional int64 next_funding_time = 6;  // microseconds
  int64 timestamp = 7;                   // microseconds
}
```

### ✅ Alignment Status: **WELL ALIGNED**

| Python Field | Proto Field | Status | Notes |
|--------------|-------------|--------|-------|
| `exchange` | `exchange` | ✅ Match | Both `string` |
| `symbol` | `symbol` | ✅ Match | Both `string` |
| `mark_price` | `mark_price` | ⚠️ Optionality | Python allows `None`, proto required field |
| `rate` | `rate` | ⚠️ Optionality | Python allows `None`, proto required field |
| `predicted_rate` | `predicted_rate` | ✅ Match | Both optional |
| `next_funding_time` | `next_funding_time` | ✅ Match | Both optional, float → int64 µs |
| `timestamp` | `timestamp` | ✅ Match | float seconds → int64 microseconds |
| `raw` | ❌ **MISSING** | ⚠️ Not Mapped | Raw data not persisted |

### 🔍 Discrepancies

1. **`mark_price` Optionality**: Python allows `None`, proto field is required
   - **Impact**: Cannot represent missing mark price
   - **Recommendation**: Change to `optional string mark_price = 3;`

2. **`rate` Optionality**: Python allows `None`, proto field is required
   - **Impact**: Cannot represent missing funding rate
   - **Recommendation**: Change to `optional string rate = 4;`

---

## 4. Liquidation

### Python Type (`types.pyx`)
```python
cdef class Liquidation:
    cdef readonly str exchange
    cdef readonly str symbol
    cdef readonly str side
    cdef readonly object quantity  # Decimal
    cdef readonly object price     # Decimal
    cdef readonly str id
    cdef readonly str status
    cdef readonly object timestamp # None or float
    cdef readonly dict raw
```

### Proto Schema (`liquidation.proto`)
```protobuf
message Liquidation {
  string exchange = 1;
  string symbol = 2;
  TradeSide side = 3;
  string quantity = 4;
  string price = 5;
  optional string liquidation_id = 6;
  optional string status = 7;
  int64 timestamp = 8;  // microseconds
}
```

### ✅ Alignment Status: **WELL ALIGNED**

| Python Field | Proto Field | Status | Notes |
|--------------|-------------|--------|-------|
| `exchange` | `exchange` | ✅ Match | Both `string` |
| `symbol` | `symbol` | ✅ Match | Both `string` |
| `side` | `side` | ⚠️ Type Change | Python: `str` → Proto: `TradeSide` enum |
| `quantity` | `quantity` | ✅ Match | Decimal → string |
| `price` | `price` | ✅ Match | Decimal → string |
| `id` | `liquidation_id` | ✅ Match | Field renamed, marked optional |
| `status` | `status` | ✅ Match | Both optional string |
| `timestamp` | `timestamp` | ⚠️ Optionality | Python allows `None`, proto required |
| `raw` | ❌ **MISSING** | ⚠️ Not Mapped | Raw dict not persisted |

### 🔍 Discrepancies

1. **Timestamp Optionality**: Python allows `None`, proto `int64` defaults to `0`
   - **Impact**: Cannot distinguish missing timestamp from epoch 0
   - **Recommendation**: Consider `optional int64 timestamp = 8;`

---

## 5. Candle

### Python Type (`types.pyx`)
```python
cdef class Candle:
    cdef readonly str exchange
    cdef readonly str symbol
    cdef readonly double start
    cdef readonly double stop
    cdef readonly str interval
    cdef readonly object trades    # None or int
    cdef readonly object open      # Decimal
    cdef readonly object close     # Decimal
    cdef readonly object high      # Decimal
    cdef readonly object low       # Decimal
    cdef readonly object volume    # Decimal
    cdef readonly bint closed
    cdef readonly object timestamp # None or float
    cdef readonly object raw
```

### Proto Schema (`candle.proto`)
```protobuf
message Candle {
  string exchange = 1;
  string symbol = 2;
  int64 start = 3;            // microseconds
  int64 stop = 4;             // microseconds
  string interval = 5;
  optional int32 trades = 6;
  string open = 7;            // Decimal as string
  string close = 8;
  string high = 9;
  string low = 10;
  string volume = 11;
  bool closed = 12;
  optional int64 timestamp = 13;  // microseconds
}
```

### ✅ Alignment Status: **FULLY ALIGNED**

| Python Field | Proto Field | Status | Notes |
|--------------|-------------|--------|-------|
| `exchange` | `exchange` | ✅ Match | Both `string` |
| `symbol` | `symbol` | ✅ Match | Both `string` |
| `start` | `start` | ✅ Match | float seconds → int64 microseconds |
| `stop` | `stop` | ✅ Match | float seconds → int64 microseconds |
| `interval` | `interval` | ✅ Match | Both `string` |
| `trades` | `trades` | ✅ Match | Both optional, int |
| `open` | `open` | ✅ Match | Decimal → string |
| `close` | `close` | ✅ Match | Decimal → string |
| `high` | `high` | ✅ Match | Decimal → string |
| `low` | `low` | ✅ Match | Decimal → string |
| `volume` | `volume` | ✅ Match | Decimal → string |
| `closed` | `closed` | ✅ Match | bool in both |
| `timestamp` | `timestamp` | ✅ Match | Both optional, float → int64 µs |
| `raw` | ❌ **MISSING** | ⚠️ Not Mapped | Raw data not persisted |

### 🔍 Discrepancies

**None** - Excellent alignment! This is the best-aligned schema.

---

## 6. OrderBook

### Python Type (`types.pyx`)
```python
cdef class OrderBook:
    cdef readonly str exchange
    cdef readonly str symbol
    cdef readonly object book          # OrderBook instance
    cdef public dict delta
    cdef public object sequence_number
    cdef public object checksum
    cdef public object timestamp
    cdef public object raw
```

### Proto Schema (`order_book.proto`)
```protobuf
message OrderBook {
  string exchange = 1;
  string symbol = 2;
  repeated PriceLevel bids = 3;
  repeated PriceLevel asks = 4;
  optional int64 sequence_number = 5;
  optional string checksum = 6;
  int64 timestamp = 7;  // microseconds
}
```

### ⚠️ Alignment Status: **STRUCTURAL MISMATCH**

| Python Field | Proto Field | Status | Notes |
|--------------|-------------|--------|-------|
| `exchange` | `exchange` | ✅ Match | Both `string` |
| `symbol` | `symbol` | ✅ Match | Both `string` |
| `book` | `bids`/`asks` | ⚠️ Structural | Python wraps OrderBook, proto flattens |
| `delta` | ❌ **MISSING** | ⚠️ Not Mapped | Delta updates not in proto |
| `sequence_number` | `sequence_number` | ✅ Match | Both optional |
| `checksum` | `checksum` | ✅ Match | Both optional |
| `timestamp` | `timestamp` | ⚠️ Optionality | Python allows `None`, proto required |
| `raw` | ❌ **MISSING** | ⚠️ Not Mapped | Raw data not persisted |

### 🔍 Discrepancies

1. **Book Representation**: Python wraps `order_book.OrderBook`, proto uses repeated `PriceLevel`
   - **Impact**: Conversion requires iterating OrderBook.to_dict()['bids'/'asks']
   - **Recommendation**: Document conversion pattern in migration guide

2. **Delta Field Missing**: Python has `delta` dict for incremental updates
   - **Impact**: Cannot represent L2 deltas, only snapshots
   - **Recommendation**: Review `level2_delta.proto` - may need alignment

3. **Timestamp Optionality**: Python allows `None`, proto required
   - **Recommendation**: Consider `optional int64 timestamp = 7;`

---

## Summary: Alignment Issues by Category

### 🔴 Critical Issues (Block Migration)

1. **Trade.type Missing**: Loss of trade type information (market/limit)
2. **Funding Mark Price/Rate Not Optional**: Cannot represent missing values
3. **OrderBook Delta Not Represented**: Incremental updates unsupported

### 🟡 Medium Issues (May Cause Data Loss)

1. **Raw Field Universally Missing**: Cannot reconstruct original exchange messages
2. **Timestamp Optionality**: Several types allow `None` but proto uses required `int64`
3. **Trade.raw_id Unclear**: Field exists in proto but not in Python

### 🟢 Minor Issues (Documentation Needed)

1. **Side Field Type Change**: String → Enum (requires mapping documentation)
2. **Field Renames**: `id` → `trade_id`, `id` → `liquidation_id` (consistent pattern)
3. **Timestamp Precision**: float seconds → int64 microseconds (lossy for >2^53 µs)

---

## Recommendations

### Immediate Actions

1. **Add Missing Fields to Proto**:
   ```protobuf
   // trade.proto
   optional string trade_type = 9;  // e.g., "market", "limit"
   
   // funding.proto
   optional string mark_price = 3;  // Change from required
   optional string rate = 4;        // Change from required
   ```

2. **Add Raw Field Strategy**:
   - Option A: Add `optional bytes raw = N;` to all messages
   - Option B: Document that raw data is not persisted in normalized schemas

3. **Fix Timestamp Optionality**:
   - Change all required `int64 timestamp` to `optional` where Python allows `None`

### Testing Actions

1. **Create Conversion Tests**: Validate Python → Proto → Python round-trip
2. **Field Coverage Tests**: Ensure all Python fields mapped or documented as excluded
3. **Precision Tests**: Verify Decimal scale (1e-8) sufficient for all exchanges

### Documentation Actions

1. **Migration Guide**: Document field mappings and type conversions
2. **Precision Policy**: Document Decimal scale rationale and edge cases
3. **Raw Data Policy**: Clarify why raw exchange data is not persisted

---

## 7. Index (IndexPrice)

### Python Type (`types.pyx`)
```python
cdef class Index:
    cdef readonly str exchange
    cdef readonly str symbol
    cdef readonly object price     # Decimal
    cdef readonly double timestamp # float
    cdef readonly dict raw
```

### Proto Schema (`index_price.proto`)
```protobuf
message IndexPrice {
  string exchange = 1;
  string symbol = 2;
  string price = 3;       // Decimal as string
  int64 timestamp = 4;    // microseconds
}
```

### ✅ Alignment Status: **FULLY ALIGNED**

| Python Field | Proto Field | Status | Notes |
|--------------|-------------|--------|-------|
| `exchange` | `exchange` | ✅ Match | Both `string` |
| `symbol` | `symbol` | ✅ Match | Both `string` |
| `price` | `price` | ✅ Match | Decimal → string |
| `timestamp` | `timestamp` | ✅ Match | float → int64 µs |
| `raw` | ❌ **MISSING** | ⚠️ Not Mapped | Raw dict not persisted |

### 🔍 Discrepancies
**None** - Excellent alignment!

---

## 8. OpenInterest

### Python Type (`types.pyx`)
```python
cdef class OpenInterest:
    cdef readonly str exchange
    cdef readonly str symbol
    cdef readonly object open_interest # Decimal
    cdef readonly object timestamp     # None or float
    cdef readonly dict raw
```

### Proto Schema (`open_interest.proto`)
```protobuf
message OpenInterest {
  string exchange = 1;
  string symbol = 2;
  string open_interest = 3;
  optional int64 timestamp = 4;  // microseconds
}
```

### ✅ Alignment Status: **FULLY ALIGNED**

| Python Field | Proto Field | Status | Notes |
|--------------|-------------|--------|-------|
| `exchange` | `exchange` | ✅ Match | Both `string` |
| `symbol` | `symbol` | ✅ Match | Both `string` |
| `open_interest` | `open_interest` | ✅ Match | Decimal → string |
| `timestamp` | `timestamp` | ✅ Match | Both optional, float → int64 µs |
| `raw` | ❌ **MISSING** | ⚠️ Not Mapped | Raw dict not persisted |

### 🔍 Discrepancies
**None** - Excellent alignment! Optionality correctly handled.

---

## 9. Order

### Python Type (`types.pyx`)
```python
cdef class Order:
    cdef readonly str exchange
    cdef readonly str symbol
    cdef readonly str client_order_id
    cdef readonly str side
    cdef readonly str type
    cdef readonly object price     # Decimal
    cdef readonly object amount    # Decimal
    cdef readonly str account
    cdef readonly object timestamp # None or float
```

### Proto Schema (`order.proto`)
```protobuf
message Order {
  string exchange = 1;
  string symbol = 2;
  string side = 3;
  string order_type = 4;
  string price = 5;
  string amount = 6;
  int64 timestamp = 7;
  optional string client_order_id = 8;
  optional string account = 9;
}
```

### ✅ Alignment Status: **WELL ALIGNED**

| Python Field | Proto Field | Status | Notes |
|--------------|-------------|--------|-------|
| `exchange` | `exchange` | ✅ Match | Both `string` |
| `symbol` | `symbol` | ✅ Match | Both `string` |
| `client_order_id` | `client_order_id` | ✅ Match | Both optional |
| `side` | `side` | ✅ Match | Both `string` |
| `type` | `order_type` | ✅ Match | Field renamed for clarity |
| `price` | `price` | ✅ Match | Decimal → string |
| `amount` | `amount` | ✅ Match | Decimal → string |
| `account` | `account` | ✅ Match | Both optional |
| `timestamp` | `timestamp` | ⚠️ Optionality | Python allows `None`, proto required |

### 🔍 Discrepancies

1. **Timestamp Optionality**: Python allows `None`, proto required
   - **Recommendation**: Consider `optional int64 timestamp = 7;`

---

## 10. OrderInfo

### Python Type (`types.pyx`)
```python
cdef class OrderInfo:
    cdef readonly str exchange
    cdef readonly str symbol
    cdef readonly str id
    cdef readonly str client_order_id
    cdef readonly str side
    cdef readonly str status
    cdef readonly str type
    cdef readonly object price      # Decimal
    cdef readonly object amount     # Decimal
    cdef readonly object remaining  # None or Decimal
    cdef readonly str account
    cdef readonly object timestamp  # None or float
    cdef readonly object raw
```

### Proto Schema (`order_info.proto`)
```protobuf
message OrderInfo {
  string exchange = 1;
  string symbol = 2;
  string order_id = 3;
  string side = 4;
  string status = 5;
  string order_type = 6;
  string price = 7;
  string amount = 8;
  optional string remaining = 9;
  int64 timestamp = 10;
  optional string client_order_id = 11;
  optional string account = 12;
}
```

### ✅ Alignment Status: **WELL ALIGNED**

| Python Field | Proto Field | Status | Notes |
|--------------|-------------|--------|-------|
| `exchange` | `exchange` | ✅ Match | Both `string` |
| `symbol` | `symbol` | ✅ Match | Both `string` |
| `id` | `order_id` | ✅ Match | Field renamed |
| `client_order_id` | `client_order_id` | ✅ Match | Both optional |
| `side` | `side` | ✅ Match | Both `string` |
| `status` | `status` | ✅ Match | Both `string` |
| `type` | `order_type` | ✅ Match | Field renamed |
| `price` | `price` | ✅ Match | Decimal → string |
| `amount` | `amount` | ✅ Match | Decimal → string |
| `remaining` | `remaining` | ✅ Match | Both optional, Decimal → string |
| `account` | `account` | ✅ Match | Both optional |
| `timestamp` | `timestamp` | ⚠️ Optionality | Python allows `None`, proto required |
| `raw` | ❌ **MISSING** | ⚠️ Not Mapped | Raw data not persisted |

### 🔍 Discrepancies

1. **Timestamp Optionality**: Python allows `None`, proto required
   - **Recommendation**: Consider `optional int64 timestamp = 10;`

---

## Summary Update: Alignment Issues by Category

### 🔴 Critical Issues (Block Migration) - **RESOLVED**

1. ✅ **FIXED**: Trade.type Missing → Added `optional string trade_type = 9;`
2. ✅ **FIXED**: Funding Mark Price/Rate Not Optional → Changed to `optional`
3. ✅ **FIXED**: OrderBook Delta Not Represented → Documented limitation

### 🟡 Medium Issues (May Cause Data Loss)

1. **Raw Field Universally Missing**: Cannot reconstruct original exchange messages (10 types affected)
2. **Timestamp Optionality**: Several types allow `None` but proto uses required `int64` (Order, OrderInfo, possibly others)
3. ~~Trade.raw_id Unclear~~ → Clarified as CCXT/tardis trace identifier

### 🟢 Minor Issues (Documentation Needed)

1. **Side Field Type Change**: String → Enum (requires mapping documentation)
2. **Field Renames**: Consistent pattern across types
3. **Timestamp Precision**: float seconds → int64 microseconds

---

## Next Steps

1. ✅ **COMPLETED**: P0 fixes (trade_type, funding optionality, orderbook docs)
2. ✅ **COMPLETED**: Review 10/15 core types (67% complete)
3. 🔄 **IN PROGRESS**: Complete remaining 5 types (Fill, Balance, Position, Transaction, NBBO)
4. 🔄 Create GitHub issue for P1/P2 improvements
5. 🔄 Update RELEASE_v0.1.0.md with migration notes
6. 🔄 Implement conversion library with tests

---

**Review Status**: 10/15 core types reviewed (67%)  
**P0 Issues**: 3/3 resolved ✅  
**Last Updated**: 2025-10-25  
**Reviewer**: Claude Code (AI Development Workflow)
