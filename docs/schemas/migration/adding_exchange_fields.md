# Migration Guide: Adding Exchange Field Extraction

This guide provides a step-by-step template for implementing protobuf v2beta1 optional field support for a new exchange. Use this as a checklist when extending field coverage beyond the reference Binance implementation.

**Target Audience:** Cryptofeed contributors implementing exchange-specific field extraction
**Prerequisite Knowledge:** Python asyncio, exchange WebSocket APIs, protobuf basics
**Reference Implementation:** Binance (see `cryptofeed/exchanges/binance.py` and `docs/schemas/mappings/binance_field_mapping.md`)

---

## Overview

Implementing v2beta1 field support for a new exchange involves four main steps:

1. **Research:** Identify which fields the exchange provides via its API
2. **Extraction:** Extend exchange handler to populate Trade/OrderBook attributes
3. **Testing:** Verify field extraction with unit and integration tests
4. **Documentation:** Update field availability matrix and create exchange-specific mapping

**Estimated Effort:** 3-5 hours per exchange (depending on API complexity)

**Success Criteria:**
- All available v2beta1 fields extracted from exchange API
- Unit tests cover field extraction and edge cases
- Integration test validates end-to-end field transmission via Kafka
- Documentation updated to reflect implementation status

---

## Step 1: Research Exchange API Capabilities

### 1.1 Review Official API Documentation

Identify which v2beta1 fields the exchange provides in raw WebSocket/REST responses:

**Trade Fields to Research:**
- `maker` (bool): Does the exchange indicate maker/taker role?
- `event_time` (timestamp): Does the exchange provide matching engine event time?
- `match_id` (string): Does the exchange provide match/fill identifier separate from trade ID?
- `liquidity_flag` (string): Does the exchange provide liquidity indicators (e.g., "M"/"T", "maker"/"taker")?

**OrderBook Fields to Research:**
- `event_time` (timestamp): Does the exchange provide order book update event time?
- `last_update_id` (int64): Does the exchange provide sequence numbers for gap detection?

### 1.2 Analyze Raw WebSocket Messages

**Action Items:**
1. Subscribe to exchange WebSocket trade and order book channels
2. Capture sample messages (use `wscat` or Python WebSocket client)
3. Document field names, data types, and value ranges
4. Note any field transformations needed (e.g., milliseconds → seconds)

**Example Investigation:**
```python
# Quick test script to inspect exchange WebSocket messages
import asyncio
import websockets
import json

async def inspect_exchange_feed():
    uri = "wss://exchange.example.com/ws"  # Replace with exchange WebSocket URL
    async with websockets.connect(uri) as websocket:
        # Subscribe to trade channel
        await websocket.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": ["BTCUSDT@trade"]
        }))

        # Capture 10 messages
        for i in range(10):
            msg = await websocket.recv()
            print(f"Message {i+1}:")
            print(json.dumps(json.loads(msg), indent=2))
            print("-" * 80)

asyncio.run(inspect_exchange_feed())
```

### 1.3 Document Findings

Create a research notes file documenting:
- Available fields and their API names
- Data types and conversion requirements
- Fields NOT available (mark as NOT_AVAILABLE)
- Edge cases (e.g., field missing in some markets, conditional presence)

**Template:**
```markdown
## Exchange XYZ Field Research

### Trade Fields
- ✅ maker: Available via 'is_maker' field (boolean)
- ✅ event_time: Available via 'event_ts' field (milliseconds)
- ❌ match_id: Not provided in public trade feed
- ✅ liquidity_flag: Available via 'liquidity' field (string: "M"/"T")

### OrderBook Fields
- ✅ event_time: Available via 'timestamp' field (microseconds)
- ✅ last_update_id: Available via 'seq_num' field (int64)

### Data Type Conversions
- event_ts (ms) → event_time (seconds): divide by 1000
- seq_num (int64) → last_update_id (int64): direct mapping
```

---

## Step 2: Implement Field Extraction

### 2.1 Extend Trade Handler

**File:** `cryptofeed/exchanges/<exchange_name>.py`

**Pattern:** Follow the Binance reference implementation structure:

```python
async def _trade(self, msg: dict, timestamp: float):
    """
    Extract trade with v2beta1 optional fields.

    Raw message structure:
    {
        "price": "50000.00",
        "amount": "0.015",
        "is_maker": true,          # Maker flag
        "event_ts": 1672531200000, # Event timestamp (ms)
        "match_id": "abc123",      # Match identifier
        "liquidity": "M"           # Liquidity flag
    }
    """
    # Step 1: Extract core fields (existing logic)
    # ... existing extraction code ...

    # Step 2: Extract v2beta1 optional fields
    maker = None
    if 'is_maker' in msg:
        maker = bool(msg['is_maker'])  # Convert to boolean

    event_time = None
    if 'event_ts' in msg:
        event_time = msg['event_ts'] / 1000  # Milliseconds to seconds

    match_id = None
    if 'match_id' in msg:
        match_id = str(msg['match_id'])  # Ensure string type

    liquidity_flag = None
    if 'liquidity' in msg:
        liquidity_flag = str(msg['liquidity'])  # Keep as-is

    # Step 3: Create Trade object with optional fields
    return Trade(
        exchange=self.id,
        symbol=self.exchange_symbol_to_std_symbol(msg['symbol']),
        side='buy' if msg['side'] == 'buy' else 'sell',
        price=Decimal(msg['price']),
        amount=Decimal(msg['amount']),
        timestamp=msg['trade_ts'] / 1000,  # Trade timestamp
        id=str(msg.get('trade_id', '')),

        # v2beta1 Optional Fields
        maker=maker,
        event_time=event_time,
        match_id=match_id,
        liquidity_flag=liquidity_flag,
    )
```

**Best Practices:**
- **Graceful Degradation:** Use `.get()` method with None fallback for optional fields
- **Type Safety:** Explicitly convert types (bool(), str(), float())
- **Timestamp Precision:** Convert exchange timestamps to float seconds (intermediate format)
- **Null Handling:** Set field to None if not present (don't populate with defaults)

### 2.2 Extend OrderBook Handler

**File:** `cryptofeed/exchanges/<exchange_name>.py`

**Pattern:**

```python
async def _book(self, msg: dict, timestamp: float):
    """
    Extract order book with v2beta1 optional fields.

    Raw message structure:
    {
        "symbol": "BTCUSDT",
        "bids": [["50000.00", "0.5"]],
        "asks": [["50001.00", "0.3"]],
        "timestamp": 1672531200000000,  # Microseconds
        "seq_num": 160                  # Sequence number
    }
    """
    # Step 1: Parse order book levels (existing logic)
    # ... existing book parsing code ...

    # Step 2: Extract v2beta1 optional fields
    event_time = None
    if 'timestamp' in msg:
        event_time = msg['timestamp'] / 1_000_000  # Microseconds to seconds

    last_update_id = None
    if 'seq_num' in msg:
        last_update_id = int(msg['seq_num'])  # Ensure int64 type

    # Step 3: Create OrderBook object with optional fields
    return OrderBook(
        exchange=self.id,
        symbol=self.exchange_symbol_to_std_symbol(msg['symbol']),
        book=parsed_book,
        timestamp=timestamp,

        # v2beta1 Optional Fields
        event_time=event_time,
        last_update_id=last_update_id,
    )
```

**Common Pitfalls:**
- **Timestamp Units:** Exchanges use seconds, milliseconds, microseconds, or nanoseconds - verify units!
- **Sequence Gaps:** Ensure `last_update_id` is monotonically increasing per symbol
- **Missing Fields:** Some exchanges only provide fields in specific market types (spot vs futures)

### 2.3 Verify Protobuf Converter Handles New Fields

**File:** `cryptofeed/backends/protobuf/converters.py`

**Note:** The converter should already handle v2beta1 fields via `hasattr()` checks. Verify:

```python
# In trade_to_proto()
if hasattr(trade_obj, 'maker') and trade_obj.maker is not None:
    proto.maker = bool(trade_obj.maker)  # ✅ Should work

if hasattr(trade_obj, 'event_time') and trade_obj.event_time is not None:
    proto.event_time = int(trade_obj.event_time * 1_000_000)  # ✅ Should work

# ... similar for match_id, liquidity_flag, OrderBook fields ...
```

**Action:** No code changes needed in converters.py (designed for forward compatibility)

---

## Step 3: Create Comprehensive Test Suite

### 3.1 Unit Tests for Field Extraction

**File:** `tests/unit/test_<exchange>_field_extraction.py`

**Template:**

```python
"""Unit tests for Exchange XYZ v2beta1 field extraction."""

import pytest
from decimal import Decimal
from cryptofeed.exchanges.<exchange> import <ExchangeClass>


@pytest.mark.asyncio
class TestExchangeTradeFieldExtraction:
    """Test v2beta1 Trade field extraction from Exchange XYZ."""

    async def test_trade_with_all_fields_populated(self):
        """Verify all available v2beta1 fields are extracted."""
        exchange = <ExchangeClass>()

        # Mock WebSocket message with all fields
        msg = {
            'symbol': 'BTCUSDT',
            'price': '50000.00',
            'amount': '0.015',
            'side': 'buy',
            'trade_id': '12345',
            'trade_ts': 1672531199987,
            # v2beta1 fields
            'is_maker': True,
            'event_ts': 1672531200000,
            'match_id': 'abc123',
            'liquidity': 'M',
        }

        trade = await exchange._trade(msg, timestamp=1672531200.0)

        # Verify core fields
        assert trade.exchange == '<exchange-id>'
        assert trade.symbol == 'BTC-USDT'
        assert trade.price == Decimal('50000.00')

        # Verify v2beta1 fields
        assert trade.maker is True
        assert trade.event_time == 1672531200.0  # 1672531200000ms / 1000
        assert trade.match_id == 'abc123'
        assert trade.liquidity_flag == 'M'

    async def test_trade_with_missing_optional_fields(self):
        """Verify graceful degradation when fields missing."""
        exchange = <ExchangeClass>()

        # Mock message with only core fields
        msg = {
            'symbol': 'BTCUSDT',
            'price': '50000.00',
            'amount': '0.015',
            'side': 'sell',
            'trade_id': '12346',
            'trade_ts': 1672531199987,
            # No v2beta1 fields
        }

        trade = await exchange._trade(msg, timestamp=1672531200.0)

        # Verify v2beta1 fields are None (not populated with defaults)
        assert trade.maker is None
        assert trade.event_time is None
        assert trade.match_id is None
        assert trade.liquidity_flag is None

    async def test_trade_maker_flag_semantics(self):
        """Verify maker flag correctly indicates buyer role."""
        exchange = <ExchangeClass>()

        # Buyer is maker (sell order matched)
        msg_maker = {
            'symbol': 'BTCUSDT',
            'price': '50000.00',
            'amount': '0.015',
            'side': 'buy',
            'is_maker': True,
            'trade_ts': 1672531199987,
        }
        trade_maker = await exchange._trade(msg_maker, timestamp=1672531200.0)
        assert trade_maker.maker is True

        # Buyer is taker (buy order matched)
        msg_taker = {
            'symbol': 'BTCUSDT',
            'price': '50000.00',
            'amount': '0.015',
            'side': 'buy',
            'is_maker': False,
            'trade_ts': 1672531199987,
        }
        trade_taker = await exchange._trade(msg_taker, timestamp=1672531200.0)
        assert trade_taker.maker is False

    async def test_trade_timestamp_conversion_accuracy(self):
        """Verify timestamp conversion maintains precision."""
        exchange = <ExchangeClass>()

        msg = {
            'symbol': 'BTCUSDT',
            'price': '50000.00',
            'amount': '0.015',
            'side': 'buy',
            'event_ts': 1672531200123,  # Milliseconds with precision
            'trade_ts': 1672531199987,
        }

        trade = await exchange._trade(msg, timestamp=1672531200.0)

        # Verify millisecond precision preserved
        assert trade.event_time == 1672531200.123  # 1672531200123ms / 1000
        # Verify no rounding errors
        assert int(trade.event_time * 1000) == 1672531200123


@pytest.mark.asyncio
class TestExchangeOrderBookFieldExtraction:
    """Test v2beta1 OrderBook field extraction from Exchange XYZ."""

    async def test_orderbook_with_event_time_and_sequence(self):
        """Verify order book fields extracted correctly."""
        exchange = <ExchangeClass>()

        msg = {
            'symbol': 'BTCUSDT',
            'bids': [['50000.00', '0.5']],
            'asks': [['50001.00', '0.3']],
            'timestamp': 1672531200000000,  # Microseconds
            'seq_num': 160,
        }

        book = await exchange._book(msg, timestamp=1672531200.0)

        # Verify v2beta1 fields
        assert book.event_time == 1672531200.0  # 1672531200000000µs / 1_000_000
        assert book.last_update_id == 160

    async def test_orderbook_sequence_monotonicity(self):
        """Verify sequence numbers are monotonically increasing."""
        exchange = <ExchangeClass>()

        # Simulate consecutive updates
        msg1 = {
            'symbol': 'BTCUSDT',
            'bids': [['50000.00', '0.5']],
            'asks': [['50001.00', '0.3']],
            'seq_num': 157,
        }
        msg2 = {
            'symbol': 'BTCUSDT',
            'bids': [['50000.50', '0.6']],
            'asks': [['50001.50', '0.4']],
            'seq_num': 158,
        }

        book1 = await exchange._book(msg1, timestamp=1672531200.0)
        book2 = await exchange._book(msg2, timestamp=1672531201.0)

        assert book2.last_update_id == book1.last_update_id + 1
```

**Coverage Requirements:**
- Field extraction with all fields present
- Field extraction with missing fields (None handling)
- Data type conversions (timestamps, booleans, strings)
- Edge cases (zero values, negative numbers, empty strings)
- Timestamp precision preservation

### 3.2 Integration Tests for End-to-End Field Transmission

**File:** `tests/integration/test_<exchange>_field_e2e.py`

**Template:**

```python
"""Integration tests for Exchange XYZ field transmission via Kafka."""

import pytest
import asyncio
from decimal import Decimal
from cryptofeed.backends.kafka import Kafka
from cryptofeed.backends.protobuf.converters import trade_to_proto
from cryptofeed.types import Trade, OrderBook


@pytest.mark.integration
@pytest.mark.asyncio
async def test_exchange_trade_fields_kafka_transmission():
    """Verify Trade fields flow through Kafka pipeline."""
    # Setup Kafka backend (assumes Kafka running locally)
    kafka_backend = Kafka(
        bootstrap_servers='localhost:9092',
        topic_prefix='test_exchange_fields'
    )

    # Create Trade with all v2beta1 fields
    trade = Trade(
        exchange='exchange-xyz',
        symbol='BTC-USDT',
        side='buy',
        price=Decimal('50000.00'),
        amount=Decimal('0.015'),
        timestamp=1672531200.0,
        id='12345',
        maker=True,
        event_time=1672531200.123,
        match_id='abc123',
        liquidity_flag='M',
    )

    # Convert to protobuf
    proto = trade_to_proto(trade)

    # Verify protobuf fields populated
    assert proto.maker is True
    assert proto.event_time == 1672531200123000  # Microseconds
    assert proto.match_id == 'abc123'
    assert proto.liquidity_flag == 'M'

    # Send via Kafka
    await kafka_backend(trade, receipt_timestamp=1672531200.0)

    # ... (Additional verification: consume from Kafka and parse back) ...
```

**Integration Test Checklist:**
- [ ] Trade protobuf message contains all populated fields
- [ ] OrderBook protobuf message contains all populated fields
- [ ] Kafka consumer can parse protobuf messages
- [ ] Field values match source data (no corruption)
- [ ] Missing fields remain unset in protobuf (not populated with defaults)

---

## Step 4: Update Documentation

### 4.1 Update Field Availability Matrix

**File:** `docs/schemas/mappings/field_availability_matrix.md`

**Action:** Update the exchange row to reflect implementation status:

```markdown
| Exchange | maker | event_time | match_id | liquidity_flag | Implementation Status |
|----------|-------|------------|----------|----------------|----------------------|
| **Exchange XYZ** | ✅ SUPPORTED | ✅ SUPPORTED | ❌ NOT_AVAILABLE | ✅ SUPPORTED | **PRODUCTION** |
```

### 4.2 Create Exchange-Specific Field Mapping Document

**File:** `docs/schemas/mappings/<exchange>_field_mapping.md`

**Template:** Follow the Binance field mapping structure:

**Sections to Include:**
1. **Overview:** Exchange name, API version, implementation status
2. **Trade Field Mapping:** Table mapping exchange API fields to protobuf fields
3. **Data Type Conversions:** Detailed conversion logic for each field
4. **Code Implementation Reference:** Code snippets showing extraction logic
5. **OrderBook Field Mapping:** Similar structure for order book fields
6. **Field Availability Summary:** Quick reference table
7. **Testing & Validation:** Link to test files
8. **Common Issues & Troubleshooting:** Known edge cases
9. **API Documentation References:** Links to official exchange docs

**Example:**
```markdown
# Exchange XYZ Field Mapping Specification

## Trade Field Mapping

| Exchange Field | Type | Protobuf Field | Conversion Logic |
|---------------|------|----------------|------------------|
| 'is_maker' | boolean | Trade.maker | Direct mapping |
| 'event_ts' | int64 (ms) | Trade.event_time | Divide by 1000 |
| 'match_id' | string | Trade.match_id | Direct mapping |
| 'liquidity' | string | Trade.liquidity_flag | Direct mapping |
```

### 4.3 Update Field Population Metrics

**File:** `tools/validate_field_population.py` (monitoring script)

**Action:** Add exchange to tracking:

```python
# Add exchange to EXCHANGES list
EXCHANGES = ['binance', 'okx', 'coinbase', 'exchange-xyz', ...]

# Monitoring script will automatically track field population rates
```

---

## Step 5: Submit Pull Request

### 5.1 PR Checklist

- [ ] **Code:**
  - [ ] Exchange handler updated with v2beta1 field extraction
  - [ ] All fields extracted that exchange provides
  - [ ] Fields NOT available marked as None (not populated with defaults)
- [ ] **Tests:**
  - [ ] Unit tests cover field extraction (10+ test cases)
  - [ ] Integration tests validate end-to-end transmission
  - [ ] All tests passing (`pytest tests/unit/test_<exchange>_field_extraction.py -v`)
  - [ ] No regressions in existing exchange tests
- [ ] **Documentation:**
  - [ ] Field availability matrix updated
  - [ ] Exchange-specific field mapping document created
  - [ ] Code comments explain field semantics
- [ ] **CI/CD:**
  - [ ] All CI checks passing
  - [ ] Code coverage maintained or improved

### 5.2 PR Title and Description Template

**Title:**
```
feat(exchanges): add v2beta1 field extraction for Exchange XYZ
```

**Description:**
```markdown
## Summary
Implements protobuf v2beta1 optional field extraction for Exchange XYZ, following the pattern established by Binance reference implementation.

## Fields Implemented
- ✅ Trade.maker (via 'is_maker' field)
- ✅ Trade.event_time (via 'event_ts' field, milliseconds to seconds conversion)
- ❌ Trade.match_id (NOT_AVAILABLE - exchange does not provide)
- ✅ Trade.liquidity_flag (via 'liquidity' field)
- ✅ OrderBook.event_time (via 'timestamp' field, microseconds to seconds)
- ✅ OrderBook.last_update_id (via 'seq_num' field)

## Testing
- Unit tests: 12 test cases covering all fields and edge cases
- Integration tests: End-to-end Kafka transmission validated
- All tests passing: `pytest tests/unit/test_exchange_xyz_field_extraction.py -v`

## Documentation
- Field availability matrix updated: docs/schemas/mappings/field_availability_matrix.md
- Exchange-specific mapping created: docs/schemas/mappings/exchange_xyz_field_mapping.md
- Code comments added to explain field extraction logic

## Related Issues
- Closes #XXX (if applicable)
- Ref: REQ-1.18 (Field availability documentation requirement)
```

---

## Reference Resources

### Code Examples

**Canonical Reference:** Binance implementation
- Handler: `cryptofeed/exchanges/binance.py` (`_trade()` and `_book()` methods)
- Tests: `tests/unit/test_binance_field_extraction.py`
- Docs: `docs/schemas/mappings/binance_field_mapping.md`

### Documentation Templates

- **Field Availability Matrix:** `docs/schemas/mappings/field_availability_matrix.md`
- **Exchange Field Mapping:** `docs/schemas/mappings/binance_field_mapping.md`
- **Trade Mapping:** `docs/schemas/mappings/trade_mapping.md`
- **OrderBook Mapping:** `docs/schemas/mappings/order_book_mapping.md`

### Testing Patterns

**Unit Test Patterns:**
```python
# Pattern 1: All fields populated
async def test_all_fields_populated(): ...

# Pattern 2: Missing fields (None handling)
async def test_missing_fields_graceful_degradation(): ...

# Pattern 3: Data type conversions
async def test_timestamp_conversion_accuracy(): ...

# Pattern 4: Edge cases (zero, negative, empty)
async def test_edge_case_values(): ...
```

**Integration Test Patterns:**
```python
# Pattern 1: End-to-end Kafka flow
@pytest.mark.integration
async def test_kafka_field_transmission(): ...

# Pattern 2: Protobuf field validation
def test_protobuf_field_presence(): ...
```

---

## Common Patterns & Best Practices

### Pattern 1: Timestamp Conversion

**Problem:** Exchanges use different timestamp units (seconds, milliseconds, microseconds)

**Solution:**
```python
# Step 1: Identify exchange timestamp unit (check API docs)
# Step 2: Convert to float seconds (Cryptofeed intermediate format)
event_time = msg['event_ts'] / 1000  # Milliseconds to seconds
event_time = msg['timestamp'] / 1_000_000  # Microseconds to seconds

# Step 3: Protobuf converter handles seconds → microseconds automatically
proto.event_time = int(trade_obj.event_time * 1_000_000)
```

### Pattern 2: Optional Field Extraction

**Problem:** Field may not be present in all message types or markets

**Solution:**
```python
# Use .get() with None fallback
maker = None
if 'is_maker' in msg:
    maker = bool(msg['is_maker'])

# Or dict.get() method
maker = msg.get('is_maker')  # Returns None if missing
if maker is not None:
    maker = bool(maker)  # Convert to boolean only if present
```

### Pattern 3: Type Safety

**Problem:** Exchange may return unexpected types (string instead of int)

**Solution:**
```python
# Explicit type conversion with error handling
match_id = None
if 'match_id' in msg:
    try:
        match_id = str(msg['match_id'])  # Ensure string type
    except (ValueError, TypeError):
        match_id = None  # Graceful fallback on conversion error
```

### Pattern 4: Maker Flag Semantics

**Problem:** Exchanges define maker flag differently (buyer perspective vs seller perspective)

**Solution:**
```python
# Standardize on "buyer is maker" semantics
# - True: Buyer placed limit order (maker), matched by seller (taker)
# - False: Buyer aggressively crossed spread (taker), matched existing sell order

# Example: OKX provides 'side' field ("buy"/"sell" for taker)
if msg['side'] == 'buy':
    maker = False  # Buyer is taker (aggressive buy)
elif msg['side'] == 'sell':
    maker = True   # Seller is taker, buyer must be maker
```

---

## Troubleshooting

### Issue: Tests Fail with "AttributeError: 'Trade' object has no attribute 'maker'"

**Cause:** Trade class not extended with v2beta1 attributes

**Solution:** Verify `cryptofeed/types.pyx` includes v2beta1 field declarations:
```python
cdef class Trade:
    cdef public object maker  # Should be present
    cdef public object event_time
    cdef public object match_id
    cdef public object liquidity_flag
```

---

### Issue: Protobuf Fields Not Populated Despite Correct Extraction

**Cause:** Protobuf converter not using `hasattr()` checks

**Solution:** Verify `cryptofeed/backends/protobuf/converters.py` includes:
```python
if hasattr(trade_obj, 'maker') and trade_obj.maker is not None:
    proto.maker = bool(trade_obj.maker)
```

---

### Issue: Timestamp Precision Loss

**Cause:** Float precision limits or incorrect conversion

**Solution:**
1. Verify source timestamp unit (seconds vs milliseconds vs microseconds)
2. Use explicit division (not multiplication for seconds)
3. Test round-trip conversion:
   ```python
   original_ms = 1672531200123
   event_time_s = original_ms / 1000  # Seconds (float)
   proto_us = int(event_time_s * 1_000_000)  # Microseconds (int64)
   assert proto_us == 1672531200123000  # Precision preserved
   ```

---

## Success Criteria Validation

Before marking implementation complete, verify:

1. **Field Extraction:**
   - [ ] All available exchange fields extracted
   - [ ] Unavailable fields explicitly set to None (not populated with defaults)
   - [ ] Data type conversions correct (timestamps, booleans, strings)

2. **Testing:**
   - [ ] 10+ unit tests covering fields and edge cases
   - [ ] Integration test validates end-to-end Kafka transmission
   - [ ] 100% test pass rate
   - [ ] No regressions in existing exchange tests

3. **Documentation:**
   - [ ] Field availability matrix updated with implementation status
   - [ ] Exchange-specific field mapping document created
   - [ ] Code comments explain extraction logic
   - [ ] Migration guide checklist completed

4. **Code Quality:**
   - [ ] Follows Binance reference implementation pattern
   - [ ] Type hints used for all new functions
   - [ ] Error handling for missing/malformed fields
   - [ ] Consistent naming conventions

---

## Support & Questions

**Stuck on implementation?**
- Review Binance reference: `cryptofeed/exchanges/binance.py`
- Check existing tests: `tests/unit/test_binance_field_extraction.py`
- Read field mapping: `docs/schemas/mappings/binance_field_mapping.md`

**API documentation unclear?**
- Open GitHub issue with label `documentation`
- Tag with `exchange:<exchange-name>` for visibility

**Need code review?**
- Submit draft PR for early feedback
- Tag maintainers for exchange-specific guidance

---

## Version History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-12-14 | 1.0 | Initial migration guide template | PR #16 Code Review Remediation |
