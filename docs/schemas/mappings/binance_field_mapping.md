# Binance Field Mapping Specification

This document provides a detailed mapping of Binance WebSocket API fields to Cryptofeed protobuf v2beta1 schema fields. It serves as the canonical reference for Binance field extraction and as a template for implementing other exchanges.

**Exchange:** Binance Spot
**API Version:** v3 (WebSocket Market Streams)
**Implementation Status:** Production (as of 2025-12-14)
**Schema Version:** v2beta1

---

## Trade Field Mapping

### Overview

Binance provides rich trade metadata through the `aggTrade` (Aggregate Trade Streams) WebSocket channel. The implementation extracts four v2beta1 optional fields beyond core price/volume data.

### WebSocket Channel

**Subscribe Format:**
```json
{
  "method": "SUBSCRIBE",
  "params": ["btcusdt@aggTrade"],
  "id": 1
}
```

**Stream Name:** `<symbol>@aggTrade` (lowercase symbol, e.g., `btcusdt@aggTrade`)

---

### Raw WebSocket Message Example

```json
{
  "e": "aggTrade",       // Event type
  "E": 1672531200000,    // Event time (milliseconds)
  "s": "BTCUSDT",        // Symbol
  "a": 12345,            // Aggregate trade ID
  "p": "16950.50",       // Price
  "q": "0.015",          // Quantity
  "f": 100,              // First trade ID
  "l": 105,              // Last trade ID
  "T": 1672531199987,    // Trade time (milliseconds)
  "m": true,             // Is the buyer the market maker?
  "M": true              // Ignore (always true for aggTrade)
}
```

---

### Field Extraction Specification

| Binance Field | Type | Protobuf Field | Cryptofeed Attribute | Conversion Logic | Notes |
|---------------|------|----------------|---------------------|------------------|-------|
| **'m'** (maker flag) | `boolean` | `Trade.maker` | `Trade.maker` | Direct mapping: `maker = msg['m']` | True if buyer is maker (sell order matched), False if buyer is taker (buy order matched) |
| **'E'** (event time) | `int64` (ms) | `Trade.event_time` | `Trade.event_time` | Milliseconds to seconds: `event_time = msg['E'] / 1000` → Protobuf: `int(event_time * 1_000_000)` µs | Exchange matching engine event timestamp |
| **'a'** (aggregate trade ID) | `int64` | `Trade.match_id` | `Trade.match_id` | Integer to string: `match_id = str(msg['a'])` | Unique aggregate trade identifier (combines multiple fills) |
| **liquidity_flag** | - | `Trade.liquidity_flag` | `Trade.liquidity_flag` | ⚠️ NOT AVAILABLE | Binance aggTrade does not provide this field |

---

### Data Type Conversions

#### 1. Boolean Conversion (maker field)

**Binance API:**
- Type: `boolean`
- Values: `true` (buyer is maker) or `false` (buyer is taker)

**Cryptofeed Conversion:**
```python
# cryptofeed/exchanges/binance.py
maker = msg.get('m')  # Direct boolean extraction
```

**Protobuf Encoding:**
```python
# cryptofeed/backends/protobuf/converters.py
if hasattr(trade_obj, 'maker') and trade_obj.maker is not None:
    proto.maker = bool(trade_obj.maker)
```

**Semantic Meaning:**
- `maker = True`: Buyer placed a limit order that rested on the book (maker), matched by seller (taker)
- `maker = False`: Buyer aggressively crossed spread to match existing sell order

---

#### 2. Timestamp Conversion (event_time field)

**Binance API:**
- Type: `int64` (milliseconds since Unix epoch)
- Example: `1672531200000` = `2023-01-01 00:00:00 UTC`

**Cryptofeed Conversion:**
```python
# cryptofeed/exchanges/binance.py
event_time = msg.get('E', 0) / 1000  # Milliseconds to seconds (float)
```

**Protobuf Encoding:**
```python
# cryptofeed/backends/protobuf/converters.py
if hasattr(trade_obj, 'event_time') and trade_obj.event_time is not None:
    proto.event_time = int(trade_obj.event_time * 1_000_000)  # Seconds to microseconds
```

**Precision:**
- Binance provides millisecond precision (`E` field)
- Cryptofeed stores as float seconds (intermediate representation)
- Protobuf encodes as int64 microseconds (final representation)
- **Loss:** None (milliseconds → microseconds maintains precision)

**Timestamp Comparison:**
- `'E'` (event_time): Exchange matching engine event timestamp
- `'T'` (trade_time): Trade execution timestamp (used for `Trade.timestamp`)
- Difference typically <10ms, but can diverge during high load

---

#### 3. String Conversion (match_id field)

**Binance API:**
- Type: `int64` (aggregate trade ID)
- Example: `12345`

**Cryptofeed Conversion:**
```python
# cryptofeed/exchanges/binance.py
match_id = str(msg.get('a'))  # Integer to string
```

**Protobuf Encoding:**
```python
# cryptofeed/backends/protobuf/converters.py
if hasattr(trade_obj, 'match_id') and trade_obj.match_id is not None:
    proto.match_id = str(trade_obj.match_id)
```

**Rationale for String Type:**
- Cross-exchange compatibility (some exchanges use alphanumeric IDs)
- Prevents integer overflow issues for large IDs
- Protobuf schema uses `string` type for extensibility

---

### Code Implementation Reference

**Exchange Handler (cryptofeed/exchanges/binance.py):**
```python
async def _trade(self, msg: dict, timestamp: float):
    """
    Extract Binance trade with v2beta1 fields.

    Raw message structure documented at:
    https://binance-docs.github.io/apidocs/spot/en/#aggregate-trade-streams
    """
    return Trade(
        exchange='binance',
        symbol=msg['s'],
        side='buy' if msg['m'] else 'sell',  # m=true means buyer is maker
        price=Decimal(msg['p']),
        amount=Decimal(msg['q']),
        timestamp=msg['T'] / 1000,  # Trade time (milliseconds to seconds)
        id=str(msg['t']),           # Trade ID (not aggregate ID)

        # v2beta1 Optional Fields (REQ-1)
        maker=msg.get('m'),                    # Boolean maker flag
        event_time=msg.get('E', 0) / 1000,    # Event time (ms → s)
        match_id=str(msg.get('a')),            # Aggregate trade ID
    )
```

**Protobuf Converter (cryptofeed/backends/protobuf/converters.py):**
```python
def trade_to_proto(trade_obj) -> trade_pb2.Trade:
    """Convert Trade to protobuf with v2beta1 fields."""
    proto = trade_pb2.Trade()

    # ... existing field population ...

    # v2beta1 Optional Fields
    if hasattr(trade_obj, 'maker') and trade_obj.maker is not None:
        proto.maker = bool(trade_obj.maker)

    if hasattr(trade_obj, 'event_time') and trade_obj.event_time is not None:
        proto.event_time = int(trade_obj.event_time * 1_000_000)

    if hasattr(trade_obj, 'match_id') and trade_obj.match_id is not None:
        proto.match_id = str(trade_obj.match_id)

    # Note: liquidity_flag not populated (Binance does not provide)

    return proto
```

---

## OrderBook Field Mapping

### Overview

Binance provides order book metadata through the `depth` (Partial Book Depth Streams) and `depthUpdate` (Diff. Depth Stream) WebSocket channels.

### WebSocket Channel

**Subscribe Format:**
```json
{
  "method": "SUBSCRIBE",
  "params": ["btcusdt@depth"],
  "id": 1
}
```

**Stream Name:** `<symbol>@depth` (lowercase symbol, e.g., `btcusdt@depth`)

---

### Raw WebSocket Message Example

```json
{
  "e": "depthUpdate",   // Event type
  "E": 1672531200000,   // Event time (milliseconds)
  "s": "BTCUSDT",       // Symbol
  "U": 157,             // First update ID in event
  "u": 160,             // Final update ID in event
  "b": [                // Bids to be updated
    ["16950.50", "0.015"],
    ["16950.00", "1.500"]
  ],
  "a": [                // Asks to be updated
    ["16951.00", "0.500"],
    ["16951.50", "2.000"]
  ]
}
```

---

### Field Extraction Specification

| Binance Field | Type | Protobuf Field | Cryptofeed Attribute | Conversion Logic | Notes |
|---------------|------|----------------|---------------------|------------------|-------|
| **'E'** (event time) | `int64` (ms) | `OrderBook.event_time` | `OrderBook.event_time` | Milliseconds to seconds: `event_time = msg['E'] / 1000` → Protobuf: `int(event_time * 1_000_000)` µs | Exchange order book update event timestamp |
| **'u'** (final update ID) | `int64` | `OrderBook.last_update_id` | `OrderBook.last_update_id` | Direct mapping: `last_update_id = msg['u']` | Sequence number for gap detection (monotonically increasing) |

---

### Code Implementation Reference

**Exchange Handler (cryptofeed/exchanges/binance.py):**
```python
async def _book(self, msg: dict, timestamp: float):
    """
    Extract Binance order book with v2beta1 fields.

    Raw message structure documented at:
    https://binance-docs.github.io/apidocs/spot/en/#diff-depth-stream
    """
    return OrderBook(
        exchange='binance',
        symbol=msg['s'],
        book=self._parse_book_levels(msg),
        timestamp=msg['E'] / 1000,  # Event time as primary timestamp

        # v2beta1 Optional Fields (REQ-1)
        event_time=msg.get('E', 0) / 1000,  # Event time (ms → s)
        last_update_id=msg.get('u'),        # Final update ID
    )
```

**Protobuf Converter (cryptofeed/backends/protobuf/converters.py):**
```python
def orderbook_to_proto(orderbook_obj) -> order_book_pb2.OrderBook:
    """Convert OrderBook to protobuf with v2beta1 fields."""
    proto = order_book_pb2.OrderBook()

    # ... existing field population ...

    # v2beta1 Optional Fields
    if hasattr(orderbook_obj, 'event_time') and orderbook_obj.event_time is not None:
        proto.event_time = int(orderbook_obj.event_time * 1_000_000)

    if hasattr(orderbook_obj, 'last_update_id') and orderbook_obj.last_update_id is not None:
        proto.last_update_id = int(orderbook_obj.last_update_id)

    return proto
```

---

## Field Availability Summary

| Data Type | Field | Binance API Support | Implementation Status |
|-----------|-------|--------------------|-----------------------|
| Trade | maker | ✅ Yes ('m' field) | ✅ Production |
| Trade | event_time | ✅ Yes ('E' field) | ✅ Production |
| Trade | match_id | ✅ Yes ('a' field) | ✅ Production |
| Trade | liquidity_flag | ❌ No | ⚠️ Not Available |
| OrderBook | event_time | ✅ Yes ('E' field) | ✅ Production |
| OrderBook | last_update_id | ✅ Yes ('u' field) | ✅ Production |

---

## Testing & Validation

### Unit Test Coverage

**Test File:** `tests/unit/test_binance_field_extraction.py`

**Test Cases:**
1. Trade extraction with all fields populated
2. Trade extraction with missing optional fields (graceful degradation)
3. Timestamp conversion accuracy (milliseconds → seconds → microseconds)
4. Maker flag boolean semantics (buyer is maker = sell order matched)
5. Match ID string conversion (integer → string)
6. OrderBook event time and sequence number extraction
7. Missing field scenarios (None handling)

### Integration Test Coverage

**Test File:** `tests/integration/test_kafka_field_population_e2e.py`

**Test Scenarios:**
1. End-to-end Trade flow: Binance WebSocket → Kafka → Protobuf consumer
2. End-to-end OrderBook flow: Binance WebSocket → Kafka → Protobuf consumer
3. Field presence validation in Kafka messages
4. Protobuf field unset behavior when exchange doesn't provide data

---

## Common Issues & Troubleshooting

### Issue 1: Event Time vs Trade Time Confusion

**Symptom:** Timestamps don't match expected values

**Root Cause:** Binance provides two timestamps in trade messages:
- `'E'` (event_time): Matching engine event timestamp
- `'T'` (trade_time): Trade execution timestamp

**Resolution:**
- Use `'E'` for `Trade.event_time` (v2beta1 field)
- Use `'T'` for `Trade.timestamp` (core field)
- Expect <10ms difference under normal conditions

---

### Issue 2: Aggregate Trade ID vs Trade ID

**Symptom:** `match_id` doesn't match `trade_id`

**Root Cause:** Binance aggregates multiple individual fills into single aggTrade message:
- `'a'` (aggregate trade ID): Unique ID for aggregated event
- `'t'` (individual trade ID): Not provided in aggTrade stream

**Resolution:**
- Use `'a'` for `Trade.match_id` (v2beta1 field)
- Use `'a'` for `Trade.id` if individual trade ID not needed
- Subscribe to individual trade stream if granular IDs required

---

### Issue 3: Order Book Gap Detection

**Symptom:** Missing order book updates not detected

**Root Cause:** Relying only on timestamps without sequence validation

**Resolution:**
- Use `last_update_id` for gap detection:
  ```python
  if current_update_id != previous_update_id + 1:
      # Gap detected - re-request snapshot
  ```
- Store `previous_update_id` per symbol
- Request snapshot from REST API when gap detected

---

## API Documentation References

**Official Binance Documentation:**
- [WebSocket Market Streams](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams)
- [Aggregate Trade Streams](https://binance-docs.github.io/apidocs/spot/en/#aggregate-trade-streams)
- [Diff. Depth Stream](https://binance-docs.github.io/apidocs/spot/en/#diff-depth-stream)
- [REST API - Exchange Information](https://binance-docs.github.io/apidocs/spot/en/#exchange-information)

**Cryptofeed Documentation:**
- [Field Availability Matrix](field_availability_matrix.md)
- [Trade Mapping](trade_mapping.md)
- [OrderBook Mapping](order_book_mapping.md)
- [Migration Guide](../migration/adding_exchange_fields.md)

---

## Related Work

### Other Binance Markets

This specification covers **Binance Spot** only. Other Binance markets have similar but distinct field mappings:

- **Binance Futures (USDT-M):** Similar aggTrade structure, additional funding fields
- **Binance Futures (COIN-M):** Coin-margined contracts use same WebSocket schema
- **Binance Options:** Different WebSocket structure, limited metadata

### Future Enhancements

Potential improvements to Binance field extraction:

1. **liquidity_flag Field:** Derive from maker flag (`"M"` if maker, `"T"` if taker)
2. **Individual Trade Stream:** Subscribe to `<symbol>@trade` for non-aggregated fills
3. **Funding Rate:** Extract from `markPrice@1s` stream (futures-specific)
4. **Liquidation Data:** Extract from `forceOrder` stream (futures-specific)

---

## Version History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-12-14 | 1.0 | Initial specification for v2beta1 fields | PR #16 Code Review Remediation |

---

## Feedback & Contributions

**Maintained by:** Cryptofeed Development Team

**Report Issues:**
- Field extraction bugs: GitHub Issues with label `exchange:binance`
- Documentation updates: Pull requests to `docs/schemas/mappings/`

**Questions:**
- Implementation details: See `cryptofeed/exchanges/binance.py` source code
- Field semantics: Refer to official Binance API documentation
