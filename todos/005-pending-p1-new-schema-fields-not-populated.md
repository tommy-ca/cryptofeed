---
status: pending
priority: p1
issue_id: "005"
tags: [code-review, data-integrity, protobuf, schema]
dependencies: []
---

# New Schema Fields Not Populated (Silent Data Loss)

## Problem Statement

**CRITICAL DATA INTEGRITY ISSUE**: The protobuf v2beta1 schema defines new optional fields (`maker`, `event_time`, `match_id`, `liquidity_flag` for Trade; `event_time`, `last_update_id` for OrderBook), but these fields are never populated by the converter functions or extracted from exchange data. This creates **silent data loss** where the schema promises metadata that is always empty.

**Why This Matters**:
- Consumers reading protobuf messages will NEVER receive maker/taker information, event timestamps, or match IDs
- Valuable venue-specific metadata is defined in schema but discarded at source
- Violates schema contract and creates immediate technical debt
- Business impact: Cannot distinguish maker vs taker (critical for fee calculation, market analysis)

## Findings from Review Agents

**Data Integrity Guardian** identified this as the #1 critical issue:
- Trade schema defines 4 new fields (lines in `proto/cryptofeed/normalized/v2beta1/trade.proto`)
- Converter `trade_to_proto()` in `cryptofeed/backends/protobuf/converters.py` (lines 24-58) does NOT populate these fields
- Binance integration captures raw data with `m` (maker), `E` (event_time), `a` (match_id) but doesn't extract them
- Same issue for OrderBook fields (`event_time`, `last_update_id`)

**Example Data Loss Scenario**:
```
Binance WS: {"m": true, "E": 123456789, "a": 12345}
  → Trade(exchange="binance", price=50000, ...)  [maker/event_time/match_id LOST]
    → trade_to_proto() [ignores new fields]
      → Protobuf{maker: unset, event_time: unset, match_id: unset}
        → Consumer reads empty fields → 31% of schema fields permanently empty
```

## Proposed Solutions

### Solution 1: Full Field Population (Recommended)
**Pros**: Complete schema implementation, no data loss
**Cons**: Requires changes across 3 layers (types, exchanges, converters)
**Effort**: Large (2-3 days)
**Risk**: Medium (schema/converter/exchange alignment)

**Implementation Steps**:
1. Extend `Trade` class in `cryptofeed/types.pyx` with new attributes:
   ```python
   cdef class Trade:
       cdef public str maker  # or bool
       cdef public object event_time
       cdef public str match_id
       cdef public str liquidity_flag
   ```

2. Update Binance integration (`cryptofeed/exchanges/binance.py`) to extract fields:
   ```python
   def _trade(self, msg: dict, timestamp: float):
       trade = Trade(
           # ... existing fields ...
           maker=msg.get('m'),  # true/false
           event_time=msg.get('E'),  # milliseconds
           match_id=str(msg.get('a')),  # aggregate trade ID
       )
   ```

3. Update `trade_to_proto()` converter to populate protobuf fields:
   ```python
   if trade_obj.maker is not None:
       proto.maker = trade_obj.maker
   if trade_obj.event_time is not None:
       proto.event_time = int(trade_obj.event_time * 1000)  # to microseconds
   # ... etc
   ```

4. Repeat for other exchanges (OKX, Coinbase, etc.) as data becomes available

5. Add comprehensive tests for new fields

### Solution 2: Mark Fields as Deprecated in Schema
**Pros**: Quick fix, no code changes
**Cons**: Breaks schema promise, admits incomplete implementation
**Effort**: Small (1 hour)
**Risk**: Low

**Implementation**:
- Add comments to proto files: `// DEPRECATED: Not currently populated`
- Update schema documentation

### Solution 3: Phased Rollout (Pragmatic)
**Pros**: Delivers value incrementally, manageable scope
**Cons**: Partial implementation for period of time
**Effort**: Medium (1-2 days per exchange)
**Risk**: Low

**Implementation**:
- Phase 1: Implement for Binance only (highest volume)
- Phase 2: Add OKX, Coinbase
- Phase 3: Add remaining exchanges as data mapping is documented
- Document which exchanges support which fields

## Recommended Action

**SOLUTION 1 (Full Field Population)** - This is the only acceptable solution for a production system.

**Rationale**:
- Schema contract must be honored
- 31% of schema fields being permanently empty is unacceptable
- Downstream consumers depend on this metadata for analytics

**Immediate Next Steps**:
1. Block PR #16 merge until this is resolved
2. Create spike ticket to map all exchange raw data fields to new schema fields
3. Implement for Binance (reference exchange) with full test coverage
4. Document field availability per exchange

## Technical Details

**Affected Files**:
- `cryptofeed/types.pyx` - Trade/OrderBook class definitions
- `cryptofeed/exchanges/binance.py` - Data extraction from WebSocket messages
- `cryptofeed/exchanges/okx.py` - Same
- `cryptofeed/backends/protobuf/converters.py` - Protobuf conversion
- `proto/cryptofeed/normalized/v2beta1/trade.proto` - Schema definition
- `proto/cryptofeed/normalized/v2beta1/order_book.proto` - Schema definition

**Database Changes**: None (protobuf schema already supports these fields)

**API Changes**: Additive only (new fields are optional)

## Acceptance Criteria

- [ ] Trade class has attributes: `maker`, `event_time`, `match_id`, `liquidity_flag`
- [ ] OrderBook class has attributes: `event_time`, `last_update_id`
- [ ] Binance integration extracts all available fields from raw messages
- [ ] `trade_to_proto()` populates all new fields when data is available
- [ ] `orderbook_to_proto()` populates all new fields when data is available
- [ ] Unit tests verify field population for each data type
- [ ] Integration tests confirm end-to-end field transmission via Kafka
- [ ] Documentation updated with field availability per exchange
- [ ] No silent data loss (all extracted fields are transmitted)

## Work Log

**2025-12-14**: Issue identified during PR #16 code review by data-integrity-guardian agent
- Severity: CRITICAL (P1)
- Status: Pending triage and assignment
- Recommendation: BLOCK MERGE until resolved

## Resources

- PR #16: https://github.com/tommy-ca/cryptofeed/pull/16
- Protobuf v2beta1 schema: `proto/cryptofeed/normalized/v2beta1/trade.proto`
- Converter implementation: `cryptofeed/backends/protobuf/converters.py:24-58`
- Binance integration: `cryptofeed/exchanges/binance.py:255-267`
- Data Integrity Guardian review: See agent output (ac60f94)
- Schema mapping docs: `docs/schemas/mappings/trade_mapping.md`
