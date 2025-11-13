# Cryptofeed Schema v0.1.0 Release

**Release Date**: October 20, 2025  
**Version**: v0.1.0  
**Status**: Baseline Cryptofeed Schemas (Production)

## Overview

v0.1.0 represents the **baseline release** of the normalized-data-schema-crypto initiative. This release contains canonical Protobuf schemas derived from Cryptofeed dataclasses, published to the Buf Schema Registry (BSR) for consumption by downstream services.

### Scope

**In This Release**:
- ✅ Cryptofeed dataclass schemas (Trade, Ticker, OrderBook, Funding, NBBO, etc.)
- ✅ Canonical Protobuf definitions via Buf
- ✅ Python and JSON Schema code generation
- ✅ Comprehensive test coverage and validation

**Planned for Later Releases**:
- 🔄 tardis-node JSON schema alignment (v0.2.0)
- 🔄 DBN fixed layout alignment (v1.0.0)
- 🔄 Governance and monitoring infrastructure (v1.x+)

## Module Location

**Production BSR Namespace**:
```
buf.build/tommyk/crypto-market-data
```

**Version Tag**:
```
buf.build/tommyk/crypto-market-data:v0.1.0
```

## Included Schemas

### Market Data Events
- **Trade**: Individual trade execution (side, amount, price, timestamp, ID)
- **Ticker**: NBBO ticker quote (bid, ask, timestamp)
- **OrderBook**: Level 2/3 order book snapshot
- **L2 Delta**: Order book depth changes
- **NBBO**: National best bid/offer aggregation

### Funding & Derivatives
- **Funding**: Perpetual swap funding rates and rates
- **OpenInterest**: Aggregate open interest by symbol
- **Liquidation**: Liquidation event details
- **Candle**: OHLCV candlestick data

### Account Data
- **Order**: Active order state
- **Position**: Perpetual position tracking
- **Fill**: Trade execution fill details
- **Balance**: Account balance snapshot

### Utilities
- **TradeSide**: Enum for buy/sell side
- **PriceLevel**: Shared price level structure

## Migration Guide

### For Python Consumers

**Step 1: Install Buf module**
```bash
# Add to pyproject.toml or requirements
buf:buf.build/tommyk/crypto-market-data:v0.1.0
```

**Step 2: Generate Python bindings**
```bash
buf generate buf.build/tommyk/crypto-market-data
```

**Step 3: Import and use**
```python
from cryptofeed.normalized.v1 import trade_pb2

# Create a trade message
trade = trade_pb2.Trade()
trade.exchange = "BINANCE"
trade.symbol = "BTC-USDT"
trade.side = trade_pb2.SIDE_BUY
trade.amount = "1.5"
trade.price = "45000.50"
trade.timestamp = int(time.time() * 1_000_000)  # microseconds

# Serialize
serialized = trade.SerializeToString()

# Deserialize
trade2 = trade_pb2.Trade()
trade2.ParseFromString(serialized)
```

### For Go Consumers

**Step 1: Add module dependency**
```bash
go get buf.build/gen/go/tommyk/crypto-market-data/protobuf/go
```

**Step 2: Import and use**
```go
import "cryptofeed/normalized/v1/trade.pb.go"

trade := &cryptofeedv1.Trade{
    Exchange:  "BINANCE",
    Symbol:    "BTC-USDT",
    Side:      cryptofeedv1.TradeSide_SIDE_BUY,
    Amount:    "1.5",
    Price:     "45000.50",
    Timestamp: time.Now().UnixMicro(),
}
```

### For JSON Schema Consumers

**Step 1: Download schema**
```bash
buf export buf.build/tommyk/crypto-market-data:v0.1.0 --output schemas/
```

**Step 2: Use with JSON validation**
```python
import jsonschema
from pathlib import Path

schema = json.loads(Path("schemas/Trade.json").read_text())
trade_data = {
    "exchange": "BINANCE",
    "symbol": "BTC-USDT",
    "side": "BUY",
    "amount": "1.5",
    "price": "45000.50",
    "timestamp": "1729432800000000"
}

jsonschema.validate(trade_data, schema)
```

## Schema Coverage

### Field Precision

**Decimal Fields** (Price, Amount, Rate):
- Represented as strings to preserve arbitrary precision
- Example: `"45000.123456789"`
- No implicit scaling or rounding

**Timestamp Fields**:
- Represented as int64 microseconds since epoch
- Deterministic alignment with tardis-node and DBN
- Example: `1729432800123456`

**Enum Fields** (Side):
- Standard enum values: `BUY`, `SELL`
- Reserved numbers for future extension
- Extensible without breaking existing messages

### Optional Fields

Some fields are optional (oneof, optional keywords):
- API ID fields (not always available)
- Next funding times (may be None)
- Account-specific fields (position-only)

Check message definitions for `optional` markers.

## Breaking Changes

**None** - This is v0.1.0, the baseline release. All subsequent v0.x releases will maintain backward compatibility.

## Known Limitations

### Schema Coverage
1. **tardis-node Alignment**: Not yet aligned with tardis-node JSON schemas (planned for v0.2.0)
2. **DBN Alignment**: Not yet mapped to DBN fixed layouts (planned for v1.0.0)
3. **Governance**: Monitoring and governance infrastructure planned for v1.x+
4. **Account Data**: Limited to essential fields; portfolio data deferred

### Python-Proto Alignment (78%+ aligned)

For detailed alignment analysis, see `docs/schemas/PYTHON_PROTO_ALIGNMENT.md`.

#### 1. Raw Exchange Data Not Persisted
All Python types include a `raw` field containing the original exchange message. **Proto schemas do not include this field** to keep normalized data lean.

**Impact**: Cannot reconstruct original exchange messages from proto data.

**Workaround**: Store raw messages separately if needed for debugging or audit trails.

**Future**: May add `optional bytes raw_data` in v0.2.0 if demand warrants.

#### 2. OrderBook Delta Updates  
The `OrderBook` Python type includes a `delta` field for incremental updates. **Proto `Level2Book` only supports snapshots**.

**Impact**: Incremental orderbook updates must use the `Level2Delta` message instead.

**Workaround**: Use `level2_delta.proto` for delta updates, `order_book.proto` for snapshots.

**Status**: Working as designed - see proto comments for details.

#### 3. Timestamp Optionality
Python types `Order` and `OrderInfo` allow `timestamp=None`, but proto uses required `int64 timestamp`.

**Impact**: Cannot distinguish "no timestamp" from "epoch 0" (zero value).

**Workaround**: Use timestamp=0 to represent missing timestamps, document this convention.

**Future**: May change to `optional int64 timestamp` in v0.2.0 if acceptable as non-breaking enhancement.

#### 4. Recent Fixes (v0.1.0)
The following issues were identified and resolved during alignment review:
- ✅ **Trade.trade_type**: Added `optional string trade_type` field
- ✅ **Funding optionality**: Changed `mark_price` and `rate` to `optional`
- ✅ **OrderBook delta**: Documented delta limitation in proto comments

## Validation & Testing

All schemas have been:
- ✅ Validated with `buf lint` (STANDARD rules)
- ✅ Tested for field precision with regression tests
- ✅ Verified for code generation across Python, Go, JSON Schema
- ✅ Checked against Cryptofeed dataclass definitions

**Test Results**:
- Proto Integration Tests: 23 passed, 1 skipped
- Staging Publication Tests: 22 passed
- Regression Tests: All trade, ticker, and funding samples passed

## Support & Feedback

### Adoption Checklist

- [ ] Install buf CLI: `brew install bufbuild/buf/buf`
- [ ] Authenticate with BSR: `buf registry login`
- [ ] Add module to your project
- [ ] Generate bindings in your language
- [ ] Review migration guide for your stack
- [ ] Validate with sample data

### Getting Help

- **Documentation**: See `docs/schemas/migration.md` for detailed integration guides
- **Examples**: Check `examples/` for sample code
- **Issues**: Report via GitHub with `[Schema v0.1.0]` tag
- **Feedback**: Share feedback in engineering channels

## Release Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| **Protobuf Definitions** | `proto/cryptofeed/normalized/v1/` | Source `.proto` files |
| **Buf Module** | `buf.build/tommyk/crypto-market-data:v0.1.0` | Published module on BSR |
| **Tests** | `tests/proto_integration/test_production_release.py` | Validation suite |
| **Migration Guide** | `docs/schemas/migration.md` | Integration instructions |
| **Examples** | `docs/schemas/examples/` | Sample payloads and code |

## Future Roadmap

### v0.2.0 (Planned)
- tardis-node JSON schema alignment
- Field mapping documentation
- Historical data validation

### v1.0.0 (Planned)
- DBN fixed layout alignment
- Byte offset mapping to Protobuf fields
- Full canonical alignment

### v1.x+ (Planned)
- Governance and monitoring infrastructure
- BSR metrics and adoption tracking
- Consumer feedback loop and SLAs

---

**Release Notes**: See `docs/schemas/migration.md` for detailed migration instructions.

**Stability**: This is a production release. Breaking changes will only be introduced in major versions (v2.0.0+).

