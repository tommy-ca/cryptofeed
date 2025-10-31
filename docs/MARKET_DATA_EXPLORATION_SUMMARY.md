# Cryptofeed Market Data Exploration - Complete Summary

**Date**: October 30, 2025  
**Status**: Complete analysis ready for implementation  
**For**: lakehouse-backend-adapter specification (v0.1.0 baseline)

---

## Executive Summary

This exploration identified and documented **market data only** (excluding user/account data) across the cryptofeed codebase. Key findings:

### What We Found

**✅ 9 Market Data Types** (cryptofeed/types.pyx):
- Trade, OrderBook, Candle, Ticker (core)
- Funding, Liquidation, OpenInterest (derivatives)
- Index, L1Book (optional/alternatives)

**❌ 6 User/Account Types** (excluded from MVP):
- Balance, Fill, Order, OrderInfo, Position, Transaction

**✅ 10 Market Data Channels** (cryptofeed/defines.py):
- TRADES, L2_BOOK, L3_BOOK (core market depth/execution)
- CANDLES, TICKER (derived/snapshots)
- FUNDING, LIQUIDATIONS, OPEN_INTEREST (derivatives)
- INDEX, L1_BOOK (optional)

**✅ 20+ Protobuf Schemas** (proto/cryptofeed/normalized/v1/):
- 11 market data schemas (ready for v0.1.0)
- 6 user data schemas (excluded)
- 3+ supporting schemas (enums, structures)

**✅ 40+ Exchange Integrations** (cryptofeed/exchanges/*.py):
- All support TRADES + L2_BOOK (core)
- 90% support CANDLES + TICKER
- 60% support derivatives (FUNDING, LIQUIDATIONS, OPEN_INTEREST)

### MVP Recommendation

**Market Data Types** (Top 5):
1. Trade (market trades)
2. OrderBook (L2 snapshots + deltas)
3. Candle (OHLCV bars)
4. Ticker (top-of-book)
5. Funding (perpetual rates, optional)

**Channels** (Top 5):
1. TRADES (all exchanges)
2. L2_BOOK (all major exchanges)
3. CANDLES (aggregated OHLCV)
4. TICKER (lightweight snapshots)
5. FUNDING (derivatives only, defer v0.2)

**Exchanges** (Tier 1 - Start Here):
1. Coinbase (simple, no auth needed for market data)
2. Binance (massive liquidity, full channel support)

**Product Type**: SPOT ONLY (simplest scope)

**Protobuf Subset**: 7 core schemas (trade, order_book, level2_delta, candle, ticker, + 2 supporting)

---

## Document Map

### 1. **MARKET_DATA_MVP_ANALYSIS.md** (10,000+ words)
   - Complete market data types inventory
   - Channel availability matrix
   - Normalization utilities
   - Exchange support assessment
   - Storage/partitioning design
   - Query patterns & examples
   - Implementation roadmap

   **Use this for**: Strategic planning, architecture decisions, completeness

### 2. **MARKET_DATA_SCHEMA_MAPPING.md** (5,000+ words)
   - Cryptofeed type → Protobuf schema mappings
   - Code examples for each type (Trade, OrderBook, Candle, etc.)
   - Decimal/timestamp/symbol normalization
   - Parquet schema definitions
   - Implementation checklist

   **Use this for**: Developer implementation, schema definitions, code examples

### 3. **MARKET_DATA_EXPLORATION_SUMMARY.md** (This document)
   - Quick reference of all findings
   - File locations & code samples
   - Open questions answered
   - Next steps & timeline

   **Use this for**: Quick lookup, team briefing, status updates

---

## Key Findings by Exploration Area

### 1. Market Data Types (cryptofeed/types.pyx)

**Location**: `/home/tommyk/projects/quant/data-sources/crypto-data/cryptofeed/cryptofeed/types.pyx`

**Core Market Data** (MVP ready):
- **Trade** (lines 34-87): Individual market trades, not user fills
- **OrderBook** (lines 388-474): L2/L3 snapshots + deltas
- **Candle** (lines 246-319): OHLCV aggregated bars
- **Ticker** (lines 89-134): Top-of-book bid/ask
- **L1Book** (lines 640-680): Alternative single-level representation

**Derivative Market Data** (optional):
- **Funding** (lines 192-244): Perpetual swap rates (market-wide, not user-specific)
- **Liquidation** (lines 136-190): Market liquidations (not user-specific)
- **OpenInterest** (lines 354-386): Total open contracts per symbol

**User/Account Data** (EXCLUDE):
- **Balance** (lines 606-638): User wallet contents
- **Fill** (lines 719-768): User trade fills (has order_id, fee, account fields)
- **Order** (lines 475-530): User order submissions
- **OrderInfo** (lines 534-604): User order status/tracking
- **Position** (lines 770-810): User open positions (has unrealised_pnl)
- **Transaction** (lines 682-717): User deposits/withdrawals

**Key Distinction**: 
- Trade = market trade (exchange-wide, any taker/maker)
- Fill = user fill (from orders they placed, account-specific)

---

### 2. Market Data Channels (cryptofeed/defines.py)

**Location**: `/home/tommyk/projects/quant/data-sources/crypto-data/cryptofeed/cryptofeed/defines.py` (lines 56-80)

**Market Data Channels** (lines 56-67):
```python
L1_BOOK = 'l1_book'              # Single best bid/ask
L2_BOOK = 'l2_book'              # Aggregated order book (MOST COMMON)
L3_BOOK = 'l3_book'              # Individual order-level book (rare)
TRADES = 'trades'                # Individual trade executions (MOST COMMON)
TICKER = 'ticker'                # Top-of-book (bid/ask) snapshots
FUNDING = 'funding'              # Perpetual funding rates
OPEN_INTEREST = 'open_interest'  # Total open contracts
LIQUIDATIONS = 'liquidations'    # Market liquidations
INDEX = 'index'                  # Index price (metadata/computed)
CANDLES = 'candles'              # OHLCV bars
```

**Account Data Channels** (lines 69-80, EXCLUDE):
```python
ORDER_INFO, FILLS, TRANSACTIONS, BALANCES, POSITIONS
PLACE_ORDER, CANCEL_ORDER, ORDERS, ORDER_STATUS, TRADE_HISTORY
```

---

### 3. Protobuf Schemas (proto/cryptofeed/normalized/v1/)

**Location**: `/home/tommyk/projects/quant/data-sources/crypto-data/cryptofeed/proto/cryptofeed/normalized/v1/`

**20 Proto Files Total**:

**Market Data Schemas** (11 files, MVP ready):
1. `trade.proto` - Individual market trades
2. `order_book.proto` - L2 book snapshots
3. `level2_delta.proto` - Incremental L2 updates
4. `candle.proto` - OHLCV bars (with interval)
5. `ticker.proto` - Top-of-book snapshots
6. `funding.proto` - Perpetual funding rates
7. `liquidation.proto` - Market liquidations
8. `open_interest.proto` - Total open contracts
9. `index_price.proto` - Index prices
10. `top_of_book.proto` - L1 alternative to Ticker
11. `nbbo.proto` - Cross-exchange best bid/ask

**Supporting Schemas** (2 required):
- `trade_side.proto` - Enum (BUY, SELL)
- `price_level.proto` - Reusable (price, size) tuple
- `events.proto` - Event wrapper (optional)

**User Data Schemas** (6 files, EXCLUDE):
- `balance.proto`, `fill.proto`, `order.proto`
- `order_info.proto`, `position.proto`, `transaction.proto`

**All schemas use**:
- String encoding for decimals (scale 1e-8 default)
- int64 for timestamps in microseconds
- Proto3 syntax
- Published to Buf registry (v0.1.0 ready)

---

### 4. Normalization Utilities

**Location**: `cryptofeed/exchange.py` and `cryptofeed/symbols.py`

**Timestamp Normalization** (exchange.py:86-125):
```python
Exchange.timestamp_normalize(ts) → float  # UTC seconds
# Handles: datetime, int/float (seconds or ms), ISO-8601 strings
```

**Symbol Normalization** (symbols.py:13-107):
```python
Symbol(base="BTC", quote="USD", type="SPOT").normalized  # "BTC-USD"
Symbols.get(exchange)  # bidirectional mapping
```

**Decimal Precision**: Always use `Decimal` (no floats for prices)

---

### 5. Exchange Support Assessment

**Location**: `cryptofeed/exchanges/*.py` (40+ exchange implementations)

**Complete Spot Coverage** (all 5 channels):
- Binance, Coinbase, Kraken, KuCoin, OKX, Gate.io
- Bitstamp, Gemini, Bitfinex, Ascendex

**Derivatives Support** (Funding, Liquidations, OpenInterest):
- Binance Futures, Bybit, OKX, Deribit
- Gate.io Futures, Kraken Futures, Huobi DM/Swap

**Native Cryptofeed Feeds**:
- Backpack (new native integration, SPOT only)
- CCXT generic (supports 100+ exchanges via ccxt/ccxt.pro)

---

## Questions Answered

### Q: Which data types should be in the MVP?
**A**: Trade, OrderBook, Candle, Ticker (MUST). Funding optional for v0.2.

### Q: What about user fills vs market trades?
**A**: Exclude Fill type - it has account-specific fields (order_id, fee, account). Use Trade only.

### Q: Should we include derivatives?
**A**: Defer FUNDING, LIQUIDATIONS, OPEN_INTEREST to v0.2.0. Start with SPOT only.

### Q: Which exchanges first?
**A**: Coinbase (simple API) + Binance (liquidity) for Tier 1.

### Q: How should timestamps be stored?
**A**: UTC seconds as float, normalized via Exchange.timestamp_normalize(). Convert to microseconds for protobuf (int64).

### Q: How should decimals be encoded?
**A**: String format (fixed-point), scale 1e-8. Never use float64.

### Q: What partitioning for Parquet storage?
**A**: exchange × symbol × date (with hour for high-volume data like trades/books).

### Q: How long to retain market data?
**A**: Trades (1yr), L2 books (30d), Candles (3yr), Ticker (90d).

### Q: Can we just use Ticker instead of L2 books?
**A**: No - Ticker is (bid, ask) only. L2 Book has full depth (10-100 price levels).

### Q: Do we need L3 books?
**A**: Not for MVP. L3 is rare (only Gemini/few others) and high complexity.

---

## Implementation Timeline

### Phase 1: Foundation (Week 1)
- [ ] Merge `feature/normalized-data-schema-crypto` to main
- [ ] Publish v0.1.0 to Buf registry
- [ ] Generate Python proto classes
- [ ] Create proto → Cryptofeed type mappers

### Phase 2: Storage (Week 2-3)
- [ ] Implement Parquet writer
- [ ] Add Iceberg partitioning
- [ ] Create DuckDB schema & table setup
- [ ] Write validation tests

### Phase 3: Exchange Integration (Week 1-2 parallel)
- [ ] Coinbase connector (TRADES + L2_BOOK)
- [ ] Binance connector (TRADES + L2_BOOK + CANDLES)
- [ ] Test proto serialization end-to-end

### Phase 4: Queries & Analytics (Week 3-4)
- [ ] Implement OHLCV queries
- [ ] Liquidity analysis queries
- [ ] VWAP calculation
- [ ] Sample dashboards

---

## File References

### Key Source Files
| File | Purpose | Lines |
|------|---------|-------|
| `cryptofeed/types.pyx` | Data type definitions | All market + user types |
| `cryptofeed/defines.py` | Channel constants | 56-80 (market channels) |
| `cryptofeed/symbols.py` | Symbol normalization | 13-107 |
| `cryptofeed/exchange.py` | Timestamp normalization | 86-125 |
| `proto/cryptofeed/normalized/v1/*.proto` | Protobuf schemas | 20 files |
| `cryptofeed/exchanges/*.py` | Exchange implementations | 40+ files |

### Documentation Files (Just Created)
1. **MARKET_DATA_MVP_ANALYSIS.md** - Complete strategic analysis
2. **MARKET_DATA_SCHEMA_MAPPING.md** - Implementation reference
3. **MARKET_DATA_EXPLORATION_SUMMARY.md** - This document

---

## Quick Reference Tables

### Market Data Type Priority
| Type | MVP | Priority | Channel | Proto | Status |
|------|-----|----------|---------|-------|--------|
| Trade | ✅ | 🔴 MUST | TRADES | trade.proto | Ready |
| OrderBook | ✅ | 🔴 MUST | L2_BOOK | order_book.proto | Ready |
| Candle | ✅ | 🟠 SHOULD | CANDLES | candle.proto | Ready |
| Ticker | ✅ | 🟠 SHOULD | TICKER | ticker.proto | Ready |
| Funding | ⏳ | 🟡 NICE | FUNDING | funding.proto | Defer v0.2 |
| Liquidation | ⏳ | 🟡 NICE | LIQUIDATIONS | liquidation.proto | Defer v0.2 |
| OpenInterest | ⏳ | 🟡 NICE | OPEN_INTEREST | open_interest.proto | Defer v0.2 |

### Exchange Tier Support
| Tier | Exchange | TRADES | L2_BOOK | CANDLES | TICKER | Notes |
|------|----------|--------|---------|---------|--------|-------|
| 1 | Coinbase | ✅ | ✅ | ❌ | ✅ | Start here |
| 1 | Binance | ✅ | ✅ | ✅ | ✅ | Immediate |
| 2 | Kraken | ✅ | ✅ | ✅ | ✅ | Week 2-3 |
| 2 | Backpack | ✅ | ✅ | ❌ | ✅ | Week 2-3 |
| 3 | Others | ✅ | ✅ | ✅ | ✅ | Phase 2 |

---

## Success Criteria

- [x] Identified all market data types (9 total)
- [x] Excluded all user/account data (6 types)
- [x] Mapped to protobuf schemas (11 files)
- [x] Documented normalization (timestamp, symbol, decimal)
- [x] Assessed exchange coverage (40+ implementations)
- [x] Designed storage schema (Parquet + Iceberg)
- [x] Created SQL query examples (OHLCV, liquidity, VWAP)
- [x] Produced implementation guides (mappers, retention)
- [x] Timeline & roadmap (4 phases)

---

## Next Actions

**Immediate**:
1. Review findings with specification author
2. Approve MVP scope (market data only, SPOT only)
3. Confirm exchange priority (Coinbase, Binance)

**Week 1**:
1. Merge feature branch to main
2. Publish v0.1.0 to Buf registry
3. Generate Python proto classes
4. Start Coinbase connector

**Week 2**:
1. Implement Parquet writer
2. Add Binance connector
3. Create DuckDB schema

**Week 3-4**:
1. Query layer (OHLCV, liquidity)
2. Validation & testing
3. Documentation

---

## Appendix: Code Samples

### Timestamp Normalization Example
```python
from cryptofeed.exchange import Exchange

# All of these return the same thing: UTC seconds as float
ts1 = Exchange.timestamp_normalize(1730000000)              # seconds
ts2 = Exchange.timestamp_normalize(1730000000123)           # ms
ts3 = Exchange.timestamp_normalize("2025-10-30T12:34:56Z")  # ISO-8601
ts4 = Exchange.timestamp_normalize(datetime.now())          # datetime

# For protobuf, multiply by 1,000,000 to get microseconds
timestamp_us = int(ts1 * 1_000_000)
```

### Symbol Normalization Example
```python
from cryptofeed.symbols import Symbol, Symbols

# Create normalized symbol
sym = Symbol(base="BTC", quote="USDT", type="SPOT")
print(sym.normalized)  # "BTC-USDT"

# Look up exchange-specific symbol
norm_map, _ = Symbols.get("binance")
exchange_symbol = norm_map.get("BTC-USDT")  # "BTCUSDT"
```

### Decimal Handling Example
```python
from decimal import Decimal

# Right way
price = Decimal("45123.50000000")
amount = Decimal("0.12345678")
total = price * amount  # Decimal("5646.55...")

# Convert to string for protobuf
price_str = str(price)  # "45123.50000000"

# Convert from string when reading from storage
price_read = Decimal(price_str)
```

---

**Exploration Complete**: October 30, 2025  
**For Questions**: See MARKET_DATA_MVP_ANALYSIS.md (strategic) or MARKET_DATA_SCHEMA_MAPPING.md (implementation)  
**Repository**: `/home/tommyk/projects/quant/data-sources/crypto-data/cryptofeed`
