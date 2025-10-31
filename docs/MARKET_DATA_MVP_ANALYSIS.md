# Cryptofeed Market Data Only - MVP Scope Analysis

**Date**: October 30, 2025
**Branch**: feature/normalized-data-schema-crypto
**Status**: Complete baseline schema ready (v0.1.0)

---

## 1. Market Data Types in Cryptofeed (cryptofeed/types.pyx)

### Pure Market Data (MVP Candidates)

| Type | Class | Fields | Channels | Status | Notes |
|------|-------|--------|----------|--------|-------|
| **Trade** | Trade | exchange, symbol, side, amount, price, timestamp, id, type, raw | TRADES | ✅ Core | Individual market trades (NOT user fills) |
| **Ticker** | Ticker | exchange, symbol, bid, ask, timestamp, raw | TICKER | ✅ Core | Top-of-book (bid/ask only) |
| **OrderBook** | OrderBook | exchange, symbol, bids/asks, delta, sequence_number, checksum, timestamp | L2_BOOK, L3_BOOK | ✅ Core | Market depth snapshots & deltas |
| **Candle** | Candle | exchange, symbol, start, stop, interval, trades, open, close, high, low, volume, closed, timestamp | CANDLES | ✅ Core | OHLCV bars (aggregated market data) |
| **Funding** | Funding | exchange, symbol, mark_price, rate, next_funding_time, predicted_rate, timestamp | FUNDING | ✅ Derivative | Perpetual swap funding rates (market-wide) |
| **Liquidation** | Liquidation | exchange, symbol, side, quantity, price, id, status, timestamp | LIQUIDATIONS | ✅ Derivative | Market liquidations (NOT user-specific) |
| **OpenInterest** | OpenInterest | exchange, symbol, open_interest, timestamp | OPEN_INTEREST | ✅ Derivative | Total open contracts per symbol |
| **Index** | Index | exchange, symbol, price, timestamp, raw | INDEX | ⏸️ Optional | Index price (computed metric) |
| **L1Book** | L1Book | exchange, symbol, bid_price, bid_size, ask_price, ask_size, timestamp | L1_BOOK | ✅ Alternative | Single best bid/ask level |

### User/Account Data (EXCLUDE from MVP)

| Type | Class | Use Case | Status | Reason |
|------|-------|----------|--------|--------|
| **Balance** | Balance | User wallet contents | ❌ Account | User-specific, requires auth |
| **Fill** | Fill | User trade fills | ❌ Account | User-specific execution, requires auth |
| **Order** | Order | User order submission | ❌ Account | User-specific, requires auth |
| **OrderInfo** | OrderInfo | User order status | ❌ Account | User-specific, requires auth |
| **Position** | Position | User open positions | ❌ Account | User-specific, requires auth |
| **Transaction** | Transaction | User deposits/withdrawals | ❌ Account | User-specific, requires auth |

---

## 2. Core Market Data Channels (cryptofeed/defines.py)

### Market Data Channels (Lines 56-67)

```python
# Market Data
L1_BOOK = 'l1_book'           # Single best bid/ask
L2_BOOK = 'l2_book'           # Aggregated order book (MOST COMMON)
L3_BOOK = 'l3_book'           # Individual order-level book (rare)
TRADES = 'trades'             # Individual trade executions (MOST COMMON)
TICKER = 'ticker'             # Top-of-book (bid/ask) snapshots
FUNDING = 'funding'            # Perpetual funding rates
OPEN_INTEREST = 'open_interest' # Total open contracts
LIQUIDATIONS = 'liquidations'   # Market liquidations
INDEX = 'index'                 # Index price (metadata)
CANDLES = 'candles'             # OHLCV bars
```

### Account Data Channels (Lines 69-80) - EXCLUDE

```python
# Account Data / Authenticated Channels
ORDER_INFO = 'order_info'
FILLS = 'fills'
TRANSACTIONS = 'transactions'
BALANCES = 'balances'
POSITIONS = 'positions'
PLACE_ORDER = 'place_order'
CANCEL_ORDER = 'cancel_order'
ORDERS = 'orders'
ORDER_STATUS = 'order_status'
TRADE_HISTORY = 'trade_history'
```

### MVP Channel Priority Matrix

| Channel | Priority | Reason | Exchange Examples |
|---------|----------|--------|-------------------|
| **TRADES** | 🔴 MUST | Universal, minimal latency, all exchanges | Binance, Coinbase, Kraken, Backpack |
| **L2_BOOK** | 🔴 MUST | Core liquidity analysis, order book data | Binance, Coinbase, Kraken |
| **CANDLES** | 🟠 SHOULD | Analytics foundation (OHLCV bars) | Binance, Kraken, Backpack |
| **TICKER** | 🟠 SHOULD | Lightweight bid/ask reference | Coinbase, Binance, Kraken |
| **FUNDING** | 🟡 NICE | Derivatives only (Binance Futures, Bybit) | Binance Futures, Bybit, OKX |
| **LIQUIDATIONS** | 🟡 NICE | Risk monitoring (derivatives) | Binance Futures, Bybit, Deribit |
| **OPEN_INTEREST** | 🟡 NICE | Derivatives analytics | Binance Futures, Bybit |
| L1_BOOK | ⚪ ALT | Lightweight, ticker-like | Backpack, some exchanges |
| L3_BOOK | ⚪ RARE | Individual order tracking | Limited exchanges (Gemini) |
| INDEX | ⚪ SKIP | Computed/derived data | Skip for v0.1.0 |

---

## 3. Market Data Normalization (cryptofeed/exchange.py, symbols.py)

### Timestamp Normalization

**Location**: `cryptofeed/exchange.py:86-125` - `Exchange.timestamp_normalize()`

Normalizes to **UTC seconds as float**:

```python
# Accepts:
- datetime objects       → UTC timestamp
- int/float epoch        → seconds or milliseconds (>=1e12 = ms)
- ISO-8601 strings       → "2025-10-30T12:34:56Z" or "2025-10-30T12:34:56.123Z"
- Numeric strings        → epoch seconds/ms

# Returns:
float  # Always UTC seconds since epoch (e.g., 1730000000.123)
```

### Symbol Normalization

**Location**: `cryptofeed/symbols.py:13-107` - `Symbol` class

**Format**: `BASE-QUOTE` (e.g., `BTC-USD`, `ETH-USDT`, `SOL-USDC`)

**Canonical Patterns**:
```python
SPOT:       "BTC-USD"              # base-quote
PERPETUAL:  "BTC-USD-PERP"         # base-quote-PERP
FUTURES:    "BTC-USD-25Z"          # base-quote-expiry (YYMA)
OPTION:     "BTC-USD-25000-25Z-CALL"  # base-quote-strike-expiry-type
CURRENCY:   "BTC"                  # single symbol (no hyphen)
FX:         "EUR-USD-FX"           # base-quote-FX
```

**Normalization Rules**:
- Separator: `-` (consistent across all exchanges)
- Case: Uppercase (BTC, USD, USDT)
- Precision: Preserved as Decimal (no float conversion)
- Exchange-specific symbols converted via `Symbols.get(exchange)`

### Existing Normalization Utilities

| Function | Location | Purpose |
|----------|----------|---------|
| `Symbol.normalized` | symbols.py:89-107 | Generate canonical symbol string |
| `Symbol.timestamp_normalize()` | exchange.py:86-125 | Convert timestamps to UTC seconds |
| `Symbols.set()` | symbols.py:123-126 | Register exchange→normalized mapping |
| `Symbols.get()` | symbols.py:128-129 | Retrieve bidirectional symbol maps |
| `Symbols.find()` | symbols.py:134-142 | Look up normalized symbol across exchanges |

---

## 4. Protobuf Schema Scope for Market Data (proto/cryptofeed/normalized/v1/)

### Complete Protobuf Inventory

#### Market Data Schemas (MVP Ready)

| Schema | File | Status | Fields | Purpose |
|--------|------|--------|--------|---------|
| **Trade** | trade.proto | ✅ READY | exchange, symbol, side, trade_id, price, amount, timestamp | Individual market trades |
| **Level2Book** | order_book.proto | ✅ READY | exchange, symbol, bids[], asks[], timestamp, sequence, checksum | Order book snapshots |
| **Level2Delta** | level2_delta.proto | ✅ READY | bid_deltas[], ask_deltas[], sequence, checksum | Incremental updates |
| **Candle** | candle.proto | ✅ READY | exchange, symbol, start, end, interval, open/close/high/low/volume, trades, closed, timestamp | OHLCV bars |
| **Ticker** | ticker.proto | ✅ READY | exchange, symbol, bid, ask, timestamp | Top-of-book |
| **Funding** | funding.proto | ✅ READY | exchange, symbol, mark_price, rate, predicted_rate, next_funding_time, timestamp | Perpetual rates |
| **Liquidation** | liquidation.proto | ✅ READY | exchange, symbol, side, quantity, price, liquidation_id, status, timestamp | Market liquidations |
| **OpenInterest** | open_interest.proto | ✅ READY | exchange, symbol, open_interest, timestamp | Total open contracts |
| **TopOfBook** | top_of_book.proto | ✅ READY | exchange, symbol, bid_price, bid_size, ask_price, ask_size, timestamp | L1 data (alternative to Ticker) |
| **IndexPrice** | index_price.proto | ✅ READY | exchange, symbol, price, timestamp | Index price (derived) |
| **Nbbo** | nbbo.proto | ✅ READY | symbol, best_bid_exchange, best_bid_price, best_bid_size, best_ask_exchange, best_ask_price, best_ask_size, timestamp | Cross-exchange best |

#### Supporting Schemas (Required)

| Schema | File | Purpose |
|--------|------|---------|
| **TradeSide** | trade_side.proto | Enum: BUY, SELL |
| **PriceLevel** | price_level.proto | Reusable price/size pair |
| **Events** | events.proto | Event wrapper (timestamp, type, data) |

#### User Data Schemas (EXCLUDE from MVP)

| Schema | File | Reason |
|--------|------|--------|
| Balance | balance.proto | ❌ Account-specific |
| Fill | fill.proto | ❌ Account-specific |
| Order | order.proto | ❌ Account-specific |
| OrderInfo | order_info.proto | ❌ Account-specific |
| Position | position.proto | ❌ Account-specific |
| Transaction | transaction.proto | ❌ Account-specific |

### Decimal Encoding

**All numeric values use string encoding** (fixed-point scale):
- **Default scale**: 1e-8 (8 decimal places)
- **Exchange-specific overrides**: Noted per venue
- **Raw precision preserved**: Some fields use native precision (see schema comments)

**Example**:
```protobuf
message Trade {
  string price = 5;    // "45123.50000000" (scale 1e-8)
  string amount = 6;   // "0.12345678"
}
```

**Conversion**:
```python
from decimal import Decimal
price = Decimal("45123.50000000")  # Always use Decimal
amount = Decimal("0.12345678")
```

---

## 5. Exchange Support Assessment

### Complete Market Data Coverage (Spot)

| Exchange | TRADES | L2_BOOK | TICKER | CANDLES | Notes |
|----------|--------|---------|--------|---------|-------|
| **Coinbase** | ✅ | ✅ | ✅ | ❌ | Excellent L2/trade coverage |
| **Binance** | ✅ | ✅ | ✅ | ✅ | Full suite + futures |
| **Kraken** | ✅ | ✅ | ✅ | ✅ | Robust WebSocket |
| **Backpack** | ✅ | ✅ | ✅ | ❌ | Native Cryptofeed feed |
| **Bitstamp** | ✅ | ✅ | ✅ | ✅ | Stable legacy exchange |
| **Gemini** | ✅ | ✅ | ✅ | ✅ | REST-only, L3 available |
| **KuCoin** | ✅ | ✅ | ✅ | ✅ | WebSocket + REST |
| **Gate.io** | ✅ | ✅ | ✅ | ✅ | Good coverage |
| **OKX** | ✅ | ✅ | ✅ | ✅ | Full suite |
| **Bybit** | ✅ | ✅ | ✅ | ✅ | Perpetuals focus |
| **Deribit** | ✅ | ✅ | ✅ | ✅ | Options focus |

### Derivatives Support (Funding, Liquidations, OpenInterest)

| Exchange | FUNDING | LIQUIDATIONS | OPEN_INTEREST | Type | Notes |
|----------|---------|--------------|---------------|------|-------|
| **Binance Futures** | ✅ | ✅ | ✅ | PERPETUAL | Full support |
| **Bybit** | ✅ | ✅ | ✅ | PERPETUAL | Full support |
| **Deribit** | ✅ | ✅ | ✅ | OPTION | Options-focused |
| **Binance Delivery** | ✅ | ✅ | ✅ | FUTURES | Quarterly futures |
| **OKX** | ✅ | ✅ | ✅ | PERPETUAL | Full suite |
| **Gate.io Futures** | ✅ | ✅ | ✅ | PERPETUAL | Full suite |
| **Kraken Futures** | ✅ | ✅ | ✅ | FUTURES | Quarterly futures |
| **Huobi DM/Swap** | ✅ | ✅ | ✅ | PERPETUAL | Legacy but supported |
| **Backpack** | ❌ | ❌ | ❌ | SPOT ONLY | Spot-only (no perpetuals) |

### Recommended MVP Exchange Priority

#### Tier 1: Start With (0-2 weeks)
- **Coinbase** - Simplest API, excellent data quality, no auth required for market data
- **Binance Spot** - Massive liquidity, wide symbol coverage, mature WebSocket

#### Tier 2: Add Next (2-4 weeks)
- **Kraken** - Reliable, good depth, European compliance
- **Backpack** - Native Cryptofeed support (new)

#### Tier 3: Expand (4+ weeks)
- **Bitstamp**, **Gemini**, **KuCoin** (spot parity)
- **Binance Futures** (if derivatives needed)

---

## 6. Product Type Scope

### MVP Scope Recommendation: SPOT ONLY

**Rationale**:
1. **Simplest schemas** - No mark price, funding rates, liquidations
2. **Most liquid** - Tighter spreads, higher volume
3. **Wider exchange coverage** - All exchanges support spot
4. **Lowest latency** - Spot trades execute faster than futures
5. **No leverage risk** - Easier for consumers to onboard

### Product Type Support

| Product | Channels | Complexity | MVP | Timeline |
|---------|----------|-----------|-----|----------|
| **SPOT** | TRADES, L2_BOOK, TICKER, CANDLES | Low | ✅ v0.1.0 | Now |
| **PERPETUAL** | + FUNDING, LIQUIDATIONS, OPEN_INTEREST | Medium | ⏳ v0.2.0 | 2-4 weeks |
| **FUTURES** | + FUNDING, LIQUIDATIONS, OPEN_INTEREST | Medium | ⏳ v0.3.0 | 2-4 weeks |
| **OPTION** | + INDEX, GREEKS | High | ❌ v1.0.0 | 6+ weeks |

### Symbol Recommendation for MVP

**Focus on major pairs only**:
```
BTC-USD, BTC-USDT
ETH-USD, ETH-USDT
SOL-USD, SOL-USDT
(1-3 pairs per exchange to validate)
```

**Why**: Reduces schema testing burden, validates across venues, easy to expand later.

---

## 7. Storage Schema Optimization for Market Data

### Recommended Partitioning Scheme

**Parquet/Iceberg Partitioning for SPOT market data**:

```
# By exchange × symbol × date (optimal for market queries)
s3://crypto-data-lake/market/
├── exchange=binance/
│   └── symbol=BTC-USD/
│       └── data_type=trades/
│           ├── date=2025-10-30/
│           │   └── hour=12/
│           │       └── minute=30/
│           │           └── data.parquet
│       └── data_type=l2_book/
│           ├── date=2025-10-30/
│           │   └── ...
│
├── exchange=coinbase/
│   └── symbol=ETH-USD/
│       └── data_type=candles/
│           ├── date=2025-10-30/
│           │   └── ...
```

### Column Organization (Parquet Schema)

**Trades Table**:
```
REQUIRED group trade {
  REQUIRED binary exchange (STRING)
  REQUIRED binary symbol (STRING)
  REQUIRED binary side (STRING)         // BUY/SELL
  REQUIRED int64 timestamp (TIMESTAMP_MICROS)
  REQUIRED binary price (STRING)        // Decimal scale 1e-8
  REQUIRED binary amount (STRING)       // Decimal scale 1e-8
  REQUIRED binary trade_id (STRING)
  OPTIONAL binary trade_type (STRING)
}
```

**L2 Book Table** (snapshots only):
```
REQUIRED group l2_book {
  REQUIRED binary exchange (STRING)
  REQUIRED binary symbol (STRING)
  REQUIRED int64 timestamp (TIMESTAMP_MICROS)
  OPTIONAL int64 sequence (INT64)
  OPTIONAL binary checksum (STRING)
  REPEATED group bids {
    REQUIRED binary price (STRING)
    REQUIRED binary size (STRING)
  }
  REPEATED group asks {
    REQUIRED binary price (STRING)
    REQUIRED binary size (STRING)
  }
}
```

**Candles Table** (OHLCV):
```
REQUIRED group candle {
  REQUIRED binary exchange (STRING)
  REQUIRED binary symbol (STRING)
  REQUIRED binary interval (STRING)     // "1m", "5m", "1h"
  REQUIRED int64 start (TIMESTAMP_MICROS)
  REQUIRED int64 end (TIMESTAMP_MICROS)
  REQUIRED binary open (STRING)
  REQUIRED binary close (STRING)
  REQUIRED binary high (STRING)
  REQUIRED binary low (STRING)
  REQUIRED binary volume (STRING)
  OPTIONAL int64 trades (INT64)
  REQUIRED boolean closed
}
```

### Key Queries (Optimize For)

| Query | Partition Keys | Example |
|-------|-----------------|---------|
| **OHLCV bars** | exchange, symbol, interval, date | `SELECT * FROM candles WHERE symbol='BTC-USD' AND date >= '2025-10-01' AND interval='1h'` |
| **Trade stream** | exchange, symbol, date, hour | `SELECT * FROM trades WHERE symbol='BTC-USD' AND date='2025-10-30' AND hour >= 12 ORDER BY timestamp` |
| **Liquidity snapshot** | exchange, symbol, date, hour | `SELECT bids, asks FROM l2_book WHERE symbol='BTC-USD' AND timestamp BETWEEN ts1 AND ts2` |
| **Cross-exchange spread** | exchange, symbol, timestamp | `SELECT * FROM ticker WHERE symbol='BTC-USD' AND timestamp >= now() - INTERVAL 1 DAY` |
| **Volume analysis** | symbol, date | `SELECT SUM(volume) FROM candles WHERE symbol IN ('BTC-USD', 'ETH-USD') AND date >= '2025-10-01'` |

### Retention Policy

| Data Type | Retention | Reason |
|-----------|-----------|--------|
| **TRADES** | 1 year | High cardinality, used for research/backtest |
| **L2_BOOK** | 30 days | Space-intensive, older books less valuable |
| **CANDLES** | 3 years | Low cardinality, essential for analysis |
| **TICKER** | 90 days | Updates frequently, older snapshots rarely used |
| **FUNDING** | 1 year | Derivative analytics |

---

## 8. MVP Scope Summary

### Top 5 Market Data Types (Priority Order)

1. **Trade** - Core market data, universal across exchanges
   - Proto: trade.proto
   - Class: Trade
   - Priority: 🔴 MUST HAVE
   
2. **OrderBook (L2)** - Core liquidity data, essential for depth analysis
   - Proto: order_book.proto + level2_delta.proto
   - Class: OrderBook
   - Priority: 🔴 MUST HAVE
   
3. **Candle (OHLCV)** - Analytics foundation, pre-aggregated market data
   - Proto: candle.proto
   - Class: Candle
   - Priority: 🟠 SHOULD HAVE
   
4. **Ticker** - Lightweight bid/ask snapshot
   - Proto: ticker.proto
   - Class: Ticker
   - Priority: 🟠 SHOULD HAVE
   
5. **Funding** - Derivatives support (add if perpetuals included)
   - Proto: funding.proto
   - Class: Funding
   - Priority: 🟡 NICE TO HAVE (defer to v0.2.0)

### Top 5 Market Data Channels (Priority Order)

1. **TRADES** - Individual market trades (universal)
2. **L2_BOOK** - Aggregated order book (market depth)
3. **CANDLES** - OHLCV bars (analytics)
4. **TICKER** - Top-of-book snapshots (lightweight)
5. **FUNDING** - Perpetual funding rates (optional, derivatives only)

### Top 2 Recommended MVP Exchanges

1. **Coinbase** (Tier 1 - Start here)
   - ✅ TRADES, L2_BOOK, TICKER
   - Simple REST + WebSocket API
   - No authentication required for market data
   - Excellent data quality
   - Stable, mature platform
   - **Timeline**: Week 1

2. **Binance** (Tier 1 - Add immediately)
   - ✅ TRADES, L2_BOOK, TICKER, CANDLES
   - Massive liquidity & symbol coverage
   - Mature WebSocket implementation
   - Optional: Binance Futures (FUNDING, LIQUIDATIONS, OPEN_INTEREST)
   - **Timeline**: Week 1-2

**Phase 2 (weeks 2-4)**: Add Kraken + Backpack for geographic diversity

### Recommended Product Type

**SPOT ONLY** for v0.1.0

- Simplest schemas
- Widest exchange coverage
- Lowest complexity
- **Timeline**: Now (no delays)

**Perpetual futures** deferred to v0.2.0 (2-4 weeks later)

### Protobuf Schema Subset for MVP v0.1.0

**MUST INCLUDE** (5 files):
1. trade.proto - Individual trades
2. order_book.proto - Order book snapshots
3. level2_delta.proto - Incremental book updates
4. candle.proto - OHLCV bars
5. ticker.proto - Top-of-book

**SUPPORTING** (2 files):
1. trade_side.proto - Buy/sell enum
2. price_level.proto - Reusable price/size

**EXCLUDE** (all user data):
- balance.proto
- fill.proto
- order.proto
- order_info.proto
- position.proto
- transaction.proto

**DEFER to v0.2.0+**:
- funding.proto
- liquidation.proto
- open_interest.proto
- index_price.proto
- nbbo.proto (cross-exchange)

### Core Normalization Requirements

1. **Timestamp**: UTC seconds as `float` (via `Exchange.timestamp_normalize()`)
2. **Symbols**: Canonical `BASE-QUOTE` format (via `Symbol.normalized`)
3. **Decimals**: Always `Decimal` type (no floats for prices/amounts)
4. **Precision**: Default 1e-8 scale (exchange-specific overrides documented)
5. **Deltas**: Sequence numbers for book integrity (checksum validation)

### Typical Market Data Queries

**On-chain analytics & research**:
```sql
-- 1. OHLCV aggregation
SELECT timestamp, open, high, low, close, volume
FROM candles
WHERE symbol = 'BTC-USD' AND interval = '1h'
ORDER BY timestamp DESC LIMIT 100;

-- 2. Liquidity depth (bid/ask spread)
SELECT timestamp, bids[0].price, asks[0].price,
       (asks[0].price - bids[0].price) / bids[0].price as spread_bps
FROM l2_book
WHERE symbol = 'ETH-USD'
ORDER BY timestamp DESC LIMIT 1000;

-- 3. Volume-weighted price (VWAP)
SELECT SUM(price * amount) / SUM(amount) as vwap
FROM trades
WHERE symbol = 'BTC-USD'
  AND timestamp >= now() - INTERVAL 1 HOUR;

-- 4. Trade frequency (market activity)
SELECT COUNT(*) as trade_count, MIN(timestamp), MAX(timestamp)
FROM trades
WHERE symbol = 'SOL-USD'
  AND date = CURRENT_DATE()
GROUP BY symbol;

-- 5. Cross-exchange arbitrage
SELECT cb.timestamp, cb.bid, bn.ask,
       (bn.ask - cb.bid) / cb.bid as arb_bps
FROM ticker cb, ticker bn
WHERE cb.exchange = 'coinbase'
  AND bn.exchange = 'binance'
  AND cb.symbol = 'BTC-USD'
  AND bn.symbol = 'BTC-USDT'
  AND cb.timestamp = bn.timestamp;
```

---

## 9. Implementation Status & Next Steps

### Current State (Oct 30, 2025)

**✅ COMPLETED**:
- Protobuf schemas for all market data types (market data + user data)
- Cryptofeed types.pyx definitions for all data classes
- Symbol/timestamp normalization utilities
- Proto publishing infrastructure (buf.yaml, buf.gen.yaml)
- Phase 1 tests: 46/46 passing
- Phase 3 governance: 42/42 passing
- Total: 119/119 tests passing

**⏳ BLOCKED**:
- Phase 2 (external schema alignment) - awaiting tardis-node + DBN schemas

**📋 RECOMMENDED NEXT**:
1. Merge `feature/normalized-data-schema-crypto` to main
2. Publish v0.1.0 to Buf registry: `bash tools/buf_publish.sh v0.1.0`
3. Implement lakehouse MVP:
   - Consumer code for trade.proto + order_book.proto + candle.proto
   - Parquet writer with partitioning
   - SQL query layer (DuckDB)
4. Add Coinbase connector (TRADES + L2_BOOK)
5. Add Binance connector (TRADES + L2_BOOK + CANDLES)
6. Validate against proto schemas in v0.2.0 (after merger)

### Architecture for Protobuf Integration

```
Cryptofeed Data Flow:
┌─────────────────┐
│  Exchange API   │ (Binance WebSocket, Coinbase REST)
└────────┬────────┘
         │
    ┌────▼────────────────────────────┐
    │  cryptofeed/types.py             │ (Trade, OrderBook, Candle, etc.)
    │  ─ Cython optimized              │
    │  ─ Decimal precision             │
    │  ─ to_dict() serialization       │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │  Proto Mappers (NEW)               │ (custom module)
    │  ─ Trade → trade.proto             │
    │  ─ OrderBook → order_book.proto    │
    │  ─ Candle → candle.proto           │
    └────┬────────────────────────────────┘
         │
    ┌────▼─────────────────────────────┐
    │  Parquet Writer (NEW)              │ (DuckDB or Polars)
    │  ─ Protobuf serialization          │
    │  ─ Iceberg partitioning            │
    │  ─ S3 or local storage             │
    └────┬──────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │  Query Layer (NEW)                │ (DuckDB SQL)
    │  ─ Time-series queries             │
    │  ─ Liquidity analysis              │
    │  ─ OHLCV aggregation               │
    └────────────────────────────────────┘
```

---

## Summary Table: Market Data MVP Scope

| Component | Recommendation | Status | Timeline |
|-----------|----------------|--------|----------|
| **Data Types** | Trade, OrderBook, Candle, Ticker | ✅ Ready | Merge now |
| **Channels** | TRADES, L2_BOOK, CANDLES, TICKER | ✅ Ready | Merge now |
| **Exchanges** | Coinbase (tier 1), Binance (tier 1) | ✅ Supported | Implement week 1-2 |
| **Product Type** | SPOT ONLY | ✅ Scope set | v0.1.0 now |
| **Protobuf** | 7 core schemas (trade, order_book, candle, etc.) | ✅ Ready | Merge now |
| **Storage** | Parquet + Iceberg partitioning | 📋 Design | Implement week 2-3 |
| **Queries** | OHLCV, liquidity, VWAP, cross-exchange | 📋 Design | Implement week 3-4 |

---

**Document Version**: 1.0
**Last Updated**: October 30, 2025
**Prepared For**: Lakehouse backend adapter specification (normalized-data-schema-crypto v0.1.0)
