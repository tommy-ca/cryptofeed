# Market Data Schema Mapping - Cryptofeed Types to Protobuf

**Purpose**: Quick reference for mapping cryptofeed data types to protobuf schemas for lakehouse storage

**Version**: 1.0  
**Date**: October 30, 2025  
**Status**: Ready for v0.1.0 release

---

## 1. Core Data Type Mapping

### Trade (Market Trades Only)

**Cryptofeed Class**: `Trade` (cryptofeed/types.pyx:34-87)

```python
class Trade:
    exchange: str           # normalized exchange ID
    symbol: str             # normalized symbol (e.g., "BTC-USD")
    side: str              # "buy" or "sell"
    amount: Decimal        # trade quantity
    price: Decimal         # trade price
    timestamp: float       # UTC seconds (microsecond precision)
    id: str                # exchange trade ID
    type: str              # optional trade type
    raw: dict/list         # raw exchange payload
```

**Protobuf Schema**: `trade.proto`

```protobuf
message Trade {
  string exchange = 1;              // "binance", "coinbase"
  string symbol = 2;                // "BTC-USD" (normalized)
  TradeSide side = 3;               // BUY or SELL
  string trade_id = 4;              // venue-provided ID
  string price = 5;                 // decimal string, scale 1e-8
  string amount = 6;                // decimal string, scale 1e-8
  int64 timestamp = 7;              // microseconds since epoch
  optional string raw_id = 8;       // raw venue ID (for debugging)
  optional string trade_type = 9;   // "market", "limit", etc.
}
```

**Mapping Code Example**:

```python
from decimal import Decimal
from cryptofeed.types import Trade
from cryptofeed_proto.normalized.v1 import trade_pb2

def trade_to_proto(t: Trade) -> trade_pb2.Trade:
    """Convert Cryptofeed Trade to protobuf Trade."""
    return trade_pb2.Trade(
        exchange=t.exchange,
        symbol=t.symbol,
        side=trade_pb2.TradeSide.BUY if t.side == "buy" else trade_pb2.TradeSide.SELL,
        trade_id=t.id or "",
        price=str(t.price),           # Decimal → string
        amount=str(t.amount),         # Decimal → string
        timestamp=int(t.timestamp * 1_000_000),  # seconds → microseconds
        raw_id=t.id,
        trade_type=t.type or "",
    )
```

**Storage Partition Keys**: `exchange`, `symbol`, `date` (derived from timestamp)

**Retention**: 1 year (high cardinality, used for backtesting)

---

### OrderBook (L2 Book Snapshots & Deltas)

**Cryptofeed Class**: `OrderBook` (cryptofeed/types.pyx:388-474)

```python
class OrderBook:
    exchange: str                      # normalized exchange ID
    symbol: str                        # normalized symbol
    book: _OrderBook                   # underlying order book
    bids: dict[Decimal, Decimal]       # {price: size}
    asks: dict[Decimal, Decimal]       # {price: size}
    delta: dict                        # incremental changes
    timestamp: float                   # UTC seconds
    sequence_number: int               # exchange sequence (optional)
    checksum: str/int                  # integrity check (optional)
    raw: dict/list                     # raw payload
```

**Protobuf Schemas**: `order_book.proto` + `level2_delta.proto`

**Full Snapshot** (`order_book.proto`):

```protobuf
message PriceLevel {
  string price = 1;    // decimal string, scale 1e-8
  string size = 2;     // decimal string, scale 1e-8
}

message Level2Book {
  string exchange = 1;              // "binance"
  string symbol = 2;                // "BTC-USD"
  repeated PriceLevel bids = 3;     // sorted descending
  repeated PriceLevel asks = 4;     // sorted ascending
  optional int64 timestamp = 5;     // microseconds since epoch
  optional int64 sequence = 6;      // exchange sequence
  optional string checksum = 7;     // venue checksum
}
```

**Incremental Update** (`level2_delta.proto`):

```protobuf
message Level2Delta {
  string exchange = 1;              // "binance"
  string symbol = 2;                // "BTC-USD"
  repeated PriceLevel bid_deltas = 3;  // changes to bids
  repeated PriceLevel ask_deltas = 4;  // changes to asks
  optional int64 timestamp = 5;     // microseconds since epoch
  optional int64 sequence = 6;      // new sequence number
}
```

**Mapping Code Example**:

```python
from cryptofeed.types import OrderBook
from cryptofeed_proto.normalized.v1 import order_book_pb2, level2_delta_pb2

def orderbook_to_snapshot_proto(ob: OrderBook) -> order_book_pb2.Level2Book:
    """Convert full OrderBook snapshot to protobuf."""
    def price_level(price, size):
        return order_book_pb2.PriceLevel(
            price=str(price),
            size=str(size)
        )
    
    return order_book_pb2.Level2Book(
        exchange=ob.exchange,
        symbol=ob.symbol,
        bids=[price_level(p, s) for p, s in ob.bids.items()],
        asks=[price_level(p, s) for p, s in ob.asks.items()],
        timestamp=int(ob.timestamp * 1_000_000),
        sequence=ob.sequence_number,
        checksum=str(ob.checksum) if ob.checksum else None,
    )

def orderbook_delta_to_proto(ob: OrderBook) -> level2_delta_pb2.Level2Delta:
    """Convert OrderBook delta to protobuf."""
    if not ob.delta:
        return None
    
    def price_level(price, size):
        return level2_delta_pb2.PriceLevel(
            price=str(price),
            size=str(size)  # size=0 means remove
        )
    
    return level2_delta_pb2.Level2Delta(
        exchange=ob.exchange,
        symbol=ob.symbol,
        bid_deltas=[price_level(p, s) for p, s in ob.delta.get("bid", [])],
        ask_deltas=[price_level(p, s) for p, s in ob.delta.get("ask", [])],
        timestamp=int(ob.timestamp * 1_000_000),
        sequence=ob.sequence_number,
    )
```

**Storage Strategy**:
- Store snapshots every N seconds (e.g., 60s for 1h resolution)
- Store all deltas between snapshots for exact reconstruction
- Partition by: `exchange`, `symbol`, `date`, `hour`
- Retention: 30 days (space-intensive)

---

### Candle (OHLCV Bars)

**Cryptofeed Class**: `Candle` (cryptofeed/types.pyx:246-319)

```python
class Candle:
    exchange: str        # normalized exchange ID
    symbol: str          # normalized symbol
    start: float         # bar start (UTC seconds)
    stop: float          # bar end (UTC seconds)
    interval: str        # "1m", "5m", "1h", "1d"
    open: Decimal        # OHLC
    close: Decimal
    high: Decimal
    low: Decimal
    volume: Decimal      # trading volume
    trades: int          # optional trade count
    closed: bool         # true if bar is complete
    timestamp: float     # optional update timestamp
    raw: dict/list       # raw payload
```

**Protobuf Schema**: `candle.proto`

```protobuf
message Candle {
  string exchange = 1;           // "binance"
  string symbol = 2;             // "BTC-USD"
  int64 start = 3;               // bar start (microseconds)
  int64 end = 4;                 // bar end (microseconds)
  string interval = 5;           // "1m", "5m", "1h"
  optional int64 trades = 6;     // trade count
  string open = 7;               // decimal string, scale 1e-8
  string close = 8;              // decimal string, scale 1e-8
  string high = 9;               // decimal string, scale 1e-8
  string low = 10;               // decimal string, scale 1e-8
  string volume = 11;            // decimal string, scale 1e-8
  bool closed = 12;              // is bar final?
  optional int64 timestamp = 13; // update time (microseconds)
}
```

**Mapping Code Example**:

```python
from cryptofeed.types import Candle
from cryptofeed_proto.normalized.v1 import candle_pb2

def candle_to_proto(c: Candle) -> candle_pb2.Candle:
    """Convert Cryptofeed Candle to protobuf."""
    return candle_pb2.Candle(
        exchange=c.exchange,
        symbol=c.symbol,
        start=int(c.start * 1_000_000),
        end=int(c.stop * 1_000_000),
        interval=c.interval,
        trades=c.trades,
        open=str(c.open),
        close=str(c.close),
        high=str(c.high),
        low=str(c.low),
        volume=str(c.volume),
        closed=c.closed,
        timestamp=int(c.timestamp * 1_000_000) if c.timestamp else None,
    )
```

**Storage Partition Keys**: `exchange`, `symbol`, `interval`, `date`

**Retention**: 3 years (low cardinality, essential for long-term analysis)

---

### Ticker (Top-of-Book)

**Cryptofeed Class**: `Ticker` (cryptofeed/types.pyx:89-134)

```python
class Ticker:
    exchange: str        # normalized exchange ID
    symbol: str          # normalized symbol
    bid: Decimal         # best bid price
    ask: Decimal         # best ask price
    timestamp: float     # UTC seconds
    raw: dict/list       # raw payload
```

**Protobuf Schema**: `ticker.proto`

```protobuf
message Ticker {
  string exchange = 1;           // "binance"
  string symbol = 2;             // "BTC-USD"
  string bid = 3;                // decimal string, scale 1e-8
  string ask = 4;                // decimal string, scale 1e-8
  optional int64 timestamp = 5;  // microseconds since epoch
}
```

**Mapping Code Example**:

```python
from cryptofeed.types import Ticker
from cryptofeed_proto.normalized.v1 import ticker_pb2

def ticker_to_proto(t: Ticker) -> ticker_pb2.Ticker:
    """Convert Cryptofeed Ticker to protobuf."""
    return ticker_pb2.Ticker(
        exchange=t.exchange,
        symbol=t.symbol,
        bid=str(t.bid),
        ask=str(t.ask),
        timestamp=int(t.timestamp * 1_000_000) if t.timestamp else None,
    )
```

**Storage Partition Keys**: `exchange`, `symbol`, `date`

**Retention**: 90 days (lightweight, but frequent updates)

---

### Funding (Perpetual Funding Rates)

**Cryptofeed Class**: `Funding` (cryptofeed/types.pyx:192-244)

```python
class Funding:
    exchange: str               # normalized exchange ID
    symbol: str                 # perpetual symbol (e.g., "BTC-USD-PERP")
    mark_price: Decimal         # optional mark price
    rate: Decimal               # funding rate (per period)
    predicted_rate: Decimal     # optional predicted rate
    next_funding_time: float    # optional time of next funding
    timestamp: float            # UTC seconds
    raw: dict/list              # raw payload
```

**Protobuf Schema**: `funding.proto`

```protobuf
message Funding {
  string exchange = 1;                    // "binance"
  string symbol = 2;                      // "BTC-USD-PERP"
  optional string mark_price = 3;         // decimal string, scale 1e-8
  optional string rate = 4;               // decimal string, scale 1e-8
  optional string predicted_rate = 5;     // decimal string, scale 1e-8
  optional int64 next_funding_time = 6;   // microseconds
  int64 timestamp = 7;                    // microseconds since epoch
}
```

**Mapping Code Example**:

```python
from cryptofeed.types import Funding
from cryptofeed_proto.normalized.v1 import funding_pb2

def funding_to_proto(f: Funding) -> funding_pb2.Funding:
    """Convert Cryptofeed Funding to protobuf."""
    return funding_pb2.Funding(
        exchange=f.exchange,
        symbol=f.symbol,
        mark_price=str(f.mark_price) if f.mark_price else None,
        rate=str(f.rate) if f.rate else None,
        predicted_rate=str(f.predicted_rate) if f.predicted_rate else None,
        next_funding_time=int(f.next_funding_time * 1_000_000) if f.next_funding_time else None,
        timestamp=int(f.timestamp * 1_000_000),
    )
```

**Storage Partition Keys**: `exchange`, `symbol`, `date`

**Retention**: 1 year (derivative analytics)

---

### Liquidation (Market Liquidations)

**Cryptofeed Class**: `Liquidation` (cryptofeed/types.pyx:136-190)

```python
class Liquidation:
    exchange: str        # normalized exchange ID
    symbol: str          # perpetual symbol
    side: str            # liquidation direction
    quantity: Decimal    # liquidated amount
    price: Decimal       # liquidation price
    id: str              # liquidation ID
    status: str          # optional status
    timestamp: float     # UTC seconds
    raw: dict/list       # raw payload
```

**Protobuf Schema**: `liquidation.proto`

```protobuf
message Liquidation {
  string exchange = 1;                    // "binance"
  string symbol = 2;                      // "BTC-USD-PERP"
  TradeSide side = 3;                     // BUY or SELL
  string quantity = 4;                    // decimal string, scale 1e-8
  string price = 5;                       // decimal string, scale 1e-8
  optional string liquidation_id = 6;     // venue liquidation ID
  optional string status = 7;             // status (optional)
  optional int64 timestamp = 8;           // microseconds since epoch
}
```

**Mapping Code Example**:

```python
from cryptofeed.types import Liquidation
from cryptofeed_proto.normalized.v1 import liquidation_pb2

def liquidation_to_proto(l: Liquidation) -> liquidation_pb2.Liquidation:
    """Convert Cryptofeed Liquidation to protobuf."""
    return liquidation_pb2.Liquidation(
        exchange=l.exchange,
        symbol=l.symbol,
        side=liquidation_pb2.TradeSide.BUY if l.side == "buy" else liquidation_pb2.TradeSide.SELL,
        quantity=str(l.quantity),
        price=str(l.price),
        liquidation_id=l.id or "",
        status=l.status or "",
        timestamp=int(l.timestamp * 1_000_000) if l.timestamp else None,
    )
```

**Storage Partition Keys**: `exchange`, `symbol`, `date`

**Retention**: 1 year (risk monitoring)

---

### OpenInterest (Total Open Contracts)

**Cryptofeed Class**: `OpenInterest` (cryptofeed/types.pyx:354-386)

```python
class OpenInterest:
    exchange: str        # normalized exchange ID
    symbol: str          # perpetual symbol
    open_interest: Decimal  # total open contracts
    timestamp: float     # UTC seconds
    raw: dict/list       # raw payload
```

**Protobuf Schema**: `open_interest.proto`

```protobuf
message OpenInterest {
  string exchange = 1;           // "binance"
  string symbol = 2;             // "BTC-USD-PERP"
  string open_interest = 3;      // decimal string, scale 1e-8
  optional int64 timestamp = 4;  // microseconds since epoch
}
```

**Mapping Code Example**:

```python
from cryptofeed.types import OpenInterest
from cryptofeed_proto.normalized.v1 import open_interest_pb2

def openinterest_to_proto(oi: OpenInterest) -> open_interest_pb2.OpenInterest:
    """Convert Cryptofeed OpenInterest to protobuf."""
    return open_interest_pb2.OpenInterest(
        exchange=oi.exchange,
        symbol=oi.symbol,
        open_interest=str(oi.open_interest),
        timestamp=int(oi.timestamp * 1_000_000) if oi.timestamp else None,
    )
```

**Storage Partition Keys**: `exchange`, `symbol`, `date`

**Retention**: 1 year (derivatives analytics)

---

## 2. Normalization Reference

### Symbol Normalization

**Input Examples** (exchange-specific):
- Binance: `BTCUSDT` → Normalized: `BTC-USDT`
- Coinbase: `BTC-USD` → Normalized: `BTC-USD`
- Kraken: `XXBTZUSD` → Normalized: `BTC-USD`
- Backpack: `BTC_USDC` → Normalized: `BTC-USDC`

**Normalized Format**: `BASE-QUOTE` (e.g., `BTC-USDT`)

**Mapping Function**:

```python
from cryptofeed.symbols import Symbol

# Create normalized symbol
sym = Symbol(base="BTC", quote="USDT", type="SPOT")
normalized = sym.normalized  # "BTC-USDT"

# Reverse lookup (exchange-specific)
norm_map, _ = Symbols.get("binance")
exchange_symbol = norm_map.get(normalized)  # "BTCUSDT"
```

### Timestamp Normalization

**Input Examples**:
- Binance WebSocket: milliseconds (e.g., `1730000000123`)
- Coinbase REST: ISO-8601 string (e.g., `"2025-10-30T12:34:56Z"`)
- Kraken: Unix epoch seconds (e.g., `1730000000.123`)

**Normalized Format**: UTC seconds as `float` (e.g., `1730000000.123`)

**Normalization Function**:

```python
from cryptofeed.exchange import Exchange

# Convert any timestamp to UTC seconds
ts_utc = Exchange.timestamp_normalize(1730000000123)  # 1730000000.123

# All inputs supported:
Exchange.timestamp_normalize("2025-10-30T12:34:56Z")  # 1730000000.0
Exchange.timestamp_normalize(1730000000)              # 1730000000.0
Exchange.timestamp_normalize(1730000000.123)          # 1730000000.123
```

### Decimal Normalization

**Always use Decimal for prices/amounts**:

```python
from decimal import Decimal

# Correct usage
price = Decimal("45123.50000000")   # string → Decimal
amount = Decimal("0.12345678")
total = price * amount              # Decimal arithmetic

# Wrong (don't do this)
price = float("45123.50000000")     # precision loss
amount = 0.12345678                 # IEEE-754 issues
```

---

## 3. Parquet Schema Examples

### Trades Table

```parquet
message trade {
  required binary exchange (UTF8)
  required binary symbol (UTF8)
  required binary side (UTF8)
  required int64 timestamp (TIMESTAMP_MICROS)
  required binary price (UTF8)
  required binary amount (UTF8)
  required binary trade_id (UTF8)
  optional binary trade_type (UTF8)
}
```

### L2 Book Table

```parquet
message l2_book {
  required binary exchange (UTF8)
  required binary symbol (UTF8)
  required int64 timestamp (TIMESTAMP_MICROS)
  optional int64 sequence (INT64)
  optional binary checksum (UTF8)
  repeated group bids {
    required binary price (UTF8)
    required binary size (UTF8)
  }
  repeated group asks {
    required binary price (UTF8)
    required binary size (UTF8)
  }
}
```

### Candles Table

```parquet
message candle {
  required binary exchange (UTF8)
  required binary symbol (UTF8)
  required binary interval (UTF8)
  required int64 start (TIMESTAMP_MICROS)
  required int64 end (TIMESTAMP_MICROS)
  required binary open (UTF8)
  required binary close (UTF8)
  required binary high (UTF8)
  required binary low (UTF8)
  required binary volume (UTF8)
  optional int64 trades (INT64)
  required boolean closed
}
```

---

## 4. Implementation Checklist for Lakehouse

- [ ] Proto message definitions (all 7 core types)
- [ ] Python proto classes (via `buf` code generation)
- [ ] Conversion functions (cryptofeed types → proto)
- [ ] Parquet writer (proto → Parquet + Iceberg partitioning)
- [ ] DuckDB schema (Parquet table definitions)
- [ ] Sample queries (OHLCV, liquidity, VWAP, etc.)
- [ ] Validation tests (round-trip proto → Parquet → back)
- [ ] Documentation (schema, retention, queries)

---

**Document Version**: 1.0  
**Last Updated**: October 30, 2025  
**Ready for**: lakehouse-backend-adapter v0.1.0  
**File Location**: `/home/tommyk/projects/quant/data-sources/crypto-data/cryptofeed/docs/MARKET_DATA_SCHEMA_MAPPING.md`
