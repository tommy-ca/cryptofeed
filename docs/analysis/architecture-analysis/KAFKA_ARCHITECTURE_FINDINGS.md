# Cryptofeed Kafka & Serialization Architecture - Comprehensive Findings

**Exploration Date**: October 29, 2025
**Status**: Complete - All 6 focus areas analyzed

---

## 1. Current Kafka Backend Implementation

### Architecture Overview
Located: `/cryptofeed/backends/kafka.py` (159 lines)

**Current Implementation**:
```python
class KafkaCallback(BackendQueue):
    def topic(self, data: dict) -> str:
        return f"{self.key}-{data['exchange']}-{data['symbol']}"
    
    def partition_key(self, data: dict) -> Optional[bytes]:
        return None  # No partition key by default
```

**Topic Naming Pattern**: `{key}-{exchange}-{symbol}`
- Example: `trades-COINBASE-BTC-USD`, `book-BINANCE-ETH-USDT`, `funding-BYBIT-BTC-USDT-PERP`

**Data Type Implementations** (11 callback classes):
| Data Type | Class | Default Key |
|-----------|-------|------------|
| Trades | `TradeKafka` | `trades` |
| Funding Rate | `FundingKafka` | `funding` |
| Order Book | `BookKafka` | `book` |
| Ticker | `TickerKafka` | `ticker` |
| Open Interest | `OpenInterestKafka` | `open_interest` |
| Liquidations | `LiquidationsKafka` | `liquidations` |
| Candles | `CandlesKafka` | `candles` |
| Order Info | `OrderInfoKafka` | `order_info` |
| Transactions | `TransactionsKafka` | `transactions` |
| Balances | `BalancesKafka` | `balances` |
| Fills | `FillsKafka` | `fills` |

---

## 2. Topic Explosion Analysis

### Problem: Per-Symbol Topic Creation
**Severity**: MODERATE (Mitigated by override pattern)

**Issue**:
- Current default creates separate topic for every `{key}-{exchange}-{symbol}` combination
- For large exchanges (Binance: 1000+ symbols), this creates 1000+ topics per data type
- Across 13 exchange integrations × 11 data types = potential for 143,000+ topics

**Example Topic Explosion**:
```
trades-BINANCE-BTC-USDT
trades-BINANCE-ETH-USDT
trades-BINANCE-MATIC-USDT
... (1000+ more)
trades-COINBASE-BTC-USD
trades-COINBASE-ETH-USD
... (100+ more)
```

### Mitigation: Overridable Topic Strategy
**Good News**: Framework supports custom topic mapping via inheritance

**Example from `demo_kafka.py`**:
```python
class CustomTradeKafka(TradeKafka):
    def topic(self, data: dict) -> str:
        return f"{self.key}-{data['exchange']}"  # One topic per exchange!
    
    def partition_key(self, data: dict) -> Optional[bytes]:
        return f"{data['symbol']}".encode('utf-8')  # Symbol as partition key
```

**Benefits of Override Pattern**:
- Reduces topics from N symbols × M exchanges to just M exchanges
- Partitions by symbol for ordering guarantees within symbol
- Backward compatible with default behavior

---

## 3. Serialization Implementations

### Current JSON Serialization
**Default Serializer** (`_default_serializer` at line 49):
```python
def _default_serializer(self, to_bytes: dict | str) -> ByteString:
    if isinstance(to_bytes, dict):
        return json.dumpb(to_bytes)  # Uses custom JSON encoder
    elif isinstance(to_bytes, str):
        return to_bytes.encode()
```

**Why Custom JSON**:
- Uses `cryptofeed.json_utils.json.dumpb()` for specialized handling
- Supports `numeric_type` parameter (float vs Decimal preservation)
- Handles `none_to` parameter (None value mapping)
- Preserves cryptofeed-specific data types in JSON

**Extensibility Pattern** (line 95):
```python
value = updates[index] if self.producer_config.get('value_serializer') \
    else self._default_serializer(updates[index])
```

- Accepts custom `value_serializer` in producer_config
- Allows users to inject Avro, Protobuf, or other formats
- No built-in protobuf serialization yet (missing piece for Spec 1)

### Key Serializer Configuration
Line 96: `key = self.key if self.producer_config.get('key_serializer') else self._default_serializer(self.key)`
- Uses string key (e.g., "trades") as topic identifier
- Serialized same way as values

---

## 4. Data Type to Topic Mapping

### Current Mapping Pattern
All data flows through `BackendCallback.__call__()` (line 92-97 in `backend.py`):

```python
async def __call__(self, dtype, receipt_timestamp: float):
    data = dtype.to_dict(numeric_type=self.numeric_type, none_to=self.none_to)
    # ... adds receipt_timestamp ...
    await self.write(data)  # Sends to callback queue
```

### Data Types with Native to_dict() Support
**Cryptofeed Type System**:
- All data types implement `.to_dict()` method
- Called by `BackendCallback.__call__()` for serialization
- No native `.to_proto()` methods yet (will be added by Spec 1)

**Current Support**:
- Trade, Ticker, Funding, OrderBook, OpenInterest, Liquidation, Candle
- OrderInfo, Transaction, Balance, Fill
- All produce dictionaries for JSON serialization

### OrderBook Special Handling
`BackendBookCallback` (line 100-125):
- Handles both snapshots and deltas
- Delta support via configurable `snapshots_only` parameter
- Full snapshot produced at `snapshot_interval` boundaries
- Example: `BookKafka(snapshots_only=False, snapshot_interval=1000)`

---

## 5. Protobuf Integration Status

### Proto Schema Files Available
**Location**: `proto/cryptofeed/normalized/v1/` (20 files)

**Complete Schema Coverage**:
- `trade.proto` - Trade events
- `order_book.proto` (Level2Book) - Order book snapshots
- `level2_delta.proto` - Order book deltas
- `ticker.proto` - Ticker quotes
- `funding.proto` - Funding rate events
- `candle.proto` - OHLCV candles
- `liquidation.proto` - Liquidation events
- `open_interest.proto` - Open interest
- `balance.proto`, `fill.proto`, `order.proto`, `order_info.proto`
- `index_price.proto`, `nbbo.proto`, `position.proto`, `transaction.proto`

### Generated Python Bindings
**Status**: READY
- Bindings auto-generated in `gen/protobuf/cryptofeed/normalized/v1/`
- Import: `from cryptofeed.normalized.v1 import trade_pb2, order_book_pb2, etc.`
- Available fields: all typed with proper proto3 semantics

### Existing Protobuf Mapping (Partial)
**File**: `cryptofeed/proto_mappers/order_book.py` (121 lines)

**Implemented Converters**:
1. `level2_book_from_order_book()` - OrderBook → Level2Book protobuf
2. `level2_delta_from_order_book()` - OrderBook delta → Level2Delta protobuf

**Mapping Pattern**:
```python
def level2_book_from_order_book(book: OrderBook) -> order_book_pb2.Level2Book:
    message = order_book_pb2.Level2Book()
    message.exchange = book.exchange
    message.symbol = book.symbol
    _extend_price_levels(message.bids, book.book.bids, ascending=False)
    _extend_price_levels(message.asks, book.book.asks, ascending=True)
    # Handle timestamp conversion: seconds → microseconds
    message.timestamp = int(round(timestamp * 1_000_000))
    return message
```

**Key Conversions**:
- Decimal → String (1e-8 scale) via `_decimal_to_str()`
- Float seconds → Int64 microseconds (× 1,000,000)
- Numeric precision preserved with `ROUND_HALF_EVEN`

### What's Missing (Spec 1 Work)
**NOT YET IMPLEMENTED** - 18 remaining data types need converters:
- Trade, Ticker, Funding, Candle, Liquidation, OpenInterest
- OrderInfo, Transaction, Balance, Fill
- Index, NBBO, Position, and others
- No `to_proto()` methods on cryptofeed type classes yet

---

## 6. Product Type Handling

### Current Product Type Representation

**NO DEDICATED PRODUCT TYPE FIELD in Kafka topics**

**How it works today**:
- Implicit in symbol naming: `BTC-USDT` (spot), `BTC-USDT-PERP` (perpetual)
- Exchange handles product type in metadata, not in data flow
- Example from Kraken Futures (line 44-50):
  ```python
  _kraken_futures_product_type = {
      'FI': 'Inverse Futures',
      'PF': 'Perpetual Linear Multi-Collateral Futures',
      ...
  }
  # Stored in exchange info metadata, not propagated
  ```

### Define Constants
**File**: `cryptofeed/defines.py`
```python
FUTURES = 'futures'
PERPETUAL = 'perpetual'
SPOT = 'spot'
```

**Usage**: Only in exchange class definitions (BINANCE_FUTURES, KRAKEN_FUTURES, etc.)
- Not in data payload
- Not in topic names
- Implicit in symbol or exchange context

### Protobuf Schema Approach
**NO product_type FIELD in normalized protos**
- Trade.proto: has `exchange`, `symbol`, `side`, `price`, `amount`, `timestamp`
- No product_type field defined
- Design decision: normalize at schema level, not message level

### Implication for QuixStreams
- To distinguish perpetual from spot trades in aggregation:
  - Parse symbol pattern (BTC-USDT vs BTC-USDT-PERP)
  - OR store mapping externally (exchange info service)
  - OR add product_type field to protobuf (requires schema change)

---

## 7. QuixStreams Integration Readiness

### Current State
**Zero existing QuixStreams code** in cryptofeed
- No custom source implementations
- No stream topology definitions
- No aggregation pipeline

### How QuixStreams Would Integrate

**Architecture Plan** (from Spec 2: `quixstreams-integration`):
```
Kafka Topics (Protobuf)
    ↓
QuixStreams TopologyBuilder
    ├─ KafkaTopics for input (trades, books, tickers)
    ├─ StateStore for windowed aggregations (RocksDB)
    └─ Custom functions for OHLCV, VWAP, correlation
    ↓
Output Topics (Candles, Metrics, Correlation)
    ↓
Lakehouse Backend Adapter (Spec 3)
```

### Expected Topic Structure for QuixStreams

**Input Topics** (from cryptofeed Kafka backend):
- `trades-BINANCE` (with symbol partition key)
- `book-BINANCE` (order book snapshots)
- `ticker-BINANCE` (ticker updates)

**Output Topics** (created by QuixStreams):
- `candles-1m` - 1m OHLCV
- `candles-5m` - 5m OHLCV
- `metrics-vwap` - Volume-weighted prices
- `metrics-correlation` - Cross-exchange correlations

### Protobuf as Foundation
**Spec 1 Requirement** (Protobuf Callback Serialization):
- Must implement `to_proto()` on all 20 data types
- Must extend `BackendCallback` to support protobuf format option
- QuixStreams needs efficient binary format for high-throughput processing

---

## Key Findings Summary

### Strengths
1. **Flexible Topic Naming**: Override pattern allows per-exchange or custom topologies
2. **Schema-First Approach**: 20 normalized protobuf schemas already defined
3. **Partial Protobuf Implementation**: OrderBook converters exist, pattern established
4. **Extensible Serialization**: Custom value_serializer hook available
5. **Type Safety**: Python protobuf bindings generated with full type hints

### Gaps
1. **No Built-in Protobuf Serialization**: `value_serializer` must be user-provided
2. **Incomplete Type Mappings**: Only OrderBook → Protobuf; 18 types need converters
3. **No Product Type Field**: Symbol parsing required for perpetual/spot distinction
4. **No QuixStreams Integration**: Spec 2 adds this layer
5. **Topic Explosion Risk**: Default topic naming creates 1000s of topics (mitigated by override)

### Recommended Patterns

**1. For Topic Naming** (avoid explosion):
```python
class OptimizedTradeKafka(TradeKafka):
    def topic(self, data: dict) -> str:
        return f"{self.key}-{data['exchange']}"
    
    def partition_key(self, data: dict) -> Optional[bytes]:
        return data['symbol'].encode('utf-8')
```

**2. For Protobuf Serialization** (Spec 1 work):
```python
class ProtobufSerializer:
    def serialize(self, obj: Trade) -> bytes:
        proto_msg = obj.to_proto()  # Will be added by Spec 1
        return proto_msg.SerializeToString()

kafka_callback = TradeKafka(
    ...,
    value_serializer=ProtobufSerializer().serialize
)
```

**3. For Product Type Tracking** (optional enhancement):
```python
# Option A: Add to protobuf schema (breaking change)
# Option B: Embed in symbol (current approach: BTC-USDT-PERP)
# Option C: Use partition strategy with symbol prefix
```

---

## Spec Dependencies & Timeline

### Spec 1: Protobuf Callback Serialization (BLOCKING)
- **What**: Implement `to_proto()` on all 20 types, extend BackendCallback
- **Why**: QuixStreams needs efficient binary format
- **Timeline**: 1-2 weeks
- **Deliverable**: Protobuf-serialized Kafka topics ready for consumption

### Spec 2: QuixStreams Integration (BLOCKED by Spec 1)
- **What**: Stream processing layer (OHLCV, VWAP, correlation)
- **Why**: Real-time analytics foundation for lakehouse
- **Timeline**: 2-3 weeks (after Spec 1)
- **Deliverable**: Aggregated metric topics + state stores

### Spec 3: Lakehouse Backend Adapter (BLOCKED by Specs 1 & 2)
- **What**: Persistent storage (DuckDB + Parquet) + SQL interface
- **Why**: Historical analytics and backfill
- **Timeline**: 3-4 weeks (after Specs 1 & 2)
- **Deliverable**: Production lakehouse with exactly-once semantics

---

## Files Explored

**Kafka Backend** (6 files):
- `/cryptofeed/backends/kafka.py` - Main implementation
- `/examples/demo_kafka.py` - Custom topic example
- `/cryptofeed/backends/backend.py` - Callback base classes

**Protobuf Integration** (8 files):
- `/proto/cryptofeed/normalized/v1/` - 20 schema files
- `/gen/protobuf/cryptofeed/normalized/v1/` - Generated bindings
- `/cryptofeed/proto_mappers/order_book.py` - Partial mappers
- `/cryptofeed/proto_mappers/__init__.py` - Export registry

**Configuration & Specifications**:
- `.kiro/specs/protobuf-callback-serialization/requirements.md`
- `.kiro/specs/quixstreams-integration/requirements.md`
- `/docs/specs/normalized-data-schema/status.md`

**Support Code**:
- `/cryptofeed/defines.py` - SPOT, PERPETUAL, FUTURES constants
- `/cryptofeed/exchanges/kraken_futures.py` - Product type mapping example

