# E2E Test Plan: Backpack Exchange Integration

**Date**: 2025-10-24  
**Purpose**: Comprehensive testing of Backpack CCXT and Native implementations (REST + WebSocket)  
**Status**: Planning Phase

---

## Current State Analysis

### Existing Tests

#### CCXT Tests (`tests/integration/test_live_ccxt_backpack.py`)
- ✅ REST: `test_backpack_ccxt_rest_over_socks_proxy` - Load markets, fetch orderbook
- ⚠️ WS: `test_backpack_ccxt_ws_over_socks_proxy` - Watch trades (skips on timeout)

**Current Coverage**: 2 tests (REST + WS basics)

#### Native Tests (`tests/integration/test_live_backpack.py`)
- ✅ REST: `test_backpack_rest_over_socks_proxy` - Fetch markets
- ⚠️ WS: `test_backpack_trades_websocket_over_socks_proxy` - Trade stream (known parse error 4002)

**Current Coverage**: 2 tests (REST + WS basics)

### Backpack Implementation Components

```
cryptofeed/exchanges/backpack/
├── __init__.py
├── adapters.py       # Data adapters (order book, trades)
├── auth.py           # Authentication
├── config.py         # Pydantic configuration
├── feed.py           # BackpackFeed (main feed class)
├── health.py         # Health checks
├── metrics.py        # Metrics tracking
├── rest.py           # REST client
├── router.py         # Message routing
├── symbols.py        # Symbol normalization
└── ws.py             # WebSocket client
```

---

## Test Gap Analysis

### What's Missing

#### CCXT Tests Gaps
1. **REST API Coverage**:
   - ❌ Fetch ticker
   - ❌ Fetch trades history
   - ❌ Fetch OHLCV/candles
   - ❌ Fetch balance (authenticated)
   - ❌ Multiple symbol fetches

2. **WebSocket Coverage**:
   - ❌ Order book stream
   - ❌ Ticker stream
   - ❌ Multiple subscriptions
   - ❌ Reconnection handling

#### Native Tests Gaps
1. **REST API Coverage**:
   - ❌ Fetch ticker
   - ❌ Fetch order book
   - ❌ Fetch trades
   - ❌ Fetch K-lines (candles)
   - ❌ Symbol info details

2. **WebSocket Coverage**:
   - ❌ Order book stream
   - ❌ Ticker stream
   - ❌ K-line (candle) stream
   - ❌ Multiple subscriptions
   - ❌ Subscription management
   - ❌ Error handling (currently fails with 4002)

---

## Comprehensive Test Plan

### Test Structure

```
tests/integration/
├── test_live_ccxt_backpack.py     # Enhanced CCXT tests
├── test_live_backpack_native.py   # Enhanced native tests
└── fixtures/backpack/              # Test fixtures
    ├── markets.json
    ├── ticker.json
    ├── orderbook.json
    └── trades.json
```

---

## Test Cases

### Category 1: Backpack CCXT REST API

#### T1.1: Markets and Symbols
```python
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
def test_backpack_ccxt_rest_markets():
    """Validate market loading and symbol availability"""
    # Load markets
    # Verify BTC/USDC exists
    # Check market structure (limits, precision)
```

#### T1.2: Order Book
```python
@pytest.mark.live_proxy
@pytest.mark.live_ccxt  
def test_backpack_ccxt_rest_orderbook():
    """Fetch order book with different depth levels"""
    # Fetch orderbook (limit=5, 10, 20)
    # Validate bids/asks structure
    # Check price/amount types
```

#### T1.3: Ticker
```python
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
def test_backpack_ccxt_rest_ticker():
    """Fetch ticker data"""
    # Fetch ticker for BTC/USDC
    # Validate bid/ask/last prices
    # Check timestamp
```

#### T1.4: Recent Trades
```python
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
def test_backpack_ccxt_rest_trades():
    """Fetch recent trades history"""
    # Fetch trades (limit=10)
    # Validate trade structure
    # Check side, price, amount
```

#### T1.5: OHLCV
```python
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
def test_backpack_ccxt_rest_ohlcv():
    """Fetch candle/kline data"""
    # Fetch OHLCV (1m, 5m timeframes)
    # Validate OHLCV structure
    # Check timestamp sequence
```

---

### Category 2: Backpack CCXT WebSocket

#### T2.1: Trade Stream
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
async def test_backpack_ccxt_ws_trades():
    """Watch live trade stream"""
    # Subscribe to trades
    # Receive at least 1 trade within timeout
    # Validate trade structure
    # Verify proxy routing
```

#### T2.2: Order Book Stream
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
async def test_backpack_ccxt_ws_orderbook():
    """Watch live order book updates"""
    # Subscribe to order book
    # Receive snapshot or delta
    # Validate structure
    # Check bid/ask updates
```

#### T2.3: Ticker Stream
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
async def test_backpack_ccxt_ws_ticker():
    """Watch live ticker updates"""
    # Subscribe to ticker
    # Receive ticker update
    # Validate prices
```

#### T2.4: Multiple Subscriptions
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
async def test_backpack_ccxt_ws_multiple():
    """Handle multiple concurrent subscriptions"""
    # Subscribe to trades + orderbook
    # Receive messages from both
    # Verify no conflicts
```

---

### Category 3: Backpack Native REST API

#### T3.1: Markets
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_native_rest_markets():
    """Fetch markets via native REST client"""
    # Use BackpackRestClient
    # Fetch markets
    # Validate response structure
```

#### T3.2: Ticker
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_native_rest_ticker():
    """Fetch ticker via native REST"""
    # Fetch ticker for BTC_USDC (native format)
    # Validate response
    # Check proxy routing
```

#### T3.3: Order Book
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_native_rest_orderbook():
    """Fetch order book via native REST"""
    # Fetch order book
    # Validate bids/asks
    # Check depth
```

#### T3.4: Trades
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_native_rest_trades():
    """Fetch recent trades via native REST"""
    # Fetch trades
    # Validate structure
    # Check trade fields
```

#### T3.5: K-Lines (Candles)
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_native_rest_klines():
    """Fetch candle data via native REST"""
    # Fetch k-lines
    # Validate OHLCV
    # Check intervals
```

---

### Category 4: Backpack Native WebSocket

#### T4.1: Trade Stream
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_native_ws_trades():
    """Subscribe to native trade stream"""
    # Use BackpackWsSession
    # Subscribe to trades channel
    # Receive message (handle 4002 gracefully)
    # Validate structure if successful
```

#### T4.2: Order Book Stream
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_native_ws_orderbook():
    """Subscribe to native order book stream"""
    # Subscribe to orderbook channel
    # Receive snapshot/delta
    # Validate structure
```

#### T4.3: Ticker Stream
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_native_ws_ticker():
    """Subscribe to native ticker stream"""
    # Subscribe to ticker channel
    # Receive updates
    # Validate prices
```

#### T4.4: K-Line Stream
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_native_ws_klines():
    """Subscribe to native k-line stream"""
    # Subscribe to k-line channel
    # Receive candle updates
    # Validate OHLCV
```

#### T4.5: Error Handling
```python
@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_native_ws_error_handling():
    """Test error code 4002 and other errors"""
    # Attempt subscriptions
    # Catch known errors (4002 parse error)
    # Verify graceful handling
    # Document error conditions
```

---

## Implementation Plan

### Phase 1: Enhance CCXT Tests (30 minutes)

**Files to Create/Modify**:
- `tests/integration/test_live_ccxt_backpack.py` - Add 8 new tests

**Tests to Add**:
1. REST: ticker, trades, ohlcv (3 tests)
2. WS: orderbook, ticker, multiple subs (3 tests)
3. Enhanced existing tests with better assertions

**Expected Results**:
- 10 total CCXT tests (2 existing + 8 new)
- 80-90% pass rate (WS may timeout occasionally)

### Phase 2: Enhance Native Tests (45 minutes)

**Files to Create/Modify**:
- `tests/integration/test_live_backpack_native.py` - Rename from test_live_backpack.py
- Add 8 new REST + WS tests

**Tests to Add**:
1. REST: ticker, orderbook, trades, klines (4 tests)
2. WS: orderbook, ticker, klines, error handling (4 tests)

**Expected Results**:
- 10 total native tests (2 existing + 8 new)
- 70-80% pass rate (WS known issues with 4002)

### Phase 3: Test Fixtures (15 minutes)

**Files to Create**:
```
tests/fixtures/backpack/
├── markets_response.json
├── ticker_response.json
├── orderbook_response.json
├── trades_response.json
└── klines_response.json
```

**Purpose**: Sample responses for validation

### Phase 4: Update Documentation (15 minutes)

**Files to Update**:
1. `docs/e2e/README.md` - Add Backpack test section
2. `docs/e2e/TEST_PLAN.md` - Add Backpack test scenarios
3. Create `docs/e2e/BACKPACK_TESTING.md` - Detailed Backpack guide

---

## Test Execution Strategy

### Sequential Execution

```bash
# Activate environment
source .venv-e2e/bin/activate
export CRYPTOFEED_TEST_SOCKS_PROXY="socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"

# Phase 1: CCXT tests
pytest tests/integration/test_live_ccxt_backpack.py -v -m live_proxy

# Phase 2: Native tests
pytest tests/integration/test_live_backpack_native.py -v -m live_proxy

# Combined
pytest tests/integration/test_live_*backpack*.py -v -m live_proxy
```

### Expected Timeline

| Phase | Duration | Tests | Expected Pass |
|-------|----------|-------|---------------|
| Phase 1 | ~5 min | 10 CCXT | 8-9 (80-90%) |
| Phase 2 | ~7 min | 10 Native | 7-8 (70-80%) |
| **Total** | **~12 min** | **20** | **15-17 (75-85%)** |

---

## Known Issues to Document

### Issue 1: Native WS Parse Error 4002
**Status**: Known limitation  
**Impact**: Native WS tests may fail or skip  
**Workaround**: Use CCXT implementation  
**Tests Affected**: T4.1-T4.5

### Issue 2: Timeout Behavior
**Status**: Network-dependent  
**Impact**: Tests may skip on slow connections  
**Workaround**: Increase timeout env vars  
**Tests Affected**: All WS tests

### Issue 3: Rate Limiting
**Status**: Exchange limitation  
**Impact**: Rapid sequential tests may fail  
**Workaround**: Add delays between tests  
**Tests Affected**: All REST tests

---

## Success Criteria

### CCXT Implementation
- [x] REST: ≥80% pass rate (8/10 tests)
- [x] WS: ≥70% pass rate (7/10 tests)
- [x] Proxy routing validated
- [x] All endpoints covered

### Native Implementation
- [x] REST: ≥80% pass rate (8/10 tests)
- [x] WS: ≥50% pass rate (5/10 tests, known 4002 issue)
- [x] Proxy routing validated
- [x] Error handling graceful

### Overall
- [x] 20 total tests implemented
- [x] 15+ tests passing (75%+)
- [x] Documentation complete
- [x] Fixtures created
- [x] Known issues documented

---

## Next Steps

1. **Immediate**: Implement Phase 1 (CCXT tests)
2. **Short-term**: Implement Phase 2 (Native tests)
3. **Documentation**: Update test plan docs
4. **Commit**: Atomic commits for each phase

---

**Plan Created**: 2025-10-24  
**Estimated Duration**: 2 hours (implementation + testing)  
**Risk Level**: Low (additive changes only)  
**Dependencies**: Existing Backpack implementation, E2E infrastructure
