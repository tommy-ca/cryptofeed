# E2E Test Plan: Live Proxy Validation with Mullvad

## Overview

This document outlines the end-to-end test plan for validating newly introduced modules (proxy system, CCXT exchanges, native exchanges) using live Mullvad SOCKS5 proxies.

**Branch**: `feature/normalized-data-schema-crypto`  
**Date**: 2025-10-24  
**Purpose**: Validate proxy routing, exchange connectivity, and data normalization in production-like conditions

---

## Test Objectives

1. **Proxy System Validation**: Verify HTTP and WebSocket connections route correctly through SOCKS5 proxies
2. **CCXT Exchange Integration**: Validate CCXT/CCXT-Pro feeds work with proxy configuration
3. **Native Exchange Integration**: Validate native Backpack implementation with proxy support
4. **Data Normalization**: Verify timestamp normalization and symbol mapping work across exchanges
5. **Regional Behavior**: Document geofencing and regional restrictions

---

## Prerequisites

### 1. Mullvad Relay List Access

**Artifact Location**:
```bash
# Download latest Mullvad relay list
gh run download 18632839930 \
  --repo tommy-ca/mulvad-relay-list \
  -D /tmp/proxy_artifact

# View available endpoints
head /tmp/proxy_artifact/mullvad-relay-artifacts/mullvad_relays.csv
```

**Key Endpoints** (as of 2025-10-19):
| Region | Endpoint | Usage |
|--------|----------|-------|
| US East | `socks5://us-nyc-wg-socks5-301.relays.mullvad.net:1080` | Test geofencing |
| Europe | `socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080` | Primary testing |
| Asia | `socks5://sg-sin-wg-socks5-001.relays.mullvad.net:1080` | Regional validation |

### 2. Environment Setup

**Required Dependencies**:
```bash
pip install -e ".[dev]"
pip install ccxt ccxtpro aiohttp-socks python-socks
```

**Environment Variables**:
```bash
# Primary proxy endpoint (rotate between regions)
export CRYPTOFEED_TEST_SOCKS_PROXY="socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"

# Test symbols (optional overrides)
export CRYPTOFEED_TEST_BINANCE_SYMBOL="BTCUSDT"
export CRYPTOFEED_TEST_BINANCE_WS_STREAM="btcusdt@trade"
export CRYPTOFEED_TEST_CCXT_SYMBOL="BTC/USDC:USDC"
export CRYPTOFEED_TEST_BACKPACK_CCXT_SYMBOL="BTC/USDC"

# Timeout configuration
export CRYPTOFEED_TEST_BINANCE_WS_TIMEOUT="10"
export CRYPTOFEED_TEST_CCXT_WS_TIMEOUT="20"
export CRYPTOFEED_TEST_CCXT_REST_TIMEOUT="15"
export CRYPTOFEED_TEST_BACKPACK_REST_TIMEOUT="10"
```

---

## Test Categories

### Category 1: Proxy System Core (Unit → Integration)

**Purpose**: Validate proxy configuration, injection, and connection routing

#### T1.1: Proxy Configuration Loading
```bash
pytest tests/unit/test_proxy_mvp.py -v
```
**Validates**:
- Environment variable parsing
- YAML configuration loading
- Programmatic ProxySettings construction
- Precedence rules (env > YAML > programmatic)

**Success Criteria**: All 40+ unit tests pass

#### T1.2: Proxy Integration
```bash
pytest tests/integration/test_proxy_integration.py -v
```
**Validates**:
- HTTP connection proxy injection
- WebSocket connection proxy injection
- Exchange-specific proxy resolution
- Default proxy fallback

**Success Criteria**: All integration tests pass

---

### Category 2: Live Exchange Connectivity

**Purpose**: Validate real exchange connections through Mullvad proxies

#### T2.1: Binance REST + WebSocket (Established Exchange)
```bash
export CRYPTOFEED_TEST_SOCKS_PROXY="socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"

pytest tests/integration/test_live_binance.py \
  -v -m "live_proxy and live_binance" \
  -k "ticker or orderbook or websocket"
```

**Validates**:
- HTTP REST endpoint connectivity (`/api/v3/ticker/price`, `/api/v3/depth`)
- WebSocket stream connectivity (`btcusdt@trade`)
- Geofencing behavior (HTTP 451 handling)
- Proxy timeout configuration

**Expected Results**:
| Endpoint | US Proxy | EU Proxy | Asia Proxy |
|----------|----------|----------|-----------|
| REST ticker | ⚠️ 451 skip | ✅ pass | ✅ pass |
| REST depth | ⚠️ 451 skip | ✅ pass | ✅ pass |
| WS trades | ⚠️ skip | ✅ pass | ✅ pass |

**Success Criteria**: EU/Asia proxies return valid data; US proxy skips gracefully

#### T2.2: Hyperliquid (CCXT Generic Feed)
```bash
pytest tests/integration/test_live_ccxt_hyperliquid.py \
  -v -m "live_proxy and live_ccxt"
```

**Validates**:
- CCXT REST transport with proxy (`fetch_order_book`)
- CCXT.pro WebSocket transport with proxy (`watch_trades`)
- Proxy configuration via `socksProxy`/`wsSocksProxy`
- Symbol normalization (Hyperliquid perpetuals)

**Expected Results**:
| Test | EU Proxy | Status |
|------|----------|--------|
| REST order book | ✅ | BTC/USDC:USDC L2 data |
| WS trades | ✅ | Real-time trade messages |

**Success Criteria**: Both REST and WS receive valid data within timeout

#### T2.3: Backpack (Native Exchange Implementation)
```bash
# CCXT-based Backpack (working)
pytest tests/integration/test_live_ccxt_backpack.py \
  -v -m "live_proxy and live_ccxt"

# Native Backpack (known issue: parse error)
pytest tests/integration/test_live_backpack.py \
  -v -m "live_proxy and live_backpack"
```

**Validates**:
- Backpack CCXT REST (`fetch_markets`, `fetch_order_book`)
- Backpack CCXT.pro WebSocket (`watch_trades`)
- Native Backpack REST client
- Native Backpack WebSocket (⚠️ known parse error 4002)

**Expected Results**:
| Implementation | REST | WebSocket | Status |
|---------------|------|-----------|--------|
| CCXT | ✅ markets, order book | ✅ trades | Working |
| Native | ✅ markets | ⚠️ parse error (4002) | Partial |

**Success Criteria**: CCXT implementation fully functional; native WS documented as known issue

---

### Category 3: Data Normalization Validation

**Purpose**: Validate timestamp normalization and symbol mapping work correctly

#### T3.1: Timestamp Normalization
```bash
pytest tests/unit/test_exchange.py -v -k "timestamp"
```

**Validates**:
- `Exchange.timestamp_normalize()` handles datetime objects
- Millisecond detection (values >= 1e12)
- ISO-8601 string parsing (with/without 'Z')
- Numeric string parsing

**Test Matrix**:
| Input Type | Example | Expected Output |
|------------|---------|----------------|
| datetime | `datetime(2024, 10, 24, 12, 0, tzinfo=UTC)` | `1729771200.0` |
| int (seconds) | `1729771200` | `1729771200.0` |
| int (milliseconds) | `1729771200000` | `1729771200.0` |
| str (numeric) | `"1729771200"` | `1729771200.0` |
| str (ISO-8601) | `"2024-10-24T12:00:00Z"` | `1729771200.0` |

**Success Criteria**: All input formats normalize correctly

#### T3.2: Symbol Mapping (Bybit)
```bash
pytest tests/unit/test_exchange.py -v -k "bybit and symbol"
```

**Validates**:
- Spot symbols: `BTCUSDT` → `BTC-USDT`
- Perpetual symbols: `BTCUSDT` → `BTC-USDT` or `BTCUSDTPERP` → `BTCUSDTPERP`
- Futures symbols: expiry date parsing
- Heuristic fallbacks for unmapped symbols

**Success Criteria**: Symbol mapping produces consistent normalized forms

#### T3.3: Symbol Mapping (Coinbase)
```bash
pytest tests/unit/test_exchange.py -v -k "coinbase and symbol"
```

**Validates**:
- Modern API format (`base_currency_id`, `quote_currency_id`)
- Legacy API format (`base_symbol`, `quote_symbol`)
- Product ID extraction
- tick_size metadata preservation

**Success Criteria**: Both modern and legacy formats parse correctly

---

### Category 4: Concurrent Proxy Operations

**Purpose**: Validate multiple exchanges using different proxies simultaneously

#### T4.1: Multi-Exchange Concurrent Demo
```bash
# Run the existing concurrent proxy example
python examples/demo_concurrent_proxy.py
```

**Validates**:
- Per-exchange proxy configuration
- Concurrent HTTP/WS connections through different proxies
- No proxy lease conflicts
- Proper connection cleanup

**Configuration** (from example):
```python
proxy:
  enabled: true
  default:
    http:
      url: "socks5://default-proxy:1080"
  exchanges:
    binance:
      http:
        url: "socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"
    coinbase:
      http:
        url: "socks5://us-nyc-wg-socks5-301.relays.mullvad.net:1080"
```

**Success Criteria**: All feeds connect and receive data without conflicts

#### T4.2: Stress Test (20+ Concurrent Feeds)
```bash
# Create custom test script (see T4.2-stress-test.py below)
python tests/integration/T4.2-stress-test.py
```

**Validates**:
- Proxy pool behavior under load
- Connection limit handling
- Memory stability over 5-minute duration
- Graceful degradation on proxy failures

**Success Criteria**: 
- All feeds initialize successfully
- No memory leaks (<5% growth over 5 minutes)
- Clean shutdown without hanging connections

---

### Category 5: Regional Geofencing Matrix

**Purpose**: Document exchange behavior across different proxy regions

#### T5.1: Regional Validation Matrix
```bash
# Script to test all combinations
./tests/integration/regional_validation.sh
```

**Test Matrix**:
| Exchange | US Proxy | EU Proxy | Asia Proxy | Notes |
|----------|----------|----------|-----------|-------|
| Binance REST | ⚠️ 451 | ✅ | ✅ | US geofenced |
| Binance WS | ⚠️ 451 | ✅ | ✅ | US geofenced |
| Coinbase REST | ✅ | ✅ | ✅ | No restrictions |
| Coinbase WS | ✅ | ✅ | ✅ | No restrictions |
| Bybit REST | ✅ | ✅ | ✅ | No restrictions |
| Bybit WS | ✅ | ✅ | ✅ | No restrictions |
| Backpack (ccxt) | ✅ | ✅ | ✅ | No restrictions |
| Hyperliquid (ccxt) | ✅ | ✅ | ✅ | No restrictions |

**Expected Results**: Based on `docs/proxy/live-testing.md`

**Success Criteria**: Results match documented regional behavior

---

## Test Execution Strategy

### Phase 1: Smoke Tests (5-10 minutes)
```bash
# Quick validation that core functionality works
pytest tests/unit/test_proxy_mvp.py -v --tb=short
pytest tests/integration/test_proxy_integration.py -v --tb=short
pytest tests/unit/test_ccxt_*.py -v --tb=short
```

**Gates Phase 2**: All unit tests must pass

### Phase 2: Live Connectivity (15-20 minutes)
```bash
# Set EU proxy (most permissive)
export CRYPTOFEED_TEST_SOCKS_PROXY="socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"

# Run all live tests
pytest tests/integration/test_live_*.py -v -m "live_proxy"
```

**Gates Phase 3**: At least 80% of live tests pass (allowing for transient network issues)

### Phase 3: Regional Validation (30-45 minutes)
```bash
# Test each region sequentially
for region in US EU ASIA; do
  export CRYPTOFEED_TEST_SOCKS_PROXY="socks5://$(get_proxy_for_region $region)"
  pytest tests/integration/test_live_*.py -v -m "live_proxy" --junit-xml=results-${region}.xml
done
```

**Deliverable**: Regional behavior matrix documenting all combinations

### Phase 4: Stress Testing (60 minutes)
```bash
# Long-running concurrent feed validation
python tests/integration/T4.2-stress-test.py --duration=3600 --feeds=20
```

**Deliverable**: Performance report (memory, connection stability, error rates)

---

## Success Criteria

### Overall E2E Pass Criteria
- ✅ **Unit Tests**: 100% pass (tests/unit/test_proxy_mvp.py, test_ccxt_*.py)
- ✅ **Integration Tests**: 100% pass (tests/integration/test_proxy_integration.py)
- ✅ **Live Tests**: ≥80% pass with EU proxy (allowing for transient failures)
- ✅ **Regional Matrix**: Documented behavior matches observations
- ✅ **Stress Test**: <5% memory growth, 0 deadlocks over 60 minutes
- ✅ **Data Quality**: All timestamps normalize to UTC float, symbols map consistently

### Known Issues (Acceptable Failures)
- ⚠️ Binance US proxy: HTTP 451 (geofencing) - skip gracefully
- ⚠️ Backpack native WS: Parse error 4002 - documented, use CCXT fallback
- ⚠️ Transient network errors: <20% of live tests may fail due to proxy/network issues

---

## Test Artifacts

### Generated During Testing
1. **Test Results**: XML reports per phase (`pytest --junit-xml`)
2. **Regional Matrix**: CSV with exchange/region/endpoint/status
3. **Performance Metrics**: Memory profile, connection counts, error logs
4. **Network Traces**: Packet captures for debugging (optional)

### Documentation Updates
1. `docs/proxy/live-testing.md`: Update regional observations
2. `SPEC_STATUS.md`: Mark E2E validation complete
3. `IMPLEMENTATION_SUMMARY.md`: Add E2E results section

---

## Test Scripts to Create

### T4.2: Stress Test Script
```python
# tests/integration/T4.2-stress-test.py
"""
Stress test for concurrent proxy operations.
Run 20+ feeds simultaneously through Mullvad proxies for 60 minutes.
Monitor memory, connections, and error rates.
"""
import asyncio
import time
from cryptofeed import FeedHandler
from cryptofeed.defines import TRADES, L2_BOOK
# ... implementation
```

### T5.1: Regional Validation Script
```bash
# tests/integration/regional_validation.sh
"""
Test all exchange/region combinations and generate matrix.
"""
#!/bin/bash
# ... implementation
```

---

## Timeline

| Phase | Duration | Parallel Execution |
|-------|----------|-------------------|
| **Phase 1** (Smoke) | 10 min | Sequential |
| **Phase 2** (Live) | 20 min | Parallel by exchange |
| **Phase 3** (Regional) | 45 min | Parallel by region |
| **Phase 4** (Stress) | 60 min | Single long-running |
| **Total** | ~2.5 hours | With parallelization |

**Recommendation**: Run Phase 1-3 in CI/CD, Phase 4 manually before release

---

## Risk Mitigation

### Risk: Mullvad Proxies Become Unavailable
**Mitigation**: 
- Test with 3 different relays per region
- Document alternative proxy sources (ssh tunnels, commercial VPNs)
- Graceful skip on proxy unreachable

### Risk: Exchange API Changes
**Mitigation**:
- Pin exchange behavior to specific API versions in tests
- Maintain fixtures for known-good responses
- Document API version in test comments

### Risk: Transient Network Failures
**Mitigation**:
- Retry logic with exponential backoff
- Mark flaky tests with `@pytest.mark.flaky(reruns=3)`
- Report pass rate, not just pass/fail

---

## Appendix A: Quick Start Commands

```bash
# 1. Download Mullvad relay list
gh run download 18632839930 --repo tommy-ca/mulvad-relay-list -D /tmp/proxy_artifact

# 2. Set environment
export CRYPTOFEED_TEST_SOCKS_PROXY="socks5://de-fra-wg-socks5-101.relays.mullvad.net:1080"

# 3. Run smoke tests
pytest tests/unit/test_proxy_mvp.py tests/integration/test_proxy_integration.py -v

# 4. Run live tests
pytest tests/integration/test_live_*.py -v -m "live_proxy"

# 5. View results
cat results-*.xml | grep -E "testsuite|testcase"
```

---

## Appendix B: Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: aiohttp_socks` | `pip install aiohttp-socks` |
| `Connection timeout` | Verify DNS for relay hostname, check firewall |
| `HTTP 451` | Switch to EU/Asia proxy, or skip test |
| `SOCKS5 auth failed` | Mullvad relays require no auth, check URL format |
| `Parse error 4002 (Backpack)` | Expected, use CCXT implementation instead |

---

**Document Status**: ✅ Ready for execution  
**Last Updated**: 2025-10-24  
**Owner**: Engineering Team
