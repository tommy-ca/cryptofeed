# Kafka Protobuf Binance E2E - Implementation Schedule

## Overview

**Specification**: `kafka-protobuf-binance-e2e`
**Current Phase**: Tasks Approved (Ready for Implementation)
**Completion**: 28/43 tasks complete (65%)
**Remaining Work**: 15 tasks (Phase 6: Proxy-Aware Execution)

## Progress Summary

### ✅ Completed Phases (28 tasks)
- **Phase 1**: Exploration and Context Alignment (3 tasks) ✅
- **Phase 2**: Test Harness Design and Wiring (3 tasks) ✅
- **Phase 3**: Binance→Kafka Protobuf Integration Tests (3 tasks) ✅
- **Phase 4**: Environment Guards and Stability (6 tasks) ✅
- **Phase 4B**: Binance USDⓈ-M Futures E2E Extension (5 tasks) ✅
- **Phase 5**: Validation and Documentation (3 tasks) ✅
- **Phase C**: Governance & Spec Hygiene (3 tasks) ✅

### 🔧 Remaining Work: Phase 6 (15 tasks)
**Focus**: Proxy-Aware Execution (FR7)
**Goal**: Enable E2E testing through proxies for geoblocked regions

---

## Implementation Waves

### Wave 1: Fix Proxy Infrastructure Blockers (Priority: CRITICAL)
**Duration**: 2-3 days
**Dependencies**: None (blocking all proxy work)

#### Tasks:
1. **Task 6.5**: Fix proxy preflight helper
   - Initialize ProxySettings before calling get_proxy_injector()
   - Propagate leased HTTP proxy to HTTP[S]_PROXY env vars
   - **Effort**: 2-4 hours
   - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.5`

2. **Task 6.3**: REST symbol bootstrap bypasses ProxySettings (BLOCKER)
   - Route `exchange.symbol_mapping` via HTTPSync.read through ProxyInjector
   - Add timeout to sync requests.get calls
   - **Effort**: 4-6 hours
   - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.3`

3. **Task 6.4**: Listen-key generation/refresh bypasses proxies (BLOCKER)
   - Make `_generate_token` / `_refresh_token` async with aiohttp
   - Apply ProxyInjector and timeout to these flows
   - **Effort**: 4-6 hours
   - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.4`

**Wave 1 Exit Criteria**:
- ✅ Preflight proxy initialization works correctly
- ✅ Symbol bootstrap uses proxies and timeouts
- ✅ Listen-key flows use proxies and are async
- ✅ Unit tests prove proxy application on these paths

---

### Wave 2: Requests → aiohttp Migration (Priority: HIGH)
**Duration**: 3-5 days
**Dependencies**: Wave 1 complete (proxy infrastructure fixed)

#### Tasks:
4. **Task 6.6**: Requests → aiohttp migration plan
   - Inventory all production `requests` callsites
   - Classify by impact (Binance, OKX, HTTPSync, schema registry)
   - Define migration approach (prefer aiohttp + ProxyInjector + timeout)
   - **Effort**: 2-3 hours (planning)
   - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.6`

5. **Task 6.7**: Configurable timeouts for symbol bootstrap & listen-key
   - Expose timeout settings via config/env (default 10s)
   - Add unit tests for timeout overrides with proxy still applied
   - **Effort**: 2-3 hours
   - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.7`

6. **Task 6.8a**: OKX REST helper
   - Replace `_get_server_time` requests with aiohttp + ProxyInjector + timeout
   - Add unit test mocking proxy lease and timeout override
   - **Effort**: 3-4 hours
   - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.8a`

7. **Task 6.8b**: Schema registry client
   - Add proxy+timeout config to `cryptofeed/backends/kafka_schema.py`
   - Migrate to aiohttp session with ProxyInjector or add explicit proxy handling
   - Include unit tests for proxy header/auth handling
   - **Effort**: 4-5 hours
   - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.8b`

8. **Task 6.8c**: HTTPSync deprecation/migration
   - Wrap HTTPSync.read/write with proxy+timeout support (using aiohttp)
   - OR mark deprecated and replace symbol/bootstrap callers with async paths
   - Add regression test ensuring proxy application
   - **Effort**: 4-6 hours
   - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.8c`

9. **Task 6.8**: OKX / schema registry / HTTPSync follow-up (META)
   - Validate all sub-tasks (6.8a, 6.8b, 6.8c) complete
   - Ensure documentation covers remaining sync use-cases
   - **Effort**: 1 hour (validation)
   - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.8`

**Wave 2 Exit Criteria**:
- ✅ Migration plan documented with inventory
- ✅ Configurable timeouts implemented and tested
- ✅ OKX, schema registry, HTTPSync migrated or wrapped with proxy support
- ✅ All regression tests pass

---

### Wave 3: Enable Proxy E2E Testing (Priority: HIGH)
**Duration**: 2-3 days
**Dependencies**: Wave 1 + Wave 2 complete (infrastructure ready)

#### Tasks:
10. **Task 6**: Enable proxy-configured E2E runs
    - Load ProxySettings from env (CRYPTOFEED_PROXY_*, nested __)
    - Add opt-in path for proxy-enabled Binance E2E tests
    - Keep metrics disabled, reuse Redpanda/Kafka wiring
    - **Effort**: 3-4 hours
    - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6`

11. **Task 6.1**: Validate proxy/pool resolution
    - Provide test config with Binance HTTP/WS proxies (including pool)
    - Assert proxy resolution via get_proxy_injector() returns configured entries
    - Confirm direct mode when no proxy config present
    - **Effort**: 2-3 hours
    - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.1`

12. **Task 6.2**: Document proxy-enabled runs
    - Add docs/test module notes for running with proxies
    - Provide env examples, pool pattern, python-socks dependency notes
    - Reference spec name and FR7
    - **Effort**: 1-2 hours
    - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.2`

13. **Task 4.3**: Add clear skip conditions for missing Docker/Redpanda/Binance
    - Enhance skip logic for missing docker compose, Redpanda not reachable, Binance timeouts
    - Ensure clear skip messages
    - **Effort**: 1-2 hours
    - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 4.3`

**Wave 3 Exit Criteria**:
- ✅ Proxy-configured E2E tests can run with env vars
- ✅ Proxy/pool resolution validated with tests
- ✅ Documentation complete for proxy-enabled workflow
- ✅ All skip conditions clear and tested

---

### Wave 4: Optional Improvements (Priority: LOW)
**Duration**: 1-2 days (optional)
**Dependencies**: Wave 3 complete (core functionality delivered)

#### Tasks:
14. **Task 6.9**: (Optional) Symbol fetch parallelism
    - Assess startup impact of sequential symbol fetch
    - If needed: add parallel fetch with bounded concurrency (gated by config)
    - Add tests
    - **Effort**: 3-4 hours
    - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.9`

15. **Task 6.10**: Requests removal plan
    - Audit remaining runtime `requests` usages
    - Migrate to aiohttp + ProxyInjector where feasible
    - Drop `requests[socks]` from runtime dependencies
    - Update requirements/setup/docs
    - Add regression tests
    - **Effort**: 4-6 hours
    - **Command**: `/kiro:spec-impl kafka-protobuf-binance-e2e 6.10`

**Wave 4 Exit Criteria**:
- ✅ Symbol fetch parallelism implemented (if beneficial)
- ✅ Requests dependency minimized or removed
- ✅ Documentation updated
- ✅ All regression tests pass

---

## Execution Strategy

### Sequential Wave Execution (Recommended)
Execute waves in order due to dependencies:
```bash
# Wave 1 (Blockers)
/kiro:spec-impl kafka-protobuf-binance-e2e 6.5
/kiro:spec-impl kafka-protobuf-binance-e2e 6.3
/kiro:spec-impl kafka-protobuf-binance-e2e 6.4

# Wave 2 (Migration)
/kiro:spec-impl kafka-protobuf-binance-e2e 6.6
/kiro:spec-impl kafka-protobuf-binance-e2e 6.7
/kiro:spec-impl kafka-protobuf-binance-e2e 6.8a
/kiro:spec-impl kafka-protobuf-binance-e2e 6.8b
/kiro:spec-impl kafka-protobuf-binance-e2e 6.8c
/kiro:spec-impl kafka-protobuf-binance-e2e 6.8

# Wave 3 (Enable Proxy E2E)
/kiro:spec-impl kafka-protobuf-binance-e2e 6
/kiro:spec-impl kafka-protobuf-binance-e2e 6.1
/kiro:spec-impl kafka-protobuf-binance-e2e 6.2
/kiro:spec-impl kafka-protobuf-binance-e2e 4.3

# Wave 4 (Optional)
/kiro:spec-impl kafka-protobuf-binance-e2e 6.9
/kiro:spec-impl kafka-protobuf-binance-e2e 6.10
```

### Parallel Execution (Advanced)
Within Wave 2, tasks 6.8a, 6.8b, 6.8c can be executed in parallel after 6.6 and 6.7 complete.

### Validation Between Waves
After each wave:
1. Run existing E2E test suite: `make test-kafka-binance`
2. Run futures E2E suite: `make test-kafka-binance-futures`
3. Check for regressions: `pytest tests/integration/kafka/ -v`
4. Validate proxy paths: `make test-kafka-binance-mullvad` (requires proxy setup)

---

## Risk Assessment

### High Risk Areas
1. **Async/Sync Boundaries** (Tasks 6.4, 6.8c)
   - Moving sync requests to async aiohttp requires careful event loop handling
   - Mitigation: Thorough unit tests, integration validation

2. **Proxy Pool Behavior** (Tasks 6.3, 6.4, 6.5)
   - Proxy pool selection and rotation needs to work across HTTP/WS
   - Mitigation: Test with real proxy pools (Mullvad), add pool selection tests

3. **Timeout Propagation** (Tasks 6.7, 6.8a, 6.8b, 6.8c)
   - All HTTP calls need consistent timeout handling
   - Mitigation: Centralized timeout configuration, regression tests

### Medium Risk Areas
1. **HTTPSync Legacy** (Task 6.8c)
   - HTTPSync might be used in unexpected places
   - Mitigation: Comprehensive codebase search, deprecation warnings

2. **Schema Registry Client** (Task 6.8b)
   - Schema registry changes could affect Kafka serialization
   - Mitigation: Schema registry unit tests, end-to-end validation

### Low Risk Areas
1. **Documentation** (Task 6.2)
2. **Optional Improvements** (Tasks 6.9, 6.10)

---

## Success Criteria

### Must-Have (MVP for Phase 6 Complete)
- ✅ All Wave 1 tasks complete (blockers fixed)
- ✅ All Wave 2 tasks complete (migration done)
- ✅ All Wave 3 tasks complete (proxy E2E enabled)
- ✅ Existing E2E tests pass (spot + futures)
- ✅ Proxy-enabled E2E tests pass with Mullvad or equivalent
- ✅ No regressions in direct (non-proxy) mode

### Nice-to-Have (Complete Phase 6)
- ✅ Wave 4 tasks complete (parallelism, requests removal)
- ✅ Performance benchmarks show no degradation
- ✅ Documentation updated with proxy workflows

### Phase 6 Sign-Off Criteria
1. All 15 remaining tasks marked complete in tasks.md
2. Test suite passes: `make test-kafka-all`
3. Proxy E2E passes: `make test-kafka-binance-mullvad && make test-kafka-binance-futures-mullvad`
4. Code review approved
5. Spec status updated to "implementation-complete"

---

## Timeline Estimates

### Aggressive (Single Developer, Full-Time)
- Wave 1: 2 days
- Wave 2: 3 days
- Wave 3: 2 days
- Wave 4: 1 day (optional)
- **Total**: 7-8 days

### Realistic (Single Developer, Part-Time)
- Wave 1: 3-4 days
- Wave 2: 5-6 days
- Wave 3: 3-4 days
- Wave 4: 2 days (optional)
- **Total**: 11-14 days

### With Team (2-3 Developers, Parallel Waves)
- Wave 1: 2 days (sequential, blockers)
- Wave 2: 2-3 days (parallel sub-tasks)
- Wave 3: 2 days (after Wave 1+2)
- Wave 4: 1 day (parallel with Wave 3 validation)
- **Total**: 6-7 days

---

## Next Steps

1. **Review this schedule** with stakeholders
2. **Execute Wave 1** (blockers) immediately
3. **Validate after each wave** (no proceed until tests pass)
4. **Update tasks.md** as tasks complete
5. **Final validation** before marking spec complete

---

## Commands Quick Reference

```bash
# Check current status
/kiro:spec-status kafka-protobuf-binance-e2e

# Implement specific task
/kiro:spec-impl kafka-protobuf-binance-e2e <task-number>

# Validate implementation
make test-kafka-binance
make test-kafka-binance-futures
make test-kafka-binance-mullvad

# Check task completion
grep "^- \[x\]" .kiro/specs/kafka-protobuf-binance-e2e/tasks.md | wc -l
grep "^- \[ \]" .kiro/specs/kafka-protobuf-binance-e2e/tasks.md | wc -l
```

---

**Created**: 2025-12-10
**Last Updated**: 2025-12-10
**Status**: Ready for Wave 1 Execution
