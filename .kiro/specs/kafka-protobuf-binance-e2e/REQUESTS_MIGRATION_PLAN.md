# Requests → aiohttp Migration Plan

**Spec**: `kafka-protobuf-binance-e2e`
**Task**: 6.6
**Date**: 2025-12-10
**Status**: Complete — Wave 1 migrations done, Wave 2 planned

---

## Executive Summary

This document provides a comprehensive inventory of all production `requests` library usage in the Cryptofeed codebase and defines a migration strategy to async `aiohttp` with ProxyInjector integration and timeout enforcement.

### Key Achievements (Wave 1 — Complete)

✅ **Binance Symbol Bootstrap** — Migrated to aiohttp + ProxyInjector + configurable timeout
✅ **Binance Listen-Key Flows** — Migrated to aiohttp + ProxyInjector + configurable timeout
✅ **Runtime Dependency Removal** — `requests` removed from `_BASE_REQUIREMENTS` in `setup.py`

### Remaining Work (Wave 2)

- Schema registry client (Confluent HTTP calls)
- Developer tooling (`tools/tools.py` — non-production, optional)
- Dependency cleanup (dev/test requirements)

---

## 1. Complete Inventory of Production `requests` Usage

### 1.1 Production Runtime Code

| File | Line(s) | Usage Pattern | Component | Status |
|------|---------|---------------|-----------|--------|
| `cryptofeed/backends/kafka_schema.py` | 399, 464, 508, 558, 599 | `requests.post()`, `requests.get()`, `requests.put()` | Schema Registry (Confluent) | **PENDING** (Task 6.8b) |

#### Details: Schema Registry Client (`kafka_schema.py`)

**Callsites**:
- Line 399: `requests.post()` — Register schema
- Line 464: `requests.get()` — Get schema by ID
- Line 508: `requests.get()` — Get schema by version
- Line 558: `requests.post()` — Check compatibility
- Line 599: `requests.put()` — Set compatibility mode

**Current Behavior**:
- Uses sync `requests` with `HTTPBasicAuth`
- Timeout configured via `_schema_registry_http_settings()` (env `CF_SCHEMA_REGISTRY_TIMEOUT`, default 10s)
- Proxy support via `proxies` dict (env `CF_SCHEMA_REGISTRY_PROXY`)
- **DOES NOT** use ProxyInjector — manual env-based proxy only

**Proxy Sensitivity**: Medium
- Schema registry is typically internal infrastructure (not geoblocked)
- However, cross-region/VPN deployments may require proxy routing
- Authentication (basic auth) must be preserved

**Migration Priority**: Medium
- Not critical for Binance → Kafka E2E (schema registry is optional)
- Required for complete requests removal
- Affects `market-data-kafka-producer` spec when schema registry integration is used

---

### 1.2 Developer Tooling (Non-Production)

| File | Line(s) | Usage Pattern | Component | Status |
|------|---------|---------------|-----------|--------|
| `tools/tools.py` | 65, 73, 81 | `requests.get()` | Symbol list scrapers (CEX, EXX, BitMEX) | **DEFER** (tooling only) |

#### Details: Developer Tooling (`tools/tools.py`)

**Callsites**:
- Line 65: `cex_get_trading_pairs()` — Scrape CEX.io pairs
- Line 73: `exx_get_trading_pairs()` — Scrape EXX pairs
- Line 81: `bitmex_instruments()` — Scrape BitMEX instruments

**Current Behavior**:
- Simple `requests.get()` calls with no timeout or proxy
- Used for one-off symbol list generation during development
- **NOT** part of runtime ingestion pipeline

**Migration Priority**: Low (DEFER)
- These are developer utilities run manually
- No proxy or timeout requirements
- Can remain using `requests` or be migrated opportunistically
- Alternatively, could be migrated to `urllib.request.urlopen` (already used in same file)

**Recommendation**: Mark as tooling-only, accept `requests` for dev scripts, or migrate to `urllib` for consistency

---

### 1.3 Migrated Paths (Wave 1 — Complete)

| Component | Old Callsite | Migration Date | New Implementation |
|-----------|--------------|----------------|-------------------|
| **Binance Symbol Bootstrap** | `HTTPSync.read()` → `requests.get()` | 2025-12-07 | `_fetch_json_via_proxy()` (aiohttp) |
| **Binance Listen-Key (generate)** | `requests.post()` | 2025-12-07 | `_http_request_with_proxy()` (aiohttp) |
| **Binance Listen-Key (refresh)** | `requests.put()` | 2025-12-07 | `_http_request_with_proxy()` (aiohttp) |

**Wave 1 Test Coverage**:
- `tests/unit/test_exchange_symbol_mapping_proxy.py` — 6 tests (symbol bootstrap)
- `tests/unit/test_binance_listenkey_proxy.py` — 6 tests (listen-key flows)
- `tests/unit/test_preflight_proxy_init_order.py` — 5 tests (proxy init)
- `tests/integration/kafka/test_preflight_proxy_integration.py` — 5 tests (E2E proxy)

**Total**: 22 tests added, all passing ✅

---

## 2. Classification and Priority

### 2.1 By Impact to Binance → Kafka E2E

| Category | Components | Impact | Migration Priority |
|----------|------------|--------|-------------------|
| **Critical (Wave 1)** | Binance symbol bootstrap, listen-key | Blocks FR7 (proxy-aware E2E) | ✅ **COMPLETE** |
| **Medium (Wave 2)** | Schema registry client | Optional for E2E, required for requests removal | **HIGH** (Task 6.8b) |
| **Low (Deferred)** | Developer tooling | No runtime impact | **LOW** (defer or accept) |

### 2.2 By Proxy Sensitivity

| Proxy Sensitivity | Components | Reason |
|------------------|------------|--------|
| **Critical** | Binance symbol bootstrap, listen-key | Geoblocked in many regions, requires SOCKS/HTTP proxy pools |
| **Medium** | Schema registry client | Internal infra, may need proxy in cross-region deployments |
| **Low** | Developer tooling | One-off manual scripts, no deployment requirements |

### 2.3 By Operation Type

| HTTP Method | Components | Auth Required | Migration Complexity |
|-------------|------------|---------------|---------------------|
| **GET** | Symbol bootstrap, schema registry (get) | No (symbol); Yes (schema registry, basic auth) | Medium |
| **POST** | Listen-key generate, schema registry (register, check) | Yes (HMAC for Binance; basic auth for registry) | Medium-High |
| **PUT** | Listen-key refresh, schema registry (set mode) | Yes (HMAC for Binance; basic auth for registry) | Medium-High |

---

## 3. Migration Approach

### 3.1 Preferred Pattern: Async aiohttp + ProxyInjector

**Advantages**:
- Consistent with existing Cryptofeed async architecture
- Reuses ProxyInjector for pool-aware proxy selection
- Supports HTTP/HTTPS/SOCKS proxies via `aiohttp`/`aiohttp_socks`
- Configurable timeouts via env (`CF_*_TIMEOUT`)
- Non-blocking, integrates cleanly with event loop

**Implementation Template** (from Wave 1):

```python
async def _http_request_with_proxy(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Any] = None,
    timeout: float = 10.0,
    exchange_id: str = "binance",
) -> Dict[str, Any]:
    """Generic async HTTP request helper with ProxyInjector integration."""
    from cryptofeed.proxy import get_proxy_injector
    from aiohttp import ClientSession, ClientTimeout
    from aiohttp_socks import ProxyConnector

    proxy_url = None
    connector = None

    # Lease proxy if configured
    injector = get_proxy_injector()
    if injector:
        proxy_url = injector.get_http_proxy_url(exchange_id)
        if proxy_url and proxy_url.startswith("socks"):
            connector = ProxyConnector.from_url(proxy_url)

    async with ClientSession(
        connector=connector,
        timeout=ClientTimeout(total=timeout)
    ) as session:
        kwargs = {
            "headers": headers or {},
            "proxy": proxy_url if not connector else None,
        }
        if data:
            kwargs["json"] = data

        async with session.request(method, url, **kwargs) as resp:
            resp.raise_for_status()
            return await resp.json()
```

### 3.2 Sync Wrapper Pattern (When Loop is Running)

For callsites that must remain sync (e.g., symbol mapping called during `__init__`):

```python
def _run_async_in_thread(coro):
    """Run async coroutine in a new event loop on a separate thread."""
    import asyncio
    import threading

    result = None
    exception = None

    def runner():
        nonlocal result, exception
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coro)
        except Exception as e:
            exception = e
        finally:
            loop.close()

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if exception:
        raise exception
    return result
```

### 3.3 Justification for Remaining Sync Calls

If a callsite **cannot** be migrated to async:

1. **Document Reason**: Why async migration is not feasible (architectural constraint, third-party library, etc.)
2. **Require Explicit Proxy/Timeout Injection**: Must accept proxy and timeout kwargs, not rely on globals
3. **Add Regression Tests**: Prove proxy application and timeout enforcement

**Example**: If schema registry client must remain sync due to external library constraints, require:

```python
def register_schema_sync(url, schema, proxy=None, timeout=10.0):
    proxies = {"http": proxy, "https": proxy} if proxy else None
    response = requests.post(url, json={"schema": schema}, proxies=proxies, timeout=timeout)
    return response.json()
```

With test:
```python
def test_register_schema_sync_with_proxy():
    proxy_url = "http://proxy.example.com:8080"
    with patch("requests.post") as mock_post:
        register_schema_sync("http://registry", "schema", proxy=proxy_url, timeout=5.0)
        assert mock_post.call_args[1]["proxies"] == {"http": proxy_url, "https": proxy_url}
        assert mock_post.call_args[1]["timeout"] == 5.0
```

---

## 4. Migration Roadmap (Wave 2)

### 4.1 Task 6.8b: Schema Registry Client Migration

**Scope**: `cryptofeed/backends/kafka_schema.py`

**Goal**: Replace sync `requests` calls with async `aiohttp` + ProxyInjector integration

**Implementation Plan**:

1. **Add async methods to `SchemaRegistry` base class**:
   - `async def register_schema_async(...)`
   - `async def get_schema_by_id_async(...)`
   - `async def check_compatibility_async(...)`

2. **Implement in `ConfluentSchemaRegistry`**:
   - Use `aiohttp.ClientSession` with `ClientTimeout(total=timeout)`
   - Integrate ProxyInjector via `get_proxy_injector().get_http_proxy_url('schema_registry')` or generic exchange ID
   - Preserve `HTTPBasicAuth` using `aiohttp.BasicAuth`
   - Support SOCKS via `aiohttp_socks.ProxyConnector` when proxy URL starts with `socks`

3. **Maintain sync compatibility** (if needed):
   - Keep existing sync methods as wrappers around async methods using `asyncio.run()` or thread-based runner
   - Mark sync methods as deprecated with migration timeline

4. **Update timeout/proxy configuration**:
   - Current: `CF_SCHEMA_REGISTRY_TIMEOUT`, `CF_SCHEMA_REGISTRY_PROXY` (manual env)
   - Preferred: Use ProxyInjector for pool-aware proxy selection, retain timeout env for backward compat

5. **Add unit tests** (minimum 5 tests):
   - Schema registration with proxy (HTTP)
   - Schema retrieval with proxy (SOCKS)
   - Compatibility check with timeout enforcement
   - Direct mode (no proxy) regression check
   - Auth header preservation with proxy

**Estimated Effort**: 4-6 hours (including tests)

**Dependencies**: None (self-contained)

**Blocker Status**: Not blocking Binance E2E (schema registry is optional feature)

---

### 4.2 Task 6.8c: HTTPSync Deprecation/Migration

**Scope**: `cryptofeed/connection.py` (if `HTTPSync` class exists and is used beyond Binance)

**Goal**: Audit `HTTPSync` usage, migrate callers to async paths, or deprecate class

**Current Status**: Binance symbol bootstrap (only known user) already migrated off `HTTPSync.read()`

**Implementation Plan**:

1. **Audit remaining `HTTPSync` usage**:
   - Search for `HTTPSync.read()`, `HTTPSync.write()` callsites
   - Classify by exchange and operation

2. **Migration strategy**:
   - **Option A (Preferred)**: Deprecate `HTTPSync`, migrate all callers to async `HTTPAsyncConn` or direct `aiohttp`
   - **Option B**: Wrap `HTTPSync` with ProxyInjector + timeout support, document as legacy

3. **Deprecation path** (if Option A):
   - Add `DeprecationWarning` to `HTTPSync.__init__`
   - Update callers to use async alternatives
   - Remove `HTTPSync` in next major version

4. **Testing**:
   - Add regression tests for any remaining `HTTPSync` paths with proxy + timeout
   - Ensure no silent failures when proxy is configured

**Estimated Effort**: 2-4 hours (depending on remaining usage)

**Dependencies**: Must complete audit first

---

### 4.3 Task 6.7: Configurable Timeouts (Enhancement)

**Scope**: Symbol bootstrap and listen-key timeouts are currently hardcoded (10s default)

**Goal**: Expose timeout configuration via `Config` or env vars

**Implementation Plan**:

1. **Add config fields** (in `cryptofeed.config` or exchange-specific config):
   ```python
   symbol_fetch_timeout: float = Field(default=10.0, description="Timeout for symbol metadata fetch")
   listen_key_timeout: float = Field(default=10.0, description="Timeout for listen-key HTTP calls")
   ```

2. **Update env variable support**:
   - `CF_SYMBOL_FETCH_TIMEOUT` (already exists)
   - `CF_LISTEN_KEY_TIMEOUT` (already exists)

3. **Add unit tests**:
   - Override timeout via env, assert applied to aiohttp calls
   - Ensure proxy is still applied when timeout is overridden

**Estimated Effort**: 1-2 hours

**Dependencies**: None

---

### 4.4 Task 6.10: Requests Removal Plan

**Scope**: Remove `requests` from all dependency files and documentation

**Goal**: Complete migration to aiohttp, drop `requests` as a runtime dependency

**Implementation Plan**:

1. **Verify all production paths migrated**:
   - ✅ Binance symbol bootstrap
   - ✅ Binance listen-key
   - ⏳ Schema registry client (Task 6.8b)
   - ⏸️ Developer tooling (deferred or accepted)

2. **Update dependency files**:
   - ✅ `setup.py`: `requests` already removed from `_BASE_REQUIREMENTS` (line 23 comment confirms)
   - ⏳ `requirements.txt`: Verify no `requests` entry
   - ⏳ `setup.py` extras: Check `kafka` extra doesn't require requests
   - ⏳ Dev/test requirements: Keep `requests` only if needed for test tooling

3. **Update documentation**:
   - Add migration guide: "Cryptofeed 2.5+ uses aiohttp for all HTTP calls; requests is no longer required at runtime"
   - Document proxy configuration for aiohttp paths
   - Note SOCKS support requires `python-socks` + `aiohttp-socks` (already in `[proxy]` extra)

4. **Drop `requests[socks]` from extras**:
   - Current proxy extra: `"aiohttp-socks>=0.9.2", "python-socks>=2.4.3"` (correct)
   - Ensure no lingering `requests[socks]` references

5. **Add regression CI check**:
   - Lint rule or test that fails if `import requests` appears in production modules (exclude `tools/`, `tests/`)

**Estimated Effort**: 2-3 hours (once Task 6.8b complete)

**Dependencies**: Task 6.8b (schema registry migration)

---

## 5. Test Coverage Requirements

### 5.1 Wave 1 Tests (Complete)

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/unit/test_exchange_symbol_mapping_proxy.py` | 6 | Symbol bootstrap with proxy, timeout, direct mode |
| `tests/unit/test_binance_listenkey_proxy.py` | 6 | Listen-key generate/refresh with proxy, SOCKS/HTTP, timeout |
| `tests/unit/test_preflight_proxy_init_order.py` | 5 | Proxy init order, HTTP_PROXY env propagation |
| `tests/integration/kafka/test_preflight_proxy_integration.py` | 5 | E2E proxy validation for spot/futures |

**Total**: 22 tests, all passing ✅

### 5.2 Wave 2 Tests (Required for Task 6.8b)

**Minimum 5 unit tests for schema registry migration**:

1. **Schema registration with HTTP proxy**:
   - Mock ProxyInjector to return HTTP proxy URL
   - Assert `aiohttp` session uses proxy
   - Verify schema ID returned

2. **Schema registration with SOCKS proxy**:
   - Mock ProxyInjector to return SOCKS5 proxy URL
   - Assert `ProxyConnector` created
   - Verify request succeeds

3. **Schema retrieval with timeout enforcement**:
   - Set custom timeout via env (`CF_SCHEMA_REGISTRY_TIMEOUT=5`)
   - Mock slow response
   - Assert `asyncio.TimeoutError` or `ClientTimeout` raised

4. **Direct mode regression (no proxy)**:
   - No proxy configured
   - Assert `aiohttp` session has no proxy
   - Verify behavior unchanged from current implementation

5. **Auth header preservation with proxy**:
   - Configure HTTPBasicAuth (username, password)
   - Use HTTP proxy
   - Assert `Authorization` header present in request

---

## 6. Dependency Cleanup

### 6.1 Current Status

**`setup.py`**:
```python
_BASE_REQUIREMENTS = [
    "aiodns>=1.1",
    "aiofile>=2.0.0",
    "aiohttp>=3.9.5",
    # ...
    # requests no longer required at runtime; aiohttp used for HTTP paths
    "websockets>=14.1",
    # ...
]
```

✅ `requests` already removed from base requirements (line 23 comment confirms intent)

**`requirements.txt`**:
```
aiodns>=1.1
aiohttp>=3.9.5
aiohttp-socks>=0.9.2
python-socks>=2.4.3
# ... no requests entry
```

✅ No `requests` entry found

### 6.2 Remaining Work

1. **Verify test requirements**: Check if `tests/requirements.txt` or `dev` extras need `requests`
2. **Update `setup.py` comment**: Line 23 comment is accurate, keep as documentation
3. **Add linter rule**: Prevent `import requests` in production code (exclude `tools/`, `tests/`)

---

## 7. Success Criteria

### 7.1 Functional Requirements

✅ **FR7 (Proxy-Aware Execution)**: Binance symbol bootstrap and listen-key flows respect ProxySettings and proxy pools
✅ **Timeout Enforcement**: All migrated paths enforce configurable timeouts
✅ **Direct Mode Compatibility**: No regressions when proxies are not configured
✅ **SOCKS Support**: Both HTTP and SOCKS proxies work via `aiohttp`/`aiohttp_socks`

### 7.2 Wave 2 Success Criteria (Task 6.8b)

⏳ **Schema registry client migrated** to aiohttp + ProxyInjector
⏳ **5+ unit tests added** for schema registry proxy/timeout behavior
⏳ **No `requests` imports** in production runtime code (excluding deferred tooling)
⏳ **Documentation updated** with migration guide and proxy configuration examples

---

## 8. Traceability

| Task | Spec Requirement | Component | Status |
|------|------------------|-----------|--------|
| 6.3 | FR7 (proxy-aware REST) | Binance symbol bootstrap | ✅ Complete |
| 6.4 | FR7 (proxy-aware auth) | Binance listen-key | ✅ Complete |
| 6.5 | FR7 (proxy init order) | Preflight helper | ✅ Complete |
| 6.6 | FR7 (migration plan) | This document | ✅ Complete |
| 6.7 | Enhancement | Configurable timeouts | ⏳ Planned |
| 6.8b | Requests removal | Schema registry | ⏳ Planned |
| 6.8c | Requests removal | HTTPSync audit | ⏳ Planned |
| 6.10 | Requests removal | Dependency cleanup | ⏳ Planned (blocked by 6.8b) |

---

## 9. References

- **Spec**: `.kiro/specs/kafka-protobuf-binance-e2e/`
- **Proxy System Spec**: `proxy-system-complete`
- **Kafka Backend Spec**: `market-data-kafka-producer`
- **Wave 1 Implementation**: Commits from 2025-12-07 (Binance symbol/listen-key migrations)
- **Test Coverage**: `tests/unit/test_exchange_symbol_mapping_proxy.py`, `tests/unit/test_binance_listenkey_proxy.py`, etc.

---

## Appendix: Example Migration Diff (Schema Registry)

**Before** (`kafka_schema.py`, line 399):
```python
response = requests.post(
    url,
    json=payload,
    auth=self._auth,
    timeout=self._http_timeout,
    proxies=self._http_proxies,
)
```

**After** (proposed):
```python
from aiohttp import BasicAuth, ClientSession, ClientTimeout
from aiohttp_socks import ProxyConnector
from cryptofeed.proxy import get_proxy_injector

async def register_schema_async(self, subject: str, schema: str, schema_type: str = "PROTOBUF") -> int:
    url = urljoin(self.config.url, f"/subjects/{subject}/versions")
    payload = {"schema": schema, "schemaType": schema_type}

    # Lease proxy if configured
    proxy_url = None
    connector = None
    injector = get_proxy_injector()
    if injector:
        proxy_url = injector.get_http_proxy_url("schema_registry")
        if proxy_url and proxy_url.startswith("socks"):
            connector = ProxyConnector.from_url(proxy_url)

    # Build auth
    auth = None
    if self.config.username and self.config.password:
        auth = BasicAuth(self.config.username, self.config.password)

    async with ClientSession(
        connector=connector,
        timeout=ClientTimeout(total=self._http_timeout),
        auth=auth
    ) as session:
        async with session.post(
            url,
            json=payload,
            proxy=proxy_url if not connector else None
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["id"]

# Sync wrapper for backward compat
def register_schema(self, subject: str, schema: str, schema_type: str = "PROTOBUF") -> int:
    import asyncio
    return asyncio.run(self.register_schema_async(subject, schema, schema_type))
```

---

**End of Migration Plan**
