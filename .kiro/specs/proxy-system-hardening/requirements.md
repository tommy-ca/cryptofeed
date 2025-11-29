# Requirements: Proxy System Hardening (Spec: proxy-system-hardening)

## Goals
- Reliable proxy selection with clear precedence (env > explicit argument > config YAML) and validation.
- Resilient leasing with health tracking, retry/fallback, and safe release for HTTP/WS.
- Observability: structured logs and minimal metrics for selection, failures, health, and retries.
- Keep KISS/Start-Small: single retry, simple health checker, no external services.

## Scope
- IN: proxy settings loading/validation; ProxyInjector/ProxyPool behavior; HTTP/WebSocket connection usage; health checking; logging/metrics.
- OUT: external proxy service, HA control planes, advanced circuit breakers, GUI/CLI tools.

## Functional Requirements
1. **Config Precedence**: proxy settings load from env (`CRYPTOFEED_PROXY_*`), then config YAML (`proxy` key), then explicit `proxy_settings` passed to FeedHandler; later sources override earlier.
2. **Validation**: when `proxy.enabled` is true, at least one enabled proxy (default or exchange-specific) must exist; pool strategies must be supported; invalid URLs raise clear errors.
3. **Auth Support**: `ProxyUrlConfig` supports optional username/password and propagates to HTTP (aiohttp) and WS (python-socks) connectors.
4. **Health Checks**: optional periodic health checker marks proxies unhealthy/healthy using `HealthCheckConfig`; integrates with pools.
5. **Retry/Fallback**: on connection failure, mark proxy unhealthy and retry once with a different proxy (if available) for HTTP and WS flows; always release leases on failure.
6. **Observability**: structured logs for lease/select/release, failures, retries, health results; metrics counters for leases, lease_failures, retries, unhealthy_count, health_success/fail.
7. **WS Compatibility**: HTTP/SOCKS proxies validated; unknown schemes error with guidance; python-socks missing → actionable ImportError.

## Non-Functional Requirements
- Minimal overhead: health interval configurable; default healthy path unchanged.
- Backward compatible: legacy `proxy` kwarg on connections continues to work when proxy system disabled.
- Testable: unit/integration coverage for precedence, retry, health, and metrics/logs.

## Success Criteria
- Env > config > explicit precedence proven by tests.
- At least one failing-proxy retry succeeds or surfaces clear error; unhealthy list updated.
- Metrics/log lines emitted for lease/retry/health paths.
- Health checker can mark unhealthy and recover to healthy after passing checks.

## Dependencies
- Existing proxy system modules (`proxy.py`, `proxy_config.py`, `proxy_pool.py`), feedhandler initialization, connection HTTP/WS paths.
- python-socks / aiohttp dependencies already present.

