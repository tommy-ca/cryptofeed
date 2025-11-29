# Design: Proxy System Hardening

## Approach (Start Small)
- Keep existing ProxyInjector/ProxyPool; add small extensions for health, retry, auth, metrics.
- Single retry on failure to avoid complexity; no circuit breaker.
- Health checker: periodic async task owned by ProxyInjector; uses TCP/HTTP ping via existing TCPHealthChecker; marks (un)healthy in ProxyPool.
- Config loading: implement env/config/explicit precedence in `load_proxy_settings` + FeedHandler normalization; validate non-empty proxies when enabled.

## Components & Changes
1) Config Loader & Validation
- `load_proxy_settings`: build ProxySettings from env; keep existing pydantic env support; add fallback to YAML/default; merge precedence.
- Validation: raise if enabled with zero proxies; validate URLs and strategy values; support username/password on ProxyUrlConfig and propagate.

2) Health Checker
- New `ProxyHealthService` in `proxy.py` started when settings.health.enabled.
- Runs every `interval_seconds`: for each ProxyPool proxy, run TCPHealthChecker; mark unhealthy/healthy accordingly.
- Expose hooks for tests to inject a stub health checker.

3) Retry/Fallback on Failure
- HTTP: on session creation failure, mark proxy unhealthy, release, retry once with another proxy (if available); log and metric.
- WS: on connect failure after leasing, mark unhealthy, release, retry once with another proxy; ensure release in finally.

4) Observability
- Structured logs: lease/select/release, retry, health result (transport, exchange, proxy_url, status, reason).
- Metrics (minimal counters/gauges): leases_total, lease_failures_total, retries_total, unhealthy_proxies, health_success_total, health_failure_total.

5) Auth & Scheme Validation
- ProxyUrlConfig: add username/password optional; support http/https/socks4/socks4a/socks5/socks5h; error on unknown scheme with guidance.

## Testing Strategy
- Unit: precedence merge; validation errors; auth propagation; pool unhealthy fallback selection; retry-on-failure picks different proxy.
- Integration (mocked sockets): HTTP session with proxy + retry; WS connect with python-socks mocked; health checker marks/unmarks.
- Metrics/log assertions: verify counters/log entries emitted on lease/retry/health.

## Out of Scope
- External proxy manager, advanced circuit breaker, multi-region HA, GUI/CLI.

