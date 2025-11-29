# Tasks: Proxy System Hardening

## Phase 1: Config & Validation
1. Implement env/config/explicit precedence in `load_proxy_settings` and FeedHandler init; add tests for precedence and non-empty proxies when enabled.
2. Extend `ProxyUrlConfig` with username/password and scheme validation; update HTTP/WS paths to use credentials.

## Phase 2: Health Checking
3. Add `ProxyHealthService` periodic checker using `HealthCheckConfig`; mark unhealthy/healthy in ProxyPool; tests with stub checker.

## Phase 3: Retry & Observability
4. Add single-retry-on-failure for HTTP session creation with unhealthy marking/release; structured logs + metrics.
5. Add single-retry-on-failure for WS connect with unhealthy marking/release; structured logs + metrics.
6. Add metrics counters/gauges (leases, lease_failures, retries, unhealthy, health successes/failures) and minimal log formats; tests asserting emissions.

## Phase 4: Documentation
7. Update proxy docs with config examples (env/YAML/explicit), auth, health, retry behavior, metrics names.

