# Kafka Protobuf Binance E2E - Tasks

## Overview

This document enumerates the concrete tasks required to implement the `kafka-protobuf-binance-e2e` specification. Tasks are grouped into exploration, test harness implementation, and validation.

The numbering scheme follows the Kiro convention: top-level integers for major tasks, and dotted suffixes for related sub-tasks.

---

## Phase 1: Exploration and Context Alignment

- [x] 1. Review Kafka Protobuf backend implementation
  - Read `cryptofeed/backends/kafka/protobuf_callback.py` to understand how `KafkaProtobufCallback` configures serialization format, schema version, and headers.
  - Review the relevant pieces of `cryptofeed.backends.kafka` (base, callback, headers, partitioner) to understand topic and partition behavior.
  - Cross-check expectations with `docs/kafka/technical-specification.md`.

- [x] 1.1 Align with existing Kafka Protobuf tests
  - Review `tests/integration/kafka/test_kafka_protobuf_e2e.py` to understand the existing Redpanda-based Protobuf E2E tests.
  - Review `tests/unit/kafka/test_protobuf_backend.py` and related unit tests to capture header and error handling expectations.
  - Summarize key behaviors that Binance E2E tests must preserve (headers, topic naming, error semantics).

- [x] 1.2 Align with schema parity expectations
  - Review `tests/proto_integration/test_schema_parity.py` and related `proto_integration` tests.
  - Capture how `cryptofeed.types` and Protobuf schemas are expected to align for trades and (optionally) order books.
  - Ensure Binance E2E tests will not re-define or contradict parity rules from these specs.

---

## Phase 2: Test Harness Design and Wiring

- [x] 2. Implement Redpanda fixture reuse or sharing
  - Reuse the existing `redpanda` fixture from `tests/integration/kafka/test_kafka_protobuf_e2e.py`, or extract a shared fixture into a Kafka integration conftest module if necessary.
  - Ensure the fixture continues to guard against missing Docker / `docker compose` and skips tests when those prerequisites are not met.

- [x] 2.1 Define Binance + KafkaProtobufCallback wiring pattern
  - Implement a small helper or fixture that constructs:
    - A `KafkaProtobufCallback` configured for a provided bootstrap address, with metrics disabled for tests.
    - A `Binance` feed subscribed to `TRADES` for a small symbol set (e.g., `BTC-USDT`).
    - A `FeedHandler` that holds the Binance feed and wires the callback via the `callbacks` mapping.
  - Ensure the helper uses public configuration surfaces, avoiding invasive changes to production code.

- [x] 2.2 Implement Kafka consumer helper for test assertions
  - Implement a helper (similar to `_consume_one`) that:
    - Creates a `confluent_kafka.Consumer` with appropriate defaults.
    - Subscribes to a specified topic.
    - Polls until a message is available or a timeout elapses.
    - Returns a simple dataclass-like structure with `value`, `headers`, `topic`, and `key`.
  - Ensure headers are normalized into `dict[bytes, bytes]` to match existing tests.

---

## Phase 3: Binance→Kafka Protobuf Integration Tests

- [x] 3. Implement Binance trade roundtrip test (live WS)
  - Add `tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py`.
  - Implement `test_binance_trade_roundtrip_live` that:
    - Uses the Redpanda fixture and Binance + KafkaProtobufCallback wiring from Phase 2.
    - Starts the Binance feed on an event loop for a bounded period while polling Kafka for messages.
    - Consumes a message from the expected per-symbol topic and decodes it with `trade_pb2.Trade` from `cryptofeed.proto_bindings`.
    - Asserts on headers (`content-type`, `exchange`, `symbol`, `data_type`, `schema_version`, `cf.serialization_format`) and on core payload fields.
  - Mark the test with `@pytest.mark.integration`, `@pytest.mark.kafka`, and `@pytest.mark.binance_live`.

- [x] 3.1 Implement round-robin partition key variant
  - In the same test module, add a test (e.g., `test_binance_trade_roundtrip_round_robin_keyless`) that:
    - Configures `KafkaProtobufCallback` to use a round-robin partitioner (`PartitionerFactory.create("round_robin")`).
    - Produces at least one trade and consumes it from Kafka.
    - Asserts that the consumed record key is `None`, confirming round-robin semantics.

- [x] 3.2 (Optional) Implement Binance L2 order book snapshot test
  - Add an optional test (e.g., `test_binance_orderbook_snapshot_roundtrip`) that:
    - Configures Binance for `L2_BOOK` and wires `KafkaProtobufCallback` for book callbacks.
    - Waits for a snapshot and/or first delta to be produced to Kafka.
    - Consumes an order book message and decodes with the relevant Protobuf binding.
    - Asserts that exchange and symbol match and that at least one bid/ask level is present.
  - Mark this test as slow and ensure it has robust skip conditions if REST or WS connectivity fails.

---

## Phase 4: Environment Guards and Stability

- [x] 4. Add environment-based execution guards
  - Introduce an environment variable check (e.g., `CRYPTODATA_RUN_BINANCE_KAFKA_E2E`) in the new test module.
  - Skip Binance E2E tests with a clear message when the variable is not set.
  - Document the required environment variable in the test module docstring and/or Kafka docs.

- [x] 4.1 Harmonize Docker and network skip behavior
  - Ensure the new tests reuse `_docker_compose_available()` or an equivalent check for Docker + `docker compose`.
  - Ensure network-related failures (e.g., connection refused, DNS errors) are handled as skip conditions with informative messages wherever reasonable, rather than as uncaught exceptions.

- [x] 4.2 Add Makefile targets for Redpanda and Kafka tests
  - Introduce Makefile targets to manage the Redpanda lifecycle (`redpanda-up`, `redpanda-down`, `redpanda-health`) using `docker/infra/base.yml` and host port `19092` by default.
  - Add convenience targets for Kafka test entrypoints (`test-kafka-e2e`, `test-kafka-binance`, `test-kafka-unit`, `test-kafka-perf`, `test-kafka-all`) that align with this specs FR1FR6.
  - Ensure all targets fail fast with non-zero exit codes on test failure and are safe to run repeatedly in local development and CI.

- [x] 4.3 Guard port 19092 and conflicting Docker services
  - Define a non-destructive Make target (e.g., `docker-ps-19092`) that lists any containers currently bound to host port 19092 so engineers can inspect conflicts.
  - Define a guarded target (e.g., `docker-stop-19092`) that can stop conflicting containers, documenting in comments that it SHOULD be used only after manual review.
  - Document these targets in the Kafka / Redpanda test documentation to reduce accidental disruption of unrelated services.

- [ ] 4.3 Add clear skip conditions for missing Docker/Redpanda/Binance
  - Ensure skips occur when `docker compose` is unavailable, Redpanda fails to start, or Binance endpoints are unreachable within timeouts.

- [ ] 4.4 Implement topic auto-provision helper (FR8)
  - Create an idempotent helper/fixture that ensures required topics exist based on the configured topic strategy (per_symbol default, consolidated optional).
  - Support configurable partitions/replication via env/kwargs (defaults: partitions=1, replication=1 for local Redpanda).
  - On failure to create topics, skip with a clear message; do not proceed to feed start.

- [ ] 4.5 Wire auto-provision into Binance E2E
  - Invoke the provisioning helper before starting `FeedHandler` and before consumer poll in the Binance trade/L2 tests.
  - Keep behavior isolated to tests; no production Kafka code changes.

- [ ] 4.6 Document proxy and topic setup in test module/README
  - Update test module notes or supporting docs to describe proxy envs, REST proxying via HTTP(S)_PROXY, and the new auto-provision behavior.
  - Include rpk/Kafka admin command examples for manual verification.

---

## Phase 5: Validation and Documentation

- [x] 5. Validate E2E tests locally
  - Run the new Binance E2E tests against a local Redpanda instance with network access to Binance.
  - Verify that:
    - At least one trade roundtrip test passes end-to-end.
    - Header and payload assertions match expectations from existing Kafka Protobuf tests.
    - Skip behavior functions correctly when preconditions are not met.

- [x] 5.1 Integrate with documentation (lightweight)
  - Add a brief reference to the new Binance E2E tests in existing Kafka docs (e.g., test section in `docs/kafka/user-guide.md` or `INTEGRATION_GUIDE.md`), noting how to opt-in to running them.
  - Ensure the spec name (`kafka-protobuf-binance-e2e`) is mentioned so readers can trace behavior back to this specification.

- [x] 5.2 Run Binance E2E suite via Makefile and Redpanda
  - Start Redpanda using `make redpanda-up` and validate cluster health with `make redpanda-health`, skipping tests if the cluster is not reachable.
  - Execute `make test-kafka-e2e`, `make test-kafka-binance`, and `make test-kafka-unit` as the primary validation entrypoints for this spec.
  - Optionally run `make test-kafka-perf` when performance validation is in scope, treating it as non-blocking for functional readiness.

- [x] 5.3 Collect failing tests and promote them to traceable issues/tasks

### Validation Notes (2025-11-30)
- First attempt: import failure from missing `cryptofeed.backends.kafka.maintenance.doc_updater` (fixed by adding lightweight stubs).
- Second attempt (with stubs + live run): all 3 tests failed at runtime.
  - Kafka backend start now works, but consume_one raised `TypeError: must be real number, not NoneType` after Redpanda connection errors (connect refused / ApiVersion). Logs also showed `AttributeError: write` during backend shutdown (KafkaBackendBase missing write attribute when callback invoked while connections tearing down).
  - Redpanda was up via `make redpanda-up`; failures likely due to producer init/teardown and backend API mismatch (KafkaBackendBase.start signature vs FeedHandler multiprocess call handled via shim; remaining issue is backend `write` attr / producer connection handling).
  - Third attempt: cleaned header normalization paths; tests skipped due to header names arriving as byte-literal strings.
  - Fourth attempt: consumer-side header normalization added; live suite PASSED (3/3) with Redpanda up and env flag set.
  - Command: `CRYPTODATA_RUN_BINANCE_KAFKA_E2E=true KAFKA_BOOTSTRAP_SERVERS=localhost:19092 python -m pytest tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py -v -rs`
  - Result: 3 passed; only deprecation warnings from legacy kafka import path remain (non-blocking).

### Validation Notes (2025-12-06) — Review Findings
- Proxy gaps detected along Binance → Kafka path:
  - Symbol bootstrap (`exchange.symbol_mapping` via `HTTPSync.read`) uses sync `requests.get` without ProxyInjector or timeout; geoblocked regions hang and bypass configured pools.
  - User-data listen-key acquire/refresh (`_generate_token` / `_refresh_token`) uses sync `requests.post/put` without proxy/timeout, blocking the event loop and ignoring ProxySettings.
  - Test preflight `_preflight_rest_through_proxy` checks `get_proxy_injector()` before calling `init_proxy_system`, so proxy-configured runs still go direct and may skip.
- These gaps block FR7 until remediated; tracked via Tasks 6.3–6.5 in this spec, with fixes to be coordinated with the owning proxy/connector specs.

### Validation Notes (2025-12-07) — Post-migration review
- Implemented aiohttp + ProxyInjector + timeouts for symbol bootstrap and listen-key flows; added thread-based runner for symbol fetch when loop is running.
- Added unit coverage for proxy application on symbol mapping and listen-key paths.
- Preflight now initializes ProxySettings before leasing proxy.
- Remaining open items:
  - Symbol/listen-key timeouts hardcoded (10s); consider config exposure.
  - Other requests callsites (OKX, HTTPSync generic, schema registry) still pending per Task 6.6.
  - Symbol fetch is still sequential; acceptable for FR7 but perf not improved.

---

## Phase 6: Proxy-Aware Execution (FR7)

- [ ] 6. Enable proxy-configured E2E runs
  - Load `ProxySettings` from env (`CRYPTOFEED_PROXY_*`, nested `__`) in the Binance Kafka E2E harness; ensure precedence env > YAML > programmatic remains intact and direct mode still works by default.
  - Add opt-in path to run the existing Binance E2E tests with Binance HTTP/WS proxy settings applied; skip with a clear message when proxies are configured but `python-socks` is missing for SOCKS WS.
  - Keep metrics disabled and reuse existing Redpanda/Kafka wiring to avoid production code changes.

- [ ] 6.1 Validate proxy/pool resolution
  - Provide a test configuration (env or fixture) that sets Binance HTTP/WS proxies, including a pool example (e.g., `...__POOL__PROXIES__0__URL`).
  - Assert proxy resolution via `get_proxy_injector()` (HTTP and WS) returns configured entries; ensure pool selection does not crash and returns at least one proxy.
  - Confirm that when no proxy config is present, tests run direct and prior assertions remain unchanged.

- [ ] 6.2 Document proxy-enabled runs
  - Add brief docs or test module notes showing how to run the Binance Kafka E2E suite with proxies (env examples, pool pattern, dependency on `python-socks` for SOCKS WS).
  - Reference spec name (`kafka-protobuf-binance-e2e`) and FR7 in the doc note so operators can trace behavior.

- [ ] 6.3 Blocker — REST symbol bootstrap bypasses ProxySettings
  - Current path uses `HTTPSync.read` → `requests.get` without proxy or timeout, so Binance `exchangeInfo` geoblock causes hangs and violates FR7.
  - Coordinate with proxy-system-complete/connector owners to route symbol mapping through ProxyInjector (or explicitly set `HTTP[S]_PROXY` from leased proxy in tests as a stopgap).

- [ ] 6.4 Blocker — listen-key generation/refresh bypasses proxies and is synchronous
  - `_generate_token` / `_refresh_token` call `requests.post/put` without proxy or timeout, blocking the event loop and ignoring configured pools.
  - Track remediation with owning connector spec; interim mitigation is to document/skip private-channel runs when proxies are required.

- [ ] 6.5 Fix proxy preflight helper
  - `_preflight_rest_through_proxy` checks `get_proxy_injector()` before calling `init_proxy_system`, so proxy-configured runs still go direct and may skip; initialize first, then lease HTTP proxy and propagate to `HTTP[S]_PROXY`.

- [ ] 6.6 Requests → aiohttp migration plan (proxy-sensitive REST/auth paths)
  - Inventory production `requests` callsites (Binance symbol bootstrap + listen-key, OKX REST helper, HTTPSync.read/write, schema-registry client) and classify by impact to Binance → Kafka E2E.
  - Define migration approach: prefer aiohttp-based async clients with ProxyInjector + timeout support; where sync calls must remain, require explicit proxy + timeout injection and document rationale.
  - Add tests proving proxy application and timeout enforcement on migrated paths, focusing on Binance symbol bootstrap and listen-key flows.

- [ ] 6.7 Configurable timeouts for symbol bootstrap & listen-key
  - Expose timeout settings (default 10s) via config or env for symbol mapping and Binance listen-key HTTP calls.
  - Add unit tests asserting overrides are honored and proxy is still applied.

- [ ] 6.8 OKX / schema registry / HTTPSync follow-up
  - Migrate OKX REST helper and schema registry client off `requests` or add proxy+timeout plumbing with justification.
  - Evaluate generic `HTTPSync` usages; either deprecate in favor of aiohttp paths or ensure ProxyInjector + timeout support and document remaining sync use-cases.
  - Add coverage for proxy+timeout on these paths or document exclusions.

- [ ] 6.8a OKX REST helper
  - Replace `_get_server_time` requests call with aiohttp + ProxyInjector + timeout; add unit test mocking proxy lease and timeout override.

- [ ] 6.8b Schema registry client
  - Add proxy+timeout configuration (env/Config) to `cryptofeed/backends/kafka_schema.py` HTTP calls or migrate to aiohttp session with ProxyInjector; include unit tests for proxy header/auth handling.

- [ ] 6.8c HTTPSync deprecation/migration
  - Either wrap HTTPSync.read/write with proxy+timeout support (using aiohttp) or mark deprecated and replace symbol/bootstrap callers with async paths; add regression test ensuring proxy application when legacy path is used.

- [ ] 6.9 (Optional) Symbol fetch parallelism
  - Assess whether sequential symbol fetch impacts startup; if so, add optional parallel fetch with bounded concurrency and tests, gated behind config.

- [ ] 6.10 Requests removal plan
  - Audit remaining runtime `requests` usages and migrate to aiohttp + ProxyInjector where feasible; retain only in optional tooling if needed.
  - Ensure SOCKS paths rely on python-socks/aiohttp_socks; drop `requests[socks]` dependency from runtime.
  - Update requirements/setup/docs to reflect the reduced `requests` footprint and proxy/SOCKS readiness; add regression tests for migrated paths.

---

## Phase C: Governance & Spec Hygiene

- [x] C.1 Document compound workstreams and dependencies for this spec
  - Ensure the Requirements and Design documents explicitly list upstream specs (Kafka backend, Protobuf serialization, normalized schema, parity tooling) and describe which parts of the end-to-end pipeline this spec owns.
  - Confirm that topic naming, partitioning, and schema semantics are treated as *inputs* from those specs, not redefined here.

- [x] C.2 Document AI agent boundaries for this spec
  - Clarify in Requirements and Design which files and modules AI agents may modify when working under this spec (tests, fixtures, runbook notes) and which are owned by other specs (core backends, schemas, connectors).
  - Add explicit guidance that cross-spec changes require referencing and updating the owning spec before implementation.

- [x] C.3 Clarify ownership of Redpanda test harness and Makefile targets
  - State in Requirements and Design that the Redpanda Docker configuration (`docker/infra/base.yml`) and Kafka E2E Makefile targets are part of this specs validation harness, in collaboration with `market-data-kafka-producer`.
  - Clarify that changes to shared Kafka backend behavior, Protobuf serialization, or normalized schemas MUST be made under their owning specs, and that this spec only consumes those contracts via configuration and tests.
  - Ensure AI agents and human contributors treat Makefile and test harness changes that affect multiple specs as cross-spec context requiring explicit coordination.

## Traceability

- Phase 1 tasks map primarily to FR1, FR4, and the dependency alignment requirements.
- Phase 2 tasks map to FR1, FR3, FR6.
- Phase 3 tasks map to FR1, FR2 (optional), FR3, FR4.
- Phase 4 tasks map to FR5, NFR1.
- Phase 5 tasks map to NFR2 and NFR3.
- Phase C tasks map to compound engineering alignment and AI agentic implementation constraints for this spec.
