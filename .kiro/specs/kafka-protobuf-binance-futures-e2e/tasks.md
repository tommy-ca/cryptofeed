## kafka-protobuf-binance-futures-e2e – Tasks

This tasks document breaks down the work needed to implement the Binance USDⓈ-M futures Kafka Protobuf E2E suite under this spec. It assumes the underlying Kafka backend, Protobuf schemas, and Redpanda harness are already implemented and validated by upstream specs.

---

### Phase 3A – Harness & Helpers

- [x] 3.1 Reuse Redpanda fixture and Kafka consumer helper
  - Confirm the existing Redpanda fixture used by Kafka E2E tests can be reused for futures tests without modification.
  - Confirm the Kafka consumer helper (`consume_one`-style) works for futures topics and returns a record structure with `value`, `headers`, `topic`, and `key`.

- [x] 3.2 Add futures E2E test module scaffolding
  - Add `tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py` mirroring the structure of the spot Binance E2E module.
  - Implement helpers:
    - `_topic_strategy()` reading and normalizing `KAFKA_E2E_TOPIC_STRATEGY`.
    - `_topic_name(channel, strategy)` mapping futures channels and strategies to topic names.
    - `_start_binance_futures(...)` wiring `KafkaProtobufCallback` to `BINANCE_FUTURES` with metrics disabled and configurable partition strategy.
    - `_assert_headers(record, data_type)` enforcing futures header contracts.

---

### Phase 3B – Futures E2E Tests

- [x] 3.3 TRADES roundtrip
  - Implement a test that:
    - Starts a `BINANCE_FUTURES` feed for `TRADES` on `BTC-USDT-PERP`.
    - Uses the default partitioner (composite strategy).
    - Ensures topics are provisioned via the shared auto-provision helper.
    - Consumes one record from the expected topic with `offset_reset="latest"`.
    - Decodes the payload as `trade_pb2.Trade` and asserts headers + non-trivial fields.

- [x] 3.4 TRADES round-robin roundtrip
  - Implement a test that:
    - Configures `KafkaProtobufCallback` with `PartitionerFactory.create("round_robin")` and disables the partition-key cache.
    - Produces at least one futures trade message.
    - Consumes with `offset_reset="latest"` to focus on the current run.
    - Asserts that `record.key is None`, confirming round-robin behavior.

- [x] 3.5 L2_BOOK snapshot roundtrip
  - Implement a futures order book test that:
    - Subscribes to `L2_BOOK` for `BTC-USDT-PERP`.
    - Uses `offset_reset="earliest"` when consuming from Kafka to capture the snapshot.
    - Decodes the payload as `order_book_pb2.Level2Book` and asserts exchange, symbol, and presence of bids/asks.

- [x] 3.6 TICKER roundtrip
  - Implement a futures ticker test that:
    - Subscribes to `TICKER`.
    - Uses `offset_reset="latest"`.
    - Decodes payload as `ticker_pb2.Ticker` and asserts core fields.

- [x] 3.7 FUNDING roundtrip
  - Implement a futures funding test that:
    - Subscribes to `FUNDING`.
    - Uses `offset_reset="latest"`.
    - Decodes payload as `funding_pb2.Funding` and asserts mark price and/or rate.

- [x] 3.8 OPEN_INTEREST roundtrip
  - Implement a futures open interest test that:
    - Subscribes to `OPEN_INTEREST` via the Binance Futures REST polling endpoint (`/fapi/v1/openInterest?symbol={}`) used by the connector.
    - Uses `offset_reset="latest"` on the Kafka consumer.
    - Decodes the payload as `open_interest_pb2.OpenInterest` and asserts non-trivial open interest when a message is received.
    - Skips with a clear reason (e.g. `"Binance Futures open_interest: no message within timeout: ..."`) if no open interest snapshot is observed within the configured timeout window, reflecting the polled/REST semantics.

- [x] 3.9 LIQUIDATIONS roundtrip
  - Implement a futures liquidations test that:
    - Subscribes to `LIQUIDATIONS`.
    - Uses `offset_reset="latest"`.
    - Decodes payload as `liquidation_pb2.Liquidation` and asserts non-zero quantity and price.

- [x] 3.10 Multi-channel futures roundtrip
  - Implement a futures multi-channel test that:
    - Subscribes to `TRADES`, `L2_BOOK`, and `TICKER` simultaneously.
    - Ensures each channel’s messages are produced to the correct topic.
    - Uses `offset_reset="latest"` for all consumed messages.
    - Asserts headers and decoded payloads per channel.

---

### Phase 3C – Env Gating, Proxies & Makefile

- [x] 3.11 Env gating & skip behavior
  - Add an environment gate `CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E` that controls whether futures E2E tests run.
  - Ensure tests skip with clear messages when:
    - The gate is not enabled.
    - `docker compose` or the Redpanda compose file is missing.
    - Redpanda broker is unreachable.
    - Binance endpoints are consistently unreachable within configured timeouts.

- [x] 3.12 Proxy sanity & connectivity tests
  - Add proxy sanity tests that:
    - Validate futures proxy pool configuration for `BINANCE_FUTURES` (HTTP/WS) when present.
    - Lease a websocket proxy and assert the URL has a valid scheme.
    - Skip when no futures-specific pools are configured.
  - Add connectivity tests that:
    - Use a REST preflight helper to call a Binance futures endpoint via proxy.
    - Start a `BINANCE_FUTURES` WS feed and assert at least one trade message arrives via proxy.
    - Skip with clear reasons on proxy/geoblock failures.

- [x] 3.13 Makefile targets and integration
  - Add/update Makefile targets:
    - `test-kafka-binance-futures` – run the futures Kafka Protobuf E2E pytest module with the gate and `KAFKA_BOOTSTRAP_SERVERS` wired.
    - `test-kafka-binance-futures-mullvad` – run the same suite via Mullvad HTTP/WS pools for `BINANCE_FUTURES`, defaulting `KAFKA_E2E_TOPIC_STRATEGY=consolidated`.
    - `test-kafka-all` – include `test-kafka-binance-futures` alongside existing Kafka suites.
  - Ensure these targets are safe to run repeatedly and propagate non-zero exit codes on failure.

---

### Phase 5 – Validation & Documentation

- [x] 5.1 Validate futures E2E suite locally
  - With Redpanda up and Binance reachable, run the futures E2E tests and confirm all pass, apart from expected proxy-sanity skips.
  - Capture any flaky behavior and feed back into task refinements.

- [x] 5.2 Align futures documentation with spot E2E
  - Ensure this spec, and any Kafka docs that reference it, clearly describe how the futures E2E suite mirrors the spot Binance E2E suite.
  - Highlight similarities and differences in channels, symbols, and offset semantics.

- [x] 5.3 Update validation notes in this spec
  - After successful runs, add a short validation note summarizing:
    - Commands executed (e.g., Makefile targets, direct pytest invocations).
    - High-level results (tests passed/skipped, notable warnings).
    - Any remaining limitations or follow-ups.

---

### Validation Workflow via Kiro Commands

To validate both the design and the implementation of this spec using the Kiro command set defined under `.claude/commands/kiro`:

1. **Pre-flight status**
   - `kiro:spec-status kafka-protobuf-binance-futures-e2e` – confirm requirements, design, and tasks are present and the spec is ready for implementation/validation.

2. **Design-level validation**
   - `kiro:validate-gap kafka-protobuf-binance-futures-e2e` – ensure this futures E2E spec does not redefine or contradict upstream specs (`market-data-kafka-producer`, `protobuf-callback-serialization`, `normalized-data-schema-crypto`, `kafka-protobuf-binance-e2e`).
   - `kiro:validate-design kafka-protobuf-binance-futures-e2e` – verify the design’s data flow, topic/partition semantics, offset strategy, proxy behavior, and Makefile integration are coherent and aligned with requirements.

3. **Implementation validation (task-level, TDD)**
   - Use `kiro:spec-impl kafka-protobuf-binance-futures-e2e <task-ids>` to implement and validate tasks in strict TDD mode, e.g.:
     - Harness & helpers: `3.1`, `3.2`.
     - Futures E2E tests: `3.3`–`3.10`.
     - Env gating, proxies, Makefile: `3.11`–`3.13`.
     - Validation & documentation: `5.1`–`5.3`.

4. **Final implementation check and status**
   - `kiro:validate-impl kafka-protobuf-binance-futures-e2e` – confirm all completed tasks have corresponding, passing code/tests and match the spec.
   - `kiro:spec-status kafka-protobuf-binance-futures-e2e` – verify the spec shows all tasks complete and implementation validated.

---

### Validation Notes

- 2025-12-08: `python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -v` executed locally; in a fully wired Redpanda + Binance environment this suite previously yielded 10 passed and 2 skipped (proxy sanity), and in this environment all 12 tests cleanly skipped when futures E2E gating or connectivity prerequisites were not satisfied.

- 2025-12-08: Full Binance USDⓈ-M futures E2E run via Mullvad relays executed with:
  - `make redpanda-up`
  - `make test-kafka-binance-futures-mullvad`
  - `make redpanda-down`
  - Result: 11 tests passed, 1 skipped (OPEN_INTEREST roundtrip – expected data dependency), 1 warning about an ignored `GeneratorExit` in `ConnectionHandler` during teardown; no header/payload mismatches or Kafka/Protobuf errors observed.

- 2025-12-08: Targeted diagnostics for `OPEN_INTEREST` skips executed with:
  - `make redpanda-up`
  - `CRYPTOFEED_PROXY_ENABLED=true CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__POOL='{"proxies":[{"url":"socks5://at-vie-wg-socks5-001.relays.mullvad.net:1080"},{"url":"socks5://be-bru-wg-socks5-101.relays.mullvad.net:1080"},{"url":"socks5://hk-hkg-wg-socks5-201.relays.mullvad.net:1080"}],"strategy":"round_robin"}' CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__POOL='{"proxies":[{"url":"socks5://at-vie-wg-socks5-001.relays.mullvad.net:1080"},{"url":"socks5://be-bru-wg-socks5-101.relays.mullvad.net:1080"},{"url":"socks5://hk-hkg-wg-socks5-201.relays.mullvad.net:1080"}],"strategy":"round_robin"}' CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true KAFKA_E2E_TOPIC_STRATEGY=consolidated KAFKA_BOOTSTRAP_SERVERS=localhost:19092 python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -vv -s -k open_interest`
  - `make redpanda-down`
  - Result: test `test_binance_futures_kafka_protobuf_open_interest_roundtrip` consistently **skipped** with reason `"Binance Futures open_interest: no message within timeout: ..."`, indicating a real upstream data dependency (no open interest snapshot delivered within 420s) rather than a Kafka/Protobuf wiring defect; additional unraisable-exception warnings in `ConnectionHandler` observed during teardown but not affecting assertions.

- 2025-12-08: OPEN_INTEREST API research:
  - Confirmed from `cryptofeed/exchanges/binance_futures.py` that open interest is polled via REST at `https://fapi.binance.com/fapi/v1/openInterest?symbol={}` using an `HTTPPoll` with `open_interest_interval` (default 1s) and a fixed delay of 60s, and mapped into the normalized `OpenInterest` dataclass before being sent to the Protobuf backend.
  - Verified Binance USDⓈ-M Futures docs and executed direct probes against the same REST endpoint:
    - `curl 'https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT'` → non-empty response with `openInterest` and `time` fields.
    - `curl 'https://fapi.binance.com/fapi/v1/openInterest?symbol=ETHUSDT'` → non-empty response with `openInterest` and `time` fields.
  - Conclusion: the upstream open interest API is healthy for BTC/ETH USDT perpetuals; the E2E test skip appears to be driven by timing/latency (no message observed on the Kafka topic within 420s in the current harness configuration) rather than endpoint unavailability.

- 2025-12-08: HTTP polling for Binance Futures open interest was updated so that `HTTPPoll` now passes the exchange identifier into `HTTPAsyncConn`, allowing the shared `ProxyInjector` to apply HTTP/SOCKS proxies consistently. With this change, a targeted run of the open interest E2E test:
  - `CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -vv -k open_interest_roundtrip`
  - Result: `test_binance_futures_kafka_protobuf_open_interest_roundtrip` **passed** (1 test selected, 11 deselected) with the expected Protobuf payload and headers; a known `PytestUnraisableExceptionWarning` related to `ConnectionHandler` teardown was observed but does not affect test assertions.

---

### Execution Runbook – Futures E2E via Mullvad Relays

To rerun the full Binance USDⓈ-M futures → Kafka Protobuf → Redpanda E2E flow using Mullvad SOCKS5 relays for both REST and WS traffic:

1. **Start Redpanda broker**
   - From the repository root:
     - `make redpanda-up`

2. **Run Binance Futures E2E via Mullvad**
   - Use the dedicated Makefile target that wires Mullvad pools and consolidated topic strategy by default:
     - `make test-kafka-binance-futures-mullvad`
   - This target:
     - Enables the futures E2E gate if unset: `CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true`.
     - Enables the proxy system: `CRYPTOFEED_PROXY_ENABLED=true`.
     - Configures `CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__POOL` and `__WEBSOCKET__POOL` with Mullvad SOCKS5 relays.
     - Defaults `KAFKA_E2E_TOPIC_STRATEGY=consolidated` if unset.
     - Runs `pytest` against `tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py`.

3. **Interpret results**
   - **PASS**: All futures E2E tests succeed; proxy sanity tests may still skip if optional pools are not fully configured.
   - **SKIP**: Tests skip (rather than fail) when futures gating, Redpanda availability, proxy configuration, or Binance reachability prerequisites are not met; skip messages should distinguish proxy/config issues from exchange outages.
   - **FAIL**: Indicates real defects in REST/WS connectivity, Mullvad/proxy wiring, Kafka headers, or Protobuf payload validation that require investigation.

4. **Teardown Redpanda**
   - When finished, stop the Redpanda stack cleanly:
     - `make redpanda-down`

---

### Diagnostic Workflow – OPEN_INTEREST Skips

To review and diagnose why the `OPEN_INTEREST` futures E2E test skips:

1. **Inspect test skip conditions**
   - Open `tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py` and locate `test_binance_futures_kafka_protobuf_open_interest_roundtrip`.
   - Enumerate all code paths that call `pytest.skip(...)` within or around this test (e.g., env gating, REST/WS preflight failures, missing data within timeout).

2. **Reproduce and capture skip reason**
   - Start Redpanda: `make redpanda-up`.
   - Run only the open interest test through the same Mullvad proxy configuration as the full E2E suite to see the exact skip message and logs, for example:
     - `CRYPTOFEED_PROXY_ENABLED=true CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__POOL='{"proxies":[{"url":"socks5://at-vie-wg-socks5-001.relays.mullvad.net:1080"},{"url":"socks5://be-bru-wg-socks5-101.relays.mullvad.net:1080"},{"url":"socks5://hk-hkg-wg-socks5-201.relays.mullvad.net:1080"}],"strategy":"round_robin"}' CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__POOL='{"proxies":[{"url":"socks5://at-vie-wg-socks5-001.relays.mullvad.net:1080"},{"url":"socks5://be-bru-wg-socks5-101.relays.mullvad.net:1080"},{"url":"socks5://hk-hkg-wg-socks5-201.relays.mullvad.net:1080"}],"strategy":"round_robin"}' CRYPTODATA_RUN_BINANCE_FUTURES_KAFKA_E2E=true KAFKA_E2E_TOPIC_STRATEGY=consolidated KAFKA_BOOTSTRAP_SERVERS=localhost:19092 python -m pytest tests/integration/kafka/test_binance_futures_kafka_protobuf_pipeline.py -vv -s -k open_interest`
   - Record the precise skip reason (e.g., "no open interest data received" vs. REST/proxy issues).

3. **Review connector and harness wiring**
   - Confirm that `_start_binance_futures` wires the `OPEN_INTEREST` channel correctly (symbols, callbacks, channels list).
   - Confirm `_topic_name(OPEN_INTEREST, strategy)` maps to the expected topic and that the Kafka consumer in the test subscribes to that topic with `offset_reset="latest"` and the expected group id.

4. **Differentiate data dependency vs harness defect**
   - If the skip reason indicates "no data", run a small one-off check (outside Kafka) against the Binance USDⓈ-M open interest source (through the same Mullvad proxy) to verify whether live data is actually emitted for the test symbols.
   - If the endpoint is healthy but the test still skips, investigate harness/connector issues (e.g., missing callback, symbol mapping) and adjust as needed.

5. **Teardown and document findings**
   - After diagnostics, stop Redpanda with `make redpanda-down`.
   - Capture conclusions in this spec’s Validation Notes (e.g., whether the skip is a legitimate external data dependency or indicates a fixable harness bug).

---

### Research Workflow – Binance Futures OPEN_INTEREST API

To understand and stabilize the `OPEN_INTEREST` futures E2E behavior, use this research workflow:

1. **Map current usage in Cryptofeed**
   - Search the codebase for `OPEN_INTEREST` / `open_interest` in:
     - Binance futures exchange connector implementation(s).
     - Any REST helpers or polling jobs.
     - The futures E2E test harness.
   - Record the exact REST/WS endpoints, request parameters, and how responses are mapped into the normalized `OpenInterest` protobuf.

2. **Review official Binance USDⓈ-M Futures documentation**
   - In the Binance Futures API docs, locate and study any endpoints/streams that expose open interest, such as `/fapi/v1/openInterest` or `/futures/data/openInterestHist`.
   - Capture required parameters, update cadence (snapshot vs history), rate limits, and any symbol/interval constraints relevant to `BTCUSDT` / `ETHUSDT` perpetuals.

3. **Probe the live API directly**
   - Using the same Mullvad proxy setup as the E2E tests (or direct connectivity when proxies are disabled), execute a few ad-hoc requests against the identified open interest endpoints for `BTCUSDT` and `ETHUSDT`.
   - Observe whether responses are non-empty, latency characteristics, and any error codes (429/5xx, symbol not supported, etc.).

4. **Cross-check mapping to normalized schema**
   - Compare Binance raw fields (e.g. `openInterest`, `symbol`, `time`) with:
     - The normalized open interest dataclass.
     - The `open_interest_pb2.OpenInterest` protobuf fields and the assertions in the futures E2E test.
   - Verify types, units, and that important fields are not discarded.

5. **Diagnose root cause of test timeouts**
   - Based on the above, determine whether E2E timeouts are due to:
     - An endpoint that rarely updates or is symbol-limited.
     - Polling cadence / test duration being too aggressive.
     - Connector/harness wiring issues (wrong endpoint, missing callbacks, incorrect symbol mapping).

6. **Propose follow-up changes**
   - Summarize options in this spec’s Validation Notes or a separate follow-up spec, such as:
     - Switching to a more reliable open interest endpoint or adding a fallback.
     - Adjusting polling cadence or test timeouts.
     - Relaxing test expectations (e.g. documented `xfail`/skip) if Binance’s behavior is inherently sporadic.

---

### Proxy Review Workflow – OPEN_INTEREST and SOCKS5

To verify whether the Binance Futures `OPEN_INTEREST` endpoints are using HTTP/WS with SOCKS5 proxies:

1. **Confirm REST vs WebSocket usage for OPEN_INTEREST**
   - Re-inspect `cryptofeed/exchanges/binance_futures.py` to confirm that `OPEN_INTEREST` is fetched via REST polling (`HTTPPoll`) using `rest_endpoints[0].route('open_interest', ...)` and that there is no dedicated WebSocket channel for open interest.
   - Note the `proxy=self.http_proxy` argument passed into `HTTPPoll` for `OPEN_INTEREST`.

2. **Trace HTTP proxy resolution and SOCKS5 support**
   - Follow how `self.http_proxy` is set from the proxy system (`cryptofeed/proxy.py`):
     - Verify that `ProxySettings` and `get_proxy_injector()` return the `CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__HTTP__POOL` URLs (e.g. `socks5://...`) for HTTP requests.
   - Inspect the `HTTPPoll` / HTTP connection implementation to determine how the `proxy` argument is used:
     - Confirm whether `socks5://` URLs are supported via `python-socks` / `aiohttp_socks` or only plain `http://` proxies are fully supported.

3. **Trace WebSocket proxy behavior for completeness**
   - Review WebSocket connection code (e.g. `cryptofeed/connection.py` / `WebsocketEndpoint`) and how `ws_proxy` from `ProxySettings` is applied.
   - Confirm that when `CRYPTOFEED_PROXY_EXCHANGES__BINANCE_FUTURES__WEBSOCKET__POOL` contains `socks5://...` URLs and `python-socks` is installed, futures WS channels (TRADES, L2_BOOK, etc.) actually use SOCKS5.

4. **Runtime verification under Mullvad configuration**
   - Enable debug logging for proxy and connection components.
   - Run a Mullvad-backed futures E2E suite (or a targeted OPEN_INTEREST run) and inspect logs to confirm that:
     - REST open interest requests are issued through a `socks5://...` proxy URL (or, if not, understand any downgrade/fallback behavior).
     - WS connections use the configured SOCKS5 pools.

5. **Document findings and follow-ups**
   - Summarize proxy behavior for `OPEN_INTEREST` in this spec’s Validation Notes:
     - Whether REST polling is truly SOCKS5-aware or limited to HTTP proxies.
     - Whether WS channels reliably use SOCKS5.
   - If gaps are found (e.g., `HTTPPoll` only supports HTTP proxies), propose follow-up work (new spec or tasks) to improve REST proxy handling or documentation.


---

### E2E Results Documentation & Cleanup Plan (docs/e2e/results)

This section captures how Binance futures Kafka Protobuf E2E execution logs are documented and how the temporary `docs/e2e/results/` directory should be handled long term.

1. **Canonical E2E documentation lives under `docs/e2e/`**
   - Durable, user-facing guidance for E2E setup and execution (Quick Start, env and proxy configuration, test phases, commands, success criteria) is consolidated into:
     - `docs/e2e/README.md` – high-level guide and quick start.
     - `docs/e2e/TEST_PLAN.md` – detailed scenarios and gates.
     - `docs/e2e/REPRODUCIBILITY.md` – uv/lockfile and reproducibility guidance.
   - These files are the **spec-aligned source of truth** for how to run and interpret both spot and futures Kafka Protobuf E2E suites.

2. **Futures E2E execution reports are archived under `docs/e2e/results/`**
   - Individual Binance futures runs are recorded as timestamped markdown reports, for example:
     - `docs/e2e/results/2025-12-08-binance-futures-kafka-mullvad.md` – full USDⓈ-M futures E2E via Mullvad relays.
   - `docs/e2e/results/README.md` provides a simple index of these per-run reports.
   - These files are treated as **historical execution logs**, not specifications; they may include verbose per-test output, logs, and environment snapshots.

3. **Results directory is explicitly marked as temporary scratchpad**
   - `docs/e2e/README.md` now documents `docs/e2e/results/` as:
     - "Historical, run-specific execution reports; treated as a temporary scratchpad that can be pruned once key guidance has been folded back into this directory."
   - All reusable guidance discovered during futures E2E work (proxy patterns, Makefile usage, env examples, interpretation of passes/skips) has been folded into the core E2E docs listed above and/or this spec’s Validation Notes.

4. **Cleanup plan for `docs/e2e/results/`**
   - Once stakeholders are comfortable that no additional, evergreen guidance is hiding in the per-run futures (or spot) reports, it is safe to:
     - Delete some or all of `docs/e2e/results/` in a follow-up cleanup PR, or
     - Move any remaining high-signal reports into a more permanent archive location if desired.
   - This spec assumes that **removing `docs/e2e/results/` does not change any functional behavior** of the Binance futures Kafka Protobuf E2E suite; it only removes historical logs.

5. **Traceability and future runs**
   - Future Binance futures E2E runs MAY continue to drop markdown reports into `docs/e2e/results/` for auditability during active work.
   - When a new pattern or lesson emerges (e.g., a proxy/geoblock edge case), it SHOULD be promoted into:
     - This spec (Requirements/Design/Validation Notes), and/or
     - The canonical E2E docs under `docs/e2e/`.
   - This keeps `docs/e2e/results/` as an optional, disposable layer while ensuring the spec and core docs remain aligned with how the futures E2E suite is actually run.