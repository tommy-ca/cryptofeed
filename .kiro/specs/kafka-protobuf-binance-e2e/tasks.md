# Kafka Protobuf Binance E2E - Tasks

## Overview

This document enumerates the concrete tasks required to implement the `kafka-protobuf-binance-e2e` specification. Tasks are grouped into exploration, test harness implementation, and validation.

The numbering scheme follows the Kiro convention: top-level integers for major tasks, and dotted suffixes for related sub-tasks.

---

## Phase 1: Exploration and Context Alignment

- [ ] 1. Review Kafka Protobuf backend implementation
  - Read `cryptofeed/backends/kafka/protobuf_callback.py` to understand how `KafkaProtobufCallback` configures serialization format, schema version, and headers.
  - Review the relevant pieces of `cryptofeed.backends.kafka` (base, callback, headers, partitioner) to understand topic and partition behavior.
  - Cross-check expectations with `docs/kafka/technical-specification.md`.

- [ ] 1.1 Align with existing Kafka Protobuf tests
  - Review `tests/integration/kafka/test_kafka_protobuf_e2e.py` to understand the existing Redpanda-based Protobuf E2E tests.
  - Review `tests/unit/kafka/test_protobuf_backend.py` and related unit tests to capture header and error handling expectations.
  - Summarize key behaviors that Binance E2E tests must preserve (headers, topic naming, error semantics).

- [ ] 1.2 Align with schema parity expectations
  - Review `tests/proto_integration/test_schema_parity.py` and related `proto_integration` tests.
  - Capture how `cryptofeed.types` and Protobuf schemas are expected to align for trades and (optionally) order books.
  - Ensure Binance E2E tests will not re-define or contradict parity rules from these specs.

---

## Phase 2: Test Harness Design and Wiring

- [ ] 2. Implement Redpanda fixture reuse or sharing
  - Reuse the existing `redpanda` fixture from `tests/integration/kafka/test_kafka_protobuf_e2e.py`, or extract a shared fixture into a Kafka integration conftest module if necessary.
  - Ensure the fixture continues to guard against missing Docker / `docker compose` and skips tests when those prerequisites are not met.

- [ ] 2.1 Define Binance + KafkaProtobufCallback wiring pattern
  - Implement a small helper or fixture that constructs:
    - A `KafkaProtobufCallback` configured for a provided bootstrap address, with metrics disabled for tests.
    - A `Binance` feed subscribed to `TRADES` for a small symbol set (e.g., `BTC-USDT`).
    - A `FeedHandler` that holds the Binance feed and wires the callback via the `callbacks` mapping.
  - Ensure the helper uses public configuration surfaces, avoiding invasive changes to production code.

- [ ] 2.2 Implement Kafka consumer helper for test assertions
  - Implement a helper (similar to `_consume_one`) that:
    - Creates a `confluent_kafka.Consumer` with appropriate defaults.
    - Subscribes to a specified topic.
    - Polls until a message is available or a timeout elapses.
    - Returns a simple dataclass-like structure with `value`, `headers`, `topic`, and `key`.
  - Ensure headers are normalized into `dict[bytes, bytes]` to match existing tests.

---

## Phase 3: Binance→Kafka Protobuf Integration Tests

- [ ] 3. Implement Binance trade roundtrip test (live WS)
  - Add `tests/integration/kafka/test_binance_kafka_protobuf_pipeline.py`.
  - Implement `test_binance_trade_roundtrip_live` that:
    - Uses the Redpanda fixture and Binance + KafkaProtobufCallback wiring from Phase 2.
    - Starts the Binance feed on an event loop for a bounded period while polling Kafka for messages.
    - Consumes a message from the expected per-symbol topic and decodes it with `trade_pb2.Trade` from `cryptofeed.proto_bindings`.
    - Asserts on headers (`content-type`, `exchange`, `symbol`, `data_type`, `schema_version`, `cf.serialization_format`) and on core payload fields.
  - Mark the test with `@pytest.mark.integration`, `@pytest.mark.kafka`, and `@pytest.mark.binance_live`.

- [ ] 3.1 Implement round-robin partition key variant
  - In the same test module, add a test (e.g., `test_binance_trade_roundtrip_round_robin_keyless`) that:
    - Configures `KafkaProtobufCallback` to use a round-robin partitioner (`PartitionerFactory.create("round_robin")`).
    - Produces at least one trade and consumes it from Kafka.
    - Asserts that the consumed record key is `None`, confirming round-robin semantics.

- [ ] 3.2 (Optional) Implement Binance L2 order book snapshot test
  - Add an optional test (e.g., `test_binance_orderbook_snapshot_roundtrip`) that:
    - Configures Binance for `L2_BOOK` and wires `KafkaProtobufCallback` for book callbacks.
    - Waits for a snapshot and/or first delta to be produced to Kafka.
    - Consumes an order book message and decodes with the relevant Protobuf binding.
    - Asserts that exchange and symbol match and that at least one bid/ask level is present.
  - Mark this test as slow and ensure it has robust skip conditions if REST or WS connectivity fails.

---

## Phase 4: Environment Guards and Stability

- [ ] 4. Add environment-based execution guards
  - Introduce an environment variable check (e.g., `CRYPTODATA_RUN_BINANCE_KAFKA_E2E`) in the new test module.
  - Skip Binance E2E tests with a clear message when the variable is not set.
  - Document the required environment variable in the test module docstring and/or Kafka docs.

- [ ] 4.1 Harmonize Docker and network skip behavior
  - Ensure the new tests reuse `_docker_compose_available()` or an equivalent check for Docker + `docker compose`.
  - Ensure network-related failures (e.g., connection refused, DNS errors) are handled as skip conditions with informative messages wherever reasonable, rather than as uncaught exceptions.

- [ ] 4.2 Add Makefile targets for Redpanda and Kafka tests
  - Introduce Makefile targets to manage the Redpanda lifecycle (`redpanda-up`, `redpanda-down`, `redpanda-health`) using `docker/infra/base.yml` and host port `19092` by default.
  - Add convenience targets for Kafka test entrypoints (`test-kafka-e2e`, `test-kafka-binance`, `test-kafka-unit`, `test-kafka-perf`, `test-kafka-all`) that align with this specs FR1FR6.
  - Ensure all targets fail fast with non-zero exit codes on test failure and are safe to run repeatedly in local development and CI.

- [ ] 4.3 Guard port 19092 and conflicting Docker services
  - Define a non-destructive Make target (e.g., `docker-ps-19092`) that lists any containers currently bound to host port 19092 so engineers can inspect conflicts.
  - Define a guarded target (e.g., `docker-stop-19092`) that can stop conflicting containers, documenting in comments that it SHOULD be used only after manual review.
  - Document these targets in the Kafka / Redpanda test documentation to reduce accidental disruption of unrelated services.

---

## Phase 5: Validation and Documentation

- [ ] 5. Validate E2E tests locally
  - Run the new Binance E2E tests against a local Redpanda instance with network access to Binance.
  - Verify that:
    - At least one trade roundtrip test passes end-to-end.
    - Header and payload assertions match expectations from existing Kafka Protobuf tests.
    - Skip behavior functions correctly when preconditions are not met.

- [ ] 5.1 Integrate with documentation (lightweight)
  - Add a brief reference to the new Binance E2E tests in existing Kafka docs (e.g., test section in `docs/kafka/user-guide.md` or `INTEGRATION_GUIDE.md`), noting how to opt-in to running them.
  - Ensure the spec name (`kafka-protobuf-binance-e2e`) is mentioned so readers can trace behavior back to this specification.

- [ ] 5.2 Run Binance E2E suite via Makefile and Redpanda
  - Start Redpanda using `make redpanda-up` and validate cluster health with `make redpanda-health`, skipping tests if the cluster is not reachable.
  - Execute `make test-kafka-e2e`, `make test-kafka-binance`, and `make test-kafka-unit` as the primary validation entrypoints for this spec.
  - Optionally run `make test-kafka-perf` when performance validation is in scope, treating it as non-blocking for functional readiness.

- [ ] 5.3 Collect failing tests and promote them to traceable issues/tasks
  - Capture the list of failing tests (module path + test name) from the Makefile-driven runs for this spec and related Kafka specs.
  - For each distinct failing test, create a traceable task or issue summarizing the failure mode (e.g., connection error to `localhost:19092`, header assertion mismatch, Protobuf decode error) and link it back to this spec and any upstream owning spec.
  - Re-run the relevant Make targets after fixes to confirm stability, marking the associated tasks as completed when tests pass reliably.

---

## Phase C: Governance & Spec Hygiene

- [ ] C.1 Document compound workstreams and dependencies for this spec
  - Ensure the Requirements and Design documents explicitly list upstream specs (Kafka backend, Protobuf serialization, normalized schema, parity tooling) and describe which parts of the end-to-end pipeline this spec owns.
  - Confirm that topic naming, partitioning, and schema semantics are treated as *inputs* from those specs, not redefined here.

- [ ] C.2 Document AI agent boundaries for this spec
  - Clarify in Requirements and Design which files and modules AI agents may modify when working under this spec (tests, fixtures, runbook notes) and which are owned by other specs (core backends, schemas, connectors).
  - Add explicit guidance that cross-spec changes require referencing and updating the owning spec before implementation.

- [ ] C.3 Clarify ownership of Redpanda test harness and Makefile targets
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
