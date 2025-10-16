# Implementation Plan

- [ ] 1. Build unified schema inventory and adjudication workflow
  - Implement extractors that pull field metadata from Cryptofeed dataclasses (canonical source) for priority market data events (trades, order books, funding, ticker, NBBO) alongside tardis-node JSON exports and DBN fixed layouts into a normalized comparison matrix. Account/portfolio dataclasses are added in the subsequent phase.
  - Flag conflicts and coverage gaps with severity scoring; document adjudication decisions in `docs/schemas/inventory/DECISIONS.md` within two business days of discovery, explicitly recording any deviation from the Cryptofeed definition.
  - Automate validation that new fields are represented in the inventory before downstream generation steps run.
  - _Requirements: R1.1, R1.2, R1.3_

- [ ] 1.1 Automate schema inventory reporting
  - Provide a CLI (`tools/schema_inventory.py`) that emits CSV/Markdown summaries, freshness timestamps, and conflict status for CI dashboards.
  - Surface alerts when inventory entries exceed the seven-day freshness SLA or when unresolved conflicts remain past the five-day adjudication window.
  - _Requirements: R1.3_

- [ ] 2. Scaffold Buf module structure for canonical Protobuf schemas
  - Create `buf.yaml`, `buf.gen.yaml`, and module directory layout under `proto/cryptofeed/normalized/v1/` reflecting the canonical event set.
  - Generate `.proto` files from the inventory with deterministic field numbering and documented Decimal scaling guidance.
  - Ensure `buf format` and `buf lint` pass locally and in CI.
  - _Requirements: R2.1, R2.2, R2.3_

- [ ] 2.1 Integrate Buf breaking-change enforcement and generation
  - Wire `buf breaking --against` into CI using the last published module as a baseline; fail builds on incompatible changes.
  - Configure `buf generate` targets (e.g., Python, Go) for consumers and publish bindings as build artifacts.
  - _Requirements: R2.3, R2.4_

- [ ] 3. Align tardis-node JSON schemas and DBN fixed layouts with canonical Protobuf
  - Annotate tardis-node schemas with Buf field numbers and type expectations for market data events first; update DBN layout docs to map byte offsets to Protobuf message fields derived from the Cryptofeed dataclasses, noting complementary metadata each source contributes. Plan follow-up tasks for account-level datasets once market data parity is complete.
  - Produce parity mapping tables and regression fixtures demonstrating equivalence across dataclass, tardis-node, DBN, and Protobuf representations.
  - _Requirements: R4.1, R4.2_

- [ ] 3.1 Implement parity and throughput regression pipeline
  - Build automated replays that encode tardis-node/DBN samples into Protobuf messages and verify field-level parity and precision.
  - Capture throughput benchmarks for serialization/deserialization and publish reports for release readiness.
  - _Requirements: R4.3_

- [ ] 4. Publish Buf modules to the Buf Schema Registry (BSR)
  - Configure authenticated Buf CLI workflows, semantic version tagging, and release note templates.
  - Push release candidates to a staging namespace, run final lint/breaking/parity checks, and promote to production on approval.
  - _Requirements: R3.1, R3.2, R3.3_

- [ ] 4.1 Establish governance dashboards and feedback loop
  - Surface Buf module metrics (downloads, dependents, versions) in a shared dashboard; integrate consumer issue tracking with SLA monitoring.
  - Document escalation paths and resolution statuses for schema enhancement requests.
  - _Requirements: R3.4, R5.2, R5.3_

- [ ] 5. Deliver documentation and migration tooling
  - Update `docs/schemas/` with migration guides, changelog entries, FAQs, and Buf module usage examples.
  - Provide sample configuration toggles or adapters allowing services to select schema versions during rollout.
  - _Requirements: R5.1, R5.4_
