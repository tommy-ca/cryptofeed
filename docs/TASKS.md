# Exchange-Native v1 Tasks (Binance, OKX, Bybit, Bitget)

Guiding principles: Specs-first, TDD, SOLID/KISS/DRY. Keep diffs small; avoid breaking proto changes.

Cycle A (Sprint 1)
- Binance
  - [x] Add CM aggTrade fixture + test; assert side via `m` and id via `a`.
  - [x] Add incremental depth fixture; assert `finalUpdateId` → sequence_number.
- OKX
  - [x] Add incremental OrderBook delta fixture; assert `seq`/`seq_id` ordering.
  - [ ] Confirm option trade side semantics; document + test.
- Bybit
  - [x] Add snapshot + delta BookDelta fixture; assert size=0 deletion behavior.
- Bitget
  - [x] Add L2 sequencing fixture; assert `seq` mapping.
- Cross-cutting
  - [x] Add explicit `raw_data` assertions to all funding tests.
  - [x] Parametrize ticker/L2/BookDelta tests across exchanges using shared loader.
  - [x] Add Proto CI job (buf lint/generate + proto tests).

Cycle B (Sprint 2)
- Funding
  - [x] Bybit inverse funding fixture + test.
  - [x] Bitget COIN perp funding fixture + test.
- Ordering
  - [x] Add sequence ordering checks for all four exchanges using fixtures.
- Linting
  - [x] Re-enable COMMENTS lint for exchange packages (buf.yaml).

Cycle C (Sprint 3)
- Options coverage
  - [x] Binance: basic options fixtures (bookTicker/trade) + `InstrumentType.OPTION` assertions.
  - [x] OKX: document side nuance for options and add test.
 - Lakehouse SSoT
  - [x] Add schema contract tests for trades/ticker/bookdelta; assert Decimal and Symbol shapes for ETL.
- CI/Quality gates
  - [x] Add `buf breaking --against .git#branch=main` to CI with full git history; fail build on breaking changes.
  - [x] Add lakehouse schema contract CI job to run on proto/lakehouse changes.
  - [x] Add BSR publish workflow triggered by `schema-v*` tags using `BUF_TOKEN`.

Cycle D (Sprint 4) — BSR Integration
- Repo & Auth
  - [ ] Ensure BSR repo exists (buf.build/tommyk/cryptofeed-schemas) and permissions set.
  - [x] Add `BUF_TOKEN` secret in GitHub.
- Process
  - [x] Document release tag policy (`schema-vMAJOR.MINOR.PATCH`).
  - [ ] Dry-run first publish via workflow_dispatch (`dry_run=true`), then tag `schema-vX.Y.Z` and verify CI publish.
- Consumers
  - [x] Document consumer setup (`buf dep add buf.build/tommyk/cryptofeed-schemas`).
  - [ ] Add example consumer snippet in Python/Go/TS.
  - [x] Add size/headers regression check for Kafka wrappers (extend existing test with tighter thresholds per message type).
- Kafka/Protobuf
  - [x] Example: show wiring `value_serializer` in `examples/demo_kafka.py` using `examples/kafka_protobuf_serializer.py`.
  - [x] Document topic/partitioning conventions and headers in Kafka guide (expanded per-channel usage).

Cycle D.1 — BSR Workflows Hardening
- Workflows
  - [x] Add smoke workflow (`bsr-smoke.yml`) to validate BUF_TOKEN + whoami + lint/build/generate.
  - [x] Enhance publish workflow with `workflow_dispatch` inputs: version/dry_run/create/visibility; add whoami step.
  - [ ] Add caching for buf to speed CI (optional).
- Tests
  - [x] Add `tests/proto_integration/test_bsr_whoami.py` (skips without BUF_TOKEN).
  - [x] Add `tests/proto_integration/test_bsr_workflow_config.py` static checks for workflow content.

Cycle E (Sprint 5) — Registry & Packaging
- Mapper Registry
  - [x] Redesign `default_registry()` to avoid package-name conflicts with runtime modules (e.g., `cryptofeed.exchanges.binance` vs generated `cryptofeed.exchanges.binance.v1`).
  - [x] Match on protobuf `DESCRIPTOR.full_name` so tests using file loaders (`_pb2`) resolve; avoid importing pb2s by package path.
  - [x] Acceptance: Example in `docs/USAGE-MAPPERS.md` runs; pytest added: `tests/exchange_native/test_registry_fullname.py`.
- Docs
  - [x] Update `USAGE-MAPPERS.md` to document descriptor-based registry and test helper note.
- Symbol Parsing
  - [x] Extend known quote list (util.KNOWN_QUOTES) with common fiat/stable variants (TRY, BRL, DAI, BIDR, BVND, USDD, USTC, etc.).
  - [x] Add extended test cases (no network): `tests/exchange_native/test_symbol_parsing_extended.py`.
- CI/Release
  - [ ] Add BUF_TOKEN secret and perform a publish dry-run, then a tagged release (`schema-vX.Y.Z`); document outcome in BSR_WORKFLOW.

Acceptance
- `raw_data` preserved on all channels and asserted in tests.
- Correct exchange enums and instrument types from `segment` across all exchanges.
- Sequence numbers mapped consistently and validated by fixtures.
- CI runs: buf lint/build/generate + breaking check; proto/exchange-native tests green.
