## Cryptofeed Schemas Steering (Exchange-Native v1)

Objectives:
- Deliver robust exchange-native v1 coverage for Binance, OKX, Bybit, Bitget first.
- Maintain precise mappings to common `cryptofeed.v1` types with lossless `raw_data`.
- Specs-first with TDD; small, incremental changes; backward-compatible evolution.
- Apply SOLID/KISS/DRY throughout implementation and tests.

Scope (phase 1):
- Channels: trades, ticker, L2 order book; funding where applicable.
- Segments: Spot, USDT/Coin margined futures (or equivalent), Options.

Non-goals (phase 1):
- Private account events; RPC services; complex option strategies.

Design Tenets:
- Fidelity: keep native field names/semantics; do not over-normalize.
- Safety: include `raw_data` for lossless capture.
- Evolvability: new fields as optional; no breaking renames.

Workflow (Specs-Driven TDD):
1) Update exchange-native specs (fields, invariants, mapping rules, fixtures).
2) Adjust protos only if required by spec gaps (avoid churn; keep BC).
3) Write failing tests from fixtures; implement minimal mapper changes to pass.
4) `buf generate` and run tests; iterate until green.
5) DRY: factor shared helpers; parametrize cross-exchange tests.

Milestones & Tasks (Binance/OKX/Bybit/Bitget focus):
- Binance
  - [x] Trade/Ticker/L2/Funding mappers
  - [x] Fixtures + tests: ticker/L2/funding; raw_data propagation
  - [x] BookDelta mapping from DepthUpdate
  - [x] UM aggTrade fixture + test (side via `m`)
  - [x] CM aggTrade fixture + test; full-depth incremental fixture (finalUpdateId sequencing)
  - [x] Options basic fixtures and `InstrumentType.OPTION` assertions (trade + bookTicker)
- OKX
  - [x] Trade/Ticker/L2 mappers + tests; raw_data propagation
  - [x] Funding mapper + tests (swap)
  - [x] Options fixtures (ticker/trade) + tests (symbol type = OPTION)
  - [x] Document side nuance for options (confirm BUY/SELL semantics) + test
  - [x] Incremental OrderBook delta fixture with `seq` sequencing
- Bybit
  - [x] Trade/Ticker/L2 mappers + tests; raw_data propagation
  - [x] Funding mapper + test (linear)
  - [x] Inverse funding fixture + test
  - [x] Snapshot + delta BookDelta fixture to validate zero-size deletion
- Bitget
  - [x] Trade/Ticker/L2 mappers + tests; raw_data propagation
  - [x] Funding mapper + test (USDT perp)
  - [x] COIN perp funding fixture + test
  - [x] Add `seq` sequencing fixture for L2/BookDelta

Validation:
- `buf lint` clean; generated code builds in Python/Go/TS/Rust; `buf generate` produces no drift.
- Unit tests cover trades/ticker/L2 across exchanges; funding where applicable.
- Raw payloads (`raw_data`) preserved through mappers and asserted in tests.
- Sequence numbers mapped consistently: Binance `final_update_id`, OKX `seq`/`seq_id`, Bybit `seq`, Bitget `seq`.
- Options: instrument type set to OPTION; strike/expiry parsing deferred; behavior documented.



Priorities (Next 1–2 cycles):
- Cycle A (Sprint 1)
  - [x] Add raw_data asserts for remaining funding channels.
  - [x] Binance: add CM aggTrade fixture + test; document options coverage status.
  - [x] OKX: add BookDelta incremental fixture + test; confirm options trade side semantics (pending doc/test).
  - [x] Parametrize cross-exchange tests for ticker + L2 + BookDelta; consolidate helpers (DRY).
- Cycle B (Sprint 2)
  - [x] Bybit inverse funding fixture + test; [x] Bitget coin-perp funding fixture + test.
  - [x] Add sequence ordering checks across all four exchanges using fixtures.
  - [x] Re-enable strict proto COMMENTS lint as annotations are added.

Cycle C
- CI: add `buf breaking --against .git#branch=main` gate; fail on breaking changes.
- Options: Binance basic fixtures (Ticker/Trade) added; OKX side semantics covered with tests; docs note exchange-specific option symbol handling.

-Parametrized cross-exchange tests:
- [x] Ticker/L2 basic mapping with raw_data
- [x] BookDelta mapping across OKX/Bybit/Bitget
- [x] Binance UM BookDelta + aggTrade fixture
- [x] Binance CM aggTrade; incremental deltas from public docs

CI Additions:
- [x] Enforce buf lint + generate; fail on codegen drift
- [x] Breaking check against main
- [x] Run lakehouse schema contract tests on proto/lakehouse changes
- [x] Add BSR publish workflow on `schema-v*` tags with BUF_TOKEN auth
