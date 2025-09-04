## Cryptofeed Schemas Steering (Exchange-Native v1)

Objectives:
- Add native protobuf schemas for Binance, OKX, Bybit, Bitget.
- Maintain a clear mapping to common `cryptofeed.v1` types.
- Deliver with specs-first, TDD, and backward-compatible evolution.
- Adhere to SOLID/KISS/DRY in implementation and tests.

Scope (phase 1):
- Channels: trades, order book (L2), ticker; plus funding where applicable.
- Segments: Spot, USDT/Coin margined futures (or equivalent), Options.

Non-goals (phase 1):
- Private account events; RPC services; complex option strategies.

Design Tenets:
- Fidelity: keep native field names/semantics; do not over-normalize.
- Safety: include `raw_data` for lossless capture.
- Evolvability: new fields as optional; no breaking renames.

Workflow (Specs-Driven TDD):
1) Author/update spec docs per exchange (fields + examples).
2) Add/adjust proto definitions accordingly.
3) Write fixtures and unit tests mapping native -> common.
4) `buf generate`; run tests; iterate until green.
5) Refactor to shared utilities and parametrized tests where useful (DRY).

Milestones & Tasks:
- M1: Binance
  - [x] Implement native->common Trade/Ticker/L1/L2/Funding mappers
  - [x] Add fixtures + tests for ticker/L2/funding; raw_data propagation
  - [x] Add BookDelta mapping from depth updates
  - [ ] Add aggTrade fixture + tests
  - [ ] Add UM/CM depth/funding fixtures; options examples (if available)
- M2: OKX
  - [x] Trade mapper + test
  - [x] Ticker + L2 mappers + tests; raw_data propagation
  - [x] Funding mapper + tests (swap)
  - [x] Options fixtures (ticker/trade) + tests
- M3: Bybit
  - [x] Trade mapper + test
  - [x] Ticker + L2 mappers + tests; raw_data propagation
  - [ ] Funding mapper + tests (linear/inverse, if available)
- M4: Bitget
  - [x] Trade mapper + test
  - [x] Ticker + L2 mappers + tests; raw_data propagation
  - [ ] Funding mapper + tests (if available)

Validation:
- Buf lint passes; generated code builds in Python/Go/TS/Rust.
- Unit tests cover trades + books + tickers across exchanges; funding where applicable.
- Raw payloads (`raw_data`) preserved through mappers.
- Options: instrument type set correctly; no premature parsing of strike/expiry (documented non-goal).



Priorities (Next 1–2 cycles):
- Add OKX options fixtures + tests (ticker first).
- DRY tests further: shared loader adopted; next, parametrize cross-exchange ticker/L2 cases.
- Add raw_data asserts to any remaining channels.
- Incrementally annotate native protos and re-enable COMMENTS lint for exchanges.

Parametrized cross-exchange tests:
- [x] Ticker/L2 basic mapping with raw_data
- [x] BookDelta mapping across OKX/Bybit/Bitget
- [x] Binance UM/CM BookDelta fixtures
- [ ] True incremental delta fixtures from public docs where available
