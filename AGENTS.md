# Agent Guidelines

## Purpose
- Focus: exchange-native v1 schemas/mappers for Binance, OKX, Bybit, Bitget.
- Goal: accurate native→common mapping with strong tests, specs, and stability.

> See also: CLAUDE.md for a concise, Claude-oriented summary of these guidelines.

## Engineering Principles
1. FRs over NFRs: prioritize functional results and correctness first; non-functional concerns follow once behavior is locked by tests and specs.
2. SOLID, KISS, DRY: small, cohesive changes; simple over clever; de-duplicate via shared helpers and parametrized tests.
3. TDD: write a failing test, implement the smallest change to pass, then refactor safely.
4. Specs-Driven Development: update specs first to manage complexity (especially when “vibe coding”); treat docs/specs as contracts.

- Backward compatible: avoid breaking proto changes; prefer optional additions.
- Safety: preserve `raw_data` and avoid network in unit tests.

## Non-Goals
- Private account events or RPC services in this phase.
- Over-parsing option strikes/expiries (document as deferred when needed).

## Core Invariants (must always hold)
- raw_data: if set in native, equals raw_data in common.
- Instrument type from segment:
  - Binance: SPOT→SPOT, FUTURES_UM→PERPETUAL, FUTURES_CM→FUTURES, OPTIONS→OPTION.
  - OKX: SPOT/FUTURES/SWAP/OPTIONS → respective types.
  - Bybit: SPOT→SPOT, LINEAR/INVERSE→PERPETUAL, OPTIONS→OPTION.
  - Bitget: SPOT→SPOT, USDT_PERP/COIN_PERP→PERPETUAL, OPTIONS→OPTION.
- Sequence number mapping:
  - Binance: `final_update_id` → `sequence_number`.
  - OKX: `seq`/`seq_id` → `sequence_number`.
  - Bybit: `seq` → `sequence_number`.
  - Bitget: `seq`/`seq_id` → `sequence_number`.
- Side mapping:
  - Binance: `is_buyer_maker`: False=BUY, True=SELL.
  - OKX/Bybit/Bitget: text side “buy/sell” → enum; options side nuances documented in specs.

## Agent Prompts (Checklist)
- Update spec(s) first: clarify message fields/invariants/fixtures in `docs/specs/exchange-native/<exchange>.md`.
- TDD start: create/extend a fixture; write a failing test asserting mapping + invariants.
- Implement minimally: change only mapper logic required to pass; reuse; keep diffs small.
- Validate invariants: assert `raw_data`, instrument type, `sequence_number`, symbol parsing, side rules.
- DRY tests: parametrize across exchanges; use shared loader utils.
- No network in tests: pre-populate `Symbols` or use protos/fixtures only.
- Protos discipline: avoid breaking changes; add optional fields if needed; run `buf lint` and `buf generate`; ensure no drift.
- Docs/tasks: update `docs/STEERING.md` and `docs/TASKS.md` to reflect changes and remaining work.

## TDD Workflow
1) Spec: edit `docs/specs/exchange-native/<exchange>.md` with rules and acceptance.
2) Fixture: add under `tests/fixtures/exchange-native/<exchange>`.
3) Test: add targeted test(s) under `tests/exchange_native/<exchange>` and/or extend parametrized suites.
4) Code: adjust mapper under `cryptofeed/proto_mappers/<exchange>.py`.
5) Verify: run changed/related tests first; then broaden.
6) Document: mark tasks complete in `docs/STEERING.md` and `docs/TASKS.md`.

## Mappers: Required Behaviors
- Symbol mapping:
  - Binance/Bybit/Bitget: split concatenated symbols “BTCUSDT” → base=BTC, quote=USDT; `Symbol.symbol` = “BTC-USDT”.
  - OKX: split hyphen symbols “BTC-USDT”.
- Trades: map price/qty (or size) and id; set timestamp if provided; derive side correctly; propagate `raw_data`.
- Ticker/L1: support best bid/ask and bookTicker variants; propagate `raw_data`.
- L2/BookDelta: convert levels; set `sequence_number`; size=0 → deletion; propagate `raw_data`.
- Funding: map `rate`, `mark_price`, `next_funding_time` when present; `raw_data` propagate.

## Testing Strategy
- Fast unit/proto tests only; avoid network and heavy I/O.
- Parametrize:
  - ticker + L2 across exchanges.
  - BookDelta across exchanges, plus Binance DepthUpdate variant.
- Ordering checks: use paired messages/fixtures to assert increasing `sequence_number`.

## Protobuf/Buf Guidelines
- Lint: `buf lint` must pass.
- Generate: `buf generate` → no codegen drift; CI should fail on drift.
- Comments lint: re-enable incrementally as annotations are added (plan in STEERING).

## CI/Tooling
- Add steps to run `buf lint`, `buf generate`, and drift checks.
- Run targeted pytest subsets in PRs for speed.
- Keep fixtures small and representative.

## Docs & Planning
- Keep specs updated with mapping rules and invariants.
- Use `docs/STEERING.md` for milestones and sprint checklists.
- Use `docs/TASKS.md` for actionable items with acceptance.
- Note deferred items (e.g., Binance options) explicitly.

## Security & Secrets
- Never embed real API keys or depend on live endpoints in tests.
- Pre-populate `Symbols` in tests when symbol resolution is needed.

## When in Doubt
- Add a fixture + failing test to express behavior.
- Keep scope surgical; avoid broad refactors unless required by the test.
- Record assumptions in specs and mark TODOs in TASKS with acceptance criteria.

## Useful Paths/Commands
- Specs: `docs/specs/exchange-native/`
- Steering: `docs/STEERING.md`
- Tasks: `docs/TASKS.md`
- Fixtures: `tests/fixtures/exchange-native/<exchange>/`
- Proto mappers: `cryptofeed/proto_mappers/`
- Lint: `buf lint`
- Generate: `buf generate`
- Tests: `pytest -q tests/exchange_native/<area>/...`

## Context Engineering & Agent Behavior
- Preambles: before grouped tool calls, add a 1–2 sentence note stating the immediate next steps.
- Plans: for multi-step work, maintain a live plan (update incrementally) with exactly one in_progress step.
- Output discipline: keep messages concise (≤10 lines by default), using bullets; avoid heavy formatting.
- File I/O: read files in ≤250 line chunks; use `rg` for repo search; avoid dumping large blobs.
- Patches: submit minimal diffs focused on the task; don’t refactor broadly without tests driving it.
- Safety: never leak secrets; avoid network access in unit tests; respect sandbox/approval.
- Reasoning: don’t emit hidden chain-of-thought; surface decisions, invariants, and results succinctly.
