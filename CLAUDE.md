# Agent Guidelines (Claude)

This project uses shared agent guidelines to keep work consistent across tools. Claude should follow the same principles and invariants as other agents.

## Purpose
- Focus: exchange-native v1 schemas/mappers for Binance, OKX, Bybit, Bitget.
- Goal: accurate native→common mapping with strong tests, specs, and stability.

> See also: AGENTS.md for the full, canonical guidelines.

## Engineering Principles
1. FRs over NFRs: ship correct functional behavior first; layer on non-functional refinements after tests/specs are green.
2. SOLID, KISS, DRY: keep changes small, simple, and de-duplicated.
3. TDD: add failing tests before code; implement minimally; refactor with safety.
4. Specs-Driven Development: update specs to constrain scope and reduce complexity during “vibe coding”.

- Avoid breaking proto changes; prefer optional additions.
- Preserve `raw_data`; avoid network in tests.

## Context Engineering (Claude)
- Keep responses concise and structured; prefer short bullets over prose.
- Use a lightweight preamble before grouped tool calls to set immediate next actions.
- Maintain a live plan for multi-step tasks; update it atomically between phases.
- Read files in bounded chunks (≤250 lines); prefer `rg` for fast search.
- Reference files by relative path; avoid pasting large file contents unless requested.
- Make minimal, surgical patches; prefer narrow, verifiable diffs with tests.
- Do not expose hidden chain-of-thought; surface only necessary rationale, invariants, and outcomes.
- Respect sandbox/approval policies; avoid network calls in tests; never embed secrets.

## Core Invariants
- raw_data propagates unchanged from native → common.
- Instrument type derived from segment (see AGENTS.md for per-exchange rules).
- Sequence numbers mapped consistently per exchange (final_update_id/seq/seq_id).
- Side mapping per exchange (e.g., Binance `is_buyer_maker`).

## Workflow (Claude)
1) Update specs in `docs/specs/exchange-native/<exchange>.md`.
2) Add fixture in `tests/fixtures/exchange-native/<exchange>`.
3) Write failing test(s) under `tests/exchange_native/<exchange>` or parametrized suites.
4) Implement minimal mapper changes in `cryptofeed/proto_mappers/<exchange>.py`.
5) Run targeted tests; iterate.
6) Update `docs/STEERING.md` and `docs/TASKS.md`.

Guidance for tool usage
- Patches: use the provided apply_patch mechanism; avoid ad-hoc edits.
- Shell: group related commands with a short preamble; keep output noise low.
- Tests first: write the failing test that expresses the invariant, then fix.

## Testing & CI
- Prefer targeted pytest runs locally; keep fixtures minimal.
- Ensure `buf lint` and `buf generate` pass with no codegen drift.
- Re-enable proto COMMENTS lint incrementally as annotations are added.

## References
- Full shared guidelines: see `AGENTS.md`.
- Specs: `docs/specs/exchange-native/`
- Steering: `docs/STEERING.md`
- Tasks: `docs/TASKS.md`
