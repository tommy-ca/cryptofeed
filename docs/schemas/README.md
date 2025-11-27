# Normalized Schema Publishing Guide

This directory documents the workflow for maintaining canonical cryptocurrency
schemas using Cryptofeed dataclasses as the source of truth and Buf-managed
Protobuf modules for distribution.

## Directory Layout

- `inventory/` – JSON/Markdown comparison matrix generated via
  `python tools/schema_inventory.py [--tardis <path>] [--dbn <path>]`.
- `mappings/` – Field-level mapping tables linking Cryptofeed dataclasses to
  tardis-node JSON schemas and DBN layouts:
  - `trade_mapping.md`
  - `order_book_mapping.md`
  - `funding_mapping.md`
  - `ticker_mapping.md`
  - `nbbo_mapping.md`
- `examples/` – Sample tardis-node, DBN, and unified event payloads for parity
  regression (e.g., `trades.jsonl`, `nbbo.jsonl`).

## Working with the Buf Module

The canonical module lives under `proto/cryptofeed/normalized/v1/` with module
name `buf.build/tommyk/crypto-market-data` (replace the owner slug with your BSR
organization if different).

```bash
# Format and lint
buf format -w
buf lint

# Run breaking checks against the last published version
BUF_BREAKING_BASE=buf.build/tommyk/crypto-market-data:main buf breaking --against "$BUF_BREAKING_BASE"

# Generate language bindings (configured in buf.gen.yaml)
buf generate
```

## Publishing Workflow

1. Update message definitions as needed (see `mappings/` for field guidance).
2. Run inventory/regression tooling:
   ```bash
   python tools/schema_inventory.py --tardis <path> --dbn <path>
   python tools/schema_regression.py --module proto --events docs/schemas/examples/events/trades.jsonl \
     --output docs/schemas/examples/events/trade_regression_report.json
   ```
3. Review generated reports for mismatches or missing fields.
4. Execute `tools/buf_publish.sh <semver>` to lint, run breaking checks, and
   push to the Buf Schema Registry.
5. Update documentation (`docs/schemas/migration.md`, changelog, FAQs) and
   announce availability.

## Governance & Feedback

- Buf module metrics (downloads, dependents, versions) can be retrieved via
  `buf registry module get buf.build/tommyk/crypto-market-data --format json`
  for dashboards.
- Schema enhancement requests should be logged in the schema backlog with SLAs
  documented in `docs/schemas/governance.md`.
- Breaking changes require a major version bump and migration guidance tying
  fields back to Cryptofeed dataclasses.

## Migration Support

Downstream services may need to opt into specific schema versions. Suggested
configuration snippet:

```yaml
schema:
  source: buf.build/tommyk/crypto-market-data
  version: v1.2.0
  output_modes:
    - dataclass
    - protobuf
```

Consumers should validate parity using the regression script before promoting a
new version to production.

## Recent Changes

See `docs/schemas/CHANGELOG.md` for a detailed log. Highlights for the upcoming
v3.2.0 schema refresh (v2beta1):
- Trade now carries optional venue parity fields: `maker`, `event_time`,
  `match_id`, and `liquidity_flag`.
- Order book adds optional `event_time` and `last_update_id` for exchanges that
  publish them (e.g., Binance).
All additions are optional and wire-compatible; the decimal scale remains 1e-8
and `timestamp` continues to represent match time, while `event_time` captures
venue event timestamps when supplied.
