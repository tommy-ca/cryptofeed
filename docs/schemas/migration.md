# Migration Guide

This guide outlines how to transition services to new versions of the
`buf.build/cryptofeed/normalized-data` module while referencing Cryptofeed
dataclasses as the canonical schema.

## Prerequisites

- Cryptofeed version that includes the updated dataclass definitions.
- Buf CLI logged in (`buf registry whoami`).
- Inventory and regression reports regenerated (`tools/schema_inventory.py`,
  `tools/schema_regression.py`).

## Upgrade Steps

1. **Review Change Log**
   - Check release notes and `docs/schemas/mappings/` for field additions,
     renamed enums, or precision changes.

2. **Fetch Latest Protobuf Module**
   ```bash
   buf registry module pull buf.build/tommyk/crypto-market-data:vX.Y.Z
   ```

3. **Update Consumer Configuration**
   ```yaml
schema:
  source: buf.build/tommyk/crypto-market-data
     version: vX.Y.Z
     fallback_version: vX.Y.(Z-1)
     output_modes:
       - dataclass
       - protobuf
   ```

4. **Run Parity Tests**
   ```bash
   python tools/schema_regression.py --module proto \
     --events <service-specific-events.jsonl> \
     --output reports/<service>-parity.json
   ```
   - Validate `mismatches` is empty before rollout.

5. **Deploy Behind Feature Flag**
   - Enable Protobuf output for a small percentage of traffic.
   - Monitor parity reports and service health metrics.

6. **Promote to Production**
   - Remove fallback once parity is stable.
   - Archive regression report in `reports/` for audit.

## Rollback Strategy

- Revert configuration to the previous `fallback_version`.
- Regenerate regression reports to confirm parity with the older schema.
- File an issue in the schema backlog with details from the failed release.

## Troubleshooting

- **Missing Fields:** Check `docs/schemas/inventory/inventory.md` for coverage
  status; open a schema request if the field is absent in the canonical
  dataclass.
- **Precision Mismatches:** Ensure DBN scaling factors or tardis-node JSON
  string representations align with the canonical decimal scale (usually 1e-8).
- **Enum Divergence:** Update mappings to reflect Cryptofeed enum values;
  regenerate tardis-node/DBN adapters accordingly.
