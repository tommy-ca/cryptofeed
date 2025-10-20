# Tardis-Node Schema Alignment Plan (Task 7)

**Status**: ⏳ Awaiting External Schemas  
**Version**: v0.2.0  
**Target Release**: When tardis-node schemas available

## Overview

Task 7 aligns tardis-node JSON schema definitions with the canonical Protobuf schemas created in Phase 1. This enables consumers to validate historical data from tardis-node against the same standards as real-time Cryptofeed data.

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Framework** | ✅ Ready | Test infrastructure in place |
| **Directories** | ✅ Ready | `docs/schemas/examples/tardis/` created |
| **Tests** | ✅ Ready | 12 tests created (9 pass, 3 skip on missing schemas) |
| **tardis-node Schemas** | ⏳ Blocked | External dependency - not available in codebase |
| **Mapping Documentation** | ⏳ Pending | Will be generated once schemas available |
| **Regression Tests** | ⏳ Pending | Will be created once samples available |

## Blockers

### External Dependencies
The primary blocker for completing Task 7 is obtaining the tardis-node JSON schema definitions.

**Required schemas** (not currently available):
- Trade JSON schema
- Order Book JSON schema
- Ticker JSON schema
- Funding JSON schema
- NBBO JSON schema
- Other market data event types

**Action**: Obtain tardis-node schemas from:
1. tardis-node GitHub repository
2. tardis-node package documentation
3. tardis-node community resources

## Implementation Plan (When Schemas Available)

### Task 7.1: Obtain and Catalog Schemas

**What to do**:
1. Download tardis-node JSON Schema files
2. Validate schema JSON syntax
3. Store in `docs/schemas/examples/tardis/`
4. Document source version and metadata

**Files to create**:
- `docs/schemas/examples/tardis/metadata.json` - Source and version info
- `docs/schemas/examples/tardis/trade.json` - Trade schema
- `docs/schemas/examples/tardis/order_book.json` - OrderBook schema
- `docs/schemas/examples/tardis/ticker.json` - Ticker schema
- etc. for each event type

**Expected outcome**: All tardis-node schemas catalogued and validated

### Task 7.2: Create Field Mapping Documentation

**What to do**:
1. Compare tardis-node fields against Cryptofeed Protobuf definitions
2. Document field-by-field equivalence
3. Identify complementary fields (tardis-only or Protobuf-only)
4. Generate mapping tables

**Output file**: `docs/schemas/mappings/tardis_alignment.md`

**Expected content**:
```markdown
## Trade Mapping

| Cryptofeed Field | Protobuf Field | tardis-node Field | Type | Notes |
|------------------|--------|-----------|------|-------|
| exchange | exchange | exchange | string | Direct match |
| symbol | symbol | symbol | string | Direct match |
| side | side | side | enum | BUY=1, SELL=2 |
| price | price | price | string | Decimal precision preserved |
| ... | ... | ... | ... | ... |

### Complementary Fields
- tardis-node only: ...
- Protobuf only: ...
```

### Task 7.3: Validate Parity and Release v0.2.0

**What to do**:
1. Create tardis-node sample JSONL files for regression testing
2. Run regression tests through schema validation pipeline
3. Verify field-level equivalence and precision
4. Publish v0.2.0 to BSR with tardis alignment

**Expected test results**:
- All tardis samples validate against Protobuf schemas
- No precision loss in numeric conversions
- All required fields present
- Optional fields handled correctly

## Testing Strategy

### Framework Already in Place
The test framework is ready and waiting for schemas:
- ✅ 12 tests created in `test_tardis_alignment.py`
- ✅ `TardisAlignmentWorkflow` helper class ready
- ✅ Readiness assessment framework ready
- ✅ Directory structure created

### When Schemas Arrive
Once tardis-node schemas are obtained:
1. Place JSON files in `docs/schemas/examples/tardis/`
2. Existing tests will automatically detect and validate them
3. Generate mapping documentation
4. Run parity regression tests
5. Create v0.2.0 release

## Timeline

**Phase**: Post-Phase 1 (After v0.1.0 released)  
**Duration**: ~1-2 weeks once schemas obtained  
**Effort**: ~5-10 days focused work

## Success Criteria

- [ ] tardis-node schemas obtained and catalogued
- [ ] All schemas validate JSON syntax
- [ ] Field mapping documentation complete
- [ ] Regression tests all pass
- [ ] Zero precision loss in conversions
- [ ] v0.2.0 released to BSR
- [ ] Consumers can validate tardis-node data against Protobuf

## What's Ready Now

✅ **Framework**: Complete test and infrastructure framework  
✅ **Directories**: All directories created and ready  
✅ **Tests**: 12 comprehensive tests ready to run  
✅ **Documentation**: Templates ready for content  
✅ **Tools**: Regression pipeline ready for validation  

## What's Blocked

⏳ **External Schemas**: tardis-node JSON schema files needed  
⏳ **Mapping Content**: Depends on schema availability  
⏳ **Parity Tests**: Depends on sample data from tardis  

## Next Steps

**For v0.2.0 to proceed**:
1. Obtain tardis-node JSON schemas
2. Place in `docs/schemas/examples/tardis/`
3. Re-run tests (will auto-detect schemas)
4. Complete 7.2 mapping generation
5. Complete 7.3 parity validation and release

**For now**:
- Framework is ready and waiting
- Tests will automatically validate once schemas available
- All infrastructure in place for efficient completion

---

**Status**: Framework complete. Awaiting external dependencies to proceed.

**Contact**: For tardis-node schema access, consult:
- tardis-node documentation: https://docs.tardis.dev
- tardis-node GitHub: https://github.com/tardis-dev/tardis-node
- Community channels for schema availability

