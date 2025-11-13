# DBN Layout Alignment Plan (Task 8)

**Status**: ⏳ Awaiting External Specifications
**Version**: v1.0.0
**Target Release**: When DBN layout specifications available

## Overview

Task 8 aligns DBN fixed layout specifications with the canonical Protobuf schemas created in Phase 1. This enables consumers to validate historical binary data from DBN against the same standards as real-time Cryptofeed and tardis-node data.

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Framework** | ✅ Ready | Test infrastructure in place |
| **Directories** | ✅ Ready | `docs/schemas/examples/dbn/` created |
| **Tests** | ✅ Ready | 12 tests created (will auto-skip on missing specs) |
| **DBN Specifications** | ⏳ Blocked | External dependency - YAML layout files needed |
| **Mapping Documentation** | ⏳ Pending | Will be generated once specs available |
| **Regression Tests** | ⏳ Pending | Will be created once binary samples available |

## Blockers

### External Dependencies

The primary blocker for completing Task 8 is obtaining the DBN YAML layout specifications.

**Required specifications** (not currently available):
- Trade binary layout specification
- Order Book binary layout specification
- Ticker binary layout specification
- Funding binary layout specification
- OpenInterest binary layout specification
- Liquidation binary layout specification
- Candle binary layout specification
- Other derivative event types

**Action**: Obtain DBN layout specifications from:
1. DBN GitHub repository (databento/dbn)
2. DBN package documentation
3. DBN community resources or API documentation

## Implementation Plan (When Specifications Available)

### Task 8.1: Obtain and Catalog DBN Layout Specifications

**What to do**:
1. Download DBN YAML layout definition files
2. Validate YAML syntax and structure
3. Store in `docs/schemas/examples/dbn/`
4. Document source version and metadata

**Files to create**:
- `docs/schemas/examples/dbn/metadata.json` - Source and version info
- `docs/schemas/examples/dbn/trade.yaml` - Trade layout spec
- `docs/schemas/examples/dbn/order_book.yaml` - OrderBook layout spec
- `docs/schemas/examples/dbn/ticker.yaml` - Ticker layout spec
- `docs/schemas/examples/dbn/funding.yaml` - Funding layout spec
- etc. for each event type

**Expected outcome**: All DBN layout specifications catalogued and validated

### Task 8.2: Create DBN Byte Offset Mapping

**What to do**:
1. Map DBN byte offsets to Protobuf message field paths
2. Document scaling factors and encoding conversions
3. Create comprehensive mapping documentation
4. Document field interdependencies

**Output file**: `docs/schemas/mappings/dbn_alignment.md`

**Expected content**:
```markdown
## Trade Layout Mapping

| DBN Byte Offset | Size | DBN Field | Protobuf Field | Type | Scaling | Notes |
|-----------------|------|-----------|--------|------|---------|-------|
| 0-7 | 8 | ts_event | timestamp | int64 | microseconds | nanoseconds to microseconds |
| 8-15 | 8 | seq | sequence_number | int64 | 1x | Direct mapping |
| 16-23 | 8 | price | price | string | 1e-8 | Fixed-point decimal |
| 24-31 | 8 | size | amount | string | 1e-8 | Fixed-point decimal |
| ... | ... | ... | ... | ... | ... | ... |

### Complementary Fields
- DBN only: ...
- Protobuf only: ...
```

### Task 8.3: Validate DBN Parity and Release v1.0.0

**What to do**:
1. Create DBN binary sample files for regression testing
2. Run regression tests through schema validation pipeline
3. Verify binary encoding equivalence and precision
4. Capture serialization/deserialization throughput benchmarks
5. Publish v1.0.0 to BSR with full alignment

**Expected test results**:
- All DBN samples validate against Protobuf schemas
- No precision loss in numeric conversions
- All required fields present and correctly mapped
- Optional fields handled correctly
- Throughput benchmarks meet performance targets

## Testing Strategy

### Framework Already in Place
The test framework is ready and waiting for specifications:
- ✅ 12 tests created in `test_dbn_alignment.py`
- ✅ `DBNAlignmentWorkflow` helper class ready
- ✅ Readiness assessment framework ready
- ✅ Directory structure created

### When Specifications Arrive
Once DBN layout specifications are obtained:
1. Place YAML files in `docs/schemas/examples/dbn/`
2. Existing tests will automatically detect and validate them
3. Generate byte offset mapping documentation
4. Create binary regression test samples
5. Run parity regression tests
6. Create v1.0.0 release

## Timeline

**Phase**: Post-Phase 2 (After v0.2.0 released with tardis-node alignment)
**Duration**: ~2-3 weeks once specifications obtained
**Effort**: ~10-15 days focused work

## Success Criteria

- [ ] DBN layout specifications obtained and catalogued
- [ ] All specifications validate YAML syntax
- [ ] Byte offset to Protobuf field mappings complete
- [ ] Regression tests all pass
- [ ] Zero precision loss in conversions
- [ ] Throughput benchmarks captured (serialization/deserialization)
- [ ] v1.0.0 released to BSR
- [ ] Consumers can validate DBN binary data against Protobuf

## Scope: DBN Events Supported

### Market Data Events
- **Trade**: Individual trade execution with side, amount, price
- **Ticker**: NBBO ticker quote with bid/ask prices
- **OrderBook**: L2/L3 order book snapshots
- **L2 Delta**: Order book depth change events
- **NBBO**: National best bid/offer aggregation

### Derivative Events
- **Funding**: Perpetual swap funding rates
- **OpenInterest**: Aggregate open interest
- **Liquidation**: Liquidation event details

## Bytes and Precision Considerations

### Fixed-Point Encoding in DBN
DBN uses fixed-point integer encoding for decimal fields:
- Prices typically encoded at scale 1e-8 (satoshis for BTC)
- Amounts encoded with appropriate scale factor
- Timestamps encoded as nanoseconds since epoch

### Conversion to Protobuf
- Convert DBN fixed-point to Decimal strings (arbitrary precision)
- Convert nanosecond timestamps to microseconds
- Preserve scale factors in documentation

## Versioning Impact

**v1.0.0 Changes**:
- Full canonical alignment across Cryptofeed, tardis-node, and DBN
- No breaking changes from v0.2.0
- Consumers of v0.x can upgrade transparently
- New consumers can validate all three data sources

## What's Ready Now

✅ **Framework**: Complete test and infrastructure framework
✅ **Directories**: All directories created and ready
✅ **Tests**: 12 comprehensive tests ready to run
✅ **Documentation**: Templates ready for content
✅ **Tools**: Regression pipeline ready for validation

## What's Blocked

⏳ **External Specifications**: DBN YAML layout files needed
⏳ **Mapping Content**: Depends on specification availability
⏳ **Binary Samples**: Depends on sample data from DBN
⏳ **Performance Benchmarks**: Requires actual DBN serialization testing

## Next Steps

**For v1.0.0 to proceed**:
1. Obtain DBN YAML layout specifications
2. Place in `docs/schemas/examples/dbn/`
3. Re-run tests (will auto-detect specifications)
4. Complete 8.2 mapping generation
5. Complete 8.3 parity validation and release

**For now**:
- Framework is ready and waiting
- Tests will automatically validate once specifications available
- All infrastructure in place for efficient completion
- No project delays waiting for external dependencies

---

**Status**: Framework complete. Awaiting external dependencies to proceed.

**Contact**: For DBN specification access, consult:
- DBN GitHub: https://github.com/databento/dbn
- DBN Documentation: https://databento.com/docs
- Community channels for specification availability

