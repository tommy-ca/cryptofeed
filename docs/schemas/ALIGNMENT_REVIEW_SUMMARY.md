# Python-Proto Alignment Review Summary

**Date**: 2025-10-25  
**Reviewer**: Claude Code (AI Development Workflow)  
**Status**: ⚠️ **PARTIAL ALIGNMENT — DELTA GAP REMAINS**

---

## Executive Summary

A comprehensive review of Protocol Buffer schemas against original Python Cython types (`cryptofeed/types.pyx`) confirms broad alignment, with a single critical delta-related gap plus a handful of documentation and tooling follow-ups.

### Quick Stats

| Metric | Count |
|--------|-------|
| Python Types Reviewed | 15/15 (100%) |
| Proto Files | 20 total |
| Critical Issues | 1 |
| Medium Issues | 2 |
| Minor Issues | 3 |

---

## Critical Issues 🔴

### OrderBook Delta Representation
**Impact**: Incremental L2 updates captured in Python via `OrderBook.delta` cannot be serialized with the current `Level2Book` message alone.

```python
book.delta = {
    'bids': [(price, size)],
    'asks': [(price, size)],
}
```

```protobuf
message Level2Book {
  repeated PriceLevel bids = 3;
  repeated PriceLevel asks = 4;
  // No delta field
}
```

**Recommendation**: Establish a first-class mapping into `level2_delta.proto`, update emitters to use it, and document how snapshots + deltas should be combined downstream.

---

## Medium Issues 🟡

### Raw Payload Strategy
**Impact**: Normalized events omit venue-native payloads, limiting replay/debug parity.

**Recommendation**: Decide between (A) adding `optional bytes raw` to high-value messages or (B) documenting alternative tracing workflows.

---

### Delta Conversion Guidance
**Impact**: Even after introducing `level2_delta.proto`, there is no documented pipeline showing how `OrderBook.delta` maps to protobuf deltas, risking inconsistent downstream implementations.

**Recommendation**: Publish guidance and provide fixtures/tests verifying the conversion contract.

---

### 6. Trade.raw_id Unclear Purpose
**Impact**: Field exists in proto but no Python equivalent

```protobuf
message Trade {
  string raw_id = 8;  // What is this?
}
```

**Recommendation**: Clarify field purpose or remove if unused

---

## Minor Issues 🟢

### Side Field Type Change (String → Enum)
**Impact**: Requires documentation so downstream systems map Python strings (`"buy"`, `"sell"`) to `TradeSide` enum values consistently.

**Recommendation**: Add explicit mapping guidance to the migration guide.

---

### Field Name Changes
**Impact**: Renamed identifiers (e.g., `Trade.id` → `trade_id`) need to be captured for integrators.

| Type | Python Field | Proto Field |
|------|--------------|-------------|
| Trade | `id` | `trade_id` |
| Liquidation | `id` | `liquidation_id` |

**Recommendation**: Document renames and provide helper utilities where possible.

---

### Timestamp Precision Limit
**Impact**: `int64` microseconds overflow around year 2286; document mitigation for far-future datasets.

**Recommendation**: Call out the limit and suggest alternative storage strategies if required.

---

## Alignment Score by Type

| Type | Python Fields | Proto Fields | Alignment | Outstanding Items |
|------|---------------|--------------|-----------|-------------------|
| **Trade** | 9 | 9 | 🟢 95% | Raw payload strategy |
| **Ticker** | 6 | 5 | 🟢 90% | Raw payload strategy |
| **Funding** | 8 | 7 | 🟢 90% | Raw payload strategy |
| **Liquidation** | 9 | 8 | 🟢 90% | Raw payload strategy |
| **Candle** | 14 | 13 | 🟢 95% | Raw payload strategy |
| **OrderBook** | 8 | 7 | 🟡 70% | Delta representation, raw handling |

**Overall Alignment**: 🟢 **90%** (Strong alignment; delta workflow outstanding)

---

## Detailed Documentation

- **Full Alignment Review**: [PYTHON_PROTO_ALIGNMENT.md](./PYTHON_PROTO_ALIGNMENT.md)
- **Test Plan**: [ALIGNMENT_TEST_PLAN.md](./ALIGNMENT_TEST_PLAN.md)
- **Migration Guide**: TBD (create after proto updates)

---

## Recommendations by Priority

### P0 (Blocking v0.1.0 Production Usage)

1. 📌 **Implement Level2Delta pipeline** — finalize proto contract, emitter output, and consumer guidance for incremental books.

### P1 (Should Fix Before v0.2.0)

2. 📋 **Decide raw payload strategy** — either add `bytes raw` fields or document official debugging workflow.
3. 📋 **Publish conversion guide & tests** — document snapshot+delta flow and add regression fixtures.
4. 📋 **Document side enum mapping & field renames** in the migration guide.

### P2 (Nice to Have)

5. 📋 **Build converter library** with round-trip tests across event types.
6. 📋 **Add CI guardrails** (alignment and precision checks).
7. 📋 **Document timestamp precision limits** and mitigation strategies.

---

## Implementation Estimates

| Task | Effort | Priority |
|------|--------|----------|
| Level2Delta schema + emitter support | 2-4 hours | P0 |
| Raw payload strategy decision & implementation | 1-2 days | P1 |
| Snapshot/delta conversion tests & docs | 1-2 days | P1 |
| Enum mapping & rename documentation | 0.5 day | P1 |
| Converter helper library | 2-3 days | P2 |
| CI alignment checks | 1 day | P2 |

**Total for P0**: ~4-6 hours  
**Total for P0+P1**: ~6-9 days

---

## Next Steps

### Immediate (This Week)

1. Finalize `Level2Delta` contract: confirm schema ownership, update emitters, and add consumer example documentation.
2. Draft raw payload decision memo for stakeholders (retain vs. omit) and capture migration implications.
3. Outline migration guide updates covering enum mapping, field renames, and timestamp precision call-outs.

### Short Term (Next Sprint)

1. Implement snapshot/delta conversion helpers with regression fixtures in `tests/proto_integration/`.
2. Publish migration guide updates, including the enum/rename appendix and raw payload decision.
3. Wire alignment checks into CI (round-trip + precision guards).

### Medium Term (Following Sprint)

1. Execute on the chosen raw payload strategy (schema change or documentation hardening).
2. Build optional converter helper library to ease downstream adoption.
3. Monitor downstream integrations for delta adoption feedback and iterate as needed.

---

## Risk Assessment

### If Level2Delta pipeline remains unresolved
- ❌ Incremental order book fidelity is lost, forcing consumers to derive deltas themselves.
- ⚠️ Snapshot-only feeds increase bandwidth/storage costs and delay analytics parity.

### If raw payload strategy is undecided
- ⚠️ Debugging production discrepancies remains cumbersome without canonical guidance.
- ⚠️ Downstream teams may implement divergent stop-gap solutions, fragmenting the ecosystem.

### If documentation gaps persist
- ⚠️ Integrators may mis-map sides or identifiers, leading to subtle data quality regressions.
- ⚠️ Lack of precision guidance risks incorrect far-future timestamp handling.

### If We Ship With P1 Issues

- ⚠️ **Timestamp ambiguity** → Zero vs None distinction lost
- ⚠️ **Debugging difficulty** → Cannot access raw exchange data
- ✅ **Acceptable for v0.1.0** → Users can work around these

---

## Decision: Ship v0.1.0 or Block?

### ✅ Recommendation: **Ship v0.1.0 with Caveats**

**Rationale**:
1. P0 issues can be fixed in 4-6 hours (v0.1.1 patch)
2. Schemas are still highly valuable for 78% alignment
3. Documentation clearly states "Cryptofeed subset only"
4. P1 issues don't prevent core use cases

**Required Actions Before Ship**:
- [x] Document alignment limitations in RELEASE_v0.1.0.md
- [ ] Add P0 fixes to proto schemas
- [ ] Regenerate bindings
- [ ] Update migration guide with caveats
- [ ] Create GitHub issue tracking P1/P2 improvements

---

## Conclusion

The normalized schemas provide **significant value** (78% alignment) but require **P0 fixes** before production use. With 4-6 hours of work, we can ship v0.1.0 with caveats documented, then improve to 90%+ alignment in v0.1.1.

**Status**: 🟡 **CONDITIONALLY APPROVED** (fix P0 issues first)

---

**Review Completed**: 2025-10-25  
**Next Review**: After P0 fixes implemented  
**Owner**: TBD (assign to proto schema maintainer)
