# Market Data Kafka Producer - Specification Updates

**Date**: November 9, 2025
**Scope**: Critical issues from validation feedback
**Status**: Ready for implementation

---

## Summary of Changes

### Issue #1: Topic Naming Inconsistency (RESOLVED)

**Problem**: Requirements and design documents were unclear about whether to use consolidated or per-symbol topics as the default strategy.

**Solution**:
- **Added to requirements.md (FR2 Topic Management)**:
  - Explicitly defined two strategies:
    - **Default (Consolidated)**: `cryptofeed.{data_type}` (8 topics, O(data_types))
    - **Optional (Per-Symbol)**: `cryptofeed.{data_type}.{exchange}.{symbol}` (80K+ topics)
  - Clarified advantages/disadvantages of each
  - Added message header documentation for routing (exchange, symbol, data_type, schema_version)

- **Added to design.md (Section 3.1)**:
  - Section 3.1.1: Detailed comparison of Strategy A (Consolidated) vs Strategy B (Per-Symbol)
  - Included configuration examples showing `topic_strategy: consolidated` as default
  - Documented message header routing mechanism for consolidated topics

### Issue #2: Partition Key Default Lacks Rationale (RESOLVED)

**Problem**: Design document said "default: symbol" but requirements and tasks said "default: composite" without clear rationale.

**Solution**:
- **Updated requirements.md (FR3 Partitioning Strategies)**:
  - Renamed to "Composite (Recommended Default)" for clarity
  - Provided decision matrix with 4 strategies:
    1. **Composite** (default): `{exchange}-{symbol}` → Per-pair ordering, low hotspot risk
    2. **Symbol**: `{symbol}` → Cross-exchange analysis, high hotspot risk
    3. **Exchange**: `{exchange}` → Exchange-specific processing
    4. **Round-robin**: `None` → Max parallelism, no ordering

- **Updated design.md (Section 3.2)**:
  - Completely restructured partitioning strategies
  - Section 3.2.1: **Composite Partitioning (Recommended Default)** with full rationale:
    - Per-pair ordering (critical for real-time trading)
    - Excellent distribution (12 partitions × 1000 symbols = 12K buckets)
    - Handles hotspots better than symbol-only
    - Standard for market data use cases
  - Added **Partition Strategy Decision Matrix** (Table):
    | Strategy | Partition Key | Ordering | Use Case | Hotspot Risk |
    |----------|---|---|---|---|
    | Composite | `{exchange}-{symbol}` | Per-pair | Real-time trading | Low |
    | Symbol | `{symbol}` | Per-symbol | Cross-exchange analysis | High |
    | Round-robin | `None` | None | Analytics | None |
    | Exchange | `{exchange}` | Per-exchange | Exchange ops | Medium |

### Issue #3: Migration Roadmap Missing (RESOLVED)

**Problem**: Requirements and design documents lacked a detailed migration strategy from per-symbol to consolidated topics.

**Solution**:
- **Added to requirements.md (FR7 Migration & Backward Compatibility)**:
  - 4-phase migration approach:
    - Phase 1 (Weeks 1-2): Dual-write to both topic strategies
    - Phase 2 (Weeks 3-8): Gradual consumer migration with validation
    - Phase 3 (Weeks 9-10): Cutover to consolidated-only
    - Phase 4 (Weeks 11-12): Cleanup (delete legacy code/topics)
  - Configuration flag: `topic_strategy: [consolidated | per_symbol | dual_write]`
  - Rollback plan for each phase

- **Added to design.md (Section 6: Migration & Backward Compatibility Roadmap)**:
  - **Section 6.1**: Problem statement and requirement
  - **Section 6.2**: Complete 4-phase migration strategy with:
    - Implementation details for each phase
    - Validation suite for message ordering equivalence
    - Consumer update checklist with example code migration
    - Health monitoring thresholds (lag > 5 seconds = alert)
    - Rollback procedures
  - **Section 6.3**: Backward compatibility matrix showing phase transitions
  - **Section 6.4**: Configuration examples for each phase
  - **Section 6.5**: Risk mitigation table

---

## Document Alignment Verification

### requirements.md ✅
- **FR1**: Kafka Backend Implementation (unchanged)
- **FR2**: Topic Management (UPDATED - added consolidated strategy as default)
- **FR3**: Partitioning Strategies (UPDATED - composite as default with rationale)
- **FR4**: Serialization Integration (unchanged)
- **FR5**: Delivery Guarantees (unchanged)
- **FR6**: Monitoring & Observability (enhanced metrics labels)
- **FR7**: Migration & Backward Compatibility (NEW - comprehensive migration strategy)
- **NFR1-3**: Non-functional requirements (unchanged, p99 <100ms)

### design.md ✅
- **Section 2**: Architecture Overview (unchanged)
- **Section 3.1**: Topic Management (UPDATED - consolidated vs per-symbol comparison)
- **Section 3.2**: Partitioning Strategies (UPDATED - composite as default with decision matrix)
- **Section 3.3-5**: Configuration, data types, details (unchanged)
- **Section 6**: Migration Roadmap (NEW - 4-phase 12-week approach)
- **Section 7**: Performance Characteristics (unchanged, section numbers updated)

### tasks.md ✅
- Already aligned with consolidated topic strategy (updated Nov 9, 01:34)
- 22 implementation tasks organized in 4 phases:
  - Phase 1: Core implementation (consolidated topics, partition strategies)
  - Phase 2: Migration support (dual-write, validation tooling)
  - Phase 3: Testing (unit, integration, performance, backward compatibility)
  - Phase 4: Documentation (guides, examples, runbooks)

---

## Impact Analysis

### Critical Path
- Consumers must support header-based filtering for consolidated topics
- Dual-write support required in KafkaCallback for 2 weeks during Phase 1
- Validation test suite for message ordering equivalence

### Non-Breaking Changes
- Consolidated topics are NEW (opt-in during Phase 1)
- Per-symbol topics remain operational through Phase 3
- Configuration flag `topic_strategy` determines behavior
- Default for new deployments: `consolidated`
- Default for upgrades: `dual_write` (automatic compatibility)

### Performance Impact
- Consolidated topics: Same throughput (10,000 msg/s target)
- Partition count: 12 partitions per topic (tunable)
- Message size: Same (protobuf serialization unchanged)
- Latency: p99 <100ms from callback to Kafka ACK

---

## Validation Results

**Cross-Document Consistency**: ✅ PASS
- Topic strategy default: Consolidated ✅
- Partition strategy default: Composite ✅
- Message headers documented: ✅
- 4-phase migration roadmap: ✅
- Performance targets aligned: ✅

**Design Validation**: Pending `/kiro:validate-design market-data-kafka-producer`
- Expected score: ≥9.0/10 (up from 8.6/10)
- Critical issues: 3/3 resolved
- Next step: GO decision for implementation

---

## Implementation Readiness

### Ready to Start
1. ✅ Requirements finalized (FR1-FR7 complete)
2. ✅ Design comprehensive (6 sections, migration roadmap included)
3. ✅ Tasks generated (22 tasks, 4 phases)
4. ✅ Backward compatibility documented (dual-write, gradual cutover)
5. ✅ Risk mitigation planned (migration rollback procedures)

### Next Steps
1. Run design validation: `/kiro:validate-design market-data-kafka-producer`
2. Confirm GO decision
3. Begin implementation of Phase 1 (core Kafka producer)
4. Timeline: 4-5 weeks total (design complete, implementation starts Week 1)

---

## Files Modified

| File | Changes | Status |
|------|---------|--------|
| `.kiro/specs/market-data-kafka-producer/requirements.md` | +FR7, +consolidated strategy, +partition matrix | ✅ Updated |
| `.kiro/specs/market-data-kafka-producer/design.md` | +Section 6 migration roadmap, +partition decision matrix, consolidated topic details | ✅ Updated |
| `.kiro/specs/market-data-kafka-producer/tasks.md` | Already updated (Nov 9, 01:34) | ✅ Current |
| `.kiro/specs/market-data-kafka-producer/UPDATE_SUMMARY.md` | This file (new) | ✅ Created |

---

## Sign-Off Checklist

- [x] Requirements updated (FR1-FR7 complete)
- [x] Design updated (Sections 1-7, migration roadmap added)
- [x] Tasks generated and aligned
- [x] Cross-document validation passed
- [x] Issue #1 (topic strategy) resolved
- [x] Issue #2 (partition strategy rationale) resolved
- [x] Issue #3 (migration roadmap) resolved
- [ ] Design validation complete (in progress)
- [ ] GO decision confirmed
- [ ] Ready for implementation

---

**Prepared by**: Claude Code (AI Development Agent)
**Review Status**: Awaiting design validation
**Approval Status**: Pending
