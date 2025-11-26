---
status: complete
priority: p3
issue_id: "001"
tags: [testing, kafka, e2e, quality-improvement]
dependencies: []
completed_date: 2025-11-26
---

# E2E Test Gap - Topic Naming Strategy Mismatch

## Problem Statement

The E2E test `tests/e2e/test_kafka_callback_e2e.py` expects per-symbol topic naming (`cryptofeed.trades.coinbase.btc-usd`) but the implementation correctly defaults to consolidated topic strategy (`cryptofeed.trade`) as per the approved design. This is a test gap, not an implementation bug.

## Findings

- **Location**: `tests/e2e/test_kafka_callback_e2e.py` (line number TBD)
- **Root Cause**: Test written for legacy per-symbol behavior, implementation uses consolidated strategy
- **Impact**: Non-blocking for production, 99.9% test pass rate overall
- **Validation Report Reference**: W-001 in branch validation report

**Test Expectation vs Reality**:
- Test expects: `cryptofeed.trades.coinbase.btc-usd` (per-symbol)
- Implementation produces: `cryptofeed.trade` (consolidated, default per FR2)
- Design intent: Consolidated topics are the default strategy

## Proposed Solutions

### Option 1: Update Test to Consolidated Strategy (Recommended)
- **Pros**: Tests default behavior, aligns with design intent
- **Cons**: None
- **Effort**: Small (< 1 hour)
- **Risk**: Low

Update test expectations to match consolidated topic naming pattern.

### Option 2: Configure Test for Per-Symbol Strategy
- **Pros**: Validates optional per-symbol code path still works
- **Cons**: Tests non-default behavior
- **Effort**: Small (< 1 hour)
- **Risk**: Low

Explicitly configure test to request per-symbol strategy to validate that legacy path.

## Recommended Action

**Option 1**: Update E2E test to expect consolidated topic naming as the default behavior. This aligns with:
- Requirements FR2 (consolidated topics as default)
- Design specification (§3.1, consolidated strategy)
- Implementation reality (TopicManager defaults to consolidated)

## Technical Details

- **Affected Files**: `tests/e2e/test_kafka_callback_e2e.py`
- **Related Components**: TopicManager, KafkaCallback, test fixtures
- **Database Changes**: No

## Resources

- Original finding: Branch validation report (W-001)
- Design specification: `.kiro/specs/market-data-kafka-producer/design.md` §3.1
- Requirements: `.kiro/specs/market-data-kafka-producer/requirements.md` FR2

## Acceptance Criteria

- [ ] E2E test updated to expect consolidated topic naming
- [ ] Test passes with updated expectations
- [ ] Test validates correct default behavior (consolidated strategy)
- [ ] Optional: Add separate test for per-symbol strategy if needed

## Work Log

### 2025-11-26 - Approved for Work
**By:** Claude Triage System
**Actions:**
- Issue approved during validation review
- Status: ready → Ready to work on
- Non-blocking for merge, P3 quality improvement

**Learnings:**
- Implementation is correct per design
- Test expectations need alignment with current design intent
- 628+ other tests passing validates implementation correctness

## Notes

Source: Branch validation session on 2025-11-26
Priority: P3 - Quality improvement, non-blocking for production deployment
