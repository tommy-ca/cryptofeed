# Kiro Specification Critical Patterns

**Purpose:** Critical patterns that must be followed in all kiro specification workflows to ensure production readiness and prevent common pitfalls.

**Status:** Required Reading for all specification development

---

## 1. Always Run Multi-Agent Validation Before Production (ALWAYS REQUIRED)

### ❌ WRONG (Will cause documentation drift and test gaps)

```bash
# Completing implementation without validation
/kiro:spec-impl market-data-kafka-producer 1-19
# All tasks complete, tests passing
pytest tests/ -v  # 100% pass rate

# Declaring production ready without validation
git commit -m "feat: complete market-data-kafka-producer implementation"
git push
# Creating PR for production deployment
```

**Problem:** Implementation may be correct, but:
- Design.md can drift from requirements.md during development
- Tests may validate legacy behavior instead of actual defaults
- Architecture diagrams may miss recently-added features
- Performance targets may not reflect validated metrics

### ✅ CORRECT

```bash
# Phase 1-4: Implementation
/kiro:spec-impl market-data-kafka-producer 1-19
pytest tests/ -v  # Verify tests pass

# Phase 5: Multi-Agent Validation (REQUIRED before production)
/kiro:validate-design market-data-kafka-producer
# Subagent checks: requirements ↔ design alignment
# Identifies: migration strategy conflict, performance misalignment

/kiro:validate-impl market-data-kafka-producer
# Subagent checks: design ↔ implementation alignment
# Identifies: E2E test gap (validates wrong topic strategy)

# Fix all validation findings
# Update design.md to match approved requirements
# Fix E2E tests to validate actual default behavior
# Track in spec.json post_validation_refinements

# Verify fixes
pytest tests/ -v  # 100% pass rate
/kiro:spec-status market-data-kafka-producer
# Confirm: Zero validation findings, GO for production

# Now safe for production deployment
git commit -m "feat: complete market-data-kafka-producer with validation"
```

**Why:**

Multi-agent validation systematically checks alignment across all specification artifacts (requirements ↔ design ↔ implementation). Without this step:

1. **Documentation Drift**: Design.md is often drafted before requirements are finalized. When requirements change (e.g., migration strategy), design docs don't update automatically.

2. **Test Legacy Behavior**: Tests written early in development may validate legacy/experimental behavior instead of the actual production defaults.

3. **Missing Features in Diagrams**: As features are added (like message headers), architecture diagrams need updates that manual review often misses.

4. **Stale Metrics**: Performance targets from design phase may not reflect actual validated production metrics.

The validation subagents (`validate-design-agent`, `validate-impl-agent`) perform systematic line-by-line comparison that catches these gaps before production deployment.

**Placement/Context:**

- **When:** After completing all implementation tasks (Phase 1-4), before declaring "production ready" (Phase 5)
- **Required for:** All specifications marked as "execution-ready" or "production-ready"
- **Commands:**
  ```bash
  /kiro:validate-design {feature}    # Check requirements ↔ design
  /kiro:validate-impl {feature}      # Check design ↔ implementation
  ```
- **Success Criteria:** Zero validation findings, 100% test pass rate, all findings tracked in spec.json

**Documented in:** `docs/solutions/documentation-gaps/documentation-drift-spec-validation-kiro-spec-system-20251126.md`

---

## 2. Track Validation Findings in Spec.json (ALWAYS REQUIRED)

### ❌ WRONG (Will lose validation history)

```bash
# Creating standalone todo files for validation findings
echo "Fix E2E test topic naming" > todos/001-e2e-fix.md
echo "Update design.md migration" > todos/002-design-fix.md

# Fixing issues without permanent tracking
# Edit design.md, commit changes
# Delete todo files after fixes applied
rm -rf todos/

# Result: No permanent record of what was found or fixed
```

**Problem:**
- Validation findings are lost after todos are deleted
- Future audits can't see what issues were discovered
- No traceability between findings and resolution commits
- Standalone todos aren't part of specification metadata

### ✅ CORRECT

```json
// .kiro/specs/market-data-kafka-producer/spec.json
{
  "implementation_status": {
    "last_validation": "2025-11-26",
    "post_validation_refinements": {
      "date": "2025-11-26",
      "findings_addressed": 2,
      "changes": [
        "Fixed E2E test topic naming expectations (consolidated vs per-symbol)",
        "Aligned design.md with approved Blue-Green migration strategy",
        "Updated performance targets to validated 150k+ msg/s",
        "Enhanced architecture diagrams with message header specifications"
      ],
      "commit": "53f9e548",
      "test_pass_rate": "100%"
    }
  },
  "execution_readiness": {
    "validation_findings_resolved": {
      "e2e_test": "Fixed (commit 53f9e548)",
      "design_alignment": "Fixed (commit 53f9e548)",
      "test_pass_rate": "100%",
      "documentation_accuracy": "100%"
    }
  }
}
```

**Why:**

Tracking validation findings in spec.json creates permanent, auditable record of:
1. What validation discovered (specific issues)
2. How issues were resolved (specific changes)
3. When resolution occurred (date + commit hash)
4. Verification of fixes (test pass rate)

This metadata becomes part of the specification's permanent history and is visible to all reviewers and auditors.

**Placement/Context:**

- **When:** Immediately after validation subagents report findings
- **Location:** `.kiro/specs/{feature}/spec.json`
- **Sections to update:**
  - `implementation_status.post_validation_refinements`: List all findings and changes
  - `execution_readiness.validation_findings_resolved`: Status of each finding
  - `implementation_status.last_validation`: Current date
  - `implementation_status.tests_passing`: Updated count if tests added/fixed

**Documented in:** `docs/solutions/documentation-gaps/documentation-drift-spec-validation-kiro-spec-system-20251126.md`

---

## 3. Test Default Behavior, Not Legacy Options (ALWAYS REQUIRED)

### ❌ WRONG (Will validate wrong production behavior)

```python
# E2E test validates legacy per-symbol topic strategy
# tests/e2e/test_kafka_callback_e2e.py
async def test_kafka_callback_concurrent_flow_e2e():
    # ... produce messages ...

    topics = {message.topic for message in producer.messages}
    # Testing per-symbol topics (legacy behavior, not default)
    assert "cryptofeed.trades.coinbase.btc-usd" in topics
    assert "cryptofeed.trades.binance.eth-usdt" in topics
    assert len(topics) >= 2  # Expects multiple per-symbol topics
```

**Problem:**
- Test passes but validates non-default configuration
- Production deployment uses consolidated topics (default)
- Test failure won't catch if consolidated strategy breaks
- Misleading for teams expecting per-symbol as default

### ✅ CORRECT

```python
# E2E test validates actual default behavior (consolidated topics)
# tests/e2e/test_kafka_callback_e2e.py
async def test_kafka_callback_concurrent_flow_e2e():
    # ... produce messages ...

    topics = {message.topic for message in producer.messages}
    # Consolidated topic strategy (default): all trades go to single topic
    assert "cryptofeed.trade" in topics
    assert len(topics) == 1  # All messages use consolidated topic

    # Verify message headers provide routing metadata
    for message in producer.messages:
        assert message.headers and ("exchange", b"COINBASE") in message.headers
        assert ("symbol", b"BTC-USD") in message.headers or ("symbol", b"ETH-USDT") in message.headers
```

**Why:**

E2E tests must validate the production default configuration, not optional/legacy behaviors:

1. **Default Behavior = Production Reality**: Most deployments will use defaults, not custom configurations.

2. **Test Coverage Gap**: If tests validate non-default behavior, breaking changes to defaults go undetected.

3. **Documentation Alignment**: Tests serve as executable documentation. They should demonstrate the primary use case.

4. **Legacy Traps**: Tests written during development may encode experimental approaches that later changed.

**Separate tests for optional configurations:**
```python
@pytest.mark.parametrize("topic_strategy", ["consolidated", "per-symbol"])
async def test_kafka_callback_topic_strategies(topic_strategy):
    # Test both strategies explicitly
    # Mark which is default in comments
```

**Placement/Context:**

- **When:** Writing E2E and integration tests
- **Apply to:** All backend configuration (topics, partition strategies, serialization formats)
- **Check:** If `design.md` says "default", test must validate that default
- **Comments:** Add explicit comment explaining which behavior is default:
  ```python
  # Validates consolidated topic strategy (default behavior as of Phase 5)
  ```

**Documented in:** `docs/solutions/documentation-gaps/documentation-drift-spec-validation-kiro-spec-system-20251126.md`

---

## How to Use This Guide

1. **Before starting implementation:** Read all patterns to understand required workflow
2. **During implementation:** Reference patterns when making design decisions
3. **Before production:** Verify all patterns were followed (use as checklist)
4. **After issues found:** Add new patterns if they represent systematic failures

## Adding New Patterns

When adding a pattern to this file:

1. Number it sequentially (next is 4)
2. Use template from `.claude/skills/codify-docs/assets/critical-pattern-template.md`
3. Include ❌ WRONG and ✅ CORRECT code examples
4. Link to the troubleshooting doc where pattern was discovered
5. Mark as "(ALWAYS REQUIRED)" in title if non-negotiable

## Related Documentation

- **Kiro Workflow**: `.claude/commands/kiro/` (spec-init, spec-requirements, spec-design, spec-tasks, spec-impl)
- **Validation Commands**: `/kiro:validate-design`, `/kiro:validate-impl`, `/kiro:spec-status`
- **Troubleshooting**: `docs/solutions/documentation-gaps/` (specific resolution examples)
