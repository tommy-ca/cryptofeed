# ADR-002: Remove Compatibility Shim

**Date:** 2025-11-26

**Status:** accepted

## Context

During the market-data-kafka-producer implementation, a compatibility shim was created at `cryptofeed/kafka_callback.py` to redirect imports to the new backend location (`cryptofeed/backends/kafka/callback`).

The shim serves as a temporary bridge to ease migration:

- Prevents immediate import errors for existing code
- Allows gradual migration of user codebases
- Emits deprecation warnings to guide users

However, the shim also creates problems:

- **Import Path Confusion:** Two valid paths to the same code
- **Documentation Ambiguity:** Which path should be documented?
- **Maintenance Overhead:** Must keep shim in sync with backend
- **False Sense of Compatibility:** Users may not realize they need to migrate
- **Technical Debt:** Shim becomes legacy code itself

## Decision

We will remove the compatibility shim (`cryptofeed/kafka_callback.py`) after a 90-day grace period with zero observed usage.

**Removal Plan:**

1. Monitor shim usage via deprecation warning tracking
2. Ensure migration documentation clearly shows new import path
3. Track usage statistics weekly
4. Remove shim when usage has been zero for 90 consecutive days
5. Replace shim with clear import error directing to migration guide

**Target Date:** Q2 2026 (after zero usage threshold met)

## Consequences

### Positive

- **Clear Import Paths:** Single, obvious location for Kafka backend
- **Reduced Confusion:** Documentation can reference one path
- **Less Maintenance:** No need to keep shim synchronized
- **Cleaner Architecture:** No intermediate redirection layers
- **Stronger Migration Signal:** Import errors force necessary updates

### Negative

- **Breaking Change:** Code using old import path will fail
- **Potential Disruption:** Production code may break if not migrated
- **Support Burden:** Users may need help updating import paths
- **Coordination Required:** Teams must coordinate migration timing

### Mitigations

- **90-Day Grace Period:** Plenty of time for migration after zero usage
- **Clear Deprecation Warnings:** Users warned well in advance
- **Migration Guide:** Step-by-step instructions for import updates
- **Helpful Error Messages:** Failed imports direct to migration documentation
- **Rollback Plan:** Can restore shim within 30 days if critical issues

## Alternatives Considered

### Alternative 1: Keep Shim Indefinitely

**Rationale:** Avoid breaking changes, maintain backward compatibility

**Rejected Because:**
- Creates permanent technical debt
- Confuses future users about correct import path
- Prevents clean architecture
- Adds maintenance burden forever

### Alternative 2: Remove Immediately

**Rationale:** Clean break, force migration now

**Rejected Because:**
- Too aggressive, doesn't allow for migration planning
- High risk of breaking production systems
- Community backlash likely
- Doesn't align with deprecation best practices

### Alternative 3: Version-Based Removal

**Rationale:** Remove in next major version (e.g., v3.0.0)

**Rejected Because:**
- Arbitrary timeline not tied to actual usage
- May remove too early (users still migrating)
- Or too late (already at zero usage)
- Usage-based threshold is more data-driven

## Implementation Notes

### Shim Removal Process

1. **Pre-Removal Validation:**
   - Confirm zero shim usage for 90 days
   - Verify migration documentation is current
   - Check no internal code uses shim

2. **Removal:**
   - Delete `cryptofeed/kafka_callback.py`
   - Add import hook with clear error message
   - Update all documentation references
   - Create release notes entry

3. **Post-Removal:**
   - Monitor for issues/complaints
   - Respond quickly to migration help requests
   - Prepare rollback if critical issues discovered

### Error Message Template

```python
# In cryptofeed/__init__.py or via import hook
def _kafka_callback_import_error():
    raise ImportError(
        "cryptofeed.kafka_callback has been removed. "
        "Please update imports to: from cryptofeed.backends.kafka.callback import ... "
        "See migration guide: docs/kafka/migration-guide-phase2-maintenance.md"
    )
```

## References

- Deprecation Timeline: `docs/kafka/deprecation-timeline.md`
- Migration Guide: `docs/kafka/migration-guide-phase2-maintenance.md`
- ADR-001: Deprecate Legacy Kafka Backend

## Notes

Removal of the shim is dependent on achieving zero usage. Timeline is estimated but will adjust based on actual usage statistics. Monthly progress reports will track shim usage and update removal timeline accordingly.

The 90-day zero-usage threshold ensures that removal only happens when the community has fully migrated, minimizing disruption while preventing indefinite technical debt.
