# ADR-003: Modern Backend as Default

**Date:** 2025-11-26

**Status:** accepted

## Context

With the completion of the market-data-kafka-producer specification, Cryptofeed now has two Kafka backend implementations:

- **Legacy:** `cryptofeed.backends.kafka` (BackendQueue-based, JSON)
- **Modern:** `cryptofeed.backends.kafka.*` (Modular, protobuf, advanced features)

Documentation, examples, and tutorials need to choose which implementation to showcase. The default recommendation influences new user adoption and sets expectations for the ecosystem.

Current situation:

- **Examples:** Mix of legacy and modern
- **Documentation:** Primarily legacy-focused
- **New Users:** May not realize modern backend exists
- **Best Practices:** Unclear which implementation to use

## Decision

The modern Kafka backend (`cryptofeed.backends.kafka.callback`) will be the **default recommendation** for all new code, documentation, and examples.

**Implementation:**

1. **Documentation Priority:**
   - Modern backend shown first in all guides
   - Legacy backend marked as "deprecated"
   - Migration guide linked prominently

2. **Examples:**
   - Update all examples to use modern backend
   - Archive legacy examples to `examples/legacy/`
   - Add modern backend quickstart example

3. **Error Guidance:**
   - Deprecation warnings recommend modern backend
   - Error messages suggest modern implementation
   - CLI tools default to modern configuration

4. **API Documentation:**
   - Modern backend gets primary coverage
   - Legacy backend in "Deprecated APIs" section
   - Clear migration paths shown

## Consequences

### Positive

- **Clear Direction:** New users start with best-practice implementation
- **Faster Modern Adoption:** Default recommendation drives usage
- **Better First Impression:** New users see advanced features
- **Simpler Documentation:** Focus on single implementation
- **Ecosystem Alignment:** Third-party tools follow our lead

### Negative

- **Existing User Confusion:** May wonder if they need to migrate immediately
- **Documentation Churn:** Many files need updates
- **Example Maintenance:** Must update and test all examples
- **Backward Compatibility Questions:** Users may feel pressured

### Mitigations

- **Clear Migration Timeline:** Existing users see when they need to migrate
- **"Legacy Still Supported" Notice:** Reassure users they have time
- **Gradual Documentation Update:** Prioritize high-traffic pages
- **Version Tags:** Examples tagged with supported versions
- **FAQ Section:** Address "Do I need to migrate now?" questions

## Alternatives Considered

### Alternative 1: Neutral Documentation

**Rationale:** Present both equally, let users choose

**Rejected Because:**
- Confuses new users (which should I use?)
- Dilutes effort across two implementations
- Doesn't communicate deprecation clearly
- Slows modern adoption

### Alternative 2: Hide Legacy Completely

**Rationale:** Remove all legacy references immediately

**Rejected Because:**
- Existing users need legacy documentation
- Too aggressive during transition period
- May break workflows for users not ready to migrate
- Doesn't align with deprecation timeline

### Alternative 3: Wait Until Legacy Removal

**Rationale:** Keep documentation neutral until legacy is removed

**Rejected Because:**
- Slows modern adoption unnecessarily
- Misses opportunity to guide new users
- Conflicts with goal of phasing out legacy
- Extends transition period

## Implementation Checklist

### High Priority (Complete by Phase 3)

- [x] Update main README.md with modern backend examples
- [x] Create modern backend quickstart guide
- [x] Mark legacy backend as deprecated in docs
- [x] Update migration guide with clear timelines
- [ ] Add deprecation notices to legacy documentation
- [ ] Update API documentation priority

### Medium Priority (Complete by Phase 5)

- [ ] Migrate all examples/ scripts to modern backend
- [ ] Archive legacy examples to examples/legacy/
- [ ] Update all tutorial documentation
- [ ] Add modern backend to integration guides

### Low Priority (Complete by Phase 6)

- [ ] Video tutorials featuring modern backend
- [ ] Blog post announcing modern backend as default
- [ ] Community communication campaign

## References

- Modern Backend Implementation: `cryptofeed/backends/kafka/callback.py`
- Legacy Backend (deprecated): `cryptofeed.backends.kafka`
- Migration Guide: `docs/kafka/migration-guide-phase2-maintenance.md`
- ADR-001: Deprecate Legacy Kafka Backend
- ADR-002: Remove Compatibility Shim

## Notes

This decision supports the broader deprecation strategy by guiding new users to the modern implementation while maintaining support for existing legacy users during the transition period.

Setting the modern backend as default in documentation is a low-risk, high-impact change that accelerates ecosystem adoption without forcing immediate migration for existing users.
