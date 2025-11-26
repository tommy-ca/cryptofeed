# Kafka Backend Deprecation Timeline

**Last Updated:** 2025-11-26
**Status:** In Progress (Phase 5: Documentation)

This document outlines the deprecation timeline for Cryptofeed's legacy Kafka backend (`cryptofeed.backends.kafka`) and the `cryptofeed.kafka_callback` compatibility shim.

## Executive Summary

The legacy Kafka backend is being phased out in favor of a modern, modular implementation with protobuf support and advanced features. This timeline ensures a smooth, transparent migration for all users.

**Target Completion:** Q2 2026 (estimated)
**Migration Support Period:** 6 months minimum
**Deprecation Warning Active:** Yes

## Deprecation Timeline

| Phase | Description | Target Date | Status | Progress |
|-------|-------------|-------------|--------|----------|
| deprecation_warnings | Implement deprecation warning system for legacy classes | 2025-12-10 | ✅ complete | 100% |
| migration_tools | Build configuration migration and validation system | 2025-12-24 | ✅ complete | 100% |
| documentation | Create migration documentation and user guidance | 2026-01-07 | 🚧 in_progress | 90% |
| monitoring | Implement health monitoring and metrics collection | 2026-01-21 | ✅ complete | 100% |
| shim_removal | Remove compatibility shim (kafka_callback.py) | 2026-05-12 | ⏸️ pending | 0% |
| legacy_cleanup | Remove legacy Kafka backend classes | 2026-05-26 | ⏸️ pending | 0% |

## Phase Details

### Phase 1: Deprecation Warnings (✅ Complete)

**Duration:** Weeks 1-2
**Completion:** 100%

**Deliverables:**
- Centralized deprecation warning service
- Import-time and instantiation warnings for legacy classes
- Compatibility shim warnings
- Usage tracking and analytics

**Validation:**
- All legacy classes emit warnings with migration guidance
- Shim imports show clear path to modern backend
- Usage metrics collection operational

### Phase 2: Migration Tools (✅ Complete)

**Duration:** Weeks 3-4
**Completion:** 100%

**Deliverables:**
- Configuration parsing and translation engine
- Functional equivalence validation
- CLI migration tool (`kafka_config_migrate`)
- Automated configuration backup and rollback

**Validation:**
- Configuration translator handles all legacy options
- Validation confirms functional equivalence
- CLI tool tested with real-world configurations

### Phase 3: Documentation (🚧 In Progress)

**Duration:** Weeks 5-6
**Completion:** 90%

**Deliverables:**
- Comprehensive migration guide with code examples
- Troubleshooting guide for common issues
- API documentation for legacy and modern backends
- Deprecation timeline and communication plan (this document)

**Remaining Work:**
- Decision log (ADR) creation
- Progress reporting automation
- Final documentation review

### Phase 4: Monitoring (✅ Complete)

**Duration:** Weeks 7-8
**Completion:** 100%

**Deliverables:**
- Health check system for both implementations
- Usage tracking and analytics dashboard
- Alerting and escalation procedures
- Performance benchmarking

**Validation:**
- Health checks operational for legacy and modern backends
- Usage metrics tracked separately
- Alerts configured and tested

### Phase 5: Shim Removal (⏸️ Pending)

**Duration:** Weeks 9-10 (Target: Q2 2026)
**Completion:** 0%

**Prerequisites:**
- Zero observed shim usage for 90 consecutive days
- Migration documentation complete and validated
- Health monitoring shows >90% modern backend adoption

**Deliverables:**
- Remove `cryptofeed/kafka_callback.py`
- Update all internal references
- Add import errors with migration guidance
- Release notes and deprecation announcement

**Rollback Plan:**
- Git revert available for 30 days post-removal
- Emergency compatibility shim restoration procedure documented

### Phase 6: Legacy Cleanup (⏸️ Pending)

**Duration:** Weeks 11-12 (Target: Late Q2 2026)
**Completion:** 0%

**Prerequisites:**
- Shim removal complete and stable
- Zero observed legacy class usage for 90 consecutive days
- Major version release prepared

**Deliverables:**
- Remove legacy backend classes (TradeKafka, BookKafka, etc.)
- Archive legacy tests and documentation
- Update import paths and error messages
- Release notes for major version

**Rollback Plan:**
- Major version rollback procedure
- Git tag for last version with legacy support

## Communication Plan

### Communication Channels

Timeline updates are communicated through:

1. **Documentation** (Primary)
   - This timeline document
   - Migration guide updates
   - Release notes

2. **Code Warnings** (Continuous)
   - Deprecation warnings at import/instantiation
   - CLI migration tool messages
   - Health check warning outputs

3. **Release Notes** (Major Milestones)
   - Version release announcements
   - Changelog entries
   - Breaking change notices

### Update Frequency

- **Weekly:** Usage statistics review (internal)
- **Monthly:** Progress report publication
- **Milestone:** Timeline document update
- **On-Demand:** Emergency adjustments or issues

## Success Criteria

### Migration Success Metrics

- **Modern Backend Adoption:** >90% of tracked usage
- **Legacy Class Usage:** <5% of total Kafka backend instantiations
- **Shim Usage:** 0% for 90 consecutive days before removal
- **Migration Support Requests:** Trending downward
- **Zero Critical Regressions:** No production-blocking issues

### Timeline Adjustment Triggers

Timeline may be extended if:

- Modern backend adoption <50% at shim removal target date
- Critical bugs discovered in modern backend
- Major user feedback indicates need for more time
- Usage statistics show insufficient migration progress

## Rollback Procedures

### Shim Removal Rollback

If critical issues discovered within 30 days of shim removal:

1. Git revert the removal commit
2. Restore shim to codebase
3. Update deprecation warnings with new timeline
4. Communicate rollback to users
5. Investigate and fix root cause

**Maximum Time:** 4 hours from rollback decision

### Legacy Cleanup Rollback

If major version adoption blocked by legacy removal:

1. Revert to previous major version tag
2. Restore legacy classes from Git history
3. Issue patch release with legacy support
4. Extended support period (minimum 3 months)

**Maximum Time:** 24 hours from rollback decision

## Usage Statistics

Statistics are tracked automatically via:

- Deprecation warning emission counts
- Import path usage analysis
- Health check backend type tracking
- Configuration migration tool usage

**Reporting:** Monthly progress reports published to docs/kafka/progress-reports/

## Decision Log

Architectural decisions for Kafka backend evolution are tracked in ADR (Architecture Decision Record) format.

**Location:** `docs/kafka/decisions/`

**Current Decisions:**
- ADR-001: Deprecate Legacy Kafka Backend
- ADR-002: Remove Compatibility Shim
- ADR-003: Modern Backend as Default

See individual ADR files for full context and rationale.

## Contact and Support

### Migration Support

- **Documentation:** `docs/kafka/migration-guide-phase2-maintenance.md`
- **CLI Tool:** `python -m cryptofeed.tools.kafka_config_migrate --help`
- **Health Check:** `cryptofeed.backends.kafka.health`

### Questions and Issues

- **GitHub Issues:** Report bugs or migration blockers
- **Discussions:** Community support and questions
- **Email:** For critical production issues (see SUPPORT.md)

## References

- Migration Guide: `docs/kafka/migration-guide-phase2-maintenance.md`
- Health Monitoring: `cryptofeed/backends/kafka/health.py`
- Decision Log: `docs/kafka/decisions/`
- Configuration Migration: `cryptofeed/backends/kafka/migration.py`

---

**Note:** This timeline is subject to change based on usage statistics and community feedback. Check the "Last Updated" date and monitor release notes for the latest information.
