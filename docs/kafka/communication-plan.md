# Kafka Backend Deprecation Communication Plan

**Last Updated:** 2025-11-26
**Owner:** Platform Engineering Team

This document outlines the communication strategy for Kafka backend deprecation, ensuring all stakeholders are informed of timeline changes, milestones, and migration requirements.

## Communication Objectives

1. **Transparency:** Keep community informed of timeline and progress
2. **Clarity:** Provide actionable guidance at every stage
3. **Timeliness:** Communicate updates before they impact users
4. **Accessibility:** Use multiple channels to reach all stakeholders
5. **Support:** Provide resources and assistance throughout migration

## Stakeholders

### Primary Stakeholders

- **Production Users:** Teams running Cryptofeed in production
- **Development Teams:** Engineers building on Cryptofeed
- **Contributors:** Open-source contributors and maintainers
- **Documentation Users:** Developers learning from examples

### Communication Needs

| Stakeholder | Primary Need | Preferred Channel | Update Frequency |
|-------------|--------------|-------------------|------------------|
| Production Users | Stability, migration timeline | Email, GitHub Releases | Major milestones |
| Development Teams | Migration guide, code examples | Documentation, GitHub | Monthly progress |
| Contributors | Deprecation status, roadmap | GitHub Discussions | As needed |
| Documentation Users | Updated examples | Documentation | Continuous |

## Communication Channels

### 1. Documentation (Primary Channel)

**Location:** `docs/kafka/`

**Content:**
- Deprecation timeline with milestone tracking
- Migration guides with code examples
- Decision log (ADRs) for architectural choices
- Monthly progress reports

**Update Frequency:**
- Timeline: At each milestone completion
- Migration guides: As needed for clarifications
- ADRs: When significant decisions made
- Progress reports: First Monday of each month

**Responsibilities:**
- Platform Engineering Team maintains
- Technical Writing Team reviews
- Community can submit PRs for improvements

### 2. Code Warnings (Continuous Channel)

**Location:** Deprecation warning system

**Content:**
- Import-time warnings for legacy classes
- Instantiation warnings with migration guidance
- Compatibility shim warnings
- Health check warnings

**Update Frequency:**
- Continuous (emitted at runtime)
- Warning messages updated at milestone transitions

**Responsibilities:**
- Automated by deprecation warning system
- Platform Engineering Team manages warning content
- Messages link to migration documentation

### 3. Release Notes (Milestone Channel)

**Location:** `CHANGELOG.md`, GitHub Releases

**Content:**
- Major milestone completions
- Breaking change announcements
- Deprecation timeline updates
- Migration deadline reminders

**Update Frequency:**
- Every version release
- Special announcements for major deprecation events

**Responsibilities:**
- Release Manager includes in release notes
- Platform Engineering Team provides content
- Tagged releases highlight deprecation status

### 4. GitHub (Community Channel)

**Location:** Issues, Discussions, Pull Requests

**Content:**
- Migration support requests
- Bug reports related to migration
- Feature requests for migration tools
- Community questions and feedback

**Update Frequency:**
- Real-time responses to issues
- Weekly discussion monitoring
- Monthly pinned discussion updates

**Responsibilities:**
- Maintainers respond to issues
- Platform Engineering Team monitors migration questions
- Community support encouraged

### 5. Email (Critical Updates Only)

**Location:** Direct email to production users (if contact info available)

**Content:**
- Upcoming breaking changes
- Timeline extensions or accelerations
- Critical security issues requiring migration

**Update Frequency:**
- Only for critical, time-sensitive updates
- Minimum 30 days before breaking changes

**Responsibilities:**
- Platform Engineering Team drafts
- Project Lead approves
- Sent only when necessary to avoid fatigue

## Communication Timeline

### Phase 1: Deprecation Warnings (✅ Complete)

**Communications:**
- ✅ Code warnings implemented and tested
- ✅ Migration guide created
- ✅ GitHub issue created announcing deprecation
- ✅ Release notes included deprecation notice

### Phase 2: Migration Tools (✅ Complete)

**Communications:**
- ✅ CLI migration tool documented
- ✅ Examples updated to show migration
- ✅ GitHub discussion for migration questions
- ✅ Release notes highlighted migration tools

### Phase 3: Documentation (🚧 In Progress)

**Communications:**
- ✅ Comprehensive migration guide published
- 🚧 Deprecation timeline document (this document)
- 🚧 Decision log (ADRs) published
- 🚧 Progress reporting system established
- [ ] Update main README with migration notice

### Phase 4: Monitoring (✅ Complete)

**Communications:**
- ✅ Health check documentation published
- ✅ Monitoring guide for both implementations
- [ ] Dashboard templates for usage tracking

### Phase 5: Shim Removal (⏸️ Pending)

**Planned Communications:**
- [ ] 90-day notice before removal (when zero usage achieved)
- [ ] 60-day reminder
- [ ] 30-day final notice
- [ ] Email to known production users
- [ ] GitHub issue with removal timeline
- [ ] Release notes with breaking change notice
- [ ] Updated migration guide with import path changes

### Phase 6: Legacy Cleanup (⏸️ Pending)

**Planned Communications:**
- [ ] Major version announcement (v3.0.0 or similar)
- [ ] 90-day notice before legacy class removal
- [ ] Email to any remaining legacy users
- [ ] GitHub migration support period
- [ ] Final migration deadline announcement
- [ ] Post-removal support guide

## Communication Templates

### Milestone Completion Template

```markdown
## Kafka Backend Migration: [Milestone Name] Complete

We've completed the [milestone name] phase of the Kafka backend migration.

**What's New:**
- [Key deliverable 1]
- [Key deliverable 2]
- [Key deliverable 3]

**What This Means for You:**
- [Impact statement]
- [Action required/recommended]

**Timeline Update:**
- Current Phase: [Phase N]
- Next Milestone: [Milestone name] (Target: [Date])
- Overall Progress: [X]%

**Resources:**
- Migration Guide: docs/kafka/migration-guide-phase2-maintenance.md
- Timeline: docs/kafka/deprecation-timeline.md

**Questions?**
- GitHub Discussions: [link]
- Migration Support: [link to guide]
```

### Timeline Extension Template

```markdown
## Kafka Backend Migration Timeline Update

Based on usage statistics and community feedback, we're adjusting the deprecation timeline.

**Changes:**
- [Milestone Name]: [Old Date] → [New Date]
- Reason: [Clear explanation]

**Updated Timeline:**
[Include revised timeline table]

**Why This Change:**
[Detailed rationale based on data]

**What You Should Do:**
- [Recommended actions for users]
- [Updated migration deadlines if any]

**Resources:**
- Updated Timeline: docs/kafka/deprecation-timeline.md
- Migration Guide: docs/kafka/migration-guide-phase2-maintenance.md
```

### Breaking Change Notice Template

```markdown
## ⚠️ Breaking Change: [Component] Removal in [Version]

**Removal Date:** [Specific date]
**Affected Component:** [Component name and path]
**Replacement:** [New implementation path]

**What's Changing:**
[Clear description of what's being removed]

**Action Required:**
1. [Step-by-step migration instructions]
2. [Verification steps]
3. [Testing recommendations]

**Migration Support:**
- CLI Tool: python -m cryptofeed.tools.kafka_config_migrate
- Migration Guide: [link]
- Support Issues: [link]

**Rollback Plan:**
If critical issues discovered: [rollback procedure]

**Timeline:**
- T-90 days: This notice
- T-60 days: Reminder notice
- T-30 days: Final warning
- T-0 days: Breaking change deployed
```

## Communication Metrics

### Success Indicators

- **Documentation Views:** Increasing views of migration guides
- **Warning Frequency:** Decreasing legacy class warnings over time
- **Support Requests:** Decreasing migration support issues
- **Community Feedback:** Positive sentiment in discussions
- **Adoption Rate:** Increasing modern backend usage percentage

### Tracked Metrics

- Migration guide page views (monthly)
- GitHub issue velocity (migration-related)
- Deprecation warning counts (tracked automatically)
- Progress report generation (monthly)
- Timeline adherence (actual vs target dates)

## Escalation Procedures

### Critical Timeline Changes

If timeline must be extended due to critical issues:

1. **Platform Engineering Team** identifies need
2. **Project Lead** approves extension
3. **Update Timeline Document** with new dates and rationale
4. **Communicate via All Channels:**
   - Documentation update (immediate)
   - GitHub issue (within 24 hours)
   - Release notes (next release)
   - Email (if major change)
5. **Update Progress Reports** to reflect new timeline

### Emergency Rollback

If critical issues require rollback after removal:

1. **Incident declared** by on-call team
2. **Rollback executed** per ADR procedures
3. **Immediate Communication:**
   - GitHub issue (status update)
   - Code warnings (updated messages)
   - Documentation (rollback notice)
4. **Post-Mortem:**
   - Incident report published
   - Timeline adjusted
   - Prevention measures documented

## Review and Updates

This communication plan is reviewed:

- **Monthly:** During progress report generation
- **Milestone:** At each phase transition
- **As Needed:** When timeline changes or issues arise

**Last Review:** 2025-11-26
**Next Review:** 2025-12-26 (monthly)

## References

- Deprecation Timeline: `docs/kafka/deprecation-timeline.md`
- Migration Guide: `docs/kafka/migration-guide-phase2-maintenance.md`
- Decision Log: `docs/kafka/decisions/`
- Progress Reports: `docs/kafka/progress-reports/`

---

**Questions about this plan?** Open a discussion in GitHub or contact the Platform Engineering Team.
